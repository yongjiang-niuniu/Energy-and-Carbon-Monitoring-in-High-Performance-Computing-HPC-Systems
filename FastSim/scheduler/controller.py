# MIT License
#
# Copyright (c) 2023-2025 Hewlett Packard Enterprise Development LP 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os, time, copy
import datetime; from datetime import timedelta
from collections import defaultdict, Counter, OrderedDict
import dill as pickle

import pandas as pd

from config import get_config
from partition import Partitions
from job_queue import Queue, JobState
from priority_sorters import MFPrioritySorter
from fairshare import FairTree
from data_reader import SlurmDataReader


class Controller:
    def __init__(self, config_file):
        self.config = get_config(config_file)

        self.data_reader = SlurmDataReader(
            self.config.slurm_conf, self.config.node_events_dump, self.config.resv_dump,
            self.config.job_dump, self.config.qos_dump
        )

        df_jobs = self.data_reader.get_cleaned_job_df(self.config.considered_partitions, 550)

        ret = self.data_reader.get_nodes_partitions(
            self.config.considered_partitions, self.config.hpe_restrictlong_sliding_reservations,
            df_jobs.End.max(), self.config.nodes_down_in_blades
        )
        nid_data, partition_data, valid_resv, hpe_restrictlong = ret

        self.init_time = df_jobs.Start.min()
        self.time = self.init_time

        qos_data = self.data_reader.get_qos()

        self.partitions = Partitions(nid_data, partition_data)

        active_usrs = { row.User for _, row in df_jobs.iterrows() }
        self.fairtree = FairTree(
            self.config.assocs_dump, self.config.PriorityCalcPeriod,
            self.config.PriorityDecayHalfLife, self.init_time, active_usrs,
            self.config.approx_excess_assocs, self.partitions
        )

        priority_sorter = MFPrioritySorter(
            self.init_time, self.config.PriorityWeightJobSize, self.config.PriorityWeightAge,
            self.config.PriorityWeightFairshare, self.config.PriorityMaxAge,
            self.config.PriorityWeightPartition, self.config.PriorityWeightQOS,
            len({ partition.priority_tier for partition in self.partitions.partitions }) == 1,
            self.fairtree, len(self.partitions.nodes)
        )

        self.queue = Queue(
            df_jobs, self.partitions.partitions_by_name, qos_data, valid_resv, priority_sorter
        )

        # Don't start scheduling until a full system worth of jobs has been started in the data.
        # This should give an ok estimate of the initial state of the system at the sim's start.
        self.sched_start = self.init_time
        nodes, init_phase_jobs = len(self.partitions.nodes), set()
        for job in sorted(self.queue.all_jobs, key=lambda job: (job.true_job_start, job.uniq_id)):
            nodes -= job.nodes
            init_phase_jobs.add(job)
            if nodes <= 0:
                self.sched_start = job.true_job_start
                break

        for job in init_phase_jobs:
            job.ignore_in_eval = True
            new_runtime = job.runtime - (self.sched_start - job.true_job_start)
            if new_runtime <= timedelta():
                self.queue.all_jobs.remove(job)
            else:
                job.runtime = new_runtime

        self.num_sched_test_step = 0
        self.num_bf_test_step = 0

        self.times = [self.time]
        self.power_usage = 0
        self.total_energy = 0.0

        self.job_history = []
        self.running_jobs = []

        self.down_nodes = []
        self.node_down_order = sorted(
            [ node for node in self.partitions.nodes if node.down_schedule ],
            key=lambda node: (node.down_schedule[-1][0], node.nid),
            reverse=True
        )
        self.reserved_nodes = []
        self.node_reservation_order = sorted(
            [ node for node in self.partitions.nodes if node.reservation_schedule ],
            key=lambda node: (node.reservation_schedule[-1][0], node.nid),
            reverse=True
        )

        # XXX ARCHER2 specific: sliding maintenance window
        if self.config.hpe_restrictlong_sliding_reservations == "":
            self.sliding_reservations = []

        else:
            nid_to_node = { node.nid : node for node in self.partitions.nodes }
            self.sliding_reservations = [
                [
                    submitted, submitted, submitted + timedelta(hours=1),
                    submitted + timedelta(hours=1, minutes=5),
                    submitted + timedelta(days=365, hours=1, minutes=5),
                    [ nid_to_node[nid] for nid in hpe_restrictlong[submitted] ],
                    "HPE_RestrictLongJobs"
                ]
                for submitted in sorted(hpe_restrictlong)
                    if submitted >= self.init_time - timedelta(hours=1)
            ]
            self.sliding_reservations.sort(key=lambda res: res[0], reverse=True)

        self.step_cnt = 0
        self.previous_print_hour = self.time.hour

        self.sched_backfill_num = 0
        self.sched_main_num = 0

        # TODO Refactor
        # These are all backfilling parameters. Should put backfiller into its own class since
        # it needs its own state.
        self.bf_free_blocks = None
        self.bf_window = self.config.bf_window.total_seconds()
        self.bf_end_padding = (self.config.OverTimeLimit + self.config.KillWait).total_seconds()
        self.bf_resolution = self.config.bf_resolution.total_seconds()
        self.bf_max_relevant_start = (
            (self.config.bf_max_time - self.config.bf_yield_interval).total_seconds()
        )
        self.bf_loop_active = False
        self.bf_try_per_lock_hold = int(
            self.config.bf_yield_interval.total_seconds() * self.config.approx_bf_try_per_sec
        )
        self.bf_max_lock_holds = int(
            self.config.bf_max_time / (self.config.bf_yield_interval + self.config.bf_yield_sleep)
        )
        self.bf_locks_remaining = 1
        self.bf_time = None
        self.bf_nodes_free_now_max_reqtimes = None
        self.bf_max_reqtime = None
        self.bf_queue_min_reqtime = None
        self.bf_job_reqtimes = None
        self.bf_secs_past = None

    def _next_job_finish(self):
        if not self.running_jobs:
            return datetime.datetime.max

        return self.running_jobs[-1].end

    def run_sim(self, max_steps=0):
        sim_start = time.time()

        # NOTE Assuming: defer,bf_continue are always set. I think this is true for large systems
        previous_small_sched = self.sched_start
        next_bf_time = self.sched_start + self.config.bf_interval
        next_sched_time = self.sched_start + self.config.sched_interval
        small_sched_waiting_time = None
        next_fairtree_time = self.time + self.config.PriorityCalcPeriod
        while self.queue.all_jobs or self.queue.queue or self.running_jobs:
            self.time = min(
                next_bf_time, next_sched_time, next_fairtree_time, self._next_job_finish(),
                self.queue.next_newjob()
            )

            if small_sched_waiting_time is not None:
                if self.time >= small_sched_waiting_time:
                    self.time = small_sched_waiting_time

            sched, sched_depth, bf, fairtree = False, None, False, False

            if self.time == next_bf_time:
                bf = True
                next_bf_time += self.config.bf_yield_interval + self.config.bf_yield_sleep
                next_bf_sleep = self.time + self.config.bf_yield_interval

            if self.time == next_fairtree_time:
                fairtree = True
                next_fairtree_time += self.config.PriorityCalcPeriod

            if self.time == next_sched_time:
                if self.bf_loop_active and self.time < next_bf_sleep:
                    next_sched_time = next_bf_sleep
                else:
                    sched = True
                    small_sched_waiting_time = None
                    next_sched_time += self.config.sched_interval
            elif (
                self.time == small_sched_waiting_time or
                (
                    self.time > previous_small_sched + self.config.sched_min_interval and
                    self.time == self._next_job_finish()
                )
            ):
                if self.bf_loop_active and self.time < next_bf_sleep:
                    small_sched_waiting_time = next_bf_sleep
                else:
                    sched = True
                    sched_depth = self.config.default_queue_depth
                    previous_small_sched = self.time
                    small_sched_waiting_time = None

            self._step(sched, sched_depth, bf, fairtree)
            self.step_cnt += 1

            if bf and not self.bf_loop_active:
                next_bf_time += self.config.bf_interval

            if max_steps and self.step_cnt > max_steps:
                break

            print("Step: {}".format(self.step_cnt), end='\r')

        elapsed = time.time() - sim_start
        print(
            "Sim complete in {} hr {} mins".format(
                int(elapsed // (60 * 60)), int((elapsed % (60 * 60)) // 60)
            )
        )

    def _step(self, sched, sched_depth, bf, fairtree):
        self._check_finished_jobs()

        self._check_down_nodes()
        self._check_reservations()

        self.queue.step(self.time, self.running_jobs)

        if fairtree:
            self.fairtree.fairshare_calc(self.running_jobs, self.time)

        pre_submit_running_jobs_len = len(self.running_jobs)

        if sched:
            self.num_sched_test_step = 0
            sched_depth = sched_depth if sched_depth is not None else (
                sum(len(res_queue) for res_queue in self.queue.reservations.values()) +
                len(self.queue.queue)
            )

            self._sched_reservations(sched_depth)

            self._sched_main(sched_depth)

        if bf:
            if not self.bf_loop_active:
                self._prep_new_bf()
                self.bf_loop_active = True

            self._backfill(self.bf_try_per_lock_hold)

        if len(self.running_jobs) != pre_submit_running_jobs_len:
            self.running_jobs.sort(key=lambda job: job.end, reverse=True)

        self.times.append(self.time)

        self._print_stats(sched, bf)

    def _submit(self, job, nodes):
        self.running_jobs.append(job)
        self.power_usage += job.true_node_power * job.nodes / 1e+6
        self.total_energy += (
            job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
        )

        if job.assigned_nodes:
            raise Exception("UUGGHGHG")
        if len(nodes) != job.nodes:
            raise Exception("bruh")

        for node in nodes:
            self.partitions.remove_free_block(node)
            node.interval_times[0] = job.endlimit
            self.partitions.add_free_block(node)
            job.assign_node(node)

        return True

    def _check_finished_jobs(self):
        while self.running_jobs and self.running_jobs[-1].end <= self.time:
            job = self.running_jobs.pop()
            self._end_job(job)

    def _end_job(self, job):
        self.fairtree.job_finish_usage_update(job, self.time)
        self.job_history.append(job)
        self.power_usage -= job.true_node_power * job.nodes / 1e+6
        self.total_energy += (
            job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
        )
        job.end_job()
        for node in job.assigned_nodes:
            # Down nodes dont exist to free_blocks
            if node.down:
                continue
            self.partitions.remove_free_block(node)
            node.interval_times[0] = self.time
            self.partitions.add_free_block(node)

    def _sched_reservations(self, sched_depth):
        for reservation, res_queue in self.queue.reservations.items():
            if not res_queue:
                continue

            free_nodes_ready_now = [
                node
                for interval, nodes in self.partitions.free_blocks[reservation].items()
                    if interval[0] <= self.time
                    for node in nodes
                        if node.running_job is None
            ]
            free_nodes_ready_now.sort(key=lambda node: (node.weight, node.nid))

            if not free_nodes_ready_now:
                continue

            jobs_submitted, jobs_cancelled, i_job = [], [], len(res_queue)

            for job in reversed(res_queue):
                i_job -= 1

                if self.num_sched_test_step >= sched_depth:
                    continue
                self.num_sched_test_step += 1

                if job.qos.hold_job(job):
                    continue

                job_end = self.time + job.reqtime

                n_nodes, valid_nodes = 0, set()
                
                for node in free_nodes_ready_now:
                    if job_end > node.interval_times[-1]:
                        continue

                    valid_nodes.add(node)
                    if len(valid_nodes) >= job.nodes:
                        break

                if len(valid_nodes) == job.nodes:
                    if job.cancel is not None:
                        jobs_cancelled.append(i_job)
                        jobs_submitted = [ i - 1 for i in jobs_submitted ]
                        continue

                    self._submit(job.start_job(self.time), valid_nodes)
                    jobs_submitted.append(i_job)

                else:
                    # Would still need to iterate over the remaining jobs and confirm we cant
                    # schedule anymore from its reservation
                    self.num_sched_test_step += i_job 
                    break

                for node in valid_nodes:
                    free_nodes_ready_now.remove(node)

            self.sched_main_num += len(jobs_submitted)

            for i_job in jobs_cancelled:
                cancelled_job = res_queue.pop(i_job)
                cancelled_job.cancel_job()
                self.queue.jobs_to_cancel.remove(cancelled_job)

            for i_job in jobs_submitted:
                res_queue.pop(i_job)

    def _sched_main(self, sched_depth):
        free_nodes_ready_now = [
            node
            for interval, nodes in self.partitions.free_blocks[""].items()
                if interval[0] <= self.time
                for node in nodes
                    if node.running_job is None
        ]
        if not free_nodes_ready_now:
            return

        jobs_submitted, jobs_cancelled, partitions_failed = [], [], set()
        i_job = len(self.queue.queue)

        for job in reversed(self.queue.queue):
            i_job -= 1

            if self.num_sched_test_step >= sched_depth:
                break
            self.num_sched_test_step += 1

            if job.partition in partitions_failed:
                continue

            if job.qos.hold_job(job):
                continue

            job_end = self.time + job.reqtime

            n_nodes, valid_nodes = 0, set()

            for node in free_nodes_ready_now:
                if job.partition not in node.partitions:
                    continue

                if job_end > node.interval_times[-1]:
                    continue

                valid_nodes.add(node)
                if len(valid_nodes) >= job.nodes:
                    break

            if len(valid_nodes) == job.nodes:
                # Cancel early if set to be cancelled in queue at later time
                if job.cancel is not None:
                    jobs_cancelled.append(i_job)
                    jobs_submitted = [ i - 1 for i in jobs_submitted ]
                    continue

                self._submit(job.start_job(self.time), valid_nodes)
                jobs_submitted.append(i_job)

                for node in valid_nodes:
                    free_nodes_ready_now.remove(node)

            else:
                partitions_failed.add(job.partition)
                if len(partitions_failed) == len(self.partitions.partitions):
                    break

                free_nodes_ready_now = [
                    node for node in free_nodes_ready_now if job.partition not in node.partitions
                ]
                if not free_nodes_ready_now:
                    break

        self.sched_main_num += len(jobs_submitted)

        for i_job in jobs_cancelled:
            cancelled_job = self.queue.queue.pop(i_job)
            cancelled_job.cancel_job()
            self.queue.jobs_to_cancel.remove(cancelled_job)

        for i_job in jobs_submitted:
            self.queue.queue.pop(i_job)

    def _prep_new_bf(self):
        self.bf_locks_remaining = self.bf_max_lock_holds
        self.bf_time = self.time
        self.bf_secs_past = 0

        self._prep_bf_map()

        self._prep_bf_q()

    def _prep_bf_map(self):
        self.bf_free_blocks, self.bf_nodes_free_now_max_reqtimes = {}, {}

        for resv, free_block in self.partitions.free_blocks.items():
            self.bf_free_blocks[resv] = defaultdict(set)
            bf_free_blocks = self.bf_free_blocks[resv]
            self.bf_nodes_free_now_max_reqtimes[resv] = {}
            bf_nodes_free_now_max_reqtimes = self.bf_nodes_free_now_max_reqtimes[resv]

            for interval, nodes in free_block.items():
                if interval[1] == datetime.datetime.max:
                    interval_f = self.bf_window
                else:
                    interval_f = min(
                        (interval[1] - self.time).total_seconds(), self.bf_window
                    )

                if interval[0] <= self.time:
                    for node in nodes:
                        if node.running_job is not None:
                            interval_i = max(
                                (
                                    (node.running_job.endlimit - self.time).total_seconds() +
                                    self.bf_end_padding
                                ),
                                1
                            )

                        else:
                            interval_i = 0

                        bf_free_blocks[(interval_i, interval_f)].add(node)

                        if interval_i <= self.bf_max_relevant_start:
                            bf_nodes_free_now_max_reqtimes[node] = interval_f - interval_i

                else:
                    interval_i = (interval[0] - self.time).total_seconds()
                    if interval_i >= self.bf_window:
                        continue

                    bf_free_blocks[(interval_i, interval_f)].update(nodes)

                if interval_i <= self.bf_max_relevant_start:
                    for node in nodes:
                        bf_nodes_free_now_max_reqtimes[node] = interval_f - interval_i

        self.bf_max_reqtime = {
            resv : max(nodes_free_now_min_reqtimes.values()) if nodes_free_now_min_reqtimes else 0
            for resv, nodes_free_now_min_reqtimes in self.bf_nodes_free_now_max_reqtimes.items()
        }

    def _prep_bf_q(self):
        self.bf_queue, self.bf_job_ordered_reqtimes = [], {}

        # NOTE Only checking for the normal queue because resv queues are usually small. Should
        # really just have a single queue with reservation sorted to the front, this would clean
        # up some code
        max_test_from_normal_q = (
            self.config.bf_max_job_test - sum(len(q) for q in self.queue.reservations.values())
        )
        if max_test_from_normal_q > 0:
            self.bf_job_ordered_reqtimes[""] = {}
            for job in self.queue.queue[-max_test_from_normal_q:]:
                self.bf_queue.append(job)
                self.bf_job_ordered_reqtimes[""][job] = job.reqtime.total_seconds()
        for resv, resv_q in self.queue.reservations.items():
            self.bf_job_ordered_reqtimes[resv] = {}
            for job in resv_q:
                self.bf_queue.append(job)
                self.bf_job_ordered_reqtimes[resv][job] = job.reqtime.total_seconds()
        # Earliest reqtime job at front so I can call next(iter()) to grab it
        self.bf_job_ordered_reqtimes = {
            resv : OrderedDict(
                {
                    job : reqtime
                    for job, reqtime in sorted(
                        job_ordered_reqtimes.items(), key=lambda job_reqtime: job_reqtime[1]
                    )
                }
            )
            for resv, job_ordered_reqtimes in self.bf_job_ordered_reqtimes.items()
        }

        # If there are no more jobs in the queue that could possibly be started now, we shouldnt
        # waste time backfilling. Still want to go throught the yield cycles and just pretend
        # we are actually backfilling. This is more likely to flick to true as the backfill
        # schedule fills up and jobs are processed from the queue. I think this is saving time but
        # the amount saved will depend on the workload and typical requested times.
        self.bf_done = {
            resv : (
                next(iter(job_ordered_reqtimes.values())) > self.bf_max_reqtime[resv]
                if job_ordered_reqtimes
                else True
            )
            for resv, job_ordered_reqtimes in self.bf_job_ordered_reqtimes.items()
        }

    def _backfill(self, n_try):
        backfill_now, loop_finished = self._get_backfill_jobs(n_try)

        self.bf_locks_remaining -= 1
        self.bf_loop_active = bool(not loop_finished and self.bf_locks_remaining)
        self.bf_secs_past += self.config.bf_yield_interval.total_seconds()

        self.sched_backfill_num += len(backfill_now)

        for job, nodes in backfill_now:
            queue = (
                self.queue.reservations[job.reservation] if job.reservation else self.queue.queue
            )
            queue.remove(job)
            self._submit(job.start_job(self.time), nodes)

    def _get_backfill_jobs(self, n_try):
        backfill_now = []

        while n_try and self.bf_queue:
            job = self.bf_queue.pop()

            if job.qos.hold_job(job):
                continue

            # sched loop could've scheduled between loops
            if (
                job.state == JobState.RUNNING or
                job.state == JobState.COMPLETED or
                job.state == JobState.CANCELLED
            ):
                continue

            n_try -= 1

            resv = job.reservation

            if self.bf_done[resv]:
                continue

            self.bf_done[resv] = (
                next(iter(self.bf_job_ordered_reqtimes[resv].values())) >
                self.bf_max_reqtime[resv]
            )

            reqtime = self.bf_job_ordered_reqtimes[resv].pop(job)

            free_blocks = self.bf_free_blocks[resv]

            num_free_nodes, selected_intervals = 0, defaultdict(set)

            sorted_free_blocks = sorted(free_blocks.items(), key=lambda block: block[0])
            usage_block_start = sorted_free_blocks[0][0][0]
            usage_block_end = usage_block_start + reqtime

            for i_block, (interval, nodes) in enumerate(sorted_free_blocks):
                if interval[1] >= usage_block_end and interval[1] >= interval[0] + reqtime:
                    valid_nodes = { node for node in nodes if job.partition in node.partitions }
                    if valid_nodes:
                        selected_intervals[interval] = valid_nodes
                        num_free_nodes += len(valid_nodes)
                        usage_block_start = interval[0]
                        usage_block_end = usage_block_start + reqtime
                        if usage_block_end > self.bf_window:
                            break

                # Only move forward once we have gathered all valid nodes that share an interval
                # start time and have checked all nodes that we may be able to run on immediately
                if (
                    not i_block + 1 == len(sorted_free_blocks) and
                    (
                        sorted_free_blocks[i_block + 1][0][0] == interval[0] or
                        sorted_free_blocks[i_block + 1][0][0] <= self.bf_secs_past
                    )
                ):
                    continue

                if job.nodes <= num_free_nodes:
                    # Remove blocks that do not remain free long enough for job to run
                    # NOTE Still need to check this since usage_block end has moved forward from
                    # first blocks
                    for selected_interval in list(selected_intervals.keys()):
                        if usage_block_end > selected_interval[1]:
                            num_free_nodes -= len(selected_intervals.pop(selected_interval))

                    if job.nodes > num_free_nodes:
                        continue

                    # Need to sort the nodes by weight to choose which to run on
                    selected_nodes = [
                        node for nodes in selected_intervals.values() for node in nodes
                    ]

                    # Prioritise nodes that are available if job might be able to start now
                    if usage_block_start <= self.bf_secs_past:
                        selected_nodes.sort(
                            key=lambda node: (
                                node.running_job is not None, node.weight, node.nid
                            ),
                            reverse=True
                        )
                    else:
                        selected_nodes.sort(key=lambda node: (node.weight, node.nid), reverse=True)

                    selected_nodes = selected_nodes[num_free_nodes - job.nodes:]

                    # Run the job
                    if usage_block_start <= self.bf_secs_past:
                        if job.cancel is not None:
                            self.queue.cancel_job(job)
                            break

                        # Node may have been allocated by sched during the yield_sleep or a
                        # job is running overtime. Cannot schedule the job in this case
                        if all(node.running_job is None for node in selected_nodes):
                            backfill_now.append((job, selected_nodes))

                    recompute_bf_max_reqtime = False

                    usage_block_start = (
                        int(usage_block_start // self.bf_resolution * self.bf_resolution)
                    )
                    usage_block_end = max(
                        int(usage_block_end // self.bf_resolution * self.bf_resolution),
                        self.bf_resolution
                    )

                    for selected_interval, nodes in selected_intervals.items():
                        nodes = nodes.intersection(selected_nodes)
                        if not nodes:
                            continue

                        free_blocks[selected_interval] -= nodes

                        if selected_interval[0] < usage_block_start:
                            free_blocks[(selected_interval[0], usage_block_start)].update(nodes)

                        if selected_interval[1] > usage_block_end:
                            free_blocks[(usage_block_end, selected_interval[1])].update(nodes)

                        # This interval is relevant for the max reqtime so need to update tracking
                        # information
                        if selected_interval[0] <= self.bf_max_relevant_start:
                            new_reqtime_early = usage_block_start - selected_interval[0]
                            old_reqtime = selected_interval[1] - selected_interval[0]
                            # Very short job can create two possible reqtimes by finishing before
                            # max relevant start
                            if usage_block_end < self.bf_max_relevant_start:
                                new_reqtime_late = selected_interval[1] - usage_block_end
                            else:
                                new_reqtime_late = None

                            for node in nodes:
                                if (
                                    self.bf_nodes_free_now_max_reqtimes[resv][node] != old_reqtime
                                ):
                                    continue

                                if new_reqtime_late is None:
                                    if new_reqtime_early <= 0:
                                        self.bf_nodes_free_now_max_reqtimes[resv].pop(node)
                                    else:
                                        self.bf_nodes_free_now_max_reqtimes[resv][node] = (
                                            new_reqtime_early
                                        )
                                else:
                                    self.bf_nodes_free_now_max_reqtimes[resv][node] = max(
                                        new_reqtime_early, new_reqtime_late
                                    )

                                recompute_bf_max_reqtime = True

                        if not free_blocks[selected_interval]:
                            free_blocks.pop(selected_interval)

                    if recompute_bf_max_reqtime:
                        if self.bf_nodes_free_now_max_reqtimes[resv]:
                            self.bf_max_reqtime[resv] = max(
                                self.bf_nodes_free_now_max_reqtimes[resv].values()
                            )
                        else:
                            self.bf_max_reqtime[resv] = 0

                    break

        return backfill_now, not self.bf_queue

    def _check_down_nodes(self):
        node_update = False

        while self.down_nodes and self.down_nodes[-1].up_time <= self.time:
            node = self.down_nodes.pop()
            node.set_up()
            if node.reservation:
                node.interval_times[1] = node.unreserved_time
            elif node.reservation_schedule:
                node.interval_times[1] = node.reservation_schedule[-1][0]
            else:
                node.interval_times[1] = datetime.datetime.max
            node.interval_times[0] = self.time
            self.partitions.add_free_block(node)

            node_update = True

        # To avoid sorting many times in the same loop
        still_has_down_schedule, orig_len_down_nodes = set(), len(self.down_nodes)

        while self.node_down_order and self.node_down_order[-1].down_schedule[-1][0] <= self.time:
            node = self.node_down_order.pop()
            # If already down delay this new downtime until the next up to not interfere (this
            # happens because the DOWN implementation waits for current running job to finish)
            if node.down:
                node.down_schedule[-1][0] = node.up_time
                still_has_down_schedule.add(node)
                continue

            # Delay down until after current running job has finished, this call will happen
            # after _check_fininshed_jobs() and before any sched calls so this down will not be
            # getting perpetually delayed.
            if node.down_schedule[-1][2] == "DOWN" and node.running_job is not None:
                node.down_schedule[-1][0] = node.running_job.end
                still_has_down_schedule.add(node)
                continue

            down_schedule = node.down_schedule.pop()
            if len(node.down_schedule):
                still_has_down_schedule.add(node)

            up_time = self.time + down_schedule[1]

            if node.running_job and up_time <= node.running_job.end:
                continue

            self.partitions.remove_free_block(node)

            node.set_down(up_time)

            node.interval_times = [datetime.datetime.max, datetime.datetime.max]

            self.down_nodes.append(node)

            node_update = True

        # If nodes change between bf yield intervals, the bf loop breaks even with bf_continue
        if node_update and self.bf_loop_active:
            self.bf_queue = []

        if len(self.down_nodes) != orig_len_down_nodes:
            self.down_nodes.sort(key=lambda node: node.up_time, reverse=True)

        if still_has_down_schedule:
            for node in still_has_down_schedule:
                self.node_down_order.append(node)
            self.node_down_order.sort(
                key=lambda node: (node.down_schedule[-1][0], node.nid), reverse=True
            )

    def _check_reservations(self):
        resv_update = False

        # Destroy and spawn new sliding reservations
        while self.sliding_reservations and self.sliding_reservations[-1][0] <= self.time:
            _, submit, clear, start, end, nodes, name = self.sliding_reservations[-1]
            for node in nodes:
                if submit is not None: # At event where submitting a new resv block
                    node.reservation_schedule.append((start, end, name))
                    node.reservation_schedule.sort(key=lambda schedule: schedule[0], reverse=True)

                    if node.down:
                        continue

                    resv_update = True

                    self.partitions.remove_free_block(node)

                    node.interval_times[-1] = node.reservation_schedule[-1][0]
                    self.partitions.add_free_block(node)

                # At event where clearing reservation that is about to start in anticipation of the
                # next resv in the sliding resv
                else: 
                    for i_res_sched, res_sched in enumerate(node.reservation_schedule):
                        if res_sched[2] != name:
                            continue
                        node.reservation_schedule.pop(i_res_sched)

                        if node.down:
                            break

                        resv_update = True

                        self.partitions.remove_free_block(node)

                        if node.reservation_schedule:
                            node.interval_times[-1] = node.reservation_schedule[-1][0]
                        else:
                            node.interval_times[-1] = datetime.datetime.max

                        self.partitions.add_free_block(node)

                        break

            if submit is None: # Finished this resv step
                self.sliding_reservations.pop()
            else: # Created this resv step, need to clear it next
                self.sliding_reservations[-1][0] = clear
                self.sliding_reservations[-1][1] = None
                # If clear is same time as next submit, want to clear first
                self.sliding_reservations.sort(
                    key=lambda sliding_res: (sliding_res[0], sliding_res[1] is not None),
                    reverse=True
                )

        # Set/unset reservation
        while self.reserved_nodes and self.reserved_nodes[-1].unreserved_time <= self.time:
            node = self.reserved_nodes.pop()

            if node.down:
                node.set_unreserved()
                continue

            resv_update = True

            self.partitions.remove_free_block(node)

            node.set_unreserved()
            if node.reservation_schedule:
                node.interval_times[-1] = node.reservation_schedule[-1][0]
            else:
                node.interval_times[-1] = datetime.datetime.max

            self.partitions.add_free_block(node)

        while (
            self.node_reservation_order and
            self.node_reservation_order[-1].reservation_schedule[-1][0] <= self.time
        ):
            node = self.node_reservation_order[-1]

            reservation_schedule = node.reservation_schedule.pop()
            if not len(node.reservation_schedule):
                self.node_reservation_order.pop()
            else:
                self.node_reservation_order.sort(
                    key=lambda node: (node.reservation_schedule[-1][0], node.nid), reverse=True
                )

            if node.down:
                node.set_reserved(reservation_schedule[2], reservation_schedule[1])

                self.reserved_nodes.append(node)
                self.reserved_nodes.sort(key=lambda node: node.unreserved_time, reverse=True)

                break

            resv_update = True

            self.partitions.remove_free_block(node)

            node.set_reserved(reservation_schedule[2], reservation_schedule[1])
            node.interval_times[-1] = node.unreserved_time
            self.partitions.add_free_block(node)

            self.reserved_nodes.append(node)
            self.reserved_nodes.sort(key=lambda node: node.unreserved_time, reverse=True)

        # If reservations change between bf yield intervals, the bf loop breaks even with
        # bf_continue
        if resv_update and self.bf_loop_active:
            self.bf_queue = []

    def _print_stats(self, sched, bf):
        if not (self.time.hour != self.previous_print_hour and not self.time.hour % 3):
            return

        # XXX Some of this is ARCHER2 specific info
        self.previous_print_hour = self.time.hour
        print(
            "{} (step {} SchedMain {} SchedBackfill {}):\n".format(
                self.time, self.step_cnt, self.sched_main_num, self.sched_backfill_num
            ) +
            # "Idle Nodes = {} (num in free_blocks_ready {}) (highmem {})\t" \
            "Idle Nodes = {} (".format(sum(1 for node in self.partitions.nodes if node.free)) +
            " ".join(
                "{}={}".format(
                    partition.name, sum(1 for node in partition.nodes if node.free)
                ) for partition in self.partitions.partitions
            ) +
            ")\t" +
            "NodesReserved = {} (Idle = {})\t".format(
                sum(1 for node in self.partitions.nodes if node.reservation),
                sum(
                   1 for node in self.partitions.nodes if (
                        node.reservation and not node.running_job and not node.down
                    )
                )
            ) +
            (
                "NodesHPE_RestrictLongJobs = {} (Idle = {})\t".format(
                    sum(
                        1 for node in self.partitions.nodes if (
                            node.reservation_schedule and
                            "HPE_RestrictLongJobs" in [
                                res_sched[2] for res_sched in node.reservation_schedule
                            ]
                        )
                    ),
                    sum(
                        1 for node in self.partitions.nodes if (
                            node.reservation_schedule and
                            "HPE_RestrictLongJobs" in [
                                res_sched[2] for res_sched in node.reservation_schedule
                            ] and
                            not node.running_job and not node.down
                        )
                    )
                )
                if self.sliding_reservations
                else ""
            )
            +
            "NodesDown = {}\tPower = {:.4f} MW\n".format(
                sum(1 for node in self.down_nodes if not node.running_job),
                self.power_usage
            ) +
            "Priority Queue = {} (cancel {} partition ".format(
                len(self.queue.queue) + len(self.queue.waiting_dependency),
                sum(1 for job in self.queue.queue if job.cancel is not None)
            ) +
            " ".join(
                "{}={}".format(
                    partition.name,
                    sum(1 for job in self.queue.queue if job.partition is partition)
                ) for partition in self.partitions.partitions
            ) +
            " qos " +
            " ".join(
                "{}={}".format(
                    qos.name, sum(1 for job in self.queue.queue if job.qos is qos)
                )
                for qos in self.queue.qoss.values()
                    if sum(1 for job in self.queue.queue if job.qos is qos)
            ) +
            ") dependency {} ".format(
                len(self.queue.waiting_dependency)
            ) +
            "qos submit holds {} (".format(
                sum(len(jobs) for jobs in self.queue.qos_submit_held.values())
            ) +
            " ".join(
                "{}={}".format(
                    qos.name, len(jobs)
                ) for qos, jobs in self.queue.qos_submit_held.items() if len(jobs)
            ) +
            ")\tRunningJobs = {}\n".format(len(self.running_jobs))
        )

