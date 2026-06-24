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

from collections import defaultdict
import datetime; from datetime import timedelta

import pandas as pd

from aux_funcs import convert_nodelist_to_node_nums

from job_queue import Job


class Partitions:
    def __init__(self, nid_data, partition_data):
        self.partitions = {
            Partition(name, data["prio_tier"], data["prio_jobfactor"])
            for name, data in partition_data.items()
        }
        self.partitions_by_name = { partition.name : partition for partition in self.partitions }

        self.nodes = set()

        for nid, data in sorted(nid_data.items()):
            node = Node(nid, data["weight"], data["down_schedule"], data["resv_schedule"])
            for p_name in data["partitions"]:
                self.partitions_by_name[p_name].add_node(node)
            self.nodes.add(node)

        print("Using partitions:")
        for partition in self.partitions:
            print(
                partition.name, partition.priority_tier, partition.priority_weight, "-",
                len(partition.nodes), "nodes",
                sep=" "
            )
        print("With {} unique nodes total".format(len(self.nodes)))

        self.reservations = defaultdict(list)

        # { reservation : { interval : nodes, ... }, ... }
        self.free_blocks = defaultdict(lambda: defaultdict(set))
        for node in self.nodes:
            self.free_blocks[node.reservation][
                (node.interval_times[0], node.interval_times[-1])
            ].add(node)

    def remove_free_block(self, node):
        interval = (node.interval_times[0], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].remove(node)
        if not self.free_blocks[node.reservation][interval]:
            self.free_blocks[node.reservation].pop(interval)

    def add_free_block(self, node):
        interval = (node.interval_times[0], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].add(node)

    def get_partition_by_name(self, name):
        return self.partitions_by_name[name]


class Partition:
    def __init__(self, name, priority_tier, priority_weight):
        self.name = name
        self.priority_tier = priority_tier
        self.priority_weight = priority_weight # Normalised s.t. partition with greatest has 1

        self.nodes = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Partition):
            return self.name == other.name
        return False

    def add_node(self, node):
        node.partitions.append(self)
        self.nodes.append(node)
        # does this still need to be sorted? can it just be a set
        self.nodes.sort(key=lambda node: (node.weight, node.nid)) # Small weights get priority


class Node:
    def __init__(self, nid, weight, down_schedule, reservation_schedule):
        self.nid = nid
        self.nid_hash = hash(nid)
        self.weight = weight

        self.free = True
        self.running_job = None

        self.down_schedule = down_schedule
        self.down = False
        self.up_time = None

        self.reservation_schedule = reservation_schedule
        self.reservation = ""
        self.unreserved_time = None

        self.partitions = []

        # NOTE This is from when I was allowing the BF sched to plan nodes in a way that would be
        # respected by the main scheduling loop. Not doing this anymore so there is only ever 2
        # entries. Might be useful I want to implement a node going in and then out of a
        # reservation.
        self.interval_times = [
            datetime.datetime.min,
            datetime.datetime.max if not reservation_schedule else reservation_schedule[-1][0]
        ]

        self.bf_free_blocks_start = None

    def __hash__(self):
        return self.nid_hash

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.nid == other.nid
        return False

    def set_reserved(self, reservation_name, end_time):
        self.reservation = reservation_name
        self.unreserved_time = end_time
        if self.down or not self.free:
            return
        self.free = False

    def set_unreserved(self):
        self.reservation = ""
        self.unreserved_time = None
        if self.down or self.running_job:
            return
        self.free = True

    def set_down(self, up_time):
        self.down = True
        self.up_time = up_time
        # If job is already running it is allowed to finish
        if not self.free:
            return
        self.free = False

    def set_up(self):
        self.down = False
        self.up_time = None
        if self.reservation:
            return
        self.free = True

    def set_free(self):
        if self.down or self.reservation:
            return
        self.free = True

    def set_busy(self):
        if self.reservation:
            return
        self.free = False

