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

import os, argparse
import datetime; from datetime import timedelta
import dill as pickle
from collections import defaultdict
from itertools import cycle

import matplotlib.dates
from matplotlib.dates import DateFormatter
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
from tqdm import tqdm

from controller import Controller
from fairshare import FairTree
from aux_funcs import mkdir_p

# TODO
# - Total power usage plots

global bd_threshold
bd_threshold = timedelta(minutes=10)

matplotlib.use('TkAgg')
plt.style.use('tableau-colorblind10')


def to_plot_or_not_to_plot(batch):
    if batch:
        plt.close()
    else:
        plt.show()


def metric_property_hist2d(job_history, job_to_metric_sim, job_to_metric_data, property, metric):
    job_property, sim_metrics, data_metrics = [], [], []

    for job in job_history:
        if job.ignore_in_eval:
            continue

        if property == "nodes":
            if job.nodes == 0:
                continue
            job_property.append(job.nodes)
        elif property == "reqtime":
            if job.reqtime.total_seconds() == 0:
                continue
            job_property.append(job.reqtime.total_seconds() / 60)
        else:
            raise NotImplementedError(property)

        sim_metrics.append(job_to_metric_sim(job))
        data_metrics.append(job_to_metric_data(job))

    if property == "nodes":
        bins_property = np.logspace(
            np.log10(min(job_property)), np.log10(max(job_property) + 0.5), 30, dtype=int
        )
        # Merge identical bins
        _, uniq_i = np.unique(bins_property, return_index=True)
        bins_property = bins_property[np.sort(uniq_i)]
    elif property == "reqtime":
        bins_property = np.logspace(np.log10(min(job_property)), np.log10(max(job_property)), 30)

    if metric == "bdslowdown":
        min_metric = 1.0
        nbins = 20
    elif metric == "wait_time":
        min_metric = 1 / 6
        nbins = 20
    # max_metric = np.percentile(sim_metrics + data_metrics, 99)
    max_metric = max(max(sim_metrics), max(data_metrics))
    bins_metric = np.logspace(np.log10(min_metric), np.log10(max_metric), nbins)

    h_data = np.histogram2d(job_property, data_metrics, bins=[bins_property, bins_metric])
    h_sim = np.histogram2d(job_property, sim_metrics, bins=[bins_property, bins_metric])

    h_data, h_data_edges = h_data[0], (h_data[1], h_data[2])
    h_sim, h_sim_edges = h_sim[0], (h_sim[1], h_sim[2])

    h_data_col_sums = h_data.sum(axis=1)
    h_data_col_sums[(h_data_col_sums == 0)] = 1
    h_data = (h_data.T / h_data_col_sums).T
    h_sim_col_sums = h_sim.sum(axis=1)
    h_sim_col_sums[(h_sim_col_sums == 0)] = 1
    h_sim = (h_sim.T / h_sim_col_sums).T

    return h_data, h_sim, bins_property, bins_metric


def top_assoc_waits(job_history, job_to_assoc, num_top, nodehour_threshold=None):
    assoc_sim_wait, assoc_data_wait = defaultdict(list), defaultdict(list)
    assoc_nodehours = defaultdict(float)
    for job in job_history:
        if job.ignore_in_eval:
            continue

        # assoc = assoc_tree.assocs[job.assoc].parent.parent.name
        assoc = job_to_assoc(job)
        sim_wait = (job.start - job.submit).total_seconds() / 60 / 60
        data_wait = (job.true_job_start - job.true_submit).total_seconds() / 60 / 60

        assoc_nodehours[assoc] += job.nodes * job.runtime.total_seconds() / 60 / 60
        assoc_sim_wait[assoc].append(sim_wait)
        assoc_data_wait[assoc].append(data_wait)

    if nodehour_threshold is not None:
        top_assocs = [
            assoc for assoc, nodehours in assoc_nodehours.items() if nodehours > nodehour_threshold
        ]
    else:
        top_assocs = [
            assoc
            for assoc, _ in (
                sorted(
                    assoc_nodehours.items(), key=lambda keyval: keyval[1], reverse=True
                )[:num_top]
            )
        ]

    assoc_sim_wait_mean = { assoc : np.mean(waits) for assoc, waits in assoc_sim_wait.items() }
    assoc_data_wait_mean = { assoc : np.mean(waits) for assoc, waits in assoc_data_wait.items() }
    assoc_sim_wait_err = { assoc : np.std(waits) for assoc, waits in assoc_sim_wait.items() }
    assoc_data_wait_err = { assoc : np.std(waits) for assoc, waits in assoc_data_wait.items() }
    sorted_sim_wait = [
        (assoc, wait_mean, assoc_sim_wait_err[assoc], len(assoc_sim_wait[assoc]))
        for assoc, wait_mean in sorted(
            assoc_sim_wait_mean.items(), key=lambda assoc_wait: assoc_wait[1], reverse=True
        )
            if assoc in top_assocs
    ]
    sorted_data_wait = [
        (assoc, wait_mean, assoc_data_wait_err[assoc], len(assoc_data_wait[assoc]))
        for assoc, wait_mean in sorted(
            assoc_data_wait_mean.items(), key=lambda assoc_wait: assoc_wait[1], reverse=True
        )
            if assoc in top_assocs
    ]

    print(
        "Sim top assoc by mean wait times:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {} ({} jobs)".format(
                i + 1, assoc_wait[0], assoc_wait[1], assoc_wait[2], assoc_wait[3]
            )
            for i, assoc_wait in enumerate(sorted_sim_wait)
        ) +
        "\n"
    )
    print(
        "True top assoc by mean wait times:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {} ({} jobs)".format(
                i + 1, assoc_wait[0], assoc_wait[1], assoc_wait[2], assoc_wait[3]
            )
            for i, assoc_wait in enumerate(sorted_data_wait)
        ) +
        "\n"
    )

    top_assocs.sort(key=lambda assoc: assoc_data_wait_mean[assoc], reverse=True)

    sim_mean_waits = [ assoc_sim_wait_mean[assoc] for assoc in top_assocs ]
    data_mean_waits = [ assoc_data_wait_mean[assoc] for assoc in top_assocs ]

    return top_assocs, sim_mean_waits, data_mean_waits


def group_waits(job_history, job_to_group):
    sim_group_waits, data_group_waits = defaultdict(list), defaultdict(list)
    for job in job_history:
        if job.ignore_in_eval:
            continue
        sim_group_waits[job_to_group(job)].append(
            (job.start - job.submit).total_seconds() / 60 / 60
        )
        data_group_waits[job_to_group(job)].append(
            (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
        )

    print("Num Jobs by group:")
    print(
        " | ".join("{} - {}".format(group, len(waits)) for group, waits in sim_group_waits.items())
    )

    sim_group_mean_waits = { group : np.mean(waits) for group, waits in sim_group_waits.items() }
    data_group_mean_waits = { group : np.mean(waits) for group, waits in data_group_waits.items() }
    sorted_group = [
        group
        for group, _ in sorted(
            data_group_mean_waits.items(), key=lambda group_wait: group_wait[1], reverse=True
        )
    ]
    sim_mean_waits = [ sim_group_mean_waits[group] for group in sorted_group ]
    data_mean_waits = [ data_group_mean_waits[group] for group in sorted_group ]

    return sorted_group, data_mean_waits, sim_mean_waits


def rolling_window(job_history, job_to_metric, hours, window_hrs, data=False):
    if data:
        job_to_hour = lambda job: job.true_submit.replace(minute=0, second=0)
    else:
        job_to_hour = lambda job: job.submit.replace(minute=0, second=0)

    # job_to_hour = lambda job: job.true_submit.replace(minute=0, second=0)

    # NOTE wait_time stand in for any metric

    submit_hour_waits = defaultdict(list)
    for job in job_history:
        if job.ignore_in_eval:
            continue

        submit_hour_waits[job_to_hour(job)].append(job_to_metric(job))

    mean_wait_times_rolling_window = np.zeros(len(hours))
    mean_wait_times_rolling_window_err = np.zeros(len(hours))
    wait_times_rolling_window, wait_times_rolling_window_hour_lens = [], []
    for hr_num in range(window_hrs):
        wait_times_rolling_window += submit_hour_waits[hours[0] + timedelta(hours=hr_num)]
        wait_times_rolling_window_hour_lens.append(
            len(submit_hour_waits[hours[0] + timedelta(hours=hr_num)])
        )
    for i_hour, hour in enumerate(hours):
        if wait_times_rolling_window:
            mean_wait_times_rolling_window[i_hour] = np.mean(wait_times_rolling_window)
            mean_wait_times_rolling_window_err[i_hour] = np.std(wait_times_rolling_window)
        wait_times_rolling_window = (
            wait_times_rolling_window[wait_times_rolling_window_hour_lens.pop(0):]
        )
        wait_times_rolling_window += submit_hour_waits[hour + timedelta(hours=window_hrs)]
        wait_times_rolling_window_hour_lens.append(
            len(submit_hour_waits[hour + timedelta(hours=window_hrs)])
        )

    return mean_wait_times_rolling_window, mean_wait_times_rolling_window_err


def total_alloc_nodes(job_history):
    sim_max_end = max(job_history, key=lambda job: job.end).end
    sim_min_start = min(job_history, key=lambda job: job.start).start
    data_max_end_job = max(job_history, key=lambda job: job.true_job_start + job.runtime)
    data_max_end = data_max_end_job.true_job_start + data_max_end_job.runtime
    data_min_start = min(job_history, key=lambda job: job.true_job_start).true_job_start

    sim_alloc_nodes = np.zeros(int((sim_max_end - sim_min_start).total_seconds() / 60))
    data_alloc_nodes = np.zeros(int((data_max_end - data_min_start).total_seconds() / 60))

    for job in tqdm(job_history):
        l_mins = int((job.start - sim_min_start).total_seconds() / 60) + 1
        u_mins = int((job.end - sim_min_start).total_seconds() / 60)
        sim_alloc_nodes[l_mins:u_mins] += job.nodes

        l_mins = int((job.true_job_start - data_min_start).total_seconds() / 60) + 1
        u_mins = int((job.true_job_start + job.runtime - data_min_start).total_seconds() / 60)
        data_alloc_nodes[l_mins:u_mins] += job.nodes

    pad = 24 * 60 * 2
    print("Sim mean(max) allocations nodes = {} +- {} ({})".format(
        np.mean(sim_alloc_nodes[pad:-pad]), np.std(sim_alloc_nodes[pad:-pad]),
        np.max(sim_alloc_nodes)
    ))
    print("Data mean(max) allocations nodes = {} +- {} ({})".format(
        np.mean(data_alloc_nodes[pad:-pad]), np.std(data_alloc_nodes[pad:-pad]),
        np.max(data_alloc_nodes)
    ))

    print(
        "Data 75th percentile allocated nodes {}".format(
            np.percentile(data_alloc_nodes[pad:-pad], 75)
        )
    )
    print(
        "Sim 75th percentile allocated nodes {}".format(
            np.percentile(sim_alloc_nodes[pad:-pad], 75)
        )
    )
    print("Data median allocated nodes {}".format(np.percentile(data_alloc_nodes[pad:-pad], 50)))
    print("Sim median allocated nodes {}".format(np.percentile(sim_alloc_nodes[pad:-pad], 50)))

    data_minutes = [
        data_min_start + timedelta(minutes=min_num) for min_num in range(len(data_alloc_nodes))
    ]
    sim_minutes = [
        sim_min_start + timedelta(minutes=min_num) for min_num in range(len(sim_alloc_nodes))
    ]

    return data_alloc_nodes, data_minutes, sim_alloc_nodes, sim_minutes


def q_size(job_history):
    sim_min_submit = min(job_history, key=lambda job: job.submit).submit
    data_min_submit = min(job_history, key=lambda job: job.true_submit).true_submit
    sim_max_start = max(job_history, key=lambda job: job.start).start
    data_max_start = max(job_history, key=lambda job: job.true_job_start).true_job_start

    sim_q_length = np.zeros(int((sim_max_start - sim_min_submit).total_seconds() / 60))
    data_q_length = np.zeros(int((data_max_start - data_min_submit).total_seconds() / 60))
    sim_q_length_nodes = np.zeros(int((sim_max_start - sim_min_submit).total_seconds() / 60))
    data_q_length_nodes = np.zeros(int((data_max_start - data_min_submit).total_seconds() / 60))

    for job in tqdm(job_history):
        if job.ignore_in_eval:
            continue

        l_mins = int((job.submit - sim_min_submit).total_seconds() / 60) + 1
        u_mins = int((job.start - sim_min_submit).total_seconds() / 60)
        sim_q_length[l_mins:u_mins] += 1
        sim_q_length_nodes[l_mins:u_mins] += job.nodes

        l_mins = int((job.true_submit - data_min_submit).total_seconds() / 60) + 1
        u_mins = int((job.true_job_start - data_min_submit).total_seconds() / 60)
        data_q_length[l_mins:u_mins] += 1
        data_q_length_nodes[l_mins:u_mins] += job.nodes

    pad = 24 * 60 * 2
    print("Sim mean(max) queue size (jobs) = {} +- {} ({})".format(
        np.mean(sim_q_length[pad:-pad]), np.std(sim_q_length[pad:-pad]), np.max(sim_q_length)
    ))
    print("Data mean(max) queue size (jobs) = {} +- {} ({})".format(
        np.mean(data_q_length[pad:-pad]), np.std(data_q_length[pad:-pad]), np.max(data_q_length)
    ))
    print("Sim mean(max) queue size (nodes) = {} +- {} ({})".format(
        np.mean(sim_q_length_nodes[pad:-pad]), np.std(sim_q_length_nodes[pad:-pad]),
        np.max(sim_q_length_nodes)
    ))
    print("Data mean(max) queue size (nodes) = {} +- {} ({})".format(
        np.mean(data_q_length_nodes[pad:-pad]), np.std(data_q_length_nodes[pad:-pad]),
        np.max(data_q_length_nodes)
    ))

    data_minutes = [
        data_min_submit + timedelta(minutes=min_num) for min_num in range(len(data_q_length))
    ]
    sim_minutes = [
        sim_min_submit + timedelta(minutes=min_num) for min_num in range(len(sim_q_length))
    ]

    return (
        data_q_length, data_q_length_nodes, data_minutes, sim_q_length, sim_q_length_nodes,
        sim_minutes
    )


def mean_metrics(job_history, controller):
    data_bd_slowdowns = [
        max(
            (job.true_job_start + job.reqtime - job.true_submit) / max(job.reqtime, bd_threshold),
            1
        )
        for job in job_history
            if not job.ignore_in_eval
    ]
    sim_bd_slowdowns = [
        max((job.endlimit - job.submit) / max(job.reqtime, bd_threshold), 1)
        for job in job_history
            if not job.ignore_in_eval
    ]
    data_wait_times = [
        (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
        for job in job_history
            if not job.ignore_in_eval
    ]
    sim_wait_times = [
        (job.start - job.submit).total_seconds() / 60 / 60
        for job in job_history
            if not job.ignore_in_eval
    ]

    print(
        "True mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns), np.sum(data_bd_slowdowns)
        ) +
        "Sim mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(sim_bd_slowdowns), np.std(sim_bd_slowdowns), np.sum(sim_bd_slowdowns)
        ) +
        "True mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(data_wait_times), np.std(data_wait_times), np.sum(data_wait_times)
        ) +
        "Sim mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(sim_wait_times), np.std(sim_wait_times), np.sum(sim_wait_times)
        )
    )

    return data_bd_slowdowns, data_wait_times, sim_bd_slowdowns, sim_wait_times


def spider_plot_metrics(job_history):
    wait_times = [
        (job.start - job.submit).total_seconds() for job in job_history if not job.ignore_in_eval
    ]
    avg_wait = np.mean(wait_times)
    max_wait = max(wait_times)

    bd_slowdowns = [
        max((job.endlimit - job.submit) / max(job.reqtime, bd_threshold), 1)
        for job in job_history
            if not job.ignore_in_eval
    ]
    avg_slowdown = np.mean(bd_slowdowns)

    responses = [
        (job.end - job.submit).total_seconds() for job in job_history if not job.ignore_in_eval
    ]
    avg_response = np.mean(responses)

    spider_plot_data = {
        "avg_wait" : avg_wait, "max_wait" : max_wait, "avg_slowdown" : avg_slowdown,
        "avg_response" : avg_response
    }

    return spider_plot_data


def spider_plot_wait_qos(job_history):
    qos_jobs = defaultdict(list)
    for job in job_history:
        if job.ignore_in_eval or job.qos.name == "short" or job.qos.name == "reservation":
            continue

        qos_jobs[job.qos.name].append(job)

    spider_plot_data = {}

    for qos, jobs in qos_jobs.items():
        avg_wait = np.mean([ (job.start - job.submit).total_seconds() for job in jobs ])

        spider_plot_data[qos] = avg_wait

    return spider_plot_data


# Treating slurm to cab as a scaling factor + baseline power of any nodes without jobs runnning
# and so not reported by slurm. Ignore any down nodes that may not be drawing power.
# NOTE These numbers are for ARCHER2
def slurm_to_cab(slurm_power, occupancy): # MW, [0,1]
    # From comparing with cab data
    baseline_power = 1.692
    full_slurm_to_cab = 1.185

    return slurm_power * full_slurm_to_cab + (1 - occupancy) * baseline_power


def power_usage(times, job_history, max_nodes, data=False):
    power, nodes = np.zeros_like(times, dtype=float), np.zeros_like(times, dtype=int)
    tick = 0

    if not data:
        for job in job_history:
            while job.end > times[tick]:
                tick += 1

            prev_tick = tick - 1

            while job.start <= times[prev_tick]:
                power[prev_tick] += job.true_node_power * job.nodes
                nodes[prev_tick] += job.nodes
                prev_tick -= 1

    else:
        trueend_sorted_job_history = sorted(
            job_history, key=lambda job: job.true_job_start + job.runtime
        )

        for job in trueend_sorted_job_history:
            if job.true_job_start + job.runtime > times[-1]:
                continue

            while job.true_job_start + job.runtime > times[tick]:
                tick += 1

            prev_tick = tick - 1

            while job.true_job_start <= times[prev_tick]:
                power[prev_tick] += job.true_node_power * job.nodes
                nodes[prev_tick] += job.nodes
                prev_tick -= 1

    power /= 1e+6 # MW

    for tick, slurm_power in enumerate(power):
        power[tick] = slurm_to_cab(slurm_power, nodes[tick] / max_nodes)

    return power


def plot_power_diff(
    hours, hour_dates, power_baseline, power_exp, slice_l, slice_r,
    vlines=True, title=None, legend_labels=None
):
    hour_dates_slice = hour_dates[slice_l:slice_r]
    power_baseline_slice = power_baseline[slice_l:slice_r]
    power_exp_slice = power_exp[slice_l:slice_r]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot_date(hour_dates_slice, power_baseline_slice, 'C7', linewidth=0.0)
    ax.plot_date(hour_dates_slice, power_exp_slice, 'C8', linewidth=0.0)
    fb_exp_higher = ax.fill_between(
        hour_dates_slice, power_baseline_slice, power_exp_slice,
        where=power_baseline_slice<=power_exp_slice, facecolor="C1", interpolate=True, alpha=0.8
    )
    fb_baseline_higher = ax.fill_between(
        hour_dates_slice, power_baseline_slice, power_exp_slice,
        where=power_baseline_slice>=power_exp_slice, facecolor="C0", interpolate=True, alpha=0.8
    )

    day = min(hours[slice_l:slice_r]).replace(hour=0)
    max_day = max(hours[slice_l:slice_r]).replace(hour=0) + timedelta(days=1)

    while day < max_day:
        vsp_peak = ax.axvspan(
            matplotlib.dates.date2num(day + timedelta(hours=11)),
            matplotlib.dates.date2num(day + timedelta(hours=16)),
            color="gray", alpha=0.3
        )
        if vlines:
            ax.vlines(
                matplotlib.dates.date2num(day + timedelta(hours=9)),
                ymin=0, ymax=1.1 * power_baseline_slice.max(), color="b"
            )
        day += timedelta(days=1)

    if title is None:
        title = "Difference between power usage for Experiment and Baseline (sampled hourly)"
    if legend_labels is None:
        legend_labels = ["Baseline > Experiment", "Experiment > Baseline", "11am - 4pm"]

    ax.set_ylim(0.9 * power_baseline_slice.min(), 1.1 * power_baseline_slice.max())
    ax.set_ylabel("Power (MW)", fontsize=20)
    ax.set_xlabel("Date (hour resolution)", fontsize=20)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    plt.title(title, fontsize=20)
    plt.legend([fb_baseline_higher, fb_exp_higher, vsp_peak], legend_labels, fontsize=16)

    fig.tight_layout()

    power_baseline_slice_peak, power_exp_slice_peak = [], []

    for i_hour, hour in enumerate(hours[slice_l:slice_r]):
        if 11 <= hour.hour < 16:
            power_baseline_slice_peak.append(power_baseline_slice[i_hour])
            power_exp_slice_peak.append(power_exp_slice[i_hour])

    baseline_slice_peak_mu = np.mean(power_baseline_slice_peak)
    exp_slice_peak_mu = np.mean(power_exp_slice_peak)
    print("Time range slicer {}".format(slice_r))
    print("Baseline peak mean power {} MW".format(baseline_slice_peak_mu))
    print("Exp peak mean power {} MW".format(exp_slice_peak_mu))
    print(
        "Baseline - Exp {} MW ({} KW)".format(
            baseline_slice_peak_mu - exp_slice_peak_mu,
            (baseline_slice_peak_mu - exp_slice_peak_mu) * 1000
        )
    )

    return fig, ax


def main(args):
    PLOT_DIR = os.path.join(
        args.plot_dir, "-".join(os.path.basename(sim).split(".")[0] for sim in args.sim)
    )
    mkdir_p(PLOT_DIR)

    # TODO Do I still want a FIFO baseline to compare with?

    controllers = []
    for sim in args.sim:
        with open(sim, "rb") as f:
            controllers.append(pickle.load(f))

    max_submit = max(controllers[0].job_history, key=lambda job: job.true_submit).true_submit

    job_histories = [
        [
            job
            for job in controller.job_history
                if (
                    controller.init_time + timedelta(days=args.days_ignore) < job.true_submit <
                    max_submit - timedelta(days=args.days_ignore)
                )
        ]
        for controller in controllers
    ]

    # Some things (including truth data) should be the same across all controllers so want a single
    # one to reference for this stuff
    controller, job_history = controllers[0], job_histories[0]

    job_history = [
        job for job in controller.job_history if (
            controller.init_time + timedelta(days=4) < job.true_submit <
            max_submit - timedelta(days=4)
        )
    ]

    print(
        "Ignoring {} out of {} jobs in evaluation\n".format(
            sum(1 for job in job_history if job.ignore_in_eval), len(job_history)
        )
    )

    assoc_tree = FairTree(
        controller.config.assocs_dump, timedelta(minutes=1), timedelta(minutes=1),
        controller.init_time, set(), 0, controller.partitions
    )

    data_bd_slowdowns, data_wait_times, sim_bd_slowdowns, sim_wait_times = mean_metrics(
        job_history, controller
    )

    if "bdslowdowns_hist2d" in args.plots:
        job_to_bdslowdown_sim = lambda job: (
            max((job.endlimit - job.submit) / max(job.reqtime, bd_threshold), 1)
        )
        job_to_bdslowdown_data = lambda job: (
            max(
                (
                    (job.true_job_start + job.reqtime - job.true_submit) /
                    max(job.reqtime, bd_threshold)
                ),
                1
            )
        )

        h_data, h_sim, bins_allocnodes, bins_bdslowdowns = metric_property_hist2d(
            job_history, job_to_bdslowdown_sim, job_to_bdslowdown_data, "nodes", "bdslowdown"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        h_min = min(h_data[(h_data != .0)].min(), h_sim[(h_sim != .0)].min())

        ax0 = ax[0].pcolormesh(bins_allocnodes, bins_bdslowdowns, h_data.T, vmin=0.05, vmax=1.0)
        ax1 = ax[1].pcolormesh(bins_allocnodes, bins_bdslowdowns, h_sim.T, vmin=0.05, vmax=1.0)

        ax0.set_edgecolor("face")
        ax1.set_edgecolor("face")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Nodes")
        ax[0].set_ylabel("Bounded Slowdown")
        ax[0].set_title("Data")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Nodes")
        ax[1].set_ylabel("Bounded Slowdown")
        ax[1].set_title("Sim")
        cax = fig.add_axes([0.92, 0.06, 0.02, 0.84])
        fig.colorbar(ax0, cax, orientation="vertical", extend="min")

        fig.savefig(
            os.path.join(PLOT_DIR, "allocnodes_bdslowdowns_hist2d{}.pdf".format(args.save_suffix)),
            bbox_inches="tight"
        )
        to_plot_or_not_to_plot(args.batch)

        h_data, h_sim, bins_reqtime, bins_bdslowdowns= metric_property_hist2d(
            job_history, job_to_bdslowdown_sim, job_to_bdslowdown_data, "reqtime", "bdslowdown"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        h_min = min(h_data[(h_data != .0)].min(), h_sim[(h_sim != .0)].min())

        ax0 = ax[0].pcolormesh(bins_reqtime, bins_bdslowdowns, h_data.T, vmin=0.05, vmax=1.0)
        ax1 = ax[1].pcolormesh(bins_reqtime, bins_bdslowdowns, h_sim.T, vmin=0.05, vmax=1.0)

        ax0.set_edgecolor("face")
        ax1.set_edgecolor("face")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Req Time (mins)")
        ax[0].set_ylabel("Bounded Slowdown")
        ax[0].set_title("Data")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Req Time (mins)")
        ax[1].set_ylabel("Bounded Slowdown")
        ax[1].set_title("Sim")
        cax = fig.add_axes([0.92, 0.06, 0.02, 0.84])
        fig.colorbar(ax0, cax, orientation='vertical', extend="min")

        fig.savefig(
            os.path.join(PLOT_DIR, "reqtime_bdslowdowns_hist2d{}.pdf".format(args.save_suffix)),
            bbox_inches="tight"
        )
        to_plot_or_not_to_plot(args.batch)

    if "wait_times_hist2d" in args.plots:
        job_to_wait_sim = lambda job: (job.start - job.submit).total_seconds() / 60
        job_to_wait_data = lambda job: (job.true_job_start - job.true_submit).total_seconds() / 60

        h_data, h_sim, bins_allocnodes, bins_wait_times = metric_property_hist2d(
            job_history, job_to_wait_sim, job_to_wait_data, "nodes", "wait_time"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        h_min = min(h_data[(h_data != .0)].min(), h_sim[(h_sim != .0)].min())

        ax0 = ax[0].pcolormesh(bins_allocnodes, bins_wait_times, h_data.T, vmin=0.01, vmax=1.0)
        ax1 = ax[1].pcolormesh(bins_allocnodes, bins_wait_times, h_sim.T, vmin=0.01, vmax=1.0)

        ax0.set_edgecolor("face")
        ax1.set_edgecolor("face")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Nodes", fontsize=22)
        ax[0].set_ylabel("Wait (m)", fontsize=22)
        ax[0].set_title("Data", fontsize=22)
        ax[0].tick_params(axis='both', which='major', labelsize=18)
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Nodes", fontsize=22)
        ax[1].set_ylabel("Wait (m)", fontsize=22)
        ax[1].set_title("Sim", fontsize=22)
        ax[1].tick_params(axis='both', which='major', labelsize=18)
        cax = fig.add_axes([0.92, 0.10, 0.02, 0.76])
        fig.colorbar(ax0, cax, orientation="vertical", extend="min")

        fig.savefig(
            os.path.join(PLOT_DIR, "allocnodes_wait_time_hist2d{}.pdf".format(args.save_suffix)),
            bbox_inches="tight"
        )
        to_plot_or_not_to_plot(args.batch)

        h_data, h_sim, bins_reqtime, bins_wait_times = metric_property_hist2d(
            job_history, job_to_wait_sim, job_to_wait_data, "reqtime", "wait_time"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        h_min = min(h_data[(h_data != .0)].min(), h_sim[(h_sim != .0)].min())

        ax0 = ax[0].pcolormesh(bins_reqtime, bins_wait_times, h_data.T, vmin=0.01, vmax=1.0)
        ax1 = ax[1].pcolormesh(bins_reqtime, bins_wait_times, h_sim.T, vmin=0.01, vmax=1.0)

        ax0.set_edgecolor("face")
        ax1.set_edgecolor("face")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Req Time (m)", fontsize=22)
        ax[0].set_ylabel("Wait (m)", fontsize=22)
        ax[0].set_title("Data", fontsize=22)
        ax[0].tick_params(axis='both', which='major', labelsize=18)
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Req Time (m)", fontsize=22)
        ax[1].set_ylabel("Wait (m)", fontsize=22)
        ax[1].set_title("Sim", fontsize=22)
        ax[1].tick_params(axis='both', which='major', labelsize=18)
        cax = fig.add_axes([0.92, 0.10, 0.02, 0.76])
        fig.colorbar(ax0, cax, orientation="vertical", extend="min")

        fig.savefig(
            os.path.join(PLOT_DIR, "reqtime_wait_time_hist2d{}.pdf".format(args.save_suffix)),
            bbox_inches="tight"
        )
        to_plot_or_not_to_plot(args.batch)

    if "top_projs" in args.plots:
        job_to_proj = lambda job: assoc_tree.assocs[job.assoc].parent.parent.name
        top_projs, sim_mean_waits, data_mean_waits = top_assoc_waits(job_history, job_to_proj, 15)
        x = np.arange(len(top_projs))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data", color="C3")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, top_projs)
        ax.legend(prop={'size': 16})
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "top_projs_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "top_accounts" in args.plots:
        job_to_acc = lambda job: assoc_tree.assocs[job.assoc].parent.name
        top_accs, sim_mean_waits, data_mean_waits = top_assoc_waits(job_history, job_to_acc, 15)
        x = np.arange(len(top_accs))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data", color="C3")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, top_accs)
        ax.legend(prop={'size': 16})
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "top_accs_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "top_users" in args.plots:
        job_to_usr = lambda job: assoc_tree.assocs[job.assoc].name
        top_usr, sim_mean_waits, data_mean_waits = top_assoc_waits(job_history, job_to_usr, 15)
        x = np.arange(len(top_usr))

        for i in range(len(top_usr)):
            top_usr[i] = "User" + str((i + 1))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data", color="C3")
        ax.set_title("Wait times for users with highest usage", fontsize=22)
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=22)
        ax.set_xticks(x, top_usr, fontsize=18, rotation=45)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylim(top=max(max(sim_mean_waits), max(data_mean_waits)) * 1.1)
        ax.legend(prop={'size': 18})
        ax.bar_label(sim_bars, padding=3, rotation=90 ,fmt="%.1f")
        ax.bar_label(data_bars, padding=3, rotation=90, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "top_usr_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "qos_waits" in args.plots:
        job_to_qos = lambda job: job.qos.name
        sorted_qos, data_mean_waits, sim_mean_waits = group_waits(job_history, job_to_qos)

        print(
            "\nlargescale jobs "
            "(id - nodes - submit - elapsed - reqtime - sim wait - true wait - user - account)"
        )
        for job in job_history:
            if job.ignore_in_eval:
                continue

            if job.qos.name == "largescale":
                print(
                    job.jid, job.nodes, job.true_submit,
                    job.runtime, job.reqtime, (job.start - job.submit).round(freq="S"),
                    job.true_job_start - job.true_submit, job.user, job.account,
                    sep=" - "
                )

        x = np.arange(len(sim_mean_waits))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data", color="C3")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, sorted_qos)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "qos_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "partition_waits" in args.plots:
        job_to_partition = lambda job: job.partition.name
        sorted_partition, data_mean_waits, sim_mean_waits = group_waits(
            job_history, job_to_partition
        )

        x = np.arange(len(sim_mean_waits))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data", color="C3")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, sorted_partition)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "partition_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if (
        "rolling_window" in args.plots or
        "rolling_window_qos" in args.plots or
        "rolling_window_partition" in args.plots
    ):
        hours = [
            controllers[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr)
            for hr in range(
                int(
                    (
                        max_submit - timedelta(days=args.rolling_window_days) -
                        controllers[0].init_time
                    ).total_seconds() /
                    (60 * 60)
                )
            )
        ][48:-48]
        window_hrs = int(args.rolling_window_days * 24)

        # Rolling window mean wait time
        job_to_wait_sim = lambda job: (job.start - job.submit).total_seconds() / 60 / 60

        sims_mean_wait_times_rolling_window, sims_mean_wait_times_rolling_window_err = [], []

        for job_history in job_histories:
            means, errs = rolling_window(job_history, job_to_wait_sim, hours, window_hrs)
            sims_mean_wait_times_rolling_window.append(means)
            sims_mean_wait_times_rolling_window_err.append(errs)

        if not args.no_data_comparison:
            job_to_wait_data = lambda job: (
                (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
            )

            means, errs = rolling_window(
                job_history, job_to_wait_data, hours, window_hrs, data=True
            )
            data_mean_wait_times_rolling_window = means
            data_mean_wait_times_rolling_window_err = errs

        hour_dates = matplotlib.dates.date2num(
            [ hour + timedelta(hours=int(window_hrs * 2)) for hour in hours ]
        )

        for label, sim_means in zip(args.labels, sims_mean_wait_times_rolling_window):
            print(label + ":")
            mae = np.abs((sim_means - data_mean_wait_times_rolling_window)).sum() / sim_means.size
            print("MAE for {} day rolling window = {} hr".format(args.rolling_window_days, mae))
            zero_mask = (data_mean_wait_times_rolling_window != 0)
            mape = (
                np.abs(
                    (sim_means[zero_mask] - data_mean_wait_times_rolling_window[zero_mask]) /
                    data_mean_wait_times_rolling_window[zero_mask]
                ).sum() /
                sim_means[zero_mask].size
            )
            mape *= 100
            print("MAPE for {} day rolling window = {} %".format(args.rolling_window_days, mape))

        # The plot with the error band will be horrible for multiple experiments at once
        if len(sims_mean_wait_times_rolling_window) == 1:
            fig = plt.figure(1, figsize=(12, 8))

            sim_mean_wait_times_rolling_window_err = sims_mean_wait_times_rolling_window_err[0]
            sim_mean_wait_times_rolling_window = sims_mean_wait_times_rolling_window[0]

            ax_big = fig.add_axes((.1, .32, .8, .58))
            ax_big.plot_date(
                hour_dates, sim_mean_wait_times_rolling_window, 'C0', label="Sim", linewidth=1.2
            )
            ax_big.plot_date(
                hour_dates, data_mean_wait_times_rolling_window, 'C3', label="Data", linewidth=1.2
            )

            plt.legend(prop={'size' : 12})

            ax_small = fig.add_axes((.1, .1, .8, .2))
            ax_small.plot_date(
                hour_dates, sim_mean_wait_times_rolling_window_err, 'C0', label="_", linewidth=1.2
            )
            ax_small.plot_date(
                hour_dates, data_mean_wait_times_rolling_window_err,
                'C3', label="_", linewidth=1.2
            )

            ax_big.set_ylabel("Moving averge wait time (h)", fontsize=14)
            ax_big.set_xticklabels([])
            ax_big.set_ylim(bottom=0.0)
            ax_small.set_xlabel("Middle hour of window", fontsize=14)
            ax_small.set_ylabel("Moving std dev wait time (h)", fontsize=14)
            ax_small.set_ylim(bottom=0.0)

            fig.savefig(
                os.path.join(PLOT_DIR, "wait_times_rolling_window{}.pdf".format(args.save_suffix)),
                bbox_inches="tight"
            )
            to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for sim_mean_wait_times_rolling_window, label in zip(
            sims_mean_wait_times_rolling_window, args.labels
        ):
            ax.plot_date(hour_dates, sim_mean_wait_times_rolling_window, "-", label=label)
        if not args.no_data_comparison:
            ax.plot_date(hour_dates, data_mean_wait_times_rolling_window, "k--", label="Data")

        ax.set_ylabel("Mean Wait Time")
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Middle Hour of Rolling Window")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "wait_times_rolling_window_noerr{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

        # Rolling window mean bd slowdown
        job_to_bdslowdown_sim = lambda job: (
            max((job.endlimit - job.submit) / max(job.reqtime, bd_threshold), 1)
        )

        sims_mean_bdslowdowns_rolling_window, sims_mean_bdslowdowns_rolling_window_err = [], []

        for job_history in job_histories:
            means, errs = rolling_window(job_history, job_to_bdslowdown_sim, hours, window_hrs)
            sims_mean_bdslowdowns_rolling_window.append(means)
            sims_mean_bdslowdowns_rolling_window_err.append(errs)

        if not args.no_data_comparison:
            job_to_bdslowdown_data = lambda job: (
                max(
                    (
                        (job.true_job_start + job.reqtime - job.true_submit) /
                        max(job.reqtime, bd_threshold)
                    ),
                    1
                )
            )

            means, errs = rolling_window(
                job_history, job_to_bdslowdown_data, hours, window_hrs, data=True
            )
            data_mean_bdslowdowns_rolling_window = means
            data_mean_bdslowdowns_rolling_window_err = errs

        # The plot with the error band will be horrible for multiple experiments at once
        if len(sims_mean_bdslowdowns_rolling_window) == 1:
            fig = plt.figure(1, figsize=(12, 8))

            sim_mean_bdslowdowns_rolling_window_err = sims_mean_bdslowdowns_rolling_window_err[0]
            sim_mean_bdslowdowns_rolling_window = sims_mean_bdslowdowns_rolling_window[0]

            ax_big = fig.add_axes((.1, .32, .8, .58))
            ax_big.plot_date(
                hour_dates, sim_mean_bdslowdowns_rolling_window, 'C0', label="Sim", linewidth=1.2
            )
            ax_big.plot_date(
                hour_dates, data_mean_bdslowdowns_rolling_window,
                'C3', label="Data", linewidth=1.2
            )

            plt.legend()

            ax_small = fig.add_axes((.1, .1, .8, .2))
            ax_small.plot_date(
                hour_dates, sim_mean_bdslowdowns_rolling_window_err,
                'C0', label="_", linewidth=1.2
            )
            ax_small.plot_date(
                hour_dates, data_mean_bdslowdowns_rolling_window_err,
                'C3', label="_", linewidth=1.2
            )

            ax_big.set_ylabel("Mean Bounded Slowdown")
            ax_big.set_xticklabels([])
            ax_big.set_ylim(bottom=1.0)
            ax_small.set_xlabel("Middle Hour of Rolling Window")
            ax_small.set_ylabel("Std dev Bounded Slowdown")
            ax_small.set_ylim(bottom=0.0)

            fig.savefig(
                os.path.join(
                    PLOT_DIR,
                    "bd_slowdowns_rolling_window{}.pdf".format(args.save_suffix)
                ),
                bbox_inches="tight"
            )
            to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for sim_mean_bdslowdowns_rolling_window, label in zip(
            sims_mean_bdslowdowns_rolling_window, args.labels
        ):
            ax.plot_date(hour_dates, sim_mean_bdslowdowns_rolling_window, "-", label=label)
        if not args.no_data_comparison:
            ax.plot_date(hour_dates, data_mean_bdslowdowns_rolling_window, "k--", label="Data")

        ax.set_ylabel("Mean Bounded Slowdown")
        ax.set_ylim(bottom=1.0)
        ax.set_xlabel("Middle Hour of Rolling Window")
        plt.legend()

        plt.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "bd_slowdowns_rolling_window_noerr{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

    if "rolling_window_qos" in args.plots:
        qos_sim_mean_wait_times_rolling_window = {
            "all" : sims_mean_wait_times_rolling_window[0]
        }
        qos_data_mean_wait_times_rolling_window = {
            "all" : data_mean_wait_times_rolling_window
        }
        qos_sim_mean_bdslowdowns_rolling_window = {
            "all" : sims_mean_bdslowdowns_rolling_window[0]
        }
        qos_data_mean_bdslowdowns_rolling_window = {
            "all" : data_mean_bdslowdowns_rolling_window
        }

        qos_job_history = defaultdict(list)

        for job in job_history:
            qos_job_history[job.qos.name].append(job)

        for qos, jobs in qos_job_history.items():
            # short goes through instantly and there are too few largescale
            if qos == "largescale" or qos == "short" or qos == "reservation":
                continue

            means, _ = rolling_window(jobs, job_to_wait_sim, hours, window_hrs)
            qos_sim_mean_wait_times_rolling_window[qos] = means
            means, _ = rolling_window(jobs, job_to_wait_data, hours, window_hrs, data=True)
            qos_data_mean_wait_times_rolling_window[qos] = means

            means, _ = rolling_window(jobs, job_to_bdslowdown_sim, hours, window_hrs)
            qos_sim_mean_bdslowdowns_rolling_window[qos] = means
            means, _ = rolling_window(jobs, job_to_bdslowdown_data, hours, window_hrs, data=True)
            qos_data_mean_bdslowdowns_rolling_window[qos] = means

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        color_cycle = cycle(
            ("C0", "C0", "C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4", "C5", "C5", "C6", "C6")
        )

        # ax.plot_date(
        #     hour_dates, qos_sim_mean_wait_times_rolling_window.pop("all"), fmt="-k", label="all"
        # )
        # ax.plot_date(
        #     hour_dates, qos_data_mean_wait_times_rolling_window.pop("all"), fmt="--k", label="_"
        # )
        for qos in qos_sim_mean_wait_times_rolling_window:
            sim_mean_wait_times_rolling_window = qos_sim_mean_wait_times_rolling_window[qos]
            data_mean_wait_times_rolling_window = qos_data_mean_wait_times_rolling_window[qos]

            ax.plot_date(
                hour_dates, qos_sim_mean_wait_times_rolling_window[qos],
                fmt="-" + next(color_cycle), label=qos
            )
            ax.plot_date(
                hour_dates, qos_data_mean_wait_times_rolling_window[qos],
                fmt="--" + next(color_cycle), label="_"
            )

        ax.set_ylabel("Mean Wait Time", fontsize=18)
        ax.set_xlabel("Middle Hour of Rolling Window", fontsize=18)
        ax.set_yscale("log")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "wait_times_rolling_window_byqos{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

        qos_sim_mean_wait_times_rolling_window_6 = {
            qos : waits
            for qos, waits in qos_sim_mean_wait_times_rolling_window.items()
                if qos != "all"
        }
        means, _ = rolling_window(
            qos_job_history["largescale"], job_to_wait_sim, hours, window_hrs
        )
        qos_sim_mean_wait_times_rolling_window_6["largescale"] = means
        qos_data_mean_wait_times_rolling_window_6 = {
            qos : waits
            for qos, waits in qos_data_mean_wait_times_rolling_window.items()
                if qos != "all"
        }
        means, _ = rolling_window(
            qos_job_history["largescale"], job_to_wait_data, hours, window_hrs, data=True
        )
        qos_data_mean_wait_times_rolling_window_6["largescale"] = means

        fig, ax = plt.subplots(2, 3, figsize=(12, 8))

        min_wait_sim = min(
            min(waits)
            for qos, waits in qos_sim_mean_wait_times_rolling_window_6.items()
                if qos != "largescale" and qos != "reservation"
        )
        min_wait_data = min(
            min(waits)
            for qos, waits in qos_data_mean_wait_times_rolling_window_6.items()
                if qos != "largescale" and qos != "reservation"
        )
        min_wait = min(min_wait_data, min_wait_sim)
        max_wait_sim = max(
            max(waits)
            for qos, waits in qos_sim_mean_wait_times_rolling_window_6.items()
        )
        max_wait_data = max(
            max(waits)
            for qos, waits in qos_data_mean_wait_times_rolling_window_6.items()
        )
        max_wait = max(max_wait_sim, max_wait_data)

        for qos, a in zip(qos_sim_mean_wait_times_rolling_window_6, ax.flatten()):
            a.plot_date(
                hour_dates, qos_sim_mean_wait_times_rolling_window_6[qos], fmt='-C0', label="Sim"
            )
            a.plot_date(
                hour_dates, qos_data_mean_wait_times_rolling_window_6[qos], fmt='-C3', label="Data"
            )

            a.set_ylim(0.9 * min_wait, 1.1 * max_wait)
            a.set_xticks([])
            a.set_title(qos, fontsize=14)
            a.set_yscale("log")

        ax[0][2].legend(prop={"size" : 12})

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                "wait_times_rolling_window_byqos_subplots{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        color_cycle = cycle(
            ("C0", "C0", "C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4", "C5", "C5", "C6", "C6")
        )

        # ax.plot_date(
        #     hour_dates, qos_sim_mean_bdslowdowns_rolling_window.pop("all"), fmt="-k", label="all"
        # )
        # ax.plot_date(
        #     hour_dates, qos_data_mean_bdslowdowns_rolling_window.pop("all"), fmt="--k", label="_"
        # )
        for qos in qos_sim_mean_bdslowdowns_rolling_window:
            sim_mean_bdslowdowns_rolling_window = qos_sim_mean_bdslowdowns_rolling_window[qos]
            data_mean_bdslowdowns_rolling_window = qos_data_mean_bdslowdowns_rolling_window[qos]

            ax.plot_date(
                hour_dates, qos_sim_mean_bdslowdowns_rolling_window[qos],
                fmt="-" + next(color_cycle), label=qos
            )
            ax.plot_date(
                hour_dates, qos_data_mean_bdslowdowns_rolling_window[qos],
                fmt="--" + next(color_cycle), label="_"
            )

        ax.set_ylabel("Mean Bounded Slowdown", fontsize=18)
        ax.set_xlabel("Middle Hour of Rolling Window", fontsize=18)
        ax.set_yscale("log")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "bd_slowdowns_rolling_window_byqos{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

    if "rolling_window_partition" in args.plots:
        partition_sim_mean_wait_times_rolling_window = {}
        partition_data_mean_wait_times_rolling_window = {}
        partition_job_history = defaultdict(list)

        for job in job_history:
            partition_job_history[job.partition.name].append(job)

        for partition, jobs in partition_job_history.items():
            means, _ = rolling_window(jobs, job_to_wait_sim, hours, window_hrs)
            partition_sim_mean_wait_times_rolling_window[partition] = means
            means, _ = rolling_window(jobs, job_to_wait_data, hours, window_hrs, data=True)
            partition_data_mean_wait_times_rolling_window[partition] = means

        fig, ax = plt.subplots(1, 2, figsize=(12, 8))

        min_wait_sim = min(
            min(waits)
            for partition, waits in partition_sim_mean_wait_times_rolling_window.items()
        )
        min_wait_data = min(
            min(waits)
            for partition, waits in partition_data_mean_wait_times_rolling_window.items()
        )
        min_wait = min(min_wait_data, min_wait_sim)
        max_wait_sim = max(
            max(waits)
            for partition, waits in partition_sim_mean_wait_times_rolling_window.items()
        )
        max_wait_data = max(
            max(waits)
            for partition, waits in partition_data_mean_wait_times_rolling_window.items()
        )
        max_wait = max(max_wait_sim, max_wait_data)

        for partition, a in zip(partition_sim_mean_wait_times_rolling_window, ax.flatten()):
            a.plot_date(
                hour_dates, partition_sim_mean_wait_times_rolling_window[partition], fmt='-C0'
            )
            a.plot_date(
                hour_dates, partition_data_mean_wait_times_rolling_window[partition], fmt='-C3'
            )

            a.set_ylim(0.9 * min_wait, 1.1 * max_wait)
            a.set_xticklabels([])
            a.set_title(partition)
            a.set_yscale("log")

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                "wait_times_rolling_window_bypartition_subplots{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

    if "total_allocnodes_timeseries" in args.plots:
        data_alloc_nodes, data_minutes, sim_alloc_nodes, sim_minutes = total_alloc_nodes(
            job_history
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(sim_minutes, sim_alloc_nodes, "C0", label="Sim", linewidth=0.75, alpha=0.8)
        ax.plot_date(data_minutes, data_alloc_nodes, "C1", label="Data", linewidth=0.75, alpha=0.8)

        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Number of Allocated Nodes", fontsize=18)
        ax.set_ylim(
            max(data_alloc_nodes) * 0.5 if max(data_alloc_nodes) > 2000 else 0,
            min(
                len(controller.partitions.nodes),
                max(max(data_alloc_nodes), max(sim_alloc_nodes)) * 1.2
            )
        )
        ax.grid(axis="y")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOT_DIR, "total_allocnodes_bytime{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

        start_tick_sim, end_tick_sim, start_tick_data, end_tick_data = None, None, None, None
        for i_minute, minute in enumerate(sim_minutes):
            if (
                start_tick_sim is None and
                minute.month == 12 and minute.day == 9 and minute.hour == 9
            ):
                start_tick_sim = i_minute
            if (
                end_tick_sim is None and
                minute.month == 12 and minute.day == 10 and minute.hour == 8
            ):
                end_tick_sim = i_minute
        for i_minute, minute in enumerate(data_minutes):
            if (
                start_tick_data is None and
                minute.month == 12 and minute.day == 9 and minute.hour == 9
            ):
                start_tick_data = i_minute
            if (
                end_tick_data is None and
                minute.month == 12 and minute.day == 10 and minute.hour == 8
            ):
                end_tick_data = i_minute
        sim_alloc_nodes_crop = sim_alloc_nodes[start_tick_sim:end_tick_sim]
        sim_minutes_crop = sim_minutes[start_tick_sim:end_tick_sim]
        data_alloc_nodes_crop = data_alloc_nodes[start_tick_data:end_tick_data]
        data_minutes_crop = data_minutes[start_tick_data:end_tick_data]

        sim_alloc_nodes_crop *= 100 / 5860
        data_alloc_nodes_crop *= 100 / 5860

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(
            sim_minutes_crop, sim_alloc_nodes_crop, "C0", label="Sim", linewidth=0.75, alpha=0.8
        )
        ax.plot_date(
            data_minutes_crop, data_alloc_nodes_crop, "C3", label="Data", linewidth=0.75, alpha=0.8
        )

        ax.set_title("Utilisation sampled each minute", fontsize=22)
        ax.set_xlabel("Date (minute resolution)", fontsize=22)
        ax.set_ylabel("Utilisation (%)", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylim(bottom=55, top=100)
        ax.set_yticks([60,70,80,90,100])
        ax.grid(axis="both")
        plt.legend(prop={'size': 18})

        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOT_DIR, "total_allocnodes_bytime_crop{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

        sim_hours, sim_alloc_nodes_hour = [], []
        for i_minute, minute in enumerate(sim_minutes):
            if minute.minute == 0:
                sim_hours.append(minute)
                sim_alloc_nodes_hour.append(
                    np.mean(
                        [ alloc_nodes for alloc_nodes in sim_alloc_nodes[i_minute:i_minute+60] ]
                    )
                )

        data_hours, data_alloc_nodes_hour = [], []
        for i_minute, minute in enumerate(data_minutes):
            if minute.minute == 0:
                data_hours.append(minute)
                data_alloc_nodes_hour.append(
                    np.mean(
                        [ alloc_nodes for alloc_nodes in data_alloc_nodes[i_minute:i_minute+60] ]
                    )
                )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(
            sim_hours, sim_alloc_nodes_hour, "C0", label="Sim", linewidth=0.75, alpha=0.8
        )
        ax.plot_date(
            data_hours, data_alloc_nodes_hour, "C1", label="Data", linewidth=0.75, alpha=0.8
        )

        ax.set_xlabel("Date (hour resolution)", fontsize=18)
        ax.set_ylabel("Number of allocated nodes hourly average", fontsize=18)
        ax.set_ylim(
            max(data_alloc_nodes_hour) * 0.5 if max(data_alloc_nodes_hour) > 2000 else 0,
            min(
                len(controller.partitions.nodes),
                max(max(data_alloc_nodes_hour), max(sim_alloc_nodes_hour)) * 1.2
            )
        )
        ax.grid(axis="y")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                "total_allocnodes_bytime_hourlyavg{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

        start_tick_sim, end_tick_sim, start_tick_data, end_tick_data = None, None, None, None
        for i_hour, hour in enumerate(sim_hours):
            if start_tick_sim is None and (hour.month == 12 and hour.day == 9 and hour.hour == 9):
                start_tick_sim = i_hour
            if end_tick_sim is None and (hour.month == 12 and hour.day == 10 and hour.hour == 8):
                end_tick_sim = i_hour
        for i_hour, hour in enumerate(data_hours):
            if start_tick_data is None and (hour.month == 12 and hour.day == 9 and hour.hour == 9):
                start_tick_data = i_hour
            if end_tick_data is None and (hour.month == 12 and hour.day == 10 and hour.hour == 8):
                end_tick_data = i_hour
        sim_alloc_nodes_hour_crop = np.array(sim_alloc_nodes_hour[start_tick_sim:end_tick_data])
        sim_hours_crop = np.array(sim_hours[start_tick_sim:end_tick_sim])
        data_alloc_nodes_hour_crop = np.array(data_alloc_nodes_hour[start_tick_data:end_tick_data])
        data_hours_crop = np.array(data_hours[start_tick_data:end_tick_data])

        sim_alloc_nodes_hour_crop *= 100 / 5860
        data_alloc_nodes_hour_crop *= 100 / 5860

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(
            sim_hours_crop, sim_alloc_nodes_hour_crop, "C0",
            label="Sim", linewidth=0.75, alpha=0.8
        )
        ax.plot_date(
            data_hours_crop, data_alloc_nodes_hour_crop, "C3",
            label="Data", linewidth=0.75, alpha=0.8
        )

        ax.set_xlabel("Date (hourly resolution)", fontsize=16)
        ax.set_ylabel("Utilisation (%)", fontsize=16)
        ax.set_ylim(bottom=55, top=100)
        ax.set_yticks([60,70,80,90,100])
        ax.grid(axis="both")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                "total_allocnodes_bytime_hourlyaverage_crop{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

    if "queue_size_timeseries" in args.plots:
        ret = q_size(job_history)
        data_q_length, data_q_length_nodes, data_minutes = ret[0], ret[1], ret[2]
        sim_q_length, sim_q_length_nodes, sim_minutes = ret[3], ret[4], ret[5]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(sim_minutes, sim_q_length, "C0", label="Sim", linewidth=0.5)
        ax.plot_date(data_minutes, data_q_length, "C1", label="Data", linewidth=0.5)

        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Queue Size (Jobs)", fontsize=18)
        plt.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "queue_size_jobs{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(sim_minutes, sim_q_length_nodes, "C0", label="Sim", linewidth=0.5)
        ax.plot_date(data_minutes, data_q_length_nodes, "C1", label="Data", linewidth=0.5)

        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Queue Size (Nodes)", fontsize=18)
        plt.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "queue_size_nodes{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "spider_mean_metrics" in args.plots:
        baseline_data = spider_plot_metrics(job_histories[args.labels.index(args.baseline_label)])

        spider_plot_data = {}

        for job_history, label in zip(job_histories, args.labels):
            if label == args.baseline_label:
                continue

            spider_plot_data[label] = spider_plot_metrics(job_history)

            for metric in spider_plot_data[label]:
                spider_plot_data[label][metric] /= baseline_data[metric]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)

        categories = ["avg_slowdown", "avg_wait", "max_wait", "avg_response"]
        angles = [ i / float(len(categories)) * 2 * np.pi for i in range(len(categories)) ]
        angles += angles[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        category_labels = ["mean(slowdown)", "mean(wait)", "max(wait)", "mean\n(response)"]
        plt.xticks(angles[:-1], category_labels, size=18)
        half = len(ax.get_xticklabels()) // 2
        for label in ax.get_xticklabels()[1:half]:
            label.set_horizontalalignment("left")
        for label in ax.get_xticklabels()[-half+1:]:
            label.set_horizontalalignment("right")
        ax.tick_params(axis='x', which='major', pad=10)
        ax.set_rlabel_position(0)
        # plt.yticks([0.75,1,1.25], ["0.75","1","1.25"], color="grey", size=14)
        # plt.ylim(0.7,1.3)
        # For highprio experiments
        plt.yticks([0.9,1,1.1,1.2], ["0.9","1","1.1","1.2"], color="grey", size=18)
        plt.ylim(0.88,1.25)
        # For largescale experiments
        # plt.yticks([0.9,1], ["0.9","1"], color="grey", size=18)
        # plt.ylim(0.85,1.05)

        for label, plot_data in spider_plot_data.items():
            vals = [ 1 / plot_data[metric] for metric in categories ]
            vals += vals[:1]
            colour = next(ax._get_lines.prop_cycler)["color"]
            ax.plot(angles, vals, linewidth=2, linestyle='solid', c=colour, label=label)

        ax.plot(
            angles, [1] * len(angles), linewidth=3, linestyle='solid', c='k', label="Baseline"
        )

        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.085), fontsize=18, ncol=5, frameon=False)
        plt.subplots_adjust(left=0.0, top=0.9, right=1.00, bottom=0.1)
        plt.title(
            (
                r"$\mathrm{metric}_{\mathrm{Baseline}}/$" +
                r"$\mathrm{metric}_{\mathrm{Experiment}}$ for all jobs"
            ),
            fontsize=22, pad=15
        )

        fig.savefig(os.path.join(PLOT_DIR, "spider_plot_metrics{}.pdf".format(args.save_suffix)))

    if "spider_mean_wait_qos" in args.plots:
        baseline_data = spider_plot_wait_qos(job_histories[args.labels.index(args.baseline_label)])

        spider_plot_data = {}
        standard_qos, replaced_with_standard = "standard", set()

        for job_history, label in zip(job_histories, args.labels):
            if label == args.baseline_label:
                continue

            spider_plot_data[label] = spider_plot_wait_qos(job_history)

            for metric in spider_plot_data[label]:
                if metric not in baseline_data:
                    baseline_data[metric] = baseline_data[standard_qos]
                    replaced_with_standard.add(metric)

                spider_plot_data[label][metric] /= baseline_data[metric]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)

        categories = list(baseline_data)
        angles = [ i / float(len(categories)) * 2 * np.pi for i in range(len(categories)) ]
        angles += angles[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        category_labels = [
            label + "\n(relative to standard)"
            if label in replaced_with_standard else
            label
            for label in baseline_data
        ]
        plt.xticks(angles[:-1], category_labels, size=18)
        half = len(ax.get_xticklabels()) // 2
        for label in ax.get_xticklabels()[1:half]:
            label.set_horizontalalignment("left")
        for label in ax.get_xticklabels()[-half+1:]:
            label.set_horizontalalignment("right")
        ax.tick_params(axis='x', which='major', pad=15)
        ax.set_rlabel_position(0)

        log, min_val = False, 1.0

        for label, plot_data in spider_plot_data.items():
            vals = [ 1 / plot_data[metric] for metric in categories ]
            vals += vals[:1]

            colour = next(ax._get_lines.prop_cycler)["color"]
            ax.plot(angles, vals, linewidth=2, linestyle='solid', c=colour, label=label)

            print(categories)
            print(vals)

            min_val = min(min(vals), min_val)

            if any(val > 2 for val in vals):
                log = True

        ax.plot(
            angles, [1] * len(angles), linewidth=3, linestyle='solid', c='k', label="Baseline"
        )

        if log:
            ax.set_yscale("symlog", linthresh=0.1)
            # plt.yticks([0.75,1,2,4,8], ["0.75","1","2","4","8"], color="grey", size=14)
            # plt.ylim(0.55,10)
            # For highprio experiments
            plt.yticks([0.5,0.75,1,2,4,8], ["0.5","0.75","1","2","4","8"], color="grey", size=18)
            plt.ylim(0.45,10)
        else:
            if min_val > 0.5:
                plt.yticks([0.5,0.75,1], ["0.5","0.75","1"], color="grey", size=18)
                plt.ylim(0.3,1.2)
            else:
                plt.yticks([0.25,0.5,0.75,1], ["0.25","0.5","0.75","1"], color="grey", size=18)
                plt.ylim(0.0,1.2)


        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.085), fontsize=18, ncol=5, frameon=False)
        plt.subplots_adjust(left=0.0, top=0.9, right=1.00, bottom=0.1)
        plt.title(
            (
                r"$\mathrm{mean(waits)}_{\mathrm{Baseline}}/$" +
                r"$\mathrm{mean(waits)}_{\mathrm{Experiment}}$ by job QOS"
            ),
            fontsize=22, pad=15
        )

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "spider_plot_wait_qos{}.pdf".format(args.save_suffix)))

    if "power" in args.plots:
        hours = [
            controllers[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr)
            for hr in range(
                int((controller.times[-1] - controller.times[0]).total_seconds() / 60 / 60) + 1
            )
        ]
        hours = np.array(hours)

        power = power_usage(hours, job_history, len(controller.partitions.nodes))

        hour_dates = matplotlib.dates.date2num(hours)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(hour_dates, power, 'k', linewidth=0.5)

        ax.set_ylim(0.9 * power.min(), 1.1 * power.max())
        ax.set_ylabel("Power (MW)")

        # fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "power_usage{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "power_diff" in args.plots:
        if len(job_histories) != 2:
            raise NotImplementedError

        hours = [
            controllers[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr)
            for hr in range(
                int((controller.times[-1] - controller.times[0]).total_seconds() / 60 / 60) + 1
            )
        ]
        hours = np.array(hours)

        i_baseline = args.labels.index(args.baseline_label)
        i_exp = 0 if i_baseline == 1 else 1

        power_baseline = power_usage(
            hours, job_histories[i_baseline], len(controller.partitions.nodes)
        )
        power_exp = power_usage(hours, job_histories[i_exp], len(controller.partitions.nodes))

        hour_dates = matplotlib.dates.date2num(hours)

        slice_r = len(hour_dates) - 336
        while slice_r > 336:
            slice_l = slice_r - 336

            fig, ax = plot_power_diff(
                hours, hour_dates, power_baseline, power_exp, slice_l, slice_r, vlines=False
            )
            fig.savefig(
                os.path.join(
                    PLOT_DIR,
                    "power_usage_diff_2weeks_slicer{}{}.pdf".format(slice_r, args.save_suffix)
                )
            )
            to_plot_or_not_to_plot(args.batch)

            slice_r -= 336

        fig, ax = plot_power_diff(
            hours, hour_dates, power_baseline, power_exp, None, None, vlines=False
        )
        fig.savefig(os.path.join(PLOT_DIR, "power_usage_diff{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

        power_baseline_peak, power_exp_peak = [], []

        for i_hour, hour in enumerate(hours):
            if 11 <= hour.hour < 16:
                power_baseline_peak.append(power_baseline[i_hour])
                power_exp_peak.append(power_exp[i_hour])

        baseline_peak_mu, exp_peak_mu = np.mean(power_baseline_peak), np.mean(power_exp_peak)
        print("Full time range:")
        print("Baseline peak mean power {} MW".format(baseline_peak_mu))
        print("Exp peak mean power {} MW".format(exp_peak_mu))
        print(
            "Baseline - Exp {} MW ({} KW)".format(
                baseline_peak_mu - exp_peak_mu, (baseline_peak_mu - exp_peak_mu) * 1000
            )
        )

        start_date = datetime.datetime.strptime("2022-10-24", "%Y-%m-%d")

        for slice_l, hour in enumerate(hours):
            if hour.replace(hour=0, minute=0, second=0) == start_date:
                slice_r = slice_l + 336
                fig, ax = plot_power_diff(
                    hours, hour_dates, power_baseline, power_exp, slice_l, slice_r, vlines=False
                )
                fig.savefig(
                    os.path.join(
                        PLOT_DIR,
                        "power_usage_diff_2weeks_slicer{}{}.pdf".format(slice_r, args.save_suffix)
                    )
                )
                to_plot_or_not_to_plot(args.batch)

                slice_r = slice_l + 240
                fig, ax = plot_power_diff(
                    hours, hour_dates, power_baseline, power_exp, slice_l, slice_r, vlines=False
                )
                fig.savefig(
                    os.path.join(
                        PLOT_DIR,
                        "power_usage_diff_10days_slicer{}{}.pdf".format(slice_r, args.save_suffix)
                    )
                )
                to_plot_or_not_to_plot(args.batch)

                break

        start_date = datetime.datetime.strptime("2023-01-09", "%Y-%m-%d")

        for slice_l, hour in enumerate(hours):
            if hour.replace(hour=0, minute=0, second=0) == start_date:
                slice_r = slice_l + 288
                fig, ax = plot_power_diff(
                        hours, hour_dates, power_baseline, power_exp, slice_l, slice_r, vlines=False
                        )
                fig.savefig(
                        os.path.join(
                            PLOT_DIR,
                            "power_usage_diff_12days_slicer{}{}.pdf".format(slice_r, args.save_suffix)
                            )
                        )
                to_plot_or_not_to_plot(args.batch)

                break

    if "power_diff_data" in args.plots:
        if len(job_histories) != 1:
            raise NotImplementedError

        hours = [
                controllers[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr)
                for hr in range(
                    int((controller.times[-1] - controller.times[0]).total_seconds() / 60 / 60) + 1
                    )
                ]
        hours = np.array(hours)

        power_data = power_usage(hours, job_history, len(controller.partitions.nodes), data=True)
        power_sim = power_usage(hours, job_history, len(controller.partitions.nodes))

        hour_dates = matplotlib.dates.date2num(hours)

        slice_r = len(hour_dates) - 336
        while slice_r > 336:
            slice_l = slice_r - 336

            fig, ax = plot_power_diff(
                hours, hour_dates, power_data, power_sim, slice_l, slice_r,
                vlines=False,
                title=(
                    "Difference between power usage for Simulation and Experiment"
                    "(sampled hourly)"
                ),
                legend_labels=["Data > Experiment", "Simulation > Data", "11am - 4pm"]
            )
            fig.savefig(
                os.path.join(
                    PLOT_DIR,
                    "power_usage_diff_data_2weeks_slicer{}{}.pdf".format(slice_r, args.save_suffix)
                )
            )
            to_plot_or_not_to_plot(args.batch)

            slice_r -= 336

        fig, ax = plot_power_diff(
            hours, hour_dates, power_data, power_sim, None, None,
            vlines=False,
            title=(
                "Difference between power usage for Simulation and Experiment"
                "(sampled hourly)"
            ),
            legend_labels=["Data > Experiment", "Simulation > Data", "11am - 4pm"]
        )
        fig.savefig(os.path.join(PLOT_DIR, "power_usage_diff_data{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sim", type=lambda sims: [ sim for sim in sims.split(',') ],
        help="Experiment to plot, can be comma delimited list to plot multiple experiments"
    )

    parser.add_argument(
        "--labels", default=["sim"], type=lambda plots: [ plot for plot in plots.split(',') ],
        help="Labels to use when plotting multiple experiments"
    )
    parser.add_argument(
        "--plots", default=[], type=lambda plots: [ plot for plot in plots.split(',') ],
        help=(
            "comma delimited list or plots to plot:\n"
            "(bdslowdowns_hist2d|wait_times_hist2d|top_projs|top_qccounts|top_users|qos_waits|"
            "partition_waits|rolling_window|rolling_window_qos|cumulative_throughput|"
            "total_allocnodes_timeseries|queue_size_timeseries|spider_mean_metrics|"
            "spider_mean_wait_qos|power|power_diff|power_diff_data)\n"
            "Plots that work for multiple experiments:\n"
            "(rolling_window|spider_mean_metrics|spider_mean_wait_qos)"
        )
    )

    parser.add_argument("--batch", action="store_true", help="Dont draw plots, just save")
    parser.add_argument(
        "--save_suffix", type=str, default="", help="Optional suffix to add to name of saved plots"
    )
    parser.add_argument(
        "--no_data_comparison", action="store_true", help="Dont plot the data with the sim"
    )
    parser.add_argument(
        "--plot_dir", type=str, default="/work/y02/y02/awilkins/data/plots/archer2_jobdata_plots",
        help="Override ARCHER2 plot dir"
    )
    parser.add_argument(
        "--days_ignore", type=float, default=4.0,
        help="ovveride the default ignore period (4 days) at the start and end of job data"
    )
    parser.add_argument("--rolling_window_days", type=float, default=14.0)
    parser.add_argument(
        "--baseline_label", type=str, default="baseline", help="ovverride baseline label"
    )

    args = parser.parse_args()

    if len(args.sim) != len(args.labels):
        parser.error("Need a label for each experiment being plotted")

    if len(args.sim) > 1:
        print("NOTE: Not all plots have been implemented to plot multiple experiments")

    return args


if __name__ == "__main__":
    main(parse_arguments())

