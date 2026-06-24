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

import os
from datetime import timedelta
from collections import namedtuple

import yaml

# NOTE Priority weight defaults are set to ARCHER2 defaults currently
# NOTE: approx_excess_assocs remove a number of unused in workload traceassocs from the assoc tree,
# this is relevant since the fairshare factor scales with the tot number of user assocs. This
# happend because to capture all assocs they need to be dumped "withDeleted" so you end up with
# some assocs that never existed at any given time.
defaults = {
    "bd_threshold" : 60, "defer" : False, "default_queue_depth" : 100, "sched_interval" : 60,
    "sched_min_interval" : 2000000, "PriorityCalcPeriod" : 5, "bf_resolution" : 60,
    "bf_max_job_test" : 500, "bf_window" : 1440, "bf_interval" : 30, "bf_max_time" : 30,
    "bf_yield_interval" : 2000000, "bf_yield_sleep" : 500000, "bf_continue" : False,
    "hpe_restrictlong_sliding_reservations" : "const", "PriorityMaxAge" : 7,
    "PriorityDecayHalfLife" : 7, "PriorityWeightAge" : 0, "PriorityWeightFairshare" : 0,
    "PriorityWeightJobSize" : 0, "PriorityWeightPartition" : 0, "PriorityWeightQOS" : 0,
    "approx_excess_assocs" : 0, "JobRequeue" : 1, "approx_bf_try_per_sec" : 10, "KillWait" : 30,
    "OverTimeLimit" : 0, "nodes_down_in_blades" : False
}

vals_us = ["sched_min_interval", "bf_yield_interval", "bf_yield_sleep"]
vals_s = ["sched_interval", "bf_resolution", "bf_interval", "bf_max_time", "KillWait"]
vals_min = ["bd_threshold", "PriorityCalcPeriod", "bf_window", "OverTimeLimit"]
vals_days = ["PriorityMaxAge", "PriorityDecayHalfLife"]
vals_bool = ["JobRequeue"]

# TODO Include node/partition information dump once setup to read this
mandatory_fields = set(
    (
        "assocs_dump", "node_events_dump", "resv_dump", "job_dump", "slurm_conf",
        "considered_partitions", "qos_dump"
    )
)


def get_config(config_file):
    print("Reading config from {}".format(config_file))

    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    with open(config_dict["slurm_conf"], "r") as f:
        for line in f:
            if line[0] == "#":
                continue

            line = line.strip("\n")
            param = line.split("=")[0].strip(" ")

            if param == "SchedulerParameters":
                line = line.replace(" ", "")
                subparam_entries = line.lstrip(param + "=").split(",")
                for subparam_entry in subparam_entries:
                    if "=" not in subparam_entry:
                        defaults[subparam_entry] = True
                        continue
                    subparam, val = subparam_entry.split("=")
                    # NOTE: This assumes all time strings are of the form days-hrs, may also be
                    # days-hrs:mins:secs or hys:mins:secs. At least if I come accross this it
                    # will throw an error
                    if "-" in val:
                        val = int(val.split("-")[0]) + (int(val.split("-")[1]) / 24)
                    else:
                        val = int(val)
                    # Params can still be overidden in the yaml config so treat slurm.conf as
                    # the defaults for this system
                    defaults[subparam] = val

            elif param in defaults:
                val = line.split("=")[1].strip(" ")
                if "-" in val:
                    val = int(val.split("-")[0]) + (int(val.split("-")[1]) / 24)
                else:
                    val = int(val)
                defaults[param] = val

    missing_fields = mandatory_fields - set(config_dict.keys())
    if missing_fields:
        raise ValueError(
            "Missing mandatory fields {} in config file at {}".format(missing_fields, config_file)
        )

    for option in set(defaults.keys()) - set(config_dict.keys()):
        config_dict[option] = defaults[option]

    for option in vals_us:
        config_dict[option] = timedelta(microseconds=config_dict[option])
    for option in vals_s:
        config_dict[option] = timedelta(seconds=config_dict[option])
    for option in vals_min:
        config_dict[option] = timedelta(minutes=config_dict[option])
    for option in vals_days:
        config_dict[option] = timedelta(days=config_dict[option])
    for option in vals_bool:
        config_dict[option] = bool(config_dict[option])

    config_namedtuple = namedtuple("config", config_dict)
    config = config_namedtuple(**config_dict)

    return config

