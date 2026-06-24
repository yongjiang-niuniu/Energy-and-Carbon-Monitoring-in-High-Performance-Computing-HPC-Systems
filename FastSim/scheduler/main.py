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

import argparse
from datetime import timedelta
import dill as pickle

import numpy as np

from controller import Controller


def main(args):
    controller = Controller(args.config_file)

    controller.run_sim(max_steps=args.max_steps)

    print_sim_result(controller)

    if args.dump_sim_to:
        with open(args.dump_sim_to, "wb") as f:
            pickle.dump(controller, f)


def print_sim_result(controller):
    max_submit = max(controller.job_history, key=lambda job: job.true_submit).true_submit
    job_history = [
        job
        for job in controller.job_history
            if (
                (
                    controller.init_time + timedelta(days=2) < job.true_submit <
                    max_submit - timedelta(days=2)
                ) and
                not job.ignore_in_eval
            )
    ]
    data_bd_slowdowns = [
        max(
            (
                (job.true_job_start + job.runtime - job.true_submit) /
                max(job.runtime, controller.config.bd_threshold)
            ),
            1
        )
        for job in job_history
    ]
    sim_bd_slowdowns = [
        max((job.end - job.submit) / max(job.runtime, controller.config.bd_threshold), 1)
        for job in job_history
    ]
    data_wait_times = [
        (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
        for job in job_history
    ]
    sim_wait_times = [
        (job.start - job.submit).total_seconds() / 60 / 60
        for job in job_history
    ]
    print(
        "True starts mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns), np.sum(data_bd_slowdowns)
        ) +
        "Scheduling sim mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(sim_bd_slowdowns), np.std(sim_bd_slowdowns), np.sum(sim_bd_slowdowns)
        ) +
        "True starts mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(data_wait_times), np.std(data_wait_times), np.sum(data_wait_times)
        ) +
        "Scheduling sim mean wait time={}+-{}hrs (total = {} hrs)\n".format(
            np.mean(sim_wait_times), np.std(sim_wait_times), np.sum(sim_wait_times)
        )
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", type=str)

    parser.add_argument("--dump_sim_to", type=str, default="", help="Pickle Controller after sim")

    parser.add_argument("--max_steps", type=int, default=0, help="Terminate sim after max steps")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(parse_arguments())

