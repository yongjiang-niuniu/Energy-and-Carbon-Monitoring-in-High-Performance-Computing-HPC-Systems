#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析一次 Stanage 模拟运行的结果：
  slurmctld.log(调度事实) + workload_profile.csv(作业画像/功率参数)
  => 每作业 提交/开始/结束/等待时间 + 能耗与碳排放估算

能耗模型(活动数据 x 功率系数, 与真实集群记账的思路一致):
  E_job [kWh] = (cores * W_core + gpus * W_gpu) * runtime_h / 1000 * PUE
  CO2 [g]    = E_job * CI          (CI: 电网碳强度 gCO2/kWh)

默认 PUE=1.2(现代数据中心典型值), CI=150 gCO2/kWh(近年英国电网均值量级),
两者都可用命令行参数覆盖 —— 这正是本项目后续要精细化的部分。

用法:
  python3 analyze_run.py [--log ../stanage-sim/slurmctld.log]
                         [--profile ../stanage-sim/workload_profile.csv]
                         [--pue 1.2] [--ci 150]
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.normpath(os.path.join(HERE, "..", "stanage-sim"))

TS_RE = r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\]"
RE_SUBMIT = re.compile(TS_RE + r".*_slurm_rpc_submit_batch_job: JobId=(\d+) ")
RE_START1 = re.compile(TS_RE + r".*sched: Allocate JobId=(\d+) NodeList=(\S+)")
RE_START2 = re.compile(TS_RE + r".*_start_job: Started JobId=(\d+) in (\S+) on (\S+)")
RE_DONE = re.compile(TS_RE + r".*_job_complete: JobId=(\d+) done")
RE_TIMEOUT = re.compile(TS_RE + r".*Time limit exhausted for JobId=(\d+)")


def ts(s):
    return datetime.fromisoformat(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(SIM_DIR, "slurmctld.log"))
    ap.add_argument("--profile", default=os.path.join(SIM_DIR, "workload_profile.csv"))
    ap.add_argument("--out", default=os.path.join(SIM_DIR, "job_results.csv"))
    ap.add_argument("--pue", type=float, default=1.2)
    ap.add_argument("--ci", type=float, default=150.0, help="gCO2/kWh")
    args = ap.parse_args()

    submit, start, done, nodelist, timed_out = {}, {}, {}, {}, set()
    with open(args.log, errors="replace") as f:
        for line in f:
            m = RE_SUBMIT.search(line)
            if m:
                submit.setdefault(int(m.group(2)), ts(m.group(1)))
                continue
            m = RE_START1.search(line)
            if m:
                jid = int(m.group(2))
                start.setdefault(jid, ts(m.group(1)))
                nodelist.setdefault(jid, m.group(3))
                continue
            m = RE_START2.search(line)
            if m:
                jid = int(m.group(2))
                start.setdefault(jid, ts(m.group(1)))
                nodelist.setdefault(jid, m.group(4))
                continue
            m = RE_DONE.search(line)
            if m:
                done.setdefault(int(m.group(2)), ts(m.group(1)))
                continue
            m = RE_TIMEOUT.search(line)
            if m:
                timed_out.add(int(m.group(2)))

    profile = {}
    with open(args.profile) as f:
        for row in csv.DictReader(f):
            profile[int(row["job_sim_id"])] = row

    rows, agg = [], defaultdict(lambda: dict(jobs=0, wait=0.0, kwh=0.0,
                                             coreh=0.0, gpuh=0.0))
    total_kwh = 0.0
    for jid, p in sorted(profile.items()):
        sub, st, dn = submit.get(jid), start.get(jid), done.get(jid)
        wait_min = (st - sub).total_seconds() / 60.0 if sub and st else None
        run_h = (dn - st).total_seconds() / 3600.0 if st and dn else None
        cores, gpus = int(p["cores"]), int(p["gpus"])
        kwh = None
        if run_h is not None:
            power_w = cores * float(p["cpu_w_per_core"]) + gpus * float(p["gpu_w"])
            kwh = power_w * run_h / 1000.0 * args.pue
            total_kwh += kwh
            a = agg[p["class"]]
            a["jobs"] += 1
            a["wait"] += wait_min or 0.0
            a["kwh"] += kwh
            a["coreh"] += cores * run_h
            a["gpuh"] += gpus * run_h
        state = ("TIMEOUT" if jid in timed_out else
                 "COMPLETED" if dn else "RUNNING/UNKNOWN" if st else "PENDING")
        rows.append([jid, p["class"], p["user"], p["partition"], cores, gpus,
                     sub.isoformat() if sub else "", st.isoformat() if st else "",
                     dn.isoformat() if dn else "",
                     round(wait_min, 1) if wait_min is not None else "",
                     round(run_h, 3) if run_h is not None else "",
                     round(kwh, 2) if kwh is not None else "",
                     round(kwh * args.ci / 1000.0, 3) if kwh is not None else "",
                     state, nodelist.get(jid, "")])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "class", "user", "partition", "cores", "gpus",
                    "submit", "start", "end", "wait_min", "runtime_h",
                    "energy_kwh", "co2_kg", "state", "nodes"])
        w.writerows(rows)

    n_done = sum(1 for r in rows if r[13] in ("COMPLETED", "TIMEOUT"))
    n_to = sum(1 for r in rows if r[13] == "TIMEOUT")
    print("jobs in profile : %d" % len(profile))
    print("finished        : %d  (timeout: %d)" % (n_done, n_to))
    print("total energy    : %.1f kWh  (PUE=%.2f)" % (total_kwh, args.pue))
    print("total CO2       : %.1f kg   (CI=%.0f gCO2/kWh)"
          % (total_kwh * args.ci / 1000.0, args.ci))
    print()
    print("%-16s %5s %10s %10s %10s %12s" %
          ("class", "jobs", "avg_wait_m", "core-h", "gpu-h", "energy_kWh"))
    for c, a in sorted(agg.items(), key=lambda kv: -kv[1]["kwh"]):
        print("%-16s %5d %10.1f %10.1f %10.1f %12.1f" %
              (c, a["jobs"], a["wait"] / max(a["jobs"], 1),
               a["coreh"], a["gpuh"], a["kwh"]))
    print()
    print("per-job results -> %s" % args.out)


if __name__ == "__main__":
    main()
