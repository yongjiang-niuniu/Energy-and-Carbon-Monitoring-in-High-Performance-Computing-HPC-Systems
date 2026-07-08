#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic HPC workload generator for the Stanage Slurm simulator.

为什么 placeholder job 可以替换成"真实" workload：
  模拟器里作业从不真正执行（pseudo.job 从未被读取），调度器看到的只有
  资源画像：partition / 节点数 / 核数 / GPU 数 / 申请 walltime(-t) /
  实际运行时长(-sim-walltime) / 提交时刻(-dt)。因此只要这些参数服从真实
  集群的统计规律，模拟出的排队、回填、资源占用与能耗活动数据就是真实的。

统计规律来源（HPC 工作负载研究的共识结论）：
  * Feitelson Parallel Workloads Archive: 运行时呈重尾(对数正态)分布，
    节点数偏好 2 的幂，作业到达呈日间/夜间周期的泊松过程。
  * Fugaku F-DATA / NERSC / ARCHER2 负载特征分析: 大量短小作业贡献了
    作业数，少量大 MPI 作业贡献了大部分 node-hours；用户申请的 walltime
    普遍是实际用时的 2~10 倍，且集中在整点值(1h/4h/12h/24h...)；
    少部分作业撞墙超时(TIMEOUT)或启动后迅速失败。
  * AI 负载(训练/微调/推理)集中在 GPU 分区，训练作业长且 GPU 利用率高，
    交互/调试作业短且频繁。

生成的作业类别（对应 Stanage 分区，功率参数用于后续能耗估算）:
  htc_short      标准分区单节点参数扫描/生信流水线(数量最多)
  mpi_physics    2~32 节点 MPI 物理模拟(CFD/MD/格点QCD, node-hours 主力)
  mpi_capability 64+ 节点大规模模拟(少而大)
  ai_train_a100  A100 整机训练(4 GPU/节点, 长作业)
  ai_dev_gpu     GPU 调试/微调/批量推理(1-2 GPU, 短作业)
  llm_h100nvl    H100-NVL LLM 训练(4 GPU, 长作业)
  bigmem / hugemem  基因组组装、量子化学等大内存作业

输出:
  stanage-sim/sim.events.realistic   模拟器事件文件(不能含注释行!)
  stanage-sim/workload_profile.csv   每作业画像(供能耗/碳排放分析连接)

用法:
  python3 generate_workload.py [--hours 12] [--jobs 140] [--seed 42]
"""

import argparse
import csv
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "stanage-sim"))

# 用户及其活跃度权重（真实集群中少数用户贡献大部分作业）
USERS = [
    ("alice", 30), ("bob", 20), ("carol", 15), ("dave", 12),
    ("erin", 9), ("frank", 6), ("grace", 5), ("heidi", 3),
]

# 功率参数（TDP 近似，供 analyze_run.py 估算能耗）
#   Ice Lake 8358: 250W/32c ~ 7.8 W/core;  GPU 节点host CPU按 6 W/core
CPU_W_PER_CORE = {"standard": 7.8, "bigmem": 7.8, "hugemem": 7.8,
                  "gpu": 6.0, "gpu-h100": 6.0, "gpu-h100-nvl": 6.0}
GPU_W = {"a100": 400.0, "h100": 350.0, "h100nvl": 400.0}

# 用户申请 walltime 偏好的"整点"档位（分钟）
ROUND_REQ_MIN = [15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320]


def lognormal_capped(rng, median_s, sigma, cap_s, floor_s=60):
    """重尾运行时分布: median 给中位数, cap 防止超出模拟窗口太多."""
    v = rng.lognormvariate(math.log(median_s), sigma)
    return int(max(floor_s, min(v, cap_s)))


def round_up_request(actual_s, factor):
    """用户会把申请时长取整到常见档位, 且 >= 实际用时*factor."""
    want_min = actual_s * factor / 60.0
    for m in ROUND_REQ_MIN:
        if m >= want_min:
            return m
    return ROUND_REQ_MIN[-1]


# ---------------------------------------------------------------------------
# 作业类别定义。每个函数返回 dict:
#   partition, nodes, ntasks, gres(str|None), gpu_type, gpus_total,
#   actual_s(实际运行秒), req_min(申请分钟), outcome(completed|failed|timeout)
# ---------------------------------------------------------------------------

def gen_htc_short(rng):
    n = rng.choices([1, 4, 8, 16, 32], weights=[25, 25, 25, 15, 10])[0]
    if rng.random() < 0.08:  # 启动后迅速失败(配置错/输入错)
        actual = rng.randint(20, 300)
        outcome = "failed"
    else:
        actual = lognormal_capped(rng, median_s=25 * 60, sigma=1.1, cap_s=3 * 3600)
        outcome = "completed"
    req = round_up_request(actual, rng.uniform(1.5, 6.0))
    return dict(partition="standard", nodes=1, ntasks=n, gres=None,
                gpu_type=None, gpus_total=0,
                actual_s=actual, req_min=req, outcome=outcome)


def gen_mpi_physics(rng):
    nodes = rng.choices([2, 4, 8, 16, 32], weights=[30, 30, 20, 15, 5])[0]
    if rng.random() < 0.05:  # 撞墙超时: 申请时长小于实际需要
        req = rng.choice([240, 480, 720])
        actual = int(req * 60 * rng.uniform(1.05, 1.3))
        outcome = "timeout"
    else:
        actual = lognormal_capped(rng, median_s=3.5 * 3600, sigma=0.9,
                                  cap_s=18 * 3600, floor_s=600)
        req = round_up_request(actual, rng.uniform(1.3, 4.0))
        outcome = "completed"
    return dict(partition="standard", nodes=nodes, ntasks=nodes * 64,
                gres=None, gpu_type=None, gpus_total=0,
                actual_s=actual, req_min=req, outcome=outcome)


def gen_mpi_capability(rng):
    nodes = rng.choices([64, 96, 128], weights=[60, 30, 10])[0]
    actual = lognormal_capped(rng, median_s=6 * 3600, sigma=0.6,
                              cap_s=16 * 3600, floor_s=3600)
    req = round_up_request(actual, rng.uniform(1.3, 2.5))
    return dict(partition="standard", nodes=nodes, ntasks=nodes * 64,
                gres=None, gpu_type=None, gpus_total=0,
                actual_s=actual, req_min=req, outcome="completed")


def gen_ai_train_a100(rng):
    nodes = rng.choices([1, 2], weights=[70, 30])[0]
    actual = lognormal_capped(rng, median_s=5 * 3600, sigma=0.8,
                              cap_s=20 * 3600, floor_s=1800)
    req = round_up_request(actual, rng.uniform(1.5, 3.0))
    return dict(partition="gpu", nodes=nodes, ntasks=nodes * 48,
                gres="gpu:a100:4", gpu_type="a100", gpus_total=nodes * 4,
                actual_s=actual, req_min=req, outcome="completed")


def gen_ai_dev_gpu(rng):
    if rng.random() < 0.6:
        part, gtype, per_node = "gpu-h100", "h100", 2
        ngpu = rng.choices([1, 2], weights=[70, 30])[0]
    else:
        part, gtype, per_node = "gpu", "a100", 4
        ngpu = rng.choices([1, 2], weights=[80, 20])[0]
    actual = lognormal_capped(rng, median_s=20 * 60, sigma=0.9,
                              cap_s=90 * 60, floor_s=120)
    req = round_up_request(actual, rng.uniform(1.5, 4.0))
    return dict(partition=part, nodes=1, ntasks=8 * ngpu,
                gres="gpu:%s:%d" % (gtype, ngpu), gpu_type=gtype,
                gpus_total=ngpu, actual_s=actual, req_min=req,
                outcome="completed")


def gen_llm_h100nvl(rng):
    actual = lognormal_capped(rng, median_s=8 * 3600, sigma=0.6,
                              cap_s=20 * 3600, floor_s=2 * 3600)
    req = round_up_request(actual, rng.uniform(1.3, 2.0))
    return dict(partition="gpu-h100-nvl", nodes=1, ntasks=96,
                gres="gpu:h100nvl:4", gpu_type="h100nvl", gpus_total=4,
                actual_s=actual, req_min=req, outcome="completed")


def gen_bigmem(rng):
    actual = lognormal_capped(rng, median_s=2 * 3600, sigma=0.9,
                              cap_s=12 * 3600, floor_s=600)
    req = round_up_request(actual, rng.uniform(1.5, 4.0))
    n = rng.choice([16, 32, 64])
    return dict(partition="bigmem", nodes=1, ntasks=n, gres=None,
                gpu_type=None, gpus_total=0,
                actual_s=actual, req_min=req, outcome="completed")


def gen_hugemem(rng):
    actual = lognormal_capped(rng, median_s=4 * 3600, sigma=0.7,
                              cap_s=12 * 3600, floor_s=1800)
    req = round_up_request(actual, rng.uniform(1.5, 3.0))
    return dict(partition="hugemem", nodes=1, ntasks=64, gres=None,
                gpu_type=None, gpus_total=0,
                actual_s=actual, req_min=req, outcome="completed")


CLASSES = [
    ("htc_short",      38, gen_htc_short),
    ("mpi_physics",    22, gen_mpi_physics),
    ("mpi_capability",  4, gen_mpi_capability),
    ("ai_train_a100",  12, gen_ai_train_a100),
    ("ai_dev_gpu",     13, gen_ai_dev_gpu),
    ("llm_h100nvl",     3, gen_llm_h100nvl),
    ("bigmem",          6, gen_bigmem),
    ("hugemem",         2, gen_hugemem),
]


def diurnal_arrivals(rng, n_jobs, window_s, day_start_frac=0.375):
    """非齐次泊松到达: 白天(相对窗口 09:00-18:00 一段)强度约为夜间 3 倍.

    用 thinning 法采样, 返回升序的提交时刻(秒)列表.
    """
    def intensity(t):
        day_pos = (t / window_s + day_start_frac) % 1.0
        return 1.0 if 0.375 <= day_pos <= 0.75 else 0.35

    times = []
    lam_max = 1.0
    t = 0.0
    # 期望率使 thinning 后约得到 n_jobs 个点
    mean_int = 0.375 * 1.0 + 0.625 * 0.35
    base_rate = n_jobs / (window_s * mean_int)
    while len(times) < n_jobs:
        t += rng.expovariate(base_rate * lam_max)
        if t >= window_s:
            t = rng.uniform(0, window_s)  # 罕见: 率估计偏低时回填补齐
            times.append(t)
            continue
        if rng.random() <= intensity(t) / lam_max:
            times.append(t)
    times.sort()
    return [int(x) for x in times[:n_jobs]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=12.0,
                    help="提交窗口时长(模拟小时), 默认 12")
    ap.add_argument("--jobs", type=int, default=140,
                    help="作业总数, 默认 140")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(可复现)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    window_s = int(args.hours * 3600)

    arrivals = diurnal_arrivals(rng, args.jobs, window_s)
    class_names = [c[0] for c in CLASSES]
    class_weights = [c[1] for c in CLASSES]
    class_gen = {c[0]: c[2] for c in CLASSES}

    jobs = []
    for i, dt in enumerate(arrivals, start=1):
        cname = rng.choices(class_names, weights=class_weights)[0]
        j = class_gen[cname](rng)
        user = rng.choices([u for u, _ in USERS], weights=[w for _, w in USERS])[0]
        j.update(job_sim_id=i, cls=cname, user=user, submit_dt_s=dt)
        jobs.append(j)

    events_path = os.path.join(OUT_DIR, "sim.events.realistic")
    with open(events_path, "w") as f:
        for j in jobs:
            gres = (" --gres=%s" % j["gres"]) if j["gres"] else ""
            f.write(
                "-e submit_batch_job -dt %d | --uid=%s -J jobid_%d -p %s "
                "-N %d -n %d%s -t %d -sim-walltime %d pseudo.job -sleep %d\n"
                % (j["submit_dt_s"], j["user"], j["job_sim_id"],
                   j["partition"], j["nodes"], j["ntasks"], gres,
                   j["req_min"], j["actual_s"], j["actual_s"]))

    profile_path = os.path.join(OUT_DIR, "workload_profile.csv")
    with open(profile_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_sim_id", "class", "user", "partition", "nodes",
                    "cores", "gpus", "gpu_type", "submit_dt_s",
                    "req_walltime_min", "actual_walltime_s", "outcome",
                    "cpu_w_per_core", "gpu_w"])
        for j in jobs:
            w.writerow([j["job_sim_id"], j["cls"], j["user"], j["partition"],
                        j["nodes"], j["ntasks"], j["gpus_total"],
                        j["gpu_type"] or "", j["submit_dt_s"], j["req_min"],
                        j["actual_s"], j["outcome"],
                        CPU_W_PER_CORE[j["partition"]],
                        GPU_W.get(j["gpu_type"], 0.0)])

    # 摘要
    from collections import Counter
    cnt = Counter(j["cls"] for j in jobs)
    nodeh = Counter()
    for j in jobs:
        nodeh[j["cls"]] += j["nodes"] * j["actual_s"] / 3600.0
    print("wrote %s (%d jobs over %.1f h)" % (events_path, len(jobs), args.hours))
    print("wrote %s" % profile_path)
    print("%-16s %6s %12s" % ("class", "jobs", "node-hours"))
    for c in class_names:
        print("%-16s %6d %12.1f" % (c, cnt[c], nodeh[c]))
    print("%-16s %6d %12.1f" % ("TOTAL", len(jobs), sum(nodeh.values())))


if __name__ == "__main__":
    main()
