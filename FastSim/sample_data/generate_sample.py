#!/usr/bin/env python3
"""
Generate a minimal anonymised Slurm dump for FastSim smoke testing.

Output layout mirrors scripts/slurm_dump.sh but uses a tiny 4-node Stanage-like
topology so FastSim can run without real cluster access.
"""

import os
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "stanage_smoke")


def write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def energy_joules(nodes, hours, watts_per_node=250):
    return int(nodes * watts_per_node * hours * 3600)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    write(
        os.path.join(OUT_DIR, "slurm.conf"),
        [
            "ClusterName=stanage-smoke",
            "SchedulerType=sched/backfill",
            "SelectType=select/cons_tres",
            "NodeName=node[001-004] CPUs=64 RealMemory=257024 State=UNKNOWN",
            "PartitionName=standard Nodes=node[001-004] Default=YES MaxTime=96:00:00 State=UP",
        ],
    )

    write(
        os.path.join(OUT_DIR, "sacctmgr_assocs.csv"),
        [
            "User|Account|ParentName|Partition|MaxJobs|MaxSubmit|",
            "|Acc0|root|||",
            "User0|Acc0||standard||",
            "User1|Acc0||standard||",
        ],
    )

    write(
        os.path.join(OUT_DIR, "sacctmgr_qos.csv"),
        [
            "Name|Priority|GrpTRES|GrpJobs|GrpSubmit|MaxTRESPU|MaxJobsPU|MaxJobs|MaxSubmitPU|MaxSubmit|",
            "normal|1000|||||||",
        ],
    )

    write(
        os.path.join(OUT_DIR, "sacctmgr_events.csv"),
        [
            "NodeName|TimeStart|TimeEnd|State|Reason|",
        ],
    )

    write(
        os.path.join(OUT_DIR, "sinfo_resv.csv"),
        [
            "RESV_NAME|STATE|START_TIME|END_TIME|NODELIST|",
        ],
    )

    base = datetime(2024, 1, 8, 8, 0, 0)
    jobs = [
        # Two 2-node jobs start together and occupy all 4 nodes (bootstrap phase).
        dict(
            jid=1001, user="User0", nodes=2, submit_h=0, start_h=0, runtime_h=4,
            timelimit="06:00:00", reason="None",
        ),
        dict(
            jid=1002, user="User1", nodes=2, submit_h=0, start_h=0, runtime_h=4,
            timelimit="06:00:00", reason="None",
        ),
        # Queue behind the full cluster.
        dict(
            jid=1003, user="User0", nodes=1, submit_h=1, start_h=4, runtime_h=2,
            timelimit="04:00:00", reason="None",
        ),
        dict(
            jid=1004, user="User1", nodes=2, submit_h=2, start_h=4, runtime_h=3,
            timelimit="06:00:00", reason="None",
        ),
        dict(
            jid=1005, user="User0", nodes=1, submit_h=3, start_h=7, runtime_h=1,
            timelimit="02:00:00", reason="None",
        ),
    ]

    header = (
        "User|Account|AllocNodes|ConsumedEnergyRaw|ExitCode|Flags|JobID|JobName|"
        "Partition|QOS|Reason|ReqNodes|ReqCPUS|Group|ReqMem|Start|State|End|"
        "Elapsed|Submit|SubmitLine|Timelimit|"
    )
    rows = [header]
    for job in jobs:
        submit = base + timedelta(hours=job["submit_h"])
        start = base + timedelta(hours=job["start_h"])
        end = start + timedelta(hours=job["runtime_h"])
        elapsed = end - start
        energy = energy_joules(job["nodes"], job["runtime_h"])
        rows.append(
            "|".join(
                [
                    job["user"],
                    "Acc0",
                    str(job["nodes"]),
                    str(energy),
                    "0:0",
                    "",
                    str(job["jid"]),
                    f"job{job['jid']}",
                    "standard",
                    "normal",
                    job["reason"],
                    str(job["nodes"]),
                    "64",
                    "",
                    "0",
                    start.strftime("%Y-%m-%dT%H:%M:%S"),
                    "COMPLETED",
                    end.strftime("%Y-%m-%dT%H:%M:%S"),
                    f"{int(elapsed.total_seconds() // 3600):02d}:{int((elapsed.total_seconds() % 3600) // 60):02d}:{int(elapsed.total_seconds() % 60):02d}",
                    submit.strftime("%Y-%m-%dT%H:%M:%S"),
                    f"sbatch -N {job['nodes']} -t {job['timelimit']} job{job['jid']}.sh",
                    job["timelimit"],
                ]
            )
            + "|"
        )

    write(os.path.join(OUT_DIR, "sacct_jobs.csv"), rows)
    print(f"Wrote sample dump to {OUT_DIR}")


if __name__ == "__main__":
    main()
