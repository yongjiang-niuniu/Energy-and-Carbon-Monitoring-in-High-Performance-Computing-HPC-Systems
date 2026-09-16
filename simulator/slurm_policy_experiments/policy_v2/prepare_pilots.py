"""Freeze tiny engineering inputs before any simulator execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import pandas as pd
from legacy_fixed import GENERATOR, load_carbon

ROOT = Path(__file__).resolve().parents[1]
DAY = ROOT / "campaigns/multiday_20260908/days/2025-11-18"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_for(profile, carbon, epoch, synthetic=False):
    jobs = {}
    for row in profile.to_dict("records"):
        jobs[row["sim_job_id"]] = {"user": row["user_id"], "pool": "cpu",
            "fraction": row["cpus"]/(4 if synthetic else 172*64),
            "predicted_runtime_s": float(row["decision_runtime_s"]),
            "prediction_reliable": bool(row["runtime_prediction_reliable"]),
            "flexible": bool(row["is_flexible"])}
    points = [[(row.from_utc-epoch).total_seconds(),float(row.intensity_gco2_per_kwh)] for row in carbon.itertuples()]
    return {"jobs":jobs, "carbon_points": points,
            "carbon_end_s": (carbon.to_utc.max()-epoch).total_seconds(),
            "tick_s":60 if synthetic else 300, "max_delay_s":600 if synthetic else 1800,
            "runtime_delay_ratio":4 if synthetic else .5,
            "minimum_saving_pct":5, "user_budget_s":300 if synthetic else 1800,
            "audit_path":"/opt/slurm-sim/etc/decisions.jsonl",
            "signal_kind":"synthetic_test" if synthetic else "retrospective_NESO_not_as_issued_forecast",
            "energy_proxy":"capacity weighted runtime; ranking only, not measured job energy"}


def write_pair(destination, stem, profile, carbon, epoch, synthetic=False):
    for mode in ("baseline","feedback"):
        out=destination/(stem+"_"+mode)
        out.mkdir()
        for name in ("slurm.conf","sim.conf","gres.conf"):
            shutil.copy2(ROOT/"config/stanage_2025_assumed"/name,out/name)
        if synthetic:
            slurm=(out/"slurm.conf").read_text().split("# Observed CPU node ranges")[0]
            slurm += "NodeName=node001 NodeAddr=127.0.0.1 CPUs=4 Sockets=1 CoresPerSocket=4 ThreadsPerCore=1 RealMemory=64000 State=UNKNOWN\n"
            slurm += "PartitionName=sheffield Nodes=node001 Default=YES MaxTime=4-00:00:00 State=UP\n"
            (out/"slurm.conf").write_text(slurm)
            (out/"gres.conf").write_text("")
        GENERATOR.write_outputs(out,profile,{"scope":"engineering_smoke_not_five_day_result", "synthetic":synthetic,
                                            "source_date":"2025-11-18" if not synthetic else None,
                                            "no_carry_in":True,"complete_Stanage_initial_state":False})
        carbon.to_csv(out/"carbon.csv",index=False)
        config=config_for(profile,carbon,epoch,synthetic)
        config["baseline"] = mode=="baseline"
        (out/"controller.json").write_text(json.dumps(config,indent=2))
    a,b=[destination/(stem+"_"+x) for x in ("baseline","feedback")]
    for name in ("sim.events","workload_profile.csv","users.sim","slurm.conf","sim.conf","gres.conf","carbon.csv"):
        assert digest(a/name)==digest(b/name),name


def prepare(destination):
    destination=Path(destination)
    history=pd.read_csv(DAY/"history_only/workload_profile.csv")
    # Fixed, first-in-time CPU subset. Runtime bounds limit smoke-test duration only.
    sample=history.loc[history.is_evaluation & history.partition.eq("sheffield") & history.nodes.eq(1)
                       & history.runtime_s.between(60,3600)].sort_values(["eligible_dt_s","sim_job_id"]).head(64).copy()
    if len(sample)!=64: raise ValueError("Insufficient smoke sample")
    sample["source_sim_job_id"]=sample.sim_job_id
    sample["slurm_job_id_expected"]=range(1,65)
    sample["sim_job_id"]=[f"sim_{n:06d}" for n in range(1,65)]
    sample["release_dt_s"]=sample.eligible_dt_s
    sample["policy_delay_s"]=0
    sample["policy_name"]="v2_smoke_input"
    sample["is_flexible"]=sample.flexibility_score.le(.25)
    epoch=pd.Timestamp("2025-11-18T00:00:00Z")
    carbon=load_carbon(DAY/"carbon.csv")
    write_pair(destination,"real64",sample,carbon,epoch)
    toy=[]
    for n,(arrival,cpus,runtime,flex,user) in enumerate([(0,2,120,True,"u1"),(0,1,120,True,"u1"),
                                                       (60,4,1200,False,"u2")],1):
        toy.append({"sim_job_id":f"sim_{n:06d}","slurm_job_id_expected":n,"user_id":user,"cpus":cpus,
                    "partition":"sheffield","nodes":1,"memory_per_node_mib":1024,"gpu_per_node":0,
                    "scheduled_gpus":0,"requested_gpus":0,"gpu_type":"", "timelimit_min":60,
                    "runtime_s":runtime,"decision_runtime_s":runtime,"runtime_prediction_reliable":True,
                    "release_dt_s":arrival,"eligible_dt_s":arrival,"is_flexible":flex,"is_evaluation":True,
                    "is_warmup":False,"dependency_delay_s":0,"event_epoch_utc":epoch.isoformat()})
    toycarbon=pd.DataFrame({"from_utc":[epoch,epoch+pd.Timedelta(seconds=300)],
                           "to_utc":[epoch+pd.Timedelta(seconds=300),epoch+pd.Timedelta(seconds=10000)],
                           "intensity_gco2_per_kwh":[200,50]})
    write_pair(destination,"functional",pd.DataFrame(toy),toycarbon,epoch,True)
    frozen={str(p.relative_to(destination)):digest(p) for p in destination.glob("*/*") if p.is_file()
            and p.parent.name in ("real64_baseline","real64_feedback","functional_baseline","functional_feedback")}
    (destination/"pilot_inputs.json").write_text(json.dumps({"hashes":frozen,
       "real_selection":"first 64 eligible evaluation sheffield one-node jobs on Nov 18; runtime 60..3600s",
       "comparisons":"one baseline and one feedback per identical job set; no parameter tuning after outcomes",
       "scope":"functional tests; no stability or full-cluster claim",
       "protected_inputs":"all frozen seven-policy five-date sources/results remain unchanged"},indent=2))


if __name__ == "__main__":
    prepare(sys.argv[1])
