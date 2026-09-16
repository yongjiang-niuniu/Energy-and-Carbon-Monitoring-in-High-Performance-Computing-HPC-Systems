"""Audit paired smoke runs without promoting them to the frozen study."""
import hashlib
import json
from pathlib import Path
import sys
import time
import pandas as pd
from feedback import Signal
from reporting import validation_panels

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import analyze_simulation as analysis


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def analyse(directory):
    profile=pd.read_csv(directory/"workload_profile.csv")
    bridge=jsonlines(directory/"bridge.jsonl")
    release=[row for row in bridge if row["event"]=="release"]
    mapping={f"sim_{row['id']:06d}":row for row in release}
    match=len(mapping)==len(profile) and len(release)==len(profile)
    if not match: raise ValueError(f"Incomplete bridge mapping: {directory}")
    observed=profile.copy()
    observed["slurm_job_id_expected"]=[mapping[j]["slurm_id"] for j in observed.sim_job_id]
    observed["release_dt_s"]=[mapping[j]["release_s"] for j in observed.sim_job_id]
    results=analysis.build_results(observed,analysis.parse_log(directory/"slurmctld.log"))
    config=json.loads((directory/"controller.json").read_text())
    signal=Signal(config["carbon_points"],config["carbon_end_s"])
    results["model_start_s"]=results.release_dt_s+results.sim_scheduler_wait_s.clip(lower=0)
    results["capacity_carbon_proxy"]=[
        (signal.mean(row.model_start_s,row.runtime_s) or 0)*row.runtime_s/3600*config["jobs"][row.sim_job_id]["fraction"]
        for row in results.itertuples()]
    coverage=all(signal.mean(row.model_start_s,row.runtime_s) is not None for row in results.itertuples())
    results.to_csv(directory/"v2_simulation_results.csv",index=False)
    decisions=jsonlines(directory/"decisions.jsonl")
    pd.DataFrame(decisions).to_csv(directory/"decision_audit.csv",index=False)
    panels=validation_panels(results,input_match=match,carry_in=None,carbon_coverage=coverage)
    panels["initial_state"]={"status":"limited_scope","description":"Same empty initial state; not complete Stanage carry-in"}
    panels["controller_integrity"]={"status":"fail" if any(row["event"]=="failsafe" for row in bridge) else "pass"}
    (directory/"validation_panels.json").write_text(json.dumps(panels,indent=2))
    return {"case":directory.name,"jobs":len(results),"complete_jobs":int(results.terminal_status.eq("completed").sum()),
            "held_jobs":len({row["sim_job_id"] for row in decisions if row["action"]=="hold"}),
            "cancelled_jobs":len({row["sim_job_id"] for row in decisions if row["reason"]=="feedback_cancel_benefit_lost"}),
            "budget_rejected_jobs":len({row["sim_job_id"] for row in decisions if row["reason"]=="user_budget_exhausted"}),
            "mean_total_wait_s":float(results.sim_total_user_wait_s.mean()),
            "p95_total_wait_s":float(results.sim_total_user_wait_s.quantile(.95)),
            "capacity_carbon_proxy":float(results.capacity_carbon_proxy.sum()) if coverage else None,
            "timing_max_error_s":float((results.sim_observed_runtime_s-results.runtime_s).abs().max()),
            "panels":panels}


def main(root):
    root=Path(root)
    records=[]
    for stem in ("functional","functional_check2","real64_check2"):
        pair=[]
        for mode in ("baseline","feedback"):
            directory=root/(stem+"_"+mode)
            if not (directory/"bridge.jsonl").exists(): continue
            if "All done." not in (directory/"slurmctld.log").read_text(errors="replace"):
                records.append({"case":directory.name,"status":"incomplete_not_scored"})
                continue
            result=analyse(directory);records.append(result);pair.append(result)
        if len(pair)==2:
            a,b=pair
            b["paired_proxy_saving_pct"]=(a["capacity_carbon_proxy"]-b["capacity_carbon_proxy"])/a["capacity_carbon_proxy"]*100 if a["capacity_carbon_proxy"] else None
            b["paired_mean_wait_change_s"]=b["mean_total_wait_s"]-a["mean_total_wait_s"]
    frozen=json.loads((ROOT/"campaigns/multiday_20260908/code_hashes.json").read_text())
    changed=[path for path,value in frozen.items() if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=value]
    inputs=json.loads((root/"pilot_inputs.json").read_text())["hashes"]
    changed_inputs=[path for path,value in inputs.items() if hashlib.sha256((root/path).read_bytes()).hexdigest()!=value]
    budget=json.loads((root/"budget.json").read_text())
    output={"scope":"one paired functional test and one paired 64-job engineering sample; no stability claim",
            "frozen_source_changes":changed,"pilot_input_changes":changed_inputs,
            "aggregate_elapsed_seconds_at_report":time.time()-budget["started_epoch"],"experiments":records}
    failure=root/"controller_failure_check"
    if (failure/"slurmctld.log").exists():
        log=(failure/"slurmctld.log").read_text(errors="replace")
        parsed=analysis.parse_log(failure/"slurmctld.log")
        expected=len(pd.read_csv(failure/"workload_profile.csv"))
        complete=sum(analysis.terminal_status(row)=="completed" for row in parsed.values())
        output["fault_injection"]={"expected_jobs":expected,"complete_jobs":complete,
            "failsafe_logged":"POLICY_V2_FAILSAFE" in log,
            "pass":complete==expected and "All done." in log and "POLICY_V2_FAILSAFE" in log}
    output["process_exits"]=[{"phase":row["phase"],"exit_code":row["exit_code"],"timed_out":row["timed_out"]}
                             for row in jsonlines(root/"phases.jsonl") if "baseline" in row["phase"] or "feedback" in row["phase"]]
    output["exit_note"]="Exit 139 after All done is recorded, not treated as a clean process exit; job completion is checked separately."
    (root/"results.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main(sys.argv[1])
