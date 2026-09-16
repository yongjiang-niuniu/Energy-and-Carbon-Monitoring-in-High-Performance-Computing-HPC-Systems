"""Re-audit the December 8 presentation case without changing legacy evidence."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from analyze_multiday_outcomes import EXP, REPO, MODELS, decompose_carbon, save_json
from audit_exact_carryin import audit_profile, audit_results


def main():
    scenario_root = EXP / "scenarios/low_impact_dynamic_dec08_20260902"
    result_root = EXP / "results/low_impact_dynamic_dec08_20260902/final_comparison"
    output = REPO / "work_logs/carbon_increase_review_20260909"
    output.mkdir(parents=True, exist_ok=True)
    names = ["baseline", "no_policy_repeat", "low_impact_dynamic_q25"]
    intervals, records, inputs = {}, {}, []
    for name in names:
        paths = [scenario_root / name / "workload_profile.csv",
                 scenario_root / name / "job_results.csv",
                 result_root / f"{name}_carbon_intervals.csv"]
        for path in paths:
            inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        profile, jobs, intervals[name] = [pd.read_csv(path) for path in paths]
        checks = audit_profile(profile) + audit_results(profile, jobs)
        running = jobs.loc[jobs.carry_in_type.eq("running")]
        late = running.loc[running.sim_start_offset_s.gt(60)]
        records[name] = {
            "passed": all(x["passed"] for x in checks), "checks": checks,
            "running_carry_in_count": len(running), "running_carry_in_late_count": len(late),
            "max_running_start_offset_s": float(running.sim_start_offset_s.max()),
            "late_jobs": late[["sim_job_id", "slurm_job_id_expected", "partition", "sim_start_offset_s"]].to_dict("records"),
        }
        save_json(output / f"{name}_audit.json", records[name])
    diagnostics = []
    for name in names[1:]:
        for model in MODELS:
            row, _ = decompose_carbon(intervals["baseline"], intervals[name], model)
            diagnostics.append({"scenario": name, "valid_initial_state": records[name]["passed"], **row})
    pd.DataFrame(diagnostics).to_csv(output / "legacy_carbon_decomposition.csv", index=False)
    save_json(output / "legacy_review.json", {
        "scope": "Legacy December 8 replay, excluded from current five-day campaign",
        "conclusion": "Policy replay fails initial-state audit; stored carbon arithmetic is reproducible but cannot isolate the delay-policy effect.",
        "audits": records, "decomposition": diagnostics, "inputs": inputs,
        "review_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print(json.dumps({n: {"passed": r["passed"], "late_running_jobs": r["running_carry_in_late_count"]}
                      for n, r in records.items()}))


if __name__ == "__main__":
    main()
