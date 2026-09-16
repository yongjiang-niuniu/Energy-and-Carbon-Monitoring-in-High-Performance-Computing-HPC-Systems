#!/usr/bin/env python3
"""Audit whether policy scenarios differ from baseline only as intended."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INVARIANT_COLUMNS = [
    "event_epoch_utc",
    "source_submit_utc",
    "source_eligible_utc",
    "source_partition",
    "qos",
    "partition",
    "nodes",
    "cpus",
    "memory_per_node_mib",
    "requested_gpus",
    "scheduled_gpus",
    "gpu_per_node",
    "gpu_type",
    "runtime_s",
    "source_timelimit_min",
    "timelimit_min",
    "submit_dt_s",
    "eligible_dt_s",
    "user_id",
    "flexibility_score",
    "carry_in_type",
    "is_warmup",
    "is_evaluation",
]


def load_profile(scenario: Path) -> pd.DataFrame:
    profile = pd.read_csv(scenario / "workload_profile.csv")
    if profile["sim_job_id"].duplicated().any():
        raise ValueError(f"Duplicate sim_job_id in {scenario}")
    return profile.set_index("sim_job_id").sort_index()


def mismatch_count(left: pd.Series, right: pd.Series) -> int:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        equal = np.isclose(
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        )
        return int((~equal).sum())
    left_text = left.fillna("<NA>").astype(str)
    right_text = right.fillna("<NA>").astype(str)
    return int((left_text != right_text).sum())


def audit_scenario(
    baseline: pd.DataFrame,
    scenario_path: Path,
    comparison: pd.DataFrame,
    energy_tolerance_pct: float,
) -> dict[str, object]:
    name = scenario_path.name
    policy = load_profile(scenario_path)
    validation = json.loads(
        (scenario_path / "simulation_validation.json").read_text(encoding="utf-8")
    )

    same_job_ids = baseline.index.equals(policy.index)
    invariant_mismatches: dict[str, int] = {}
    if same_job_ids:
        for column in INVARIANT_COLUMNS:
            if column not in baseline and column not in policy:
                continue
            if column not in baseline or column not in policy:
                invariant_mismatches[column] = len(baseline)
            else:
                invariant_mismatches[column] = mismatch_count(
                    baseline[column], policy[column]
                )
    else:
        invariant_mismatches["sim_job_id"] = len(
            baseline.index.symmetric_difference(policy.index)
        )

    delay = pd.to_numeric(policy["release_dt_s"], errors="coerce") - pd.to_numeric(
        policy["eligible_dt_s"], errors="coerce"
    )
    delayed = delay > 0
    negative_delays = int((delay < 0).sum())
    delay_field_mismatches = 0
    if "policy_delay_s" in policy:
        reported_delay = pd.to_numeric(policy["policy_delay_s"], errors="coerce")
        delay_field_mismatches = mismatch_count(delay.clip(lower=0), reported_delay)
    component_delay_mismatches = 0
    if {"b1_delay_s", "b2_delay_s"}.issubset(policy.columns):
        component_delay = (
            pd.to_numeric(policy["b1_delay_s"], errors="coerce").fillna(0)
            + pd.to_numeric(policy["b2_delay_s"], errors="coerce").fillna(0)
        )
        component_delay_mismatches = mismatch_count(delay.clip(lower=0), component_delay)
    nonflexible_delays = 0
    if "is_flexible" in policy:
        flexible = policy["is_flexible"].fillna(False).astype(bool)
        nonflexible_delays = int((delayed & ~flexible).sum())
    carry_in_delays = 0
    if "is_warmup" in policy:
        carry_in = policy["is_warmup"].fillna(False).astype(bool)
        carry_in_delays = int((delayed & carry_in).sum())

    b1_intensity_violations = 0
    if {
        "release_intensity_before",
        "release_intensity_after_b1",
    }.issubset(policy.columns):
        before = pd.to_numeric(policy["release_intensity_before"], errors="coerce")
        after = pd.to_numeric(policy["release_intensity_after_b1"], errors="coerce")
        checked = delayed & before.notna() & after.notna()
        b1_intensity_violations = int((after.loc[checked] > before.loc[checked]).sum())

    runtime_intensity_violations = 0
    if {"runtime_intensity_before", "runtime_intensity_after"}.issubset(policy.columns):
        before = pd.to_numeric(policy["runtime_intensity_before"], errors="coerce")
        after = pd.to_numeric(policy["runtime_intensity_after"], errors="coerce")
        checked = delayed & before.notna() & after.notna()
        runtime_intensity_violations = int((after.loc[checked] > before.loc[checked]).sum())

    wait_budget_violations = 0
    if "allowed_wait_budget_s" in policy:
        budget = pd.to_numeric(policy["allowed_wait_budget_s"], errors="coerce").fillna(0)
        wait_budget_violations = int((delay > budget + 1e-6).sum())

    row = comparison.loc[comparison["scenario"].eq(name)]
    if len(row) != 1:
        raise ValueError(f"Expected one comparison row for {name}")
    metrics = row.iloc[0]
    baseline_metrics = comparison.loc[comparison["scenario"].eq("baseline")].iloc[0]
    energy_differences = {}
    energy_difference_pct = {}
    for model in ["capacity_weighted", "node_request_upper"]:
        column = f"dynamic_energy_kwh_{model}"
        baseline_energy = float(baseline_metrics[column])
        difference = abs(float(metrics[column]) - baseline_energy)
        energy_differences[model] = difference
        energy_difference_pct[model] = (
            difference / baseline_energy * 100 if baseline_energy else 0.0
        )
    reductions = {
        model: float(metrics[f"dynamic_carbon_reduction_pct_{model}"])
        for model in ["capacity_weighted", "node_request_upper"]
    }

    structural_checks = {
        "same_job_ids": same_job_ids,
        "invariant_columns_unchanged": sum(invariant_mismatches.values()) == 0,
        "strict_simulation_pass": bool(validation["strict_pass"]),
        "all_jobs_completed": validation["completed_jobs"] == len(policy),
        "no_negative_policy_delay": negative_delays == 0,
        "policy_delay_field_matches_release_once": delay_field_mismatches == 0,
        "b1_b2_components_not_reapplied": component_delay_mismatches == 0,
        "only_flexible_jobs_delayed": nonflexible_delays == 0,
        "carry_in_jobs_never_delayed": carry_in_delays == 0,
        "b1_never_moves_to_higher_release_intensity": b1_intensity_violations == 0,
        "runtime_aware_policy_never_increases_expected_runtime_intensity": runtime_intensity_violations == 0,
        "per_job_wait_budgets_respected": wait_budget_violations == 0,
    }
    outcome_diagnostics = {
        "dynamic_energy_approximately_unchanged": max(
            energy_difference_pct.values()
        )
        <= energy_tolerance_pct,
        "dynamic_carbon_reduced_in_both_models": min(reductions.values()) > 0,
    }

    return {
        "scenario": name,
        "jobs": len(policy),
        "jobs_delayed": int(delayed.sum()),
        "invariant_mismatch_total": sum(invariant_mismatches.values()),
        "invariant_mismatches": invariant_mismatches,
        "negative_policy_delays": negative_delays,
        "policy_delay_field_mismatches": delay_field_mismatches,
        "component_delay_mismatches": component_delay_mismatches,
        "nonflexible_delays": nonflexible_delays,
        "carry_in_delays": carry_in_delays,
        "b1_intensity_violations": b1_intensity_violations,
        "runtime_intensity_violations": runtime_intensity_violations,
        "wait_budget_violations": wait_budget_violations,
        "dynamic_energy_difference_kwh": energy_differences,
        "dynamic_energy_difference_pct": energy_difference_pct,
        "dynamic_carbon_reduction_pct": reductions,
        "total_user_wait_p95_s": float(metrics["total_user_wait_p95_s"]),
        "structural_checks": structural_checks,
        "outcome_diagnostics": outcome_diagnostics,
        "pass": all(structural_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--energy-tolerance-pct", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_profile(args.baseline)
    baseline_validation = json.loads(
        (args.baseline / "simulation_validation.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(args.comparison)

    policy_audits = [
        audit_scenario(
            baseline,
            scenario,
            comparison,
            args.energy_tolerance_pct,
        )
        for scenario in args.policy
    ]
    result = {
        "audit_scope": (
            "Pass/fail covers workload invariants and simulator execution. "
            "Energy and carbon outcomes are reported as diagnostics because a valid policy can have a neutral or negative result."
        ),
        "energy_tolerance_pct": args.energy_tolerance_pct,
        "baseline": {
            "scenario": args.baseline.name,
            "jobs": len(baseline),
            "strict_simulation_pass": bool(baseline_validation["strict_pass"]),
            "completed_jobs": int(baseline_validation["completed_jobs"]),
        },
        "policies": policy_audits,
        "overall_pass": bool(baseline_validation["strict_pass"])
        and all(item["pass"] for item in policy_audits),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiment_logic_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    rows = []
    for item in policy_audits:
        rows.append(
            {
                "scenario": item["scenario"],
                "jobs": item["jobs"],
                "jobs_delayed": item["jobs_delayed"],
                "invariant_mismatch_total": item["invariant_mismatch_total"],
                "dynamic_energy_difference_kwh_capacity_weighted": item[
                    "dynamic_energy_difference_kwh"
                ]["capacity_weighted"],
                "dynamic_energy_difference_pct_capacity_weighted": item[
                    "dynamic_energy_difference_pct"
                ]["capacity_weighted"],
                "dynamic_energy_difference_pct_node_request_upper": item[
                    "dynamic_energy_difference_pct"
                ]["node_request_upper"],
                "dynamic_carbon_reduction_pct_capacity_weighted": item[
                    "dynamic_carbon_reduction_pct"
                ]["capacity_weighted"],
                "dynamic_carbon_reduction_pct_node_request_upper": item[
                    "dynamic_carbon_reduction_pct"
                ]["node_request_upper"],
                "total_user_wait_p95_s": item["total_user_wait_p95_s"],
                "pass": item["pass"],
            }
        )
    pd.DataFrame(rows).to_csv(args.output / "experiment_logic_audit.csv", index=False)
    print(json.dumps(result, indent=2))
    if not result["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
