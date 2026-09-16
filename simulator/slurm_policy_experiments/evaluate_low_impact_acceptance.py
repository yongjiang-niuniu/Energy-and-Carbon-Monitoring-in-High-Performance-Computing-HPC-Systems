#!/usr/bin/env python3
"""Evaluate the frozen low-impact policy acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def evaluate_acceptance(
    comparison: pd.DataFrame,
    baseline_results: pd.DataFrame,
    policy_results: pd.DataFrame,
    manifest: dict[str, object],
    policy_name: str,
) -> dict[str, object]:
    acceptance = manifest["acceptance"]
    assert isinstance(acceptance, dict)
    baseline_matches = comparison.loc[comparison["scenario"].eq("baseline")]
    repeat_matches = comparison.loc[comparison["scenario"].eq("no_policy_repeat")]
    policy_matches = comparison.loc[comparison["scenario"].eq(policy_name)]
    if len(baseline_matches) != 1 or len(policy_matches) != 1:
        raise ValueError("comparison must contain one baseline and one named policy row")
    if len(repeat_matches) != 1:
        raise ValueError("comparison must contain one no_policy_repeat row")
    repeat_summary = repeat_matches.iloc[0]
    policy_summary = policy_matches.iloc[0]

    delayed = policy_results.loc[
        policy_results["is_evaluation"].astype(bool)
        & pd.to_numeric(policy_results["sim_policy_delay_s"], errors="coerce").gt(0)
    ].copy()
    policy_delay_s = pd.to_numeric(delayed["sim_policy_delay_s"], errors="coerce")
    runtime_s = pd.to_numeric(delayed["runtime_s"], errors="coerce")
    per_user_delay_s = delayed.assign(_delay=policy_delay_s).groupby("user_id")[
        "_delay"
    ].sum()
    total_policy_delay_s = float(policy_delay_s.sum())
    top_user_share = (
        float(per_user_delay_s.max() / total_policy_delay_s)
        if total_policy_delay_s > 0
        else 0.0
    )

    common_ids = set(baseline_results["sim_job_id"]) == set(policy_results["sim_job_id"])
    final_window = manifest.get("final_untouched_window", {})
    assert isinstance(final_window, dict)
    expected_total_jobs = int(final_window.get("expected_total_jobs", len(baseline_results)))
    expected_evaluation_jobs = int(
        final_window.get(
            "expected_evaluation_jobs",
            baseline_results["is_evaluation"].astype(bool).sum(),
        )
    )
    expected_carry_in_jobs = int(
        final_window.get(
            "expected_carry_in_jobs",
            baseline_results["is_warmup"].astype(bool).sum(),
        )
    )
    count_pass = all(
        (
            len(results) == expected_total_jobs
            and int(results["is_evaluation"].astype(bool).sum())
            == expected_evaluation_jobs
            and int(results["is_warmup"].astype(bool).sum())
            == expected_carry_in_jobs
        )
        for results in (baseline_results, policy_results)
    )
    terminal_pass = (
        baseline_results["terminal_status"].eq("completed").all()
        and policy_results["terminal_status"].eq("completed").all()
    )
    carry_in_delay_s = pd.to_numeric(
        policy_results.loc[policy_results["is_warmup"].astype(bool), "sim_policy_delay_s"],
        errors="coerce",
    )

    observed = {
        "capacity_dynamic_carbon_reduction_pct": float(
            policy_summary["dynamic_carbon_reduction_pct_capacity_weighted"]
        ),
        "capacity_dynamic_carbon_reduction_kg": float(
            policy_summary["dynamic_carbon_reduction_kg_capacity_weighted"]
        ),
        "requested_node_dynamic_carbon_reduction_pct": float(
            policy_summary["dynamic_carbon_reduction_pct_node_request_upper"]
        ),
        "allocated_distinct_node_dynamic_carbon_reduction_pct": float(
            policy_summary.get(
                "dynamic_carbon_reduction_pct_allocated_node_distinct", float("nan")
            )
        ),
        "total_user_wait_p95_change_s": float(
            policy_summary["total_wait_p95_change_s"]
        ),
        "jobs_delayed": len(delayed),
        "total_policy_delay_hours": total_policy_delay_s / 3600,
        "jobs_delayed_longer_than_actual_runtime": int((policy_delay_s > runtime_s).sum()),
        "maximum_user_policy_delay_minutes": (
            float(per_user_delay_s.max()) / 60 if not per_user_delay_s.empty else 0.0
        ),
        "maximum_single_user_share_of_policy_delay": top_user_share,
        "baseline_jobs": len(baseline_results),
        "policy_jobs": len(policy_results),
        "expected_total_jobs": expected_total_jobs,
        "expected_evaluation_jobs": expected_evaluation_jobs,
        "expected_carry_in_jobs": expected_carry_in_jobs,
    }
    repeat_capacity_pct = float(
        repeat_summary["dynamic_carbon_reduction_pct_capacity_weighted"]
    )
    repeat_p95_change_s = float(repeat_summary["total_wait_p95_change_s"])
    repeatability_diagnostics = {
        "no_policy_repeat_capacity_dynamic_carbon_reduction_pct": repeat_capacity_pct,
        "policy_minus_no_policy_repeat_capacity_pct_points": observed[
            "capacity_dynamic_carbon_reduction_pct"
        ]
        - repeat_capacity_pct,
        "policy_exceeds_same_date_no_policy_repeat": observed[
            "capacity_dynamic_carbon_reduction_pct"
        ]
        > repeat_capacity_pct,
        "no_policy_repeat_total_user_wait_p95_change_s": repeat_p95_change_s,
        "policy_minus_no_policy_repeat_p95_change_s": observed[
            "total_user_wait_p95_change_s"
        ]
        - repeat_p95_change_s,
        "interpretation": (
            "Post-hoc repeatability diagnostic only. It does not change the frozen "
            "acceptance status, but a policy result that does not exceed the same-date "
            "no-policy repeat is not distinguishable as a strategy benefit."
        ),
    }
    checks = {
        "all_expected_jobs_completed": bool(terminal_pass and common_ids and count_pass),
        "carry_in_audit": bool(carry_in_delay_s.fillna(0).eq(0).all()),
        "waiting_budget_violations": int(policy_summary["wait_budget_violations"]) == 0,
        "capacity_dynamic_carbon_reduction_pct_minimum": observed[
            "capacity_dynamic_carbon_reduction_pct"
        ]
        >= float(acceptance["capacity_dynamic_carbon_reduction_pct_minimum"]),
        "capacity_result_must_exceed_previous_noop_noise_pct": observed[
            "capacity_dynamic_carbon_reduction_pct"
        ]
        > float(acceptance["capacity_result_must_exceed_previous_noop_noise_pct"]),
        "total_user_wait_p95_increase_seconds_maximum": observed[
            "total_user_wait_p95_change_s"
        ]
        <= float(acceptance["total_user_wait_p95_increase_seconds_maximum"]),
        "jobs_delayed_longer_than_actual_runtime": observed[
            "jobs_delayed_longer_than_actual_runtime"
        ]
        <= int(acceptance["jobs_delayed_longer_than_actual_runtime"]),
        "maximum_user_policy_delay_minutes": observed[
            "maximum_user_policy_delay_minutes"
        ]
        <= float(acceptance["maximum_user_policy_delay_minutes"]),
        "maximum_single_user_share_of_policy_delay": observed[
            "maximum_single_user_share_of_policy_delay"
        ]
        <= float(acceptance["maximum_single_user_share_of_policy_delay"]),
    }
    return {
        "experiment_id": manifest["experiment_id"],
        "policy": policy_name,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "observed": observed,
        "repeatability_diagnostics": repeatability_diagnostics,
        "diagnostic_note": (
            "Requested-node and allocated-distinct-node results are reported but are not "
            "frozen admission gates; see the experiment manifest."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--policy-results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_acceptance(
        pd.read_csv(args.comparison),
        pd.read_csv(args.baseline_results),
        pd.read_csv(args.policy_results),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.policy_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
