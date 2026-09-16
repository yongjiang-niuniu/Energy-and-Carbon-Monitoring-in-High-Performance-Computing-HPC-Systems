from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_low_impact_final_report import build_report, write_acceptance_figure


def test_final_report_and_evidence_figure_are_created(tmp_path: Path) -> None:
    comparison = pd.DataFrame(
        [
            {
                "scenario": "baseline",
                "jobs": 2,
                "evaluation_jobs": 2,
                "jobs_with_policy_delay": 0,
                "policy_delay_p50_s": 0,
                "total_wait_p95_change_s": 0,
                "dynamic_carbon_reduction_pct_capacity_weighted": 0,
                "dynamic_carbon_reduction_pct_node_request_upper": 0,
                "dynamic_carbon_reduction_pct_allocated_node_distinct": 0,
                "whole_carbon_reduction_pct_capacity_weighted": 0,
            },
            {
                "scenario": "no_policy_repeat",
                "jobs": 2,
                "evaluation_jobs": 2,
                "jobs_with_policy_delay": 0,
                "policy_delay_p50_s": 0,
                "total_wait_p95_change_s": 1,
                "dynamic_carbon_reduction_pct_capacity_weighted": 0.01,
                "dynamic_carbon_reduction_pct_node_request_upper": 0.02,
                "dynamic_carbon_reduction_pct_allocated_node_distinct": -1,
                "whole_carbon_reduction_pct_capacity_weighted": 0.001,
            },
            {
                "scenario": "low_impact_dynamic_q25",
                "jobs": 2,
                "evaluation_jobs": 2,
                "jobs_with_policy_delay": 1,
                "policy_delay_p50_s": 0,
                "total_wait_p95_change_s": 30,
                "dynamic_carbon_reduction_pct_capacity_weighted": 0.2,
                "dynamic_carbon_reduction_pct_node_request_upper": -0.1,
                "dynamic_carbon_reduction_pct_allocated_node_distinct": 1,
                "whole_carbon_reduction_pct_capacity_weighted": 0.01,
            },
        ]
    )
    results = pd.DataFrame(
        {
            "is_evaluation": [True, True],
            "sim_policy_delay_s": [600, 0],
            "user_id": ["u1", "u2"],
        }
    )
    checks = {
        "all_expected_jobs_completed": True,
        "capacity_dynamic_carbon_reduction_pct_minimum": True,
        "capacity_result_must_exceed_previous_noop_noise_pct": True,
        "total_user_wait_p95_increase_seconds_maximum": True,
        "jobs_delayed_longer_than_actual_runtime": True,
        "maximum_user_policy_delay_minutes": True,
        "maximum_single_user_share_of_policy_delay": True,
    }
    acceptance = {
        "status": "pass",
        "checks": checks,
        "observed": {
            "capacity_dynamic_carbon_reduction_pct": 0.2,
            "capacity_dynamic_carbon_reduction_kg": 0.1,
            "total_user_wait_p95_change_s": 30,
            "jobs_delayed_longer_than_actual_runtime": 0,
            "maximum_user_policy_delay_minutes": 10,
            "maximum_single_user_share_of_policy_delay": 0.25,
            "policy_jobs": 2,
            "expected_total_jobs": 2,
            "total_policy_delay_hours": 1 / 6,
        },
        "repeatability_diagnostics": {
            "no_policy_repeat_capacity_dynamic_carbon_reduction_pct": 0.01,
            "policy_minus_no_policy_repeat_capacity_pct_points": 0.19,
        },
    }
    text = build_report(comparison, acceptance, results)
    assert "Overall frozen acceptance: **PASS**" in text
    assert "0.2000%" in text
    write_acceptance_figure(acceptance, tmp_path)
    assert (tmp_path / "frozen_acceptance_summary.png").stat().st_size > 0
