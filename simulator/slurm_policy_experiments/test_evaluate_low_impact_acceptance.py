import pandas as pd

from evaluate_low_impact_acceptance import evaluate_acceptance


def test_acceptance_passes_when_all_frozen_limits_are_met() -> None:
    comparison = pd.DataFrame(
        [
            {"scenario": "baseline"},
            {
                "scenario": "no_policy_repeat",
                "dynamic_carbon_reduction_pct_capacity_weighted": 0.01,
                "total_wait_p95_change_s": -5,
            },
            {
                "scenario": "policy",
                "dynamic_carbon_reduction_pct_capacity_weighted": 0.2,
                "dynamic_carbon_reduction_kg_capacity_weighted": 0.1,
                "dynamic_carbon_reduction_pct_node_request_upper": -0.1,
                "dynamic_carbon_reduction_pct_allocated_node_distinct": 1.0,
                "total_wait_p95_change_s": 60,
                "wait_budget_violations": 0,
            },
        ]
    )
    baseline = pd.DataFrame(
        {
            "sim_job_id": ["a"],
            "terminal_status": ["completed"],
            "is_evaluation": [True],
            "is_warmup": [False],
        }
    )
    policy = pd.DataFrame(
        {
            "sim_job_id": ["a"],
            "terminal_status": ["completed"],
            "is_evaluation": [True],
            "is_warmup": [False],
            "sim_policy_delay_s": [600],
            "runtime_s": [3600],
            "user_id": ["u1"],
        }
    )
    manifest = {
        "experiment_id": "test",
        "acceptance": {
            "capacity_dynamic_carbon_reduction_pct_minimum": 0.05,
            "capacity_result_must_exceed_previous_noop_noise_pct": 0.00591,
            "total_user_wait_p95_increase_seconds_maximum": 300,
            "jobs_delayed_longer_than_actual_runtime": 0,
            "maximum_user_policy_delay_minutes": 60,
            "maximum_single_user_share_of_policy_delay": 1.0,
        },
    }
    report = evaluate_acceptance(comparison, baseline, policy, manifest, "policy")
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["repeatability_diagnostics"][
        "policy_exceeds_same_date_no_policy_repeat"
    ]


def test_acceptance_rejects_matching_but_incomplete_scenario_pair() -> None:
    comparison = pd.DataFrame(
        [
            {"scenario": "baseline"},
            {
                "scenario": "no_policy_repeat",
                "dynamic_carbon_reduction_pct_capacity_weighted": 0.01,
                "total_wait_p95_change_s": -5,
            },
            {
                "scenario": "policy",
                "dynamic_carbon_reduction_pct_capacity_weighted": 0.2,
                "dynamic_carbon_reduction_kg_capacity_weighted": 0.1,
                "dynamic_carbon_reduction_pct_node_request_upper": 0.0,
                "dynamic_carbon_reduction_pct_allocated_node_distinct": 0.0,
                "total_wait_p95_change_s": 0,
                "wait_budget_violations": 0,
            },
        ]
    )
    results = pd.DataFrame(
        {
            "sim_job_id": ["a"],
            "terminal_status": ["completed"],
            "is_evaluation": [True],
            "is_warmup": [False],
            "sim_policy_delay_s": [0],
            "runtime_s": [3600],
            "user_id": ["u1"],
        }
    )
    manifest = {
        "experiment_id": "test",
        "final_untouched_window": {
            "expected_total_jobs": 2,
            "expected_evaluation_jobs": 2,
            "expected_carry_in_jobs": 0,
        },
        "acceptance": {
            "capacity_dynamic_carbon_reduction_pct_minimum": 0.05,
            "capacity_result_must_exceed_previous_noop_noise_pct": 0.00591,
            "total_user_wait_p95_increase_seconds_maximum": 300,
            "jobs_delayed_longer_than_actual_runtime": 0,
            "maximum_user_policy_delay_minutes": 60,
            "maximum_single_user_share_of_policy_delay": 1.0,
        },
    }
    report = evaluate_acceptance(
        comparison, results.copy(), results.copy(), manifest, "policy"
    )
    assert report["status"] == "fail"
    assert report["checks"]["all_expected_jobs_completed"] is False
