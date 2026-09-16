import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("generate_policy_scenario.py")
SPEC = importlib.util.spec_from_file_location("generate_policy_scenario", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def carbon_frame(values):
    start = pd.Timestamp("2025-06-10T20:00Z")
    return pd.DataFrame(
        {
            "from_utc": [start + pd.Timedelta(minutes=30 * i) for i in range(len(values))],
            "to_utc": [start + pd.Timedelta(minutes=30 * (i + 1)) for i in range(len(values))],
            "intensity_gco2_per_kwh": values,
        }
    )


class PolicyScenarioTests(unittest.TestCase):
    def test_workload_epoch_prefers_explicit_epoch(self):
        profile = pd.DataFrame(
            {
                "event_epoch_utc": ["2025-06-10T22:00:00Z"] * 2,
                "source_submit_utc": [
                    "2025-06-11T00:00:00Z",
                    "2025-06-11T03:00:00Z",
                ],
                "submit_dt_s": [7200, 9000],
            }
        )
        self.assertEqual(
            MODULE.workload_epoch(profile), pd.Timestamp("2025-06-10T22:00:00Z")
        )

    def test_workload_epoch_fallback_uses_uncompressed_warmup_rows(self):
        profile = pd.DataFrame(
            {
                "source_submit_utc": [
                    "2025-06-10T22:00:00Z",
                    "2025-06-11T03:00:00Z",
                ],
                "submit_dt_s": [0, 9000],
                "is_warmup": [True, False],
            }
        )
        self.assertEqual(
            MODULE.workload_epoch(profile), pd.Timestamp("2025-06-10T22:00:00Z")
        )

    def test_b1_selects_lower_intensity_within_deadline(self):
        carbon = carbon_frame([230, 180, 90, 100])
        eligible = pd.Timestamp("2025-06-10T20:05Z")
        release, before, after = MODULE.best_release_time(
            carbon, eligible, eligible + pd.Timedelta(hours=1)
        )
        self.assertEqual(release, pd.Timestamp("2025-06-10T21:00Z"))
        self.assertEqual(before, 230)
        self.assertEqual(after, 90)

    def test_finalize_reassigns_ids_after_policy_reordering(self):
        profile = pd.DataFrame(
            [
                {"sim_job_id": "sim_000001", "slurm_job_id_expected": 1, "release_dt_s": 30, "eligible_dt_s": 0, "source_submit_utc": "2025-06-10T20:00:00Z"},
                {"sim_job_id": "sim_000002", "slurm_job_id_expected": 2, "release_dt_s": 10, "eligible_dt_s": 10, "source_submit_utc": "2025-06-10T20:00:10Z"},
            ]
        )
        result = MODULE.finalize_profile(profile, "b1")
        self.assertEqual(result["sim_job_id"].tolist(), ["sim_000002", "sim_000001"])
        self.assertEqual(result["slurm_job_id_expected"].tolist(), [1, 2])
        self.assertEqual(result["baseline_slurm_job_id"].tolist(), [2, 1])

    def test_runtime_aware_b1_uses_the_whole_job_interval(self):
        carbon = carbon_frame([50, 500, 50, 50, 50])
        eligible = pd.Timestamp("2025-06-10T20:00Z")
        release, before, after = MODULE.best_runtime_release_time(
            carbon,
            eligible,
            eligible + pd.Timedelta(hours=1),
            runtime_s=3600,
        )
        self.assertEqual(release, pd.Timestamp("2025-06-10T21:00Z"))
        self.assertEqual(before, 275)
        self.assertEqual(after, 50)

    def test_adaptive_wait_budget_protects_short_jobs(self):
        carbon = carbon_frame([300, 50, 50, 50])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "source_eligible_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 900,
                    "partition": "sheffield",
                    "cpus": 1,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                }
            ]
        )
        result, _ = MODULE.apply_adaptive(
            profile,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=1.0,
            max_delay_hours=4.0,
            wait_budget_ratio=1.0,
            min_carbon_saving_pct=0.0,
            wait_penalty=0.0,
            congestion_penalty=0.0,
            high_quantile=0.75,
            cap_fraction=0.75,
        )
        self.assertEqual(result.loc[0, "release_dt_s"], 0)
        self.assertEqual(result.loc[0, "allowed_wait_budget_s"], 900)

    def test_adaptive_minimum_saving_avoids_marginal_delay(self):
        carbon = carbon_frame([200, 190, 190, 190])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "source_eligible_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 1800,
                    "partition": "sheffield",
                    "cpus": 1,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                }
            ]
        )
        result, _ = MODULE.apply_adaptive(
            profile,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=1.0,
            max_delay_hours=1.0,
            wait_budget_ratio=4.0,
            min_carbon_saving_pct=10.0,
            wait_penalty=0.0,
            congestion_penalty=0.0,
            high_quantile=0.75,
            cap_fraction=0.75,
        )
        self.assertEqual(result.loc[0, "release_dt_s"], 0)
        self.assertEqual(result.loc[0, "expected_runtime_carbon_saving_pct"], 0)

    def test_dynamic_marginal_delays_high_impact_job(self):
        carbon = carbon_frame([300, 50, 50, 50])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 1800,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 64,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                }
            ]
        )
        result, _ = MODULE.apply_dynamic_marginal(
            profile,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=1.0,
            max_delay_hours=1.0,
            wait_budget_ratio=2.0,
            min_carbon_saving_pct=2.0,
            wait_penalty=0.10,
            congestion_penalty=0.0,
            cap_fraction=0.75,
            min_carbon_return=0.1,
            sensitivity_penalty=0.5,
        )
        self.assertEqual(result.loc[0, "release_dt_s"], 1800)
        self.assertGreater(
            result.loc[0, "expected_risk_adjusted_carbon_saving_proxy"], 0
        )
        self.assertGreater(result.loc[0, "dynamic_marginal_utility"], 0)

    def test_dynamic_marginal_releases_low_impact_job_immediately(self):
        carbon = carbon_frame([300, 50, 50, 50])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 1800,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 1,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                }
            ]
        )
        result, _ = MODULE.apply_dynamic_marginal(
            profile,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=1.0,
            max_delay_hours=1.0,
            wait_budget_ratio=2.0,
            min_carbon_saving_pct=2.0,
            wait_penalty=0.10,
            congestion_penalty=0.0,
            cap_fraction=0.75,
            min_carbon_return=0.0,
            sensitivity_penalty=0.5,
        )
        self.assertEqual(result.loc[0, "release_dt_s"], 0)
        self.assertEqual(result.loc[0, "dynamic_marginal_utility"], 0)

    def test_dynamic_forecast_delays_when_both_proxies_improve(self):
        carbon = carbon_frame([300, 50, 50, 50])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 1800,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 64,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                }
            ]
        )
        baseline_results = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "release_dt_s": 0,
                    "sim_scheduler_wait_s": 0.0,
                }
            ]
        )
        result, _ = MODULE.apply_dynamic_forecast(
            profile,
            baseline_results,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=1.0,
            max_delay_hours=1.0,
            wait_budget_ratio=2.0,
            min_carbon_saving_pct=2.0,
            min_node_saving_pct=2.0,
            wait_penalty=0.0,
            congestion_penalty=0.0,
            cap_fraction=0.75,
            min_carbon_return=0.0,
        )
        self.assertEqual(result.loc[0, "release_dt_s"], 1800)
        self.assertGreater(
            result.loc[0, "expected_capacity_carbon_saving_pct"], 0
        )
        self.assertGreater(result.loc[0, "expected_node_carbon_saving_pct"], 0)

    def test_dynamic_forecast_rejects_node_proxy_regression(self):
        carbon = carbon_frame([300, 50, 50, 50])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_flexible",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 1800,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 64,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                },
                {
                    "sim_job_id": "sim_background",
                    "slurm_job_id_expected": 2,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 1800,
                    "nodes": 202,
                    "partition": "sheffield",
                    "cpus": 64,
                    "scheduled_gpus": 0,
                    "flexibility_score": 1.0,
                },
            ]
        )
        baseline_results = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_flexible",
                    "release_dt_s": 0,
                    "sim_scheduler_wait_s": 0.0,
                },
                {
                    "sim_job_id": "sim_background",
                    "release_dt_s": 0,
                    "sim_scheduler_wait_s": 0.0,
                },
            ]
        )
        result, _ = MODULE.apply_dynamic_forecast(
            profile,
            baseline_results,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=0.5,
            max_delay_hours=1.0,
            wait_budget_ratio=2.0,
            min_carbon_saving_pct=0.0,
            min_node_saving_pct=0.0,
            wait_penalty=0.0,
            congestion_penalty=0.0,
            cap_fraction=0.75,
            min_carbon_return=0.0,
        )
        self.assertEqual(result.loc[0, "release_dt_s"], 0)
        self.assertEqual(result.loc[0, "expected_node_carbon_saving_pct"], 0)

    def test_dynamic_forecast_runtime_factor_changes_wait_budget(self):
        carbon = carbon_frame([300, 50, 50, 50])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 1800,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 64,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                }
            ]
        )
        baseline_results = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "release_dt_s": 0,
                    "sim_scheduler_wait_s": 0.0,
                }
            ]
        )
        result, _ = MODULE.apply_dynamic_forecast(
            profile,
            baseline_results,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=1.0,
            max_delay_hours=1.0,
            wait_budget_ratio=1.0,
            min_carbon_saving_pct=0.0,
            min_node_saving_pct=0.0,
            wait_penalty=0.0,
            congestion_penalty=0.0,
            cap_fraction=0.75,
            min_carbon_return=0.0,
            runtime_estimate_factor=0.5,
        )
        self.assertEqual(result.loc[0, "decision_runtime_s"], 900)
        self.assertEqual(result.loc[0, "allowed_wait_budget_s"], 900)
        self.assertEqual(result.loc[0, "release_dt_s"], 0)

    def test_dynamic_history_uses_earlier_month_quantiles(self):
        carbon = carbon_frame([300, 50, 50, 50])
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T20:00:00Z",
                    "release_dt_s": 0,
                    "eligible_dt_s": 0,
                    "runtime_s": 9999,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 64,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                }
            ]
        )
        runtime = pd.DataFrame(
            [
                {
                    "level": "global",
                    "count": 1000,
                    "runtime_s_q50": 1800,
                    "runtime_s_q90": 2700,
                }
            ]
        )
        wait = pd.DataFrame(
            [{"level": "global", "count": 1000, "wait_s_q50": 0}]
        )
        load_rows = []
        for slot in range(40, 44):
            row = {"weekday_utc": 1, "half_hour_slot_utc": slot}
            for name in ["cpu", "a100", "h100", "h100_nvl", "node"]:
                row[f"load_{name}_q75"] = 0.0
            load_rows.append(row)
        history_model = {
            "metadata": {
                "model_type": "deterministic_history_quantiles",
                "minimum_group_count": 50,
                "maximum_training_runtime_hours": 12,
                "training_months": ["january", "february"],
                "holdout_month": "june",
                "temporal_split_pass": True,
            },
            "runtime": runtime,
            "wait": wait,
            "load": pd.DataFrame(load_rows),
        }
        result, metadata = MODULE.apply_dynamic_history(
            profile,
            history_model,
            carbon,
            pd.Timestamp("2025-06-10T20:00Z"),
            flex_fraction=1.0,
            max_delay_hours=1.0,
            wait_budget_ratio=2.0,
            min_carbon_saving_pct=2.0,
            min_node_saving_pct=2.0,
            wait_penalty=0.0,
            congestion_penalty=0.0,
            cap_fraction=0.75,
            min_carbon_return=0.0,
            runtime_quantile=0.50,
            load_quantile=0.75,
            queue_wait_quantile=0.50,
            max_queue_wait_hours=1.0,
            max_runtime_uncertainty_ratio=2.0,
        )
        self.assertEqual(result.loc[0, "decision_runtime_s"], 1800)
        self.assertEqual(result.loc[0, "release_dt_s"], 1800)
        self.assertEqual(result.loc[0, "decision_runtime_source"], "earlier_month_grouped_quantile")
        self.assertTrue(metadata["temporal_split_pass"])

    def test_warmup_and_interactive_jobs_are_never_flexible(self):
        profile = pd.DataFrame(
            [
                {
                    "partition": "sheffield",
                    "flexibility_score": 0.0,
                    "is_warmup": True,
                },
                {
                    "partition": "interactive",
                    "flexibility_score": 0.0,
                    "is_warmup": False,
                },
                {
                    "partition": "sheffield",
                    "flexibility_score": 0.0,
                    "is_warmup": False,
                },
            ]
        )
        self.assertEqual(
            MODULE.flexible_mask(profile, flex_fraction=1.0).tolist(),
            [False, False, True],
        )

    def test_time_hybrid_uses_day_b2_and_night_runtime_b1(self):
        start = pd.Timestamp("2025-06-10T07:00Z")
        carbon = pd.DataFrame(
            {
                "from_utc": [start + pd.Timedelta(minutes=30 * i) for i in range(32)],
                "to_utc": [start + pd.Timedelta(minutes=30 * (i + 1)) for i in range(32)],
                "intensity_gco2_per_kwh": [200] * 27 + [300, 40, 40, 40, 40],
            }
        )
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_day",
                    "slurm_job_id_expected": 1,
                    "source_submit_utc": "2025-06-10T09:00:00Z",
                    "release_dt_s": 7200,
                    "eligible_dt_s": 7200,
                    "runtime_s": 600,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 1,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                },
                {
                    "sim_job_id": "sim_night",
                    "slurm_job_id_expected": 2,
                    "source_submit_utc": "2025-06-10T20:30:00Z",
                    "release_dt_s": 48600,
                    "eligible_dt_s": 48600,
                    "runtime_s": 600,
                    "nodes": 1,
                    "partition": "sheffield",
                    "cpus": 1,
                    "scheduled_gpus": 0,
                    "flexibility_score": 0.0,
                },
            ]
        )
        result, metadata = MODULE.apply_time_hybrid(
            profile,
            carbon,
            start,
            flex_fraction=1.0,
            day_start_hour_utc=8,
            day_end_hour_utc=18,
            b1_max_delay_hours=1.0,
            high_quantile=0.75,
            cap_fraction=0.5,
            b2_max_delay_hours=1.0,
        )
        self.assertEqual(result.loc[0, "hybrid_mode"], "day_b2")
        self.assertEqual(result.loc[1, "hybrid_mode"], "night_runtime_b1")
        self.assertEqual(result.loc[0, "b1_delay_s"], 0)
        self.assertEqual(result.loc[1, "b1_delay_s"], 1800)
        self.assertEqual(result.loc[0, "allowed_wait_budget_s"], 3600)
        self.assertEqual(result.loc[1, "allowed_wait_budget_s"], 3600)
        self.assertEqual(metadata["day_b2_flexible_jobs"], 1)
        self.assertEqual(metadata["night_b1_flexible_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
