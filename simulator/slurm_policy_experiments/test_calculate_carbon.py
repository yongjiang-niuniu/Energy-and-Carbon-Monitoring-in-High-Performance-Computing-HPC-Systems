import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("calculate_carbon.py")
SPEC = importlib.util.spec_from_file_location("calculate_carbon", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CarbonCalculationTests(unittest.TestCase):
    def test_model_clips_negative_queue_wait_without_mutating_raw_measurement(self):
        frame = pd.DataFrame({'release_dt_s': [100.0], 'sim_scheduler_wait_s': [-83.0],
                              'sim_start_offset_s': [50.0]})
        self.assertEqual(MODULE.model_start_offsets(frame).tolist(), [100.0])
        self.assertEqual(frame.sim_scheduler_wait_s.iloc[0], -83.0)

    def test_partial_carbon_horizon_is_rejected(self):
        start = pd.Timestamp("2025-11-13T00:00Z")
        carbon = pd.DataFrame({
            "from_utc": [start],
            "to_utc": [start + pd.Timedelta(minutes=30)],
            "intensity_gco2_per_kwh": [100.0],
        })
        with self.assertRaisesRegex(ValueError, "end of the common"):
            MODULE.build_intervals(pd.DataFrame(), carbon, start,
                                   start + pd.Timedelta(hours=1), 140, 195)

    def test_interior_carbon_gap_is_rejected(self):
        start = pd.Timestamp("2025-11-13T00:00Z")
        carbon = pd.DataFrame({
            "from_utc": [start, start + pd.Timedelta(hours=1)],
            "to_utc": [start + pd.Timedelta(minutes=30), start + pd.Timedelta(minutes=90)],
            "intensity_gco2_per_kwh": [100.0, 110.0],
        })
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            MODULE.build_intervals(pd.DataFrame(), carbon, start,
                                   start + pd.Timedelta(minutes=90), 140, 195)

    def test_workload_epoch_uses_explicit_simulation_epoch(self):
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

    def test_interval_overlap_is_fractional(self):
        starts = pd.Series([pd.Timestamp("2025-06-10T20:15Z")])
        ends = pd.Series([pd.Timestamp("2025-06-10T20:45Z")])
        overlap = MODULE.interval_overlap_seconds(
            starts,
            ends,
            pd.Timestamp("2025-06-10T20:00Z"),
            pd.Timestamp("2025-06-10T20:30Z"),
        )
        self.assertEqual(overlap.tolist(), [900.0])

    def test_model_start_uses_release_plus_scheduler_wait(self):
        results = pd.DataFrame(
            {
                "release_dt_s": [100, 200],
                "sim_scheduler_wait_s": [2.5, 3.0],
                "sim_start_offset_s": [109.5, 211.0],
            }
        )
        self.assertEqual(
            MODULE.model_start_offsets(results).tolist(), [102.5, 203.0]
        )

    def test_cpu_utilization_uses_capacity_and_duration(self):
        jobs = pd.DataFrame(
            [
                {
                    "model_start_utc": pd.Timestamp("2025-06-10T20:00Z"),
                    "model_end_utc": pd.Timestamp("2025-06-10T20:30Z"),
                    "hardware_class": "cpu",
                    "cpus": 64,
                    "scheduled_gpus": 0,
                    "nodes": 1,
                    "allocated_nodelist": "node003",
                }
            ]
        )
        result = MODULE.utilization_for_interval(
            jobs,
            pd.Timestamp("2025-06-10T20:00Z"),
            pd.Timestamp("2025-06-10T20:30Z"),
        )
        self.assertAlmostEqual(result["util_cpu"], 1 / 174)
        self.assertAlmostEqual(result["util_capacity_weighted"], 1 / 202)
        self.assertAlmostEqual(result["util_allocated_node_distinct"], 1 / 202)
        self.assertAlmostEqual(result["util_node_request_upper"], 1 / 202)

    def test_distinct_node_utilization_deduplicates_shared_hosts(self):
        jobs = pd.DataFrame(
            [
                {
                    "model_start_utc": pd.Timestamp("2025-06-10T20:00Z"),
                    "model_end_utc": pd.Timestamp("2025-06-10T20:30Z"),
                    "allocated_nodelist": "node[003-004]",
                },
                {
                    "model_start_utc": pd.Timestamp("2025-06-10T20:10Z"),
                    "model_end_utc": pd.Timestamp("2025-06-10T20:20Z"),
                    "allocated_nodelist": "node003",
                },
            ]
        )
        utilization = MODULE.allocated_node_utilization_for_interval(
            jobs,
            pd.Timestamp("2025-06-10T20:00Z"),
            pd.Timestamp("2025-06-10T20:30Z"),
        )
        self.assertAlmostEqual(utilization, 2 / 202)

    def test_warmup_jobs_do_not_change_evaluation_wait_percentiles(self):
        jobs = pd.DataFrame(
            [
                {
                    "is_warmup": True,
                    "sim_scheduler_wait_s": 0,
                    "sim_policy_delay_s": 0,
                    "sim_total_user_wait_s": 0,
                    "model_start_utc": pd.Timestamp("2025-06-10T18:00Z"),
                    "model_end_utc": pd.Timestamp("2025-06-10T21:00Z"),
                },
                {
                    "is_warmup": False,
                    "sim_scheduler_wait_s": 100,
                    "sim_policy_delay_s": 200,
                    "sim_total_user_wait_s": 300,
                    "model_start_utc": pd.Timestamp("2025-06-10T20:00Z"),
                    "model_end_utc": pd.Timestamp("2025-06-10T21:00Z"),
                },
            ]
        )
        intervals = pd.DataFrame(
            [
                {
                    "duration_h": 1,
                    "util_capacity_weighted": 0.1,
                    "util_allocated_node_distinct": 0.15,
                    "util_node_request_upper": 0.2,
                    "dynamic_energy_kwh_capacity_weighted": 5,
                    "dynamic_carbon_kg_capacity_weighted": 1,
                    "whole_energy_kwh_capacity_weighted": 145,
                    "whole_carbon_kg_capacity_weighted": 29,
                    "dynamic_energy_kwh_allocated_node_distinct": 7.5,
                    "dynamic_carbon_kg_allocated_node_distinct": 1.5,
                    "whole_energy_kwh_allocated_node_distinct": 147.5,
                    "whole_carbon_kg_allocated_node_distinct": 29.5,
                    "dynamic_energy_kwh_node_request_upper": 10,
                    "dynamic_carbon_kg_node_request_upper": 2,
                    "whole_energy_kwh_node_request_upper": 150,
                    "whole_carbon_kg_node_request_upper": 30,
                }
            ]
        )
        summary = MODULE.summarize_scenario(
            "baseline",
            jobs,
            intervals,
            pd.Timestamp("2025-06-10T20:00Z"),
            pd.Timestamp("2025-06-10T21:00Z"),
        )
        self.assertEqual(summary["jobs"], 2)
        self.assertEqual(summary["evaluation_jobs"], 1)
        self.assertEqual(summary["warmup_jobs"], 1)
        self.assertEqual(summary["scheduler_wait_p50_s"], 100)
        self.assertEqual(summary["total_user_wait_p95_s"], 300)
        self.assertAlmostEqual(summary["bounded_slowdown_p95"], 3900 / 3600)
        self.assertEqual(summary["wait_budget_violations"], 0)


if __name__ == "__main__":
    unittest.main()
