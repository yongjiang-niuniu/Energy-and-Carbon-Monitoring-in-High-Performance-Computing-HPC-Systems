import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("generate_empirical_workload.py")
SPEC = importlib.util.spec_from_file_location("generate_empirical_workload", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EmpiricalWorkloadTests(unittest.TestCase):
    def test_fixed_window_uses_systematic_sample(self):
        start = pd.Timestamp("2025-06-11T00:00Z")
        frame = pd.DataFrame(
            {"_submit": [start + pd.Timedelta(minutes=i) for i in range(5)]}
        )
        window, summary = MODULE.choose_fixed_window(frame, start, 1, 3)
        self.assertEqual(len(window), 3)
        self.assertEqual(summary["source_window_jobs_before_systematic_sample"], 5)
        self.assertEqual(summary["selection_method"], "fixed_utc_window")

    def test_duplicate_headers_are_stable(self):
        self.assertEqual(
            MODULE.unique_headers(["ReqTRES", "ReqTRES", "", "ReqTRES"]),
            ["ReqTRES", "ReqTRES__2", "_blank_3", "ReqTRES__3"],
        )

    def test_gpu_tres_ignores_telemetry_fields(self):
        value = "billing=10,gres/gpu=2,gres/gpu:h100=2,gres/gpumem=8192,gres/gpuutil=95"
        self.assertEqual(MODULE.parse_tres_value(value, "gpu"), 4)

    def test_memory_tres_is_converted_to_mib(self):
        self.assertEqual(MODULE.parse_tres_value("cpu=1,mem=8G", "mem"), 8192)

    def test_historical_partition_mapping(self):
        self.assertEqual(MODULE.PARTITION_MAP["hp-a100"], "gpu")
        self.assertNotIn("gpu,gpu-h100", MODULE.PARTITION_MAP)

    def test_gpu_nodes_expand_to_hold_total_request(self):
        row = pd.Series(
            {
                "_mapped_partition": "gpu-h100",
                "_requested_nodes": 1,
                "_requested_cpus": 16,
                "_requested_gpus": 4,
                "_allocated_gpus": 4,
                "_requested_memory_mib": 32768,
                "_runtime_s": 600,
                "_timelimit_min": 30,
            }
        )
        resources = MODULE.normalize_resources(row)
        self.assertIsNotNone(resources)
        self.assertEqual(resources["nodes"], 2)
        self.assertEqual(resources["scheduled_gpus"], 4)

    def test_simulator_timelimit_has_clock_headroom(self):
        row = pd.Series(
            {
                "_mapped_partition": "sheffield",
                "_requested_nodes": 1,
                "_requested_cpus": 1,
                "_requested_gpus": 0,
                "_allocated_gpus": 0,
                "_requested_memory_mib": 2048,
                "_runtime_s": 3,
                "_timelimit_min": 2,
            }
        )
        resources = MODULE.normalize_resources(row)
        self.assertIsNotNone(resources)
        self.assertEqual(resources["source_timelimit_min"], 2)
        self.assertEqual(resources["timelimit_min"], 61)

    def test_interactive_timelimit_never_exceeds_partition_max(self):
        row = pd.Series(
            {
                "_mapped_partition": "interactive",
                "_requested_nodes": 1,
                "_requested_cpus": 4,
                "_requested_gpus": 0,
                "_allocated_gpus": 0,
                "_requested_memory_mib": 15360,
                "_runtime_s": 27175,
                "_timelimit_min": 906,
            }
        )
        resources = MODULE.normalize_resources(row)
        self.assertIsNotNone(resources)
        self.assertEqual(resources["runtime_s"], 27175)
        self.assertEqual(resources["timelimit_min"], 480)
        self.assertFalse(resources["runtime_partition_capped"])

    def test_runtime_is_capped_below_partition_max(self):
        row = pd.Series(
            {
                "_mapped_partition": "interactive",
                "_requested_nodes": 1,
                "_requested_cpus": 1,
                "_requested_gpus": 0,
                "_allocated_gpus": 0,
                "_requested_memory_mib": 1024,
                "_runtime_s": 36000,
                "_timelimit_min": 600,
            }
        )
        resources = MODULE.normalize_resources(row)
        self.assertIsNotNone(resources)
        self.assertEqual(resources["runtime_s"], 479 * 60)
        self.assertEqual(resources["timelimit_min"], 480)
        self.assertTrue(resources["runtime_partition_capped"])

    def test_event_is_released_at_eligible_time(self):
        row = pd.Series(
            {
                "user_id": "user_001",
                "sim_job_id": "sim_000001",
                "slurm_job_id_expected": 1,
                "partition": "sheffield",
                "nodes": 1,
                "cpus": 4,
                "memory_per_node_mib": 8192,
                "gpu_per_node": 0,
                "gpu_type": "",
                "timelimit_min": 30,
                "runtime_s": 300,
                "release_dt_s": 123,
            }
        )
        line = MODULE.event_line(row)
        self.assertIn("-dt 123", line)
        self.assertIn("-J jobid_1", line)
        self.assertIn("--mem=8192M", line)
        self.assertNotIn("--gres", line)


if __name__ == "__main__":
    unittest.main()
