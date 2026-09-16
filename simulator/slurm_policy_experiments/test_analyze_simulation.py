import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("analyze_simulation.py")
SPEC = importlib.util.spec_from_file_location("analyze_simulation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SimulationLogTests(unittest.TestCase):
    def single_job(self, start_seconds=7, runtime_seconds=60, timeout=False):
        submitted = pd.Timestamp('2026-08-17T10:00:00Z')
        started = submitted + pd.Timedelta(seconds=start_seconds)
        profile = pd.DataFrame([{'sim_job_id': 'sim_1', 'slurm_job_id_expected': 1,
                                'release_dt_s': 0, 'eligible_dt_s': 0,
                                'dependency_delay_s': 0, 'source_eligible_wait_s': 0,
                                'source_submit_wait_s': 0, 'runtime_s': 60}])
        record = {'submitted_at': submitted, 'started_at': started,
                  'completed_at': started + pd.Timedelta(seconds=runtime_seconds)}
        if timeout:
            record['timed_out_at'] = record['completed_at']
        return MODULE.build_results(profile, {1: record})

    def test_negative_wait_is_preserved_and_fails_timing_not_completion(self):
        result = self.single_job(start_seconds=-83)
        summary = MODULE.validation_summary(result)
        self.assertEqual(result.sim_scheduler_wait_s.iloc[0], -83)
        self.assertEqual(result.sim_total_user_wait_s.iloc[0], -83)
        self.assertTrue(summary['completion_pass'])
        self.assertTrue(summary['strict_pass'])
        self.assertFalse(summary['event_order_pass'])
        self.assertFalse(summary['timing_pass'])
        self.assertEqual(summary['negative_scheduler_wait_jobs'], 1)
        self.assertEqual(summary['negative_total_wait_jobs'], 1)

    def test_zero_wait_passes_timing(self):
        summary = MODULE.validation_summary(self.single_job(start_seconds=0))
        self.assertTrue(summary['event_order_pass'])
        self.assertTrue(summary['timing_pass'])

    def test_long_completion_error_is_not_clean_timing(self):
        summary = MODULE.validation_summary(self.single_job(runtime_seconds=121))
        self.assertTrue(summary['completion_pass'])
        self.assertTrue(summary['event_order_pass'])
        self.assertFalse(summary['runtime_tolerance_60s_pass'])

    def test_timeout_marker_is_not_hidden_by_completion_marker(self):
        summary = MODULE.validation_summary(self.single_job(timeout=True))
        self.assertEqual(summary['timeout_events'], 1)
        self.assertFalse(summary['completion_pass'])

    def test_missing_timing_does_not_pass_ordering(self):
        result = self.single_job()
        result.loc[0, 'sim_scheduler_wait_s'] = float('nan')
        summary = MODULE.validation_summary(result)
        self.assertEqual(summary['missing_or_nonfinite_timing_jobs'], 1)
        self.assertFalse(summary['event_order_pass'])

    def test_parser_separates_completion_and_timeout(self):
        log = """\
[2026-08-17T10:00:00.000000] _slurm_rpc_submit_batch_job: JobId=1 InitPrio=1
[2026-08-17T10:00:05.000000] sched: Allocate JobId=1 NodeList=node001 #CPUs=4 Partition=sheffield
[2026-08-17T10:01:05.000000] _job_complete: JobId=1 done
[2026-08-17T10:02:00.000000] _slurm_rpc_submit_batch_job: JobId=2 InitPrio=1
[2026-08-17T10:02:10.000000] sched/backfill: _start_job: Started JobId=2 in gpu on gpu01
[2026-08-17T10:03:10.000000] Time limit exhausted for JobId=2
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slurmctld.log"
            path.write_text(log, encoding="utf-8")
            records = MODULE.parse_log(path)
        self.assertEqual(MODULE.terminal_status(records[1]), "completed")
        self.assertEqual(MODULE.terminal_status(records[2]), "timed_out")
        self.assertEqual(records[1]["allocated_nodelist"], "node001")

    def test_wait_components_are_kept_separate(self):
        profile = pd.DataFrame(
            [
                {
                    "sim_job_id": "sim_000001",
                    "slurm_job_id_expected": 1,
                    "release_dt_s": 30,
                    "eligible_dt_s": 10,
                    "dependency_delay_s": 5,
                    "source_eligible_wait_s": 8,
                    "source_submit_wait_s": 13,
                    "runtime_s": 60,
                }
            ]
        )
        records = {
            1: {
                "submitted_at": pd.Timestamp("2026-08-17T10:00:00Z"),
                "started_at": pd.Timestamp("2026-08-17T10:00:07Z"),
                "completed_at": pd.Timestamp("2026-08-17T10:01:07Z"),
            }
        }
        result = MODULE.build_results(profile, records).iloc[0]
        self.assertEqual(result["sim_policy_delay_s"], 20)
        self.assertEqual(result["sim_scheduler_wait_s"], 7)
        self.assertEqual(result["sim_total_user_wait_s"], 32)

    def test_validation_wait_percentiles_exclude_warmup_jobs(self):
        results = pd.DataFrame(
            {
                "is_warmup": [True, False],
                "terminal_status": ["completed", "completed"],
                "sim_submit_log_utc": pd.to_datetime(
                    ["2025-06-10T20:00:00Z", "2025-06-10T20:00:00Z"], utc=True
                ),
                "sim_start_log_utc": pd.to_datetime(
                    ["2025-06-10T20:01:40Z", "2025-06-10T20:00:10Z"], utc=True
                ),
                "sim_end_log_utc": pd.to_datetime(
                    ["2025-06-10T20:03:20Z", "2025-06-10T20:00:20Z"], utc=True
                ),
                "sim_scheduler_wait_s": [100.0, 10.0],
                "sim_total_user_wait_s": [100.0, 10.0],
                "source_eligible_wait_s": [100.0, 10.0],
                "source_submit_wait_s": [100.0, 10.0],
                "runtime_s": [100.0, 10.0],
                "sim_observed_runtime_s": [100.0, 10.0],
                "early_completion_event": [False, False],
                "sim_start_offset_s": [100.0, 10.0],
                "sim_end_offset_s": [200.0, 20.0],
            }
        )
        summary = MODULE.validation_summary(results)
        self.assertEqual(summary["evaluation_jobs"], 1)
        self.assertEqual(summary["warmup_jobs"], 1)
        self.assertEqual(summary["wait_metric_scope"], "evaluation_jobs_only")
        self.assertEqual(summary["sim_total_wait_p95_s"], 10.0)


if __name__ == "__main__":
    unittest.main()
