import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("add_warmup_load.py")
SPEC = importlib.util.spec_from_file_location("add_warmup_load", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class WarmupLoadTests(unittest.TestCase):
    def test_carry_in_job_is_released_during_preroll(self):
        cohort = pd.DataFrame(
            [
                {
                    "_job_id": 42,
                    "_start": pd.Timestamp("2025-06-10T19:00Z"),
                    "_end": pd.Timestamp("2025-06-10T21:00Z"),
                    "_mapped_partition": "sheffield",
                    "_requested_nodes": 1,
                    "_requested_cpus": 8,
                    "_requested_gpus": 0,
                    "_allocated_gpus": 0,
                    "_requested_memory_mib": 8192,
                    "_runtime_s": 7200,
                    "_timelimit_min": 180,
                    "Partition": "sheffield",
                    "QOS": "normal",
                    "User": "private-user",
                }
            ]
        )
        profile, summary = MODULE.carry_in_profile(
            cohort,
            pd.Timestamp("2025-06-10T20:00Z"),
            warmup_hours=2,
            max_runtime_hours=12,
        )
        self.assertEqual(summary["carry_in_jobs_added"], 1)
        self.assertEqual(profile.loc[0, "release_dt_s"], 3600)
        self.assertEqual(profile.loc[0, "runtime_s"], 7200)
        self.assertEqual(
            pd.Timestamp(profile.loc[0, "event_epoch_utc"]),
            pd.Timestamp("2025-06-10T18:00Z"),
        )
        self.assertEqual(profile.loc[0, "user_id"], "warmup_user_001")
        self.assertNotIn("private-user", profile.to_csv(index=False))


if __name__ == "__main__":
    unittest.main()
