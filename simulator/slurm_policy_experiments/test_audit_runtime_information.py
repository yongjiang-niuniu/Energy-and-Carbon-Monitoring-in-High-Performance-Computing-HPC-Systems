import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).with_name("audit_runtime_information.py")
SPEC = importlib.util.spec_from_file_location("audit_runtime_information", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RuntimeInformationAuditTests(unittest.TestCase):
    def test_ratio_summary_excludes_warmup(self):
        profile = pd.DataFrame(
            {
                "runtime_s": [100, 200, 50],
                "source_timelimit_min": [10, 20, 100],
                "is_warmup": [False, False, True],
            }
        )
        summary = MODULE.ratio_summary(profile)
        self.assertEqual(summary["evaluation_jobs"], 2)
        self.assertEqual(summary["requested_to_actual_ratio_p50"], 6.0)
        self.assertEqual(summary["requested_to_actual_ratio_max"], 6.0)


if __name__ == "__main__":
    unittest.main()
