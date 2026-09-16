import unittest

import pandas as pd

from compare_rolling_origin_modes import compare_modes


class RollingOriginModeComparisonTests(unittest.TestCase):
    def test_recent_minus_expanding_is_paired_by_month(self):
        common = {
            "runtime_q75_coverage_error_pp": [5.0],
            "runtime_q75_coverage_pct": [70.0],
            "runtime_q75_median_ape_pct": [50.0],
            "wait_q75_coverage_pct": [80.0],
            "wait_q75_coverage_error_pp": [5.0],
            "cpu_q75_coverage_pct": [75.0],
            "a100_q75_coverage_pct": [75.0],
            "h100_q75_coverage_pct": [75.0],
            "node_q75_coverage_pct": [75.0],
        }
        expanding = pd.DataFrame({"holdout_month": ["june"], **common})
        recent = expanding.copy()
        recent["runtime_q75_coverage_error_pp"] = 8.0
        result = compare_modes(expanding, recent)
        self.assertEqual(
            result.loc[0, "runtime_q75_coverage_error_pp_recent_minus_expanding"],
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
