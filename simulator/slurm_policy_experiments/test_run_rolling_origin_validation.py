import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_rolling_origin_validation import (
    summarize_origin,
    training_months_for_holdout,
)


class RollingOriginValidationTests(unittest.TestCase):
    def test_expanding_split_uses_all_earlier_months(self):
        self.assertEqual(
            training_months_for_holdout("june", "expanding", 3, 2),
            ["january", "february", "march", "april", "may"],
        )

    def test_recent_split_uses_fixed_window(self):
        self.assertEqual(
            training_months_for_holdout("june", "recent", 3, 2),
            ["march", "april", "may"],
        )

    def test_summary_extracts_q75_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)
            pd.DataFrame([
                {
                    "target": "runtime", "quantile": 0.75,
                    "observed_at_or_below_prediction_pct": 60.0,
                    "median_absolute_percentage_error_pct": 90.0,
                    "mean_bias_s": -10.0,
                },
                {
                    "target": "queue_wait", "quantile": 0.75,
                    "observed_at_or_below_prediction_pct": 80.0,
                    "median_absolute_percentage_error_pct": 10.0,
                    "mean_bias_s": 5.0,
                },
            ]).to_csv(evaluation / "runtime_wait_accuracy.csv", index=False)
            pd.DataFrame([
                {
                    "target": target, "quantile": 0.75,
                    "observed_at_or_below_prediction_pct": 75.0,
                    "mean_absolute_error_fraction": 0.1,
                }
                for target in ["load_cpu", "load_a100", "load_h100", "load_node"]
            ]).to_csv(evaluation / "load_accuracy.csv", index=False)
            row = summarize_origin(
                "expanding", "june", ["january"], Path("model"), evaluation
            )
        self.assertEqual(row["runtime_q75_coverage_error_pp"], 15.0)
        self.assertEqual(row["wait_q75_coverage_pct"], 80.0)


if __name__ == "__main__":
    unittest.main()
