import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluate_carbon_forecast_sensitivity import (
    jaccard,
    perturb_intensity,
    run_history_policy,
)


class CarbonForecastSensitivityTests(unittest.TestCase):
    def test_zero_noise_preserves_signal(self):
        values = np.array([50.0, 100.0, 150.0])
        np.testing.assert_array_equal(perturb_intensity(values, 0.0, 1), values)

    def test_noise_is_reproducible_and_positive(self):
        values = np.full(20, 100.0)
        first = perturb_intensity(values, 0.2, 7)
        second = perturb_intensity(values, 0.2, 7)
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first >= 1.0).all())

    def test_jaccard_handles_empty_and_partial_sets(self):
        self.assertEqual(jaccard(set(), set()), 1.0)
        self.assertEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_runtime_uncertainty_is_the_last_policy_argument(self):
        with patch(
            "evaluate_carbon_forecast_sensitivity.POLICY.apply_dynamic_history",
            return_value=(pd.DataFrame(), {}),
        ) as mocked:
            run_history_policy(
                pd.DataFrame(),
                {},
                pd.DataFrame(),
                pd.Timestamp("2025-01-01T00:00:00Z"),
                100.0,
            )
        self.assertEqual(mocked.call_args.args[7], 2.0)
        self.assertEqual(mocked.call_args.args[-1], 100.0)


if __name__ == "__main__":
    unittest.main()
