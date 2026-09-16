import unittest

from evaluate_prior_month_calibration import selected_quantile


class PriorMonthCalibrationTests(unittest.TestCase):
    def test_selects_lowest_quantile_that_meets_coverage(self):
        result = selected_quantile({0.5: 52.0, 0.75: 74.0, 0.9: 88.0})
        self.assertEqual(result, 0.9)

    def test_falls_back_to_highest_available_quantile(self):
        result = selected_quantile({0.5: 30.0, 0.75: 50.0, 0.9: 70.0})
        self.assertEqual(result, 0.9)

    def test_keeps_q75_when_it_meets_target(self):
        result = selected_quantile({0.5: 55.0, 0.75: 78.0, 0.9: 92.0})
        self.assertEqual(result, 0.75)


if __name__ == "__main__":
    unittest.main()
