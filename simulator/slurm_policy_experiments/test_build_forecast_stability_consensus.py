import unittest

import pandas as pd

from build_forecast_stability_consensus import apply_consensus, selection_frequencies


class ForecastStabilityConsensusTests(unittest.TestCase):
    def test_selection_frequency_uses_noisy_scenarios_only(self):
        decisions = pd.DataFrame([
            {"scenario": "reference", "sim_job_id": "a"},
            {"scenario": "noise_1", "sim_job_id": "a"},
            {"scenario": "noise_2", "sim_job_id": "b"},
            {"scenario": "noise_2", "sim_job_id": "a"},
        ])
        frequencies, scenarios = selection_frequencies(
            decisions, {"noise_1", "noise_2", "noise_3"}
        )
        self.assertEqual(scenarios, 3)
        self.assertEqual(frequencies["a"], 2 / 3)
        self.assertEqual(frequencies["b"], 1 / 3)

    def test_consensus_restores_unstable_reference_job(self):
        profile = pd.DataFrame([
            {
                "sim_job_id": "stable", "eligible_dt_s": 10, "release_dt_s": 20,
                "source_submit_utc": "2025-01-01T00:00:00Z", "b1_delay_s": 10,
                "slurm_job_id_expected": 1,
            },
            {
                "sim_job_id": "unstable", "eligible_dt_s": 30, "release_dt_s": 50,
                "source_submit_utc": "2025-01-01T00:01:00Z", "b1_delay_s": 20,
                "slurm_job_id_expected": 2,
            },
        ])
        result = apply_consensus(
            profile, pd.Series({"stable": 0.9, "unstable": 0.4}), 0.8
        ).set_index("sim_job_id")
        self.assertEqual(result.loc["stable", "policy_delay_s"], 10)
        self.assertEqual(result.loc["unstable", "policy_delay_s"], 0)
        self.assertEqual(result.loc["unstable", "b1_delay_s"], 0)


if __name__ == "__main__":
    unittest.main()
