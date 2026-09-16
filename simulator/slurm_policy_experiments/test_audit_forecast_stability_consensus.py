import unittest

import pandas as pd

from audit_forecast_stability_consensus import audit_consensus_sets


class ForecastStabilityConsensusAuditTests(unittest.TestCase):
    def test_consensus_subset_and_thresholds(self):
        reference = pd.DataFrame([
            {"sim_job_id": "a", "eligible_dt_s": 0, "release_dt_s": 10},
            {"sim_job_id": "b", "eligible_dt_s": 0, "release_dt_s": 10},
        ])
        consensus = pd.DataFrame([
            {
                "sim_job_id": "a", "eligible_dt_s": 0, "release_dt_s": 10,
                "consensus_selection_fraction": 0.9,
            },
            {
                "sim_job_id": "b", "eligible_dt_s": 0, "release_dt_s": 0,
                "consensus_selection_fraction": 0.4,
            },
        ])
        result = audit_consensus_sets(reference, consensus, 0.8)
        self.assertTrue(result["pass"])
        self.assertEqual(result["consensus_delayed_ids"], ["a"])


if __name__ == "__main__":
    unittest.main()
