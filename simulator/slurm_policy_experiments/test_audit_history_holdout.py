import unittest

from audit_history_holdout import find_identifier_columns


class HistoryHoldoutAuditTests(unittest.TestCase):
    def test_direct_identifiers_are_rejected(self):
        columns = ["partition", "user_id", "job_name", "runtime_s_q75"]
        self.assertEqual(find_identifier_columns(columns), ["user_id", "job_name"])

    def test_aggregate_job_counts_are_allowed(self):
        columns = [
            "month",
            "unique_numeric_job_ids",
            "completed_jobs",
            "valid_completed_supported_jobs",
        ]
        self.assertEqual(find_identifier_columns(columns), [])


if __name__ == "__main__":
    unittest.main()
