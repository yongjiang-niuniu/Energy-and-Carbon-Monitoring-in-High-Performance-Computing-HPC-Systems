import unittest

import pandas as pd

from audit_submission_timing import audit_frame


class TimingAuditTests(unittest.TestCase):
    def test_counts_keep_raw_values_and_separate_carry_in(self):
        frame = pd.DataFrame({'is_warmup': [True, False, False],
                              'sim_submit_offset_s': [10, 20, 90],
                              'release_dt_s': [0, 20, 0],
                              'sim_scheduler_wait_s': [-126.01, -83.31, 7],
                              'sim_total_user_wait_s': [4, -83.31, 8]})
        original = frame.copy(deep=True)
        all_rows, evaluation = audit_frame(frame, '2025-11-15', 'baseline')
        self.assertEqual(all_rows['negative_queue_wait_jobs'], 2)
        self.assertEqual(evaluation['negative_queue_wait_jobs'], 1)
        self.assertEqual(all_rows['negative_total_wait_jobs'], 1)
        self.assertEqual(all_rows['over_60s'], 1)
        self.assertEqual(all_rows['minimum_queue_wait_s'], -126.01)
        pd.testing.assert_frame_equal(frame, original)

    def test_invalid_scope_flag_is_not_silently_truthy(self):
        with self.assertRaises(ValueError):
            audit_frame(pd.DataFrame({'is_warmup': ['unknown']}), 'date', 'run')
