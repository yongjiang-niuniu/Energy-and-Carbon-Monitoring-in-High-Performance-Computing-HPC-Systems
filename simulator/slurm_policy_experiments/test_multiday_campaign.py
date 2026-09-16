import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from audit_exact_carryin import audit_profile
from multiday_common import assert_profile_match, validate_carbon
from summarize_multiday_campaign import aggregate_days, reduction
from generate_policy_scenario import finalize_profile
from run_multiday_campaign import CampaignRunner, CarryInValidationError


CRITERIA = {'positive_days_required': 4, 'worst_day_carbon_increase_limit_pct': .1}


def daily_frame(values):
    return pd.DataFrame({
        'reduction_pct_capacity_weighted': values,
        'exceeds_noise_and_floor': [v > .05 for v in values],
        'baseline_carbon_kg_capacity_weighted': [100] * len(values),
        'policy_carbon_kg_capacity_weighted': [100-v for v in values],
        'reduction_kg_capacity_weighted': values,
        'valid': True, 'delayed_jobs': 2,
        'whole_reduction_pct': [v / 10 for v in values],
        'baseline_whole_carbon_kg': 1000,
        'policy_whole_carbon_kg': [1000-v for v in values],
        'policy_delay_h': 1, 'evaluation_jobs': 100,
        'p95_wait_change_s': 20, 'low_impact_screen': True,
        'top_user_delay_share': .3, 'all_models_positive': True,
    })


class MultiDayTests(unittest.TestCase):
    def carry_in_fixture(self):
        return pd.DataFrame({
            'sim_job_id': ['run1', 'run2', 'queue', 'eval'],
            'slurm_job_id_expected': [1, 2, 3, 4],
            'carry_in_type': ['running', 'running', 'queued', 'evaluation'],
            'source_submit_utc': ['2025-11-12', '2025-11-11', '2025-11-01', '2025-11-13'],
            'release_dt_s': [0, 0, 0, 60], 'eligible_dt_s': [0, 0, 0, 60],
            'is_warmup': [True, True, True, False], 'flexibility_score': [1., 1., 1., .1],
        })

    def test_finalization_preserves_running_before_earlier_submitted_queue(self):
        base = self.carry_in_fixture()
        result = finalize_profile(base.iloc[[2, 1, 0, 3]], 'test')
        self.assertEqual(result['sim_job_id'].tolist(), base['sim_job_id'].tolist())
        self.assertEqual(result['slurm_job_id_expected'].tolist(), [1, 2, 3, 4])
        assert_profile_match(base, result)

    def test_input_audit_rejects_old_sort_and_running_permutation(self):
        base = self.carry_in_fixture()
        for order in [[2, 1, 0, 3], [1, 0, 2, 3]]:
            with self.subTest(order=order):
                with self.assertRaisesRegex(ValueError, 'Running carry-in'):
                    assert_profile_match(base, base.iloc[order])

    def test_non_carry_in_finalization_keeps_legacy_tie_break(self):
        base = self.carry_in_fixture().drop(columns='carry_in_type')
        result = finalize_profile(base, 'test')
        self.assertEqual(result['sim_job_id'].tolist(), ['queue', 'run2', 'run1', 'eval'])

    def test_post_replay_carry_in_failure_is_not_accepted(self):
        runner = CampaignRunner.__new__(CampaignRunner)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch('run_multiday_campaign.pd.read_csv', return_value=pd.DataFrame()), \
                 patch('audit_exact_carryin.audit_profile', return_value=[]), \
                 patch('audit_exact_carryin.audit_results', return_value=[{'check': 'running_carry_in_started_within_60s', 'passed': False}]):
                with self.assertRaises(CarryInValidationError):
                    runner.check_carry_in(directory)
            self.assertTrue((directory / 'carry_in_validation.json').exists())

    def test_outlier_mean_is_not_stability(self):
        row = aggregate_days(daily_frame([10, -.3, -.2, -.1, -.2]), 5, CRITERIA)
        self.assertGreater(row['mean_reduction_pct'], 0)
        self.assertEqual(row['stability_label'], 'mixed_direction')
        self.assertLess(row['leave_one_out_min_mean_pct'], 0)

    def test_consistent_effect_passes_directional_rule(self):
        row = aggregate_days(daily_frame([.2, .1, .3, .15, .12]), 5, CRITERIA)
        self.assertEqual(row['stability_label'], 'directionally_stable_on_tested_days')

    def test_missing_and_invalid_dates_do_not_pass(self):
        frame = daily_frame([1, 1, 1, 1])
        self.assertEqual(aggregate_days(frame, 5, CRITERIA)['stability_label'], 'incomplete_or_invalid')
        frame = daily_frame([1, 1, 1, 1, 1])
        frame.loc[2, 'valid'] = False
        self.assertEqual(aggregate_days(frame, 5, CRITERIA)['stability_label'], 'incomplete_or_invalid')

    def test_pooled_reduction_is_not_average_percentage(self):
        self.assertAlmostEqual(reduction(110, 99), 10)
        frame = daily_frame([10, -10])
        frame['baseline_carbon_kg_capacity_weighted'] = [100, 10]
        frame['policy_carbon_kg_capacity_weighted'] = [90, 11]
        row = aggregate_days(frame, 2, CRITERIA)
        self.assertEqual(row['mean_reduction_pct'], 0)
        self.assertAlmostEqual(row['pooled_carbon_weighted_reduction_pct'], 900/110)

    def test_identical_counts_but_different_job_ids_fail(self):
        left = pd.DataFrame({'sim_job_id': ['a', 'b']})
        with self.assertRaisesRegex(ValueError, 'population'):
            assert_profile_match(left, pd.DataFrame({'sim_job_id': ['a', 'c']}))

    def test_resource_change_fails(self):
        left = pd.DataFrame({'sim_job_id': ['a'], 'cpus': [2]})
        right = pd.DataFrame({'sim_job_id': ['a'], 'cpus': [4]})
        with self.assertRaisesRegex(ValueError, 'cpus'):
            assert_profile_match(left, right)

    def test_carbon_gap_rejected(self):
        start = pd.Timestamp('2025-11-13', tz='UTC')
        frame = pd.DataFrame({'from_utc': [start], 'to_utc': [start + pd.Timedelta(minutes=30)], 'intensity_gco2_per_kwh': [100]})
        validate_carbon(frame, start, start + pd.Timedelta(minutes=30))
        with self.assertRaisesRegex(ValueError, 'gap'):
            validate_carbon(frame, start, start + pd.Timedelta(hours=1))

    def test_window_without_held_jobs_is_valid(self):
        start = pd.Timestamp('2025-11-13', tz='UTC')
        frame = pd.DataFrame({
            'sim_job_id': ['a', 'b'], 'event_epoch_utc': [start] * 2,
            'carry_in_type': ['running', 'evaluation'],
            'source_eligible_utc': [start, start + pd.Timedelta(minutes=1)],
            'release_dt_s': [0, 60], 'policy_delay_s': [0, 0], 'flexibility_score': [1, .1],
        })
        self.assertTrue(all(item['passed'] for item in audit_profile(frame)))


if __name__ == '__main__':
    unittest.main()
