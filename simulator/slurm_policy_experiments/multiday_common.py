"""Shared configuration and validation for the frozen multi-day campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CAMPAIGN = ROOT / 'campaigns/multiday_20260908'
DATES = ['2025-11-13', '2025-11-15', '2025-11-18', '2025-12-09', '2025-12-10']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    temporary.replace(path)


def policy_specs() -> list[dict]:
    standard = json.loads((ROOT / 'manifests/untouched_24h_20260902_standard_matrix.json').read_text())['scenarios']
    names = ['b1_runtime_4h', 'b2_q75_cap50', 'adaptive_balanced', 'time_hybrid_08_18', 'history_only']
    selected = [dict(next(x for x in standard if x['name'] == name)) for name in names]
    for spec in selected:
        spec['generator'] = 'generate_policy_scenario.py'
        spec['parameters'] = dict(spec['parameters'])
        spec['parameters'].pop('history_model_dir', None)
        spec['uses_history'] = spec['name'] == 'history_only'
        spec['information'] = 'earlier-month runtime/load estimates' if spec['uses_history'] else 'observed runtime; retrospective carbon signal'
    oracle = json.loads((ROOT / 'manifests/untouched_24h_20260902_oracle_matrix.json').read_text())['scenarios'][1]
    oracle['generator'] = 'generate_policy_scenario.py'
    oracle['parameters'].pop('schedule_forecast')
    oracle['uses_baseline_schedule'] = True
    oracle['information'] = 'observed runtime plus completed baseline schedule; oracle diagnostic'
    selected.append(oracle)
    selected.append({
        'name': 'low_impact_dynamic_q25', 'generator': 'generate_low_impact_dynamic_scenario.py',
        'uses_history': True, 'information': 'earlier-month runtime/load estimates',
        'parameters': {
            'flex_fraction': 0.3, 'max_delay_minutes': 60, 'runtime_budget_ratio': 0.5,
            'delay_runtime_quantile': 0.25, 'minimum_capacity_saving_pct': 1.0,
            'minimum_carbon_return': 0.001, 'wait_penalty': 0.01, 'congestion_penalty': 0.25,
            'cap_fraction': 0.9, 'max_runtime_uncertainty_ratio': 20,
            'long_gpu_hours': 12, 'long_gpu_uncertainty_limit': 1.5,
            'user_delay_budget_minutes': 60, 'user_delayed_job_limit': 2,
            'candidate_step_minutes': 5,
        },
    })
    return selected


def model_for_date(date: str) -> Path:
    name = 'history_quantiles_jan_oct_2025_nov_holdout' if date[5:7] == '11' else 'history_quantiles_jan_nov_2025_dec_holdout'
    return ROOT / 'models' / name


def assert_profile_match(baseline: pd.DataFrame, policy: pd.DataFrame) -> None:
    from validate_experiment import INVARIANT_COLUMNS, mismatch_count
    left = baseline.set_index('sim_job_id').sort_index()
    right = policy.set_index('sim_job_id').sort_index()
    if not left.index.is_unique or not right.index.is_unique or not left.index.equals(right.index):
        raise ValueError('The job population changed')
    for col in INVARIANT_COLUMNS:
        if col in left or col in right:
            if col not in left or col not in right or mismatch_count(left[col], right[col]):
                raise ValueError(f'Workload invariant changed: {col}')
    delay = right['release_dt_s'] - right['eligible_dt_s']
    if (delay < 0).any() or delay[right['is_warmup']].ne(0).any():
        raise ValueError('Invalid release or carry-in delay')
    if (delay[right['flexibility_score'].ge(0.3)] > 0).any():
        raise ValueError('Non-flexible job delayed')
    if 'carry_in_type' in baseline:
        running_ids = baseline.loc[baseline['carry_in_type'].eq('running'), 'sim_job_id'].tolist()
        if policy.iloc[:len(running_ids)]['sim_job_id'].tolist() != running_ids:
            raise ValueError('Running carry-in must be first, in baseline order')
        for category in ['queued', 'held']:
            original = baseline.loc[baseline['carry_in_type'].eq(category), 'sim_job_id'].tolist()
            changed = policy.loc[policy['carry_in_type'].eq(category), 'sim_job_id'].tolist()
            if changed != original:
                raise ValueError(f'Carry-in order changed: {category}')


def validate_carbon(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    frame = frame.copy()
    frame['from_utc'] = pd.to_datetime(frame['from_utc'], utc=True)
    frame['to_utc'] = pd.to_datetime(frame['to_utc'], utc=True)
    frame = frame.loc[frame['from_utc'].ge(start) & frame['to_utc'].le(end)].sort_values('from_utc')
    expected = pd.date_range(start, end, freq='30min', inclusive='left')
    if len(frame) != len(expected) or not pd.DatetimeIndex(frame['from_utc']).equals(expected):
        raise ValueError('Carbon signal has a gap, duplicate or missing boundary')
    if not (frame['to_utc'] - frame['from_utc']).eq(pd.Timedelta(minutes=30)).all():
        raise ValueError('Carbon interval is not 30 minutes')
    values = pd.to_numeric(frame['intensity_gco2_per_kwh'], errors='coerce')
    if not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError('Carbon signal contains invalid values')
    return frame
