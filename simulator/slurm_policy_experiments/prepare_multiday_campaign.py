#!/usr/bin/env python3
"""Freeze dates, policies and input hashes, then prepare five matched workloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

from multiday_common import CAMPAIGN, DATES, ROOT, REPO, model_for_date, policy_specs, save_json, sha256, validate_carbon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    manifest_path = CAMPAIGN / 'manifest.json'
    if not manifest_path.exists():
        image = subprocess.check_output(['docker', 'image', 'inspect', 'slurm-sim-scheduler-fix:latest', '--format', '{{.Id}}'], text=True).strip()
        save_json(manifest_path, {
            'campaign_id': CAMPAIGN.name, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'dates': DATES, 'date_selection': 'Calendar and existing trace availability, before new policy outcome inspection; includes weekday/weekend and November/December.',
            'cohort_rule': 'All eligible supported completed trace jobs in each UTC day, plus exact running/queued/held carry-in; same IDs/resources/runtime/flexibility within a date.',
            'dates_have_equal_counts': False, 'reason': 'Natural daily workload sizes preserve queue pressure; all strategies have equal counts on each date.',
            'archive': '/Users/yongjiangliu/Downloads/stanage_2025_queue_record.zip',
            'seed': 42, 'duration_hours': 24, 'runtime_cap_hours': 96,
            'carbon_window_days': 14, 'carbon_information': 'Retrospective regional forecast/modelled series, not archived decision-time issued forecasts; all policies share it.',
            'image_id': image, 'timeout_seconds': 21600, 'attempts_per_run': 2,
            'sequential_execution': True, 'baseline_runs_per_day': 3, 'adaptive_runs_per_day': 2,
            'planned_simulator_runs': 55, 'policies': policy_specs(),
            'energy': {'idle_kw': 140, 'regular_kw': 195, 'primary': 'capacity_weighted', 'sensitivity': ['node_request_upper', 'allocated_node_distinct'], 'pue': None},
            'stability': {
                'minimum_complete_days': 5, 'positive_days_required': 4,
                'material_saving_floor_pct': 0.05,
                'noise_rule': 'Daily maximum pairwise baseline carbon range / median baseline carbon; also include Adaptive repeated-run range when available.',
                'worst_day_carbon_increase_limit_pct': 0.1,
                'service_p95_increase_limit_s': 300,
                'service_delayed_beyond_runtime_limit': 0,
                'fairness_top_user_delay_share_limit': 0.4,
                'fairness_max_user_delay_minutes': 60,
                'fairness_max_delayed_jobs_per_user': 2,
                'aggregation': ['equal-day mean', 'median', 'sample standard deviation', 'IQR', 'min/max', 'positive/noise-exceeding days', 'pooled baseline-carbon-weighted reduction', 'leave-one-day-out sign'],
                'interpretation': 'Screening thresholds for this project, not established deployment standards. Five dates are descriptive; overlapping carry-in makes them non-independent. No significance claim or invented population confidence interval.',
            },
            'excluded': {
                'b1_1h_2h_6h_and_longer': 'Parameter sweeps represented by the previously presented 4h reference; budget sweep can follow this comparison.',
                'b1_point_intensity_and_b1_b2': 'Earlier simpler composition; runtime-aware B1 and B2 give clearer mechanism comparison.',
                'dynamic_marginal': 'Marginal-cost family represented by balanced forecast and two history-based rules.',
                'dynamic_forecast_strict_oracle': 'Same family as balanced oracle with stricter gates.',
                'forecast_consensus_80pct_and_safe_rolling': 'Uncertainty/abstention variants deferred to a later ablation; prior one-day no-action results do not establish that these are useless.',
                'low_impact_parameter_iterations': 'Retain one previously frozen q25 final rule; do not re-tune on these dates.',
            },
        })
    manifest = json.loads(manifest_path.read_text())
    freeze = CAMPAIGN / 'code_hashes.json'
    if not freeze.exists():
        files = sorted(ROOT.glob('*.py')) + sorted(ROOT.glob('*.sh')) + sorted((ROOT / 'config/stanage_2025_assumed').glob('*'))
        files += [p for date in DATES for p in model_for_date(date).glob('*') if p.is_file()]
        hashes = {}
        for p in sorted(set(files)):
            if not p.is_file():
                continue
            relative = p.relative_to(ROOT)
            hashes[str(relative)] = sha256(p)
            target = CAMPAIGN / 'frozen_source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        save_json(freeze, hashes)
        save_json(CAMPAIGN / 'archive_hash.json', {'path': manifest['archive'], 'sha256': sha256(Path(manifest['archive']))})
    if args.freeze_only:
        print(f'Frozen: {manifest_path}', flush=True)
        return
    prepare(manifest)


def prepare(manifest):
    from build_exact_24h_workload import load_trace, classify_jobs, build_profile, GENERATOR, MONTHS
    from audit_exact_carryin import audit_profile
    from download_neso_carbon import download
    for date in manifest['dates']:
        directory = CAMPAIGN / 'days' / date
        if (directory / 'prepared.json').exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        save_json(CAMPAIGN / 'preparation_status.json', {'date': date, 'stage': 'reading_trace'})
        t0 = pd.Timestamp(date, tz='UTC')
        # Use the same established full carry-in builder without reducing the cohort.
        trace, trace_audit = load_trace(Path(manifest['archive']), MONTHS[:t0.month], manifest['runtime_cap_hours'])
        selected, category_audit = classify_jobs(trace, t0, t0 + pd.Timedelta(days=1))
        del trace
        profile, profile_audit = build_profile(selected, t0, t0 + pd.Timedelta(days=1), manifest['seed'])
        del selected
        summary = {'method': 'Matched multi-day exact carry-in workload', 'trace_audit': trace_audit, 'category_audit': category_audit, 'profile_audit': profile_audit, 'max_runtime_hours': manifest['runtime_cap_hours']}
        GENERATOR.write_outputs(directory / 'baseline', profile, summary)
        save_json(directory / 'baseline/exact_carry_in_summary.json', summary)
        checks = audit_profile(profile)
        if not all(x['passed'] for x in checks):
            raise ValueError(f'Carry-in preparation audit failed: {checks}')
        save_json(directory / 'profile_audit.json', checks)
        save_json(CAMPAIGN / 'preparation_status.json', {'date': date, 'stage': 'carbon_data', 'jobs': len(profile)})
        end = t0 + pd.Timedelta(days=manifest['carbon_window_days'])
        start_string, end_string = t0.strftime('%Y-%m-%dT%H:%MZ'), end.strftime('%Y-%m-%dT%H:%MZ')
        carbon_path = directory / 'carbon.csv'
        if carbon_path.exists():
            carbon = validate_carbon(pd.read_csv(carbon_path), t0, end)
        else:
            for attempt in range(3):
                try:
                    carbon, metadata = download(start_string, end_string, 5)
                    carbon = validate_carbon(carbon, t0, end)
                    carbon.to_csv(carbon_path, index=False)
                    save_json(directory / 'carbon.metadata.json', metadata)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5)
        model_metadata = json.loads((model_for_date(date) / 'model_metadata.json').read_text())
        if not model_metadata['temporal_split_pass'] or model_metadata['holdout_month'] != MONTHS[t0.month - 1]:
            raise ValueError('History model temporal split mismatch')
        record = {
            'date': date, 'jobs': len(profile), 'evaluation_jobs': int(profile['is_evaluation'].sum()),
            'carry_in_jobs': int(profile['is_warmup'].sum()), 'gpu_jobs': int(profile['scheduled_gpus'].gt(0).sum()),
            'model': str(model_for_date(date)), 'profile_sha256': sha256(directory / 'baseline/workload_profile.csv'),
            'carbon_sha256': sha256(carbon_path), 'carbon_rows': len(carbon),
        }
        save_json(directory / 'prepared.json', record)
        print(json.dumps(record), flush=True)
    save_json(CAMPAIGN / 'preparation_status.json', {'stage': 'complete', 'dates': manifest['dates']})


if __name__ == '__main__':
    main()
