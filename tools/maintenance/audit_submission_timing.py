"""Read final replay tables and quantify the existing submission-time anchoring."""
from pathlib import Path
import argparse
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'simulator/slurm_policy_experiments/campaigns/multiday_20260908'


def audit_frame(frame, date, name):
    warmup = frame['is_warmup'].astype(str).str.lower()
    if not warmup.isin(['true', 'false']).all():
        raise ValueError('is_warmup must contain Boolean values')
    rows = []
    for scope, part in [('all', frame), ('evaluation', frame.loc[warmup.eq('false')])]:
        residual = part['sim_submit_offset_s'] - part['release_dt_s']
        absolute = residual.abs()
        queue = part['sim_scheduler_wait_s']
        total = part['sim_total_user_wait_s']
        rows.append({'date': date, 'run': name, 'scope': scope, 'jobs': len(part),
                     'abs_p50_s': absolute.median(), 'abs_p95_s': absolute.quantile(.95),
                     'abs_max_s': absolute.max(), 'over_60s': int(absolute.gt(60).sum()),
                     'signed_min_s': residual.min(), 'signed_max_s': residual.max(),
                     'negative_queue_wait_jobs': int(queue.lt(0).sum()),
                     'negative_total_wait_jobs': int(total.lt(0).sum()),
                     'minimum_queue_wait_s': queue.min(), 'minimum_total_wait_s': total.min()})
    return rows


def collect(campaign):
    state = json.loads((campaign / 'state.json').read_text())
    manifest = json.loads((campaign / 'manifest.json').read_text())
    names = ['baseline', 'baseline_repeat_1', 'baseline_repeat_2',
             'adaptive_balanced_repeat'] + [x['name'] for x in manifest['policies']]
    rows = []
    for date in manifest['dates']:
        for name in names:
            key = f'{date}/{name}'
            if state['runs'][key]['status'] != 'complete':
                raise ValueError(f'Not final complete: {key}')
            frame = pd.read_csv(campaign / 'days' / key / 'job_results.csv',
                                usecols=['sim_submit_offset_s', 'release_dt_s', 'is_warmup',
                                         'sim_scheduler_wait_s', 'sim_total_user_wait_s'])
            rows.extend(audit_frame(frame, date, name))
    result = pd.DataFrame(rows)
    all_rows = result.loc[result.scope.eq('all')]
    evaluation = result.loc[result.scope.eq('evaluation')]
    summary = {'final_replays': len(all_rows),
               'runs_with_any_absolute_residual_above_60s': int(all_rows.over_60s.gt(0).sum()),
               'largest_absolute_residual_s': float(all_rows.abs_max_s.max()),
               'largest_per_run_absolute_p95_s': float(all_rows.abs_p95_s.max()),
               'runs_with_negative_queue_wait': int(all_rows.negative_queue_wait_jobs.gt(0).sum()),
               'negative_queue_wait_records': int(all_rows.negative_queue_wait_jobs.sum()),
               'negative_evaluation_queue_wait_records': int(evaluation.negative_queue_wait_jobs.sum()),
               'negative_total_wait_records': int(all_rows.negative_total_wait_jobs.sum()),
               'minimum_queue_wait_s': float(all_rows.minimum_queue_wait_s.min()),
               'minimum_total_wait_s': float(all_rows.minimum_total_wait_s.min()),
               'counting_unit': 'Replay records, not unique production jobs; raw waiting values are not changed.',
               'definition': 'logged submission offset minus intended release offset, after the existing per-replay median epoch alignment',
               'interpretation': 'Non-constant within-replay timing residual; distinct from configured-versus-logged runtime error. Not a new simulation or a recalculated carbon result.'}
    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    parser.add_argument('--output', type=Path, required=True,
                        help='New audit directory; must not already exist.')
    args = parser.parse_args()
    if args.output.resolve() == args.campaign.resolve() or args.campaign.resolve() in args.output.resolve().parents:
        raise ValueError('Audit output must be outside the frozen campaign')
    if args.output.exists():
        raise FileExistsError(args.output)
    result, summary = collect(args.campaign)
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / 'submission_time_residual_audit.csv', index=False)
    (args.output / 'submission_time_residual_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(result.sort_values('abs_max_s', ascending=False).head(6).to_string(index=False))


if __name__ == '__main__':
    main()
