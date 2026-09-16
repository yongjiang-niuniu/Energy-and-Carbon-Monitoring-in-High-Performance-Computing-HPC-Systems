"""Preserve invalid replays and record a narrowly scoped carry-in correction."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'simulator/slurm_policy_experiments'
sys.path.insert(0, str(EXP))
from audit_exact_carryin import audit_profile, audit_results
from multiday_common import CAMPAIGN, save_json, sha256


def main():
    archive = CAMPAIGN / 'corrections/20260909_carry_in_order'
    if archive.exists():
        raise RuntimeError('Correction archive exists; inspect it before any retry')
    state = json.loads((CAMPAIGN / 'state.json').read_text())
    try:
        os.kill(state['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Stop the runner before changing campaign state')
    hashes = json.loads((CAMPAIGN / 'code_hashes.json').read_text())
    changed = {name for name, expected in hashes.items() if sha256(EXP / name) != expected}
    expected_changes = {
        'generate_policy_scenario.py', 'multiday_common.py',
        'run_multiday_campaign.py', 'test_multiday_campaign.py',
    }
    if changed != expected_changes:
        raise RuntimeError(f'Unexpected source changes: {changed}')

    retained, excluded = {}, {}
    for key, record in state['runs'].items():
        directory = CAMPAIGN / 'days' / key
        profile = pd.read_csv(directory / 'workload_profile.csv')
        audit = audit_profile(profile)
        if record['status'] == 'complete':
            for name, expected in record['artifact_hashes'].items():
                assert sha256(directory / name) == expected, (key, name)
            audit += audit_results(profile, pd.read_csv(directory / 'job_results.csv'))
        if key.split('/')[1].startswith('baseline'):
            assert record['status'] == 'complete' and all(x['passed'] for x in audit), key
            retained[key] = record
        else:
            if record['status'] == 'complete':
                assert any(not x['passed'] for x in audit), key
            excluded[key] = {'original_run': record, 'checks': audit,
                             'reason': 'Changed T0 event ordering; invalid initial-state reconstruction'}

    archive.mkdir(parents=True)
    for name in ['state.json', 'code_hashes.json', 'manifest.json', 'STATUS.md', 'LIVE_PROGRESS.md']:
        shutil.copy2(CAMPAIGN / name, archive / name)
    save_json(archive / 'excluded_runs.json', excluded)
    save_json(archive / 'retained_runs.json', retained)

    changes = []
    for name in sorted(changed):
        before, after = hashes[name], sha256(EXP / name)
        old = archive / 'frozen_source_before' / name
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CAMPAIGN / 'frozen_source' / name, old)
        shutil.copy2(EXP / name, CAMPAIGN / 'frozen_source' / name)
        hashes[name] = after
        changes.append({'path': name, 'before': before, 'after': after})

    for key in excluded:
        date, name = key.split('/')
        target = archive / 'days' / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(CAMPAIGN / 'days' / key, target)
        for log in (CAMPAIGN / 'days' / date / 'logs').glob(name + '_*.log'):
            target_log = archive / 'days' / date / 'logs' / log.name
            target_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(log, target_log)
    for date in state['dates']:
        comparison = CAMPAIGN / 'days' / date / 'comparison'
        if comparison.exists():
            shutil.move(comparison, archive / 'days' / date / 'comparison')
    if (CAMPAIGN / 'summary').exists():
        shutil.move(CAMPAIGN / 'summary', archive / 'summary')

    amendment = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Policy finalization sorted T0 jobs by original submit time instead of retaining the baseline running/queued order. This produced an unequal initial state.',
        'outcomes_had_been_observed': True,
        'observed_evidence': 'Nov13: 1/305 running carry-in jobs started around 1722 seconds late in every policy replay. Nov15: 25/337 started over 60 seconds late in each of six completed policy replays; maximum around 82482 seconds. All five completed baselines passed the unchanged 60-second audit.',
        'repair': 'Preserve baseline category and same-category job order at equal release times; add pre-replay input-order validation and per-replay carry-in fail-fast auditing.',
        'not_changed': ['dates', 'job populations', 'runtime and resources', 'policy parameters',
                        'carbon signal', 'simulator image', 'energy model', 'stability thresholds'],
        'manifest_sha256': sha256(CAMPAIGN / 'manifest.json'),
        'source_changes': changes, 'retained_complete_runs': len(retained),
        'excluded_complete_runs': sum(x['original_run']['status'] == 'complete' for x in excluded.values()),
        'interrupted_runs': sum(x['original_run']['status'] == 'running' for x in excluded.values()),
        'archive': str(archive), 'tests_passed': 72,
        'interpretation': 'Excluded outputs are diagnostic evidence only. Regenerate and replay all affected policies on the same dates. Do not count old outputs as evidence for or against a strategy.',
    }
    save_json(CAMPAIGN / 'carry_in_order_amendment.json', amendment)
    save_json(CAMPAIGN / 'code_hashes.json', hashes)
    state.update(runs=retained, dates={}, status='repair_ready', current='初始队列顺序已修复，等待恢复',
                 updated_utc=amendment['timestamp_utc'], step_elapsed_seconds=0,
                 correction=str(CAMPAIGN / 'carry_in_order_amendment.json'))
    state.pop('error', None)
    save_json(CAMPAIGN / 'state.json', state)
    print(json.dumps(amendment, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
