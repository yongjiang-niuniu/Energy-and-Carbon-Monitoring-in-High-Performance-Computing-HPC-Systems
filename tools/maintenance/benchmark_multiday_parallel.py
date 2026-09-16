"""Measure three concurrent containers using disposable, exact-input startup probes."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'simulator/slurm_policy_experiments'
sys.path.insert(0, str(EXP))
from analyze_simulation import build_results, parse_log
from multiday_common import CAMPAIGN, save_json

OUT = REPO / 'work_logs/parallel_execution_20260909/preflight'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', default='colima')
    parser.add_argument('--output-tag', default='preflight')
    parser.add_argument('--observe-context', action='append', default=[])
    args = parser.parse_args()
    output = OUT.parent / args.output_tag
    docker = ['docker', '--context', args.context]
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    probes = []
    samples = []
    try:
        for date in ['2025-11-13', '2025-11-15']:
            folder = output / date
            folder.mkdir(exist_ok=True)
            for filename in ['workload_profile.csv', 'sim.events', 'users.sim']:
                shutil.copy2(CAMPAIGN / 'days' / date / 'baseline' / filename, folder / filename)
            subprocess.run([sys.executable, EXP / 'prepare_scenario.py', folder], check=True)
            name = 'stanage-parallel-probe-' + date.replace('-', '')
            subprocess.run(docker + ['run', '-d', '--rm', '--name', name, '--hostname', 'node001',
                            '--label', 'stanage.parallel_probe=20260909', '--ulimit', 'core=0',
                            '-v', f'{folder}:/opt/slurm-sim/etc', manifest['image_id'],
                            'bash', '-c', 'cd /opt/slurm-sim/etc && timeout -s KILL 150 slurmctld -D -i >/dev/null 2>&1; true'], check=True)
            probes.append((name, folder))
        started = time.monotonic()
        for _ in range(12):
            containers = []
            for context in [args.context, *args.observe_context]:
                stats = subprocess.check_output(['docker', '--context', context, 'stats', '--no-stream', '--format', '{{json .}}'], text=True)
                containers += [dict(json.loads(line), docker_context=context) for line in stats.splitlines()]
            sample = {'wall_s': time.monotonic() - started,
                      'containers': containers}
            sample['log_times'] = {}
            for name, folder in probes:
                if (folder / 'slurmctld.log').exists():
                    matches = re.findall(r'^\[([^]]+)\]', (folder / 'slurmctld.log').read_text(), re.M)
                    if matches:
                        sample['log_times'][name] = matches[-1]
            samples.append(sample)
            time.sleep(5)
        findings = []
        for name, folder in probes:
            profile = pd.read_csv(folder / 'workload_profile.csv')
            result = build_results(profile, parse_log(folder / 'slurmctld.log'))
            running = result.loc[result.carry_in_type.eq('running')]
            times = [(s['wall_s'], pd.Timestamp(s['log_times'][name])) for s in samples if name in s['log_times']]
            rate = (times[-1][1] - times[0][1]).total_seconds() / (times[-1][0] - times[0][0])
            steady = [row for row in times if row[0] >= 20]
            steady_rate = (steady[-1][1] - steady[0][1]).total_seconds() / (steady[-1][0] - steady[0][0])
            findings.append({'container': name, 'running_jobs': len(running),
                             'running_started': int(running.sim_start_offset_s.notna().sum()),
                             'running_max_offset_s': float(running.sim_start_offset_s.max()),
                             'running_within_60s': bool(running.sim_start_offset_s.le(60).all()),
                             'sim_seconds_per_wall_second': rate,
                             'steady_sim_seconds_per_wall_second': steady_rate,
                             'completed_in_probe': int(result.terminal_status.eq('completed').sum())})
        summary = {'time_utc': datetime.now(timezone.utc).isoformat(),
                   'context': args.context,
                   'scope': 'Startup and resource probe only; not a completed research replay or full equivalence proof.',
                   'samples': samples, 'probes': findings,
                   'startup_pass': all(x['running_within_60s'] for x in findings),
                   'clock_rate_pass': all(95 <= x['sim_seconds_per_wall_second'] <= 105 for x in findings),
                   'steady_clock_rate_pass': all(95 <= x['steady_sim_seconds_per_wall_second'] <= 105 for x in findings)}
        save_json(output / 'results.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'samples'}, indent=2), flush=True)
    finally:
        for name, _ in probes:
            subprocess.run(docker + ['stop', '-t', '1', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
