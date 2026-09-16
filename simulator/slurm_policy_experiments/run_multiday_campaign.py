#!/usr/bin/env python3
"""Run a resumable, serial campaign independently of an interactive chat."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import pandas as pd

from multiday_common import CAMPAIGN, REPO, ROOT, assert_profile_match, model_for_date, save_json, sha256
from process_status import record_process_status, simulator_command


def now():
    return datetime.now(timezone.utc).isoformat()


class CarryInValidationError(RuntimeError):
    """An invalid initial state must stop the campaign, not produce comparisons."""


class CampaignRunner:
    def __init__(self):
        self.manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
        path = CAMPAIGN / 'state.json'
        self.state = json.loads(path.read_text()) if path.exists() else {'runs': {}, 'dates': {}, 'created_utc': now()}
        self.state.update({'pid': os.getpid(), 'status': 'running', 'updated_utc': now()})

    def update(self, **values):
        self.state.update(values, updated_utc=now())
        save_json(CAMPAIGN / 'state.json', self.state)
        completed = sum(x.get('status') == 'complete' for x in self.state['runs'].values())
        failed = sum(x.get('status') == 'failed' for x in self.state['runs'].values())
        durations = [x['wall_seconds'] for x in self.state['runs'].values() if x.get('status') == 'complete']
        remaining_h = (self.manifest['planned_simulator_runs'] - completed) * (sum(durations) / len(durations)) / 3600 if durations else None
        lines = [
            '# 多日实验进度', '', f"状态：{self.state['status']}", f"更新时间（UTC）：{self.state['updated_utc']}",
            f"完整回放：{completed}/{self.manifest['planned_simulator_runs']}；当前失败项：{failed}",
            f"当前任务：{self.state.get('current', '等待')}",
            f"当前步骤已用时：{self.state.get('step_elapsed_seconds', 0):.0f} 秒",
        ]
        if remaining_h is not None:
            lines += [f'按已完成回放粗估剩余：{remaining_h:.1f} 小时（各日期负载不同，会变化）']
        if self.state.get('error'):
            lines += ['', '错误：' + self.state['error']]
        lines += ['', '运行期间接通电源并保持电脑开机、Colima运行。程序会阻止空闲睡眠；合盖或关机仍可能暂停。',
                  '', '结束后查看 `summary/RESULTS_ZH.md`。失败项会保留日志，不会替换日期或隐藏不利结果。']
        (CAMPAIGN / 'STATUS.md').write_text('\n\n'.join(lines) + '\n')

    def docker_command(self):
        return ['docker']

    def verify_container_inputs(self, directory):
        """Multi-backend runners can verify host-to-guest mounts before launch."""

    def check_frozen(self):
        for relative, expected in json.loads((CAMPAIGN / 'code_hashes.json').read_text()).items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f'Frozen code/model changed: {relative}')
        image = subprocess.check_output(self.docker_command() + ['image', 'inspect', 'slurm-sim-scheduler-fix:latest', '--format', '{{.Id}}'], text=True).strip()
        if image != self.manifest['image_id']:
            raise RuntimeError('Simulator image changed since freeze')

    def command(self, command, log, timeout=None, allow_nonzero=False):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.last_command_returncode = None
        started = time.monotonic()
        environment = os.environ.copy()
        environment.update({'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(CAMPAIGN / 'matplotlib_cache')})
        with log.open('a') as output:
            output.write(f'\n{now()} COMMAND {json.dumps([str(x) for x in command])}\n')
            output.flush()
            process = subprocess.Popen([str(x) for x in command], cwd=REPO, env=environment, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                self.update(step_elapsed_seconds=elapsed)
                if timeout and elapsed > timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    self.last_command_returncode = process.returncode
                    raise TimeoutError(f'Command exceeded {timeout}s; see {log}')
                time.sleep(10)
        self.last_command_returncode = process.returncode
        if process.returncode and not allow_nonzero:
            raise RuntimeError(f'Command failed ({process.returncode}); see {log}')
        return time.monotonic() - started

    def prepare(self):
        self.update(current='准备5个完整作业窗口及碳强度数据')
        self.command([sys.executable, ROOT / 'prepare_multiday_campaign.py'], CAMPAIGN / 'logs/preparation.log')

    def generate(self, date, spec):
        directory = CAMPAIGN / 'days' / date
        target = directory / spec['name']
        command = [sys.executable, ROOT / spec['generator'], '--baseline-profile', directory / 'baseline/workload_profile.csv', '--carbon', directory / 'carbon.csv', '--output', target]
        if spec.get('strategy'):
            command += ['--strategy', spec['strategy']]
        for key, value in spec['parameters'].items():
            command += ['--' + key.replace('_', '-'), str(value)]
        if spec.get('uses_history'):
            command += ['--history-model-dir', model_for_date(date)]
        if spec.get('uses_baseline_schedule'):
            command += ['--schedule-forecast', directory / 'baseline/job_results.csv']
        self.update(current=f'{date} / 生成 {spec["name"]}')
        self.command(command, directory / 'logs' / (spec['name'] + '_generate.log'))
        assert_profile_match(pd.read_csv(directory / 'baseline/workload_profile.csv'), pd.read_csv(target / 'workload_profile.csv'))

    def repeat_inputs(self, source, target):
        target.mkdir(parents=True, exist_ok=True)
        for name in ['workload_profile.csv', 'sim.events', 'users.sim', 'workload_summary.json', 'exact_carry_in_summary.json', 'policy_summary.json']:
            if (source / name).exists():
                shutil.copy2(source / name, target / name)

    def check_carry_in(self, directory):
        from audit_exact_carryin import audit_profile, audit_results
        profile = pd.read_csv(directory / 'workload_profile.csv')
        checks = audit_profile(profile) + audit_results(profile, pd.read_csv(directory / 'job_results.csv'))
        save_json(directory / 'carry_in_validation.json', checks)
        failures = [item for item in checks if not item['passed']]
        if failures:
            raise CarryInValidationError(f'Initial-state audit failed: {directory.name}: {failures}')

    def replay(self, date, name):
        key = f'{date}/{name}'
        directory = CAMPAIGN / 'days' / date / name
        record = self.state['runs'].get(key, {})
        if record.get('status') == 'complete':
            for relative, expected in record['artifact_hashes'].items():
                if sha256(directory / relative) != expected:
                    raise RuntimeError(f'Completed artifact changed: {key}/{relative}')
            self.check_carry_in(directory)
            return
        started = time.monotonic()
        for attempt in range(record.get('attempts', 0) + 1, self.manifest['attempts_per_run'] + 1):
            process_status = None
            self.update(current=key, step_elapsed_seconds=0)
            self.state['runs'][key] = {'status': 'running', 'attempts': attempt, 'started_utc': now()}
            self.update()
            log = CAMPAIGN / 'days' / date / 'logs' / f'{name}_attempt{attempt}.log'
            container = 'stanage-multiday-' + date.replace('-', '') + '-' + name.replace('_', '-')
            try:
                existing = subprocess.run(self.docker_command() + ['inspect', container], capture_output=True)
                if existing.returncode == 0:
                    subprocess.run(self.docker_command() + ['rm', '-f', container], check=True, stdout=subprocess.DEVNULL)
                if (directory / 'slurmctld.log').exists():
                    archived = directory / 'previous_attempts' / f'{int(time.time())}'
                    archived.mkdir(parents=True)
                    for filename in ['slurmctld.log', 'sched.log', 'job_results.csv', 'simulation_validation.json', 'simulation_validation.csv', 'simulator_exit_code.txt', 'process_status.json']:
                        if (directory / filename).exists():
                            shutil.move(directory / filename, archived / filename)
                self.command([sys.executable, ROOT / 'prepare_scenario.py', directory], log)
                self.verify_container_inputs(directory)
                for filename in ['simulator_exit_code.txt', 'process_status.json']:
                    (directory / filename).unlink(missing_ok=True)
                timeout = self.manifest['timeout_seconds']
                # Container-scoped shared memory and a fixed image permit independent clean replays.
                try:
                    self.command(self.docker_command() + [
                        'run', '--rm', '--name', container, '--hostname', 'node001',
                        '--label', f'stanage.campaign={self.manifest["campaign_id"]}', '--ulimit', 'core=0',
                        '-v', f'{directory}:/opt/slurm-sim/etc', self.manifest['image_id'],
                        'bash', '-c', simulator_command(timeout),
                    ], log, timeout + 180, allow_nonzero=True)
                finally:
                    process_status = record_process_status(directory, self.last_command_returncode)
                self.command([sys.executable, ROOT / 'analyze_simulation.py', directory], log, 300)
                validation = json.loads((directory / 'simulation_validation.json').read_text())
                if not validation.get('completion_pass', validation.get('strict_pass')):
                    raise ValueError(f'Strict completion failed: {key}')
                self.check_carry_in(directory)
                names = ['workload_profile.csv', 'sim.events', 'users.sim', 'job_results.csv', 'simulation_validation.json', 'process_status.json']
                self.state['runs'][key] = {
                    'status': 'complete', 'attempts': attempt, 'finished_utc': now(),
                    'wall_seconds': time.monotonic() - started, 'jobs': validation['expected_jobs'],
                    'status_scope': 'job_completion_only', 'process_status': process_status,
                    'timing_pass': validation.get('timing_pass'),
                    'artifact_hashes': {n: sha256(directory / n) for n in names},
                }
                self.update()
                return
            except CarryInValidationError as exc:
                self.state['runs'][key] = {'status': 'invalid', 'attempts': attempt, 'error': str(exc)}
                self.update()
                raise
            except Exception as exc:
                subprocess.run(self.docker_command() + ['rm', '-f', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.state['runs'][key] = {'status': 'failed', 'attempts': attempt, 'error': str(exc),
                                          'process_status': process_status}
                self.update()
        raise RuntimeError(f'Run exhausted attempts: {key}')

    def analyse(self, date, names):
        directory = CAMPAIGN / 'days' / date
        result = directory / 'comparison'
        paths = [directory / name for name in names]
        self.update(current=f'{date} / 碳排、等待、公平性与完整性分析')
        command = [sys.executable, ROOT / 'calculate_carbon.py']
        for p in paths:
            command += ['--scenario', p]
        command += ['--carbon', directory / 'carbon.csv', '--horizon-start-utc', date + 'T00:00:00Z', '--output', result]
        self.command(command, directory / 'logs/analysis.log')
        command = [sys.executable, ROOT / 'validate_experiment.py', '--baseline', directory / 'baseline']
        for p in paths[1:]:
            command += ['--policy', p]
        command += ['--comparison', result / 'scenario_comparison.csv', '--output', result]
        self.command(command, directory / 'logs/analysis.log')
        command = [sys.executable, ROOT / 'build_extended_analysis.py']
        for p in paths:
            command += ['--scenario', p]
        command += ['--comparison', result / 'scenario_comparison.csv', '--carbon-result-dir', result, '--output', result]
        self.command(command, directory / 'logs/analysis.log')
        # Check reconstructed initial state in every run, including controls.
        from audit_exact_carryin import audit_profile, audit_results
        audits = {}
        for p in paths:
            profile = pd.read_csv(p / 'workload_profile.csv')
            audits[p.name] = audit_profile(profile) + audit_results(profile, pd.read_csv(p / 'job_results.csv'))
        save_json(result / 'carry_in_audits.json', audits)
        self.state['dates'][date] = {
            'status': 'complete' if all(x['passed'] for entries in audits.values() for x in entries) else 'carry_in_audit_failed',
            'comparison': str(result / 'scenario_comparison.csv'),
        }
        self.update()

    def run(self):
        self.check_frozen()
        self.prepare()
        for index, date in enumerate(self.manifest['dates']):
            self.check_frozen()
            if self.state['dates'].get(date, {}).get('status') == 'complete':
                continue
            directory = CAMPAIGN / 'days' / date
            prepared = json.loads((directory / 'prepared.json').read_text())
            if sha256(directory / 'baseline/workload_profile.csv') != prepared['profile_sha256'] or sha256(directory / 'carbon.csv') != prepared['carbon_sha256']:
                raise RuntimeError('Prepared input changed')
            try:
                self.replay(date, 'baseline')
                specs = self.manifest['policies'].copy()
                random.Random(self.manifest['seed'] + index).shuffle(specs)
                names = ['baseline']
                # Controls before and after policies estimate within-date replay variation.
                self.repeat_inputs(directory / 'baseline', directory / 'baseline_repeat_1')
                self.replay(date, 'baseline_repeat_1')
                names += ['baseline_repeat_1']
                errors = []
                for spec in specs:
                    name = spec['name']
                    try:
                        if self.state['runs'].get(f'{date}/{name}', {}).get('status') != 'complete':
                            self.generate(date, spec)
                        self.replay(date, name)
                        names.append(name)
                    except CarryInValidationError:
                        raise
                    except Exception as exc:
                        errors.append(str(exc))
                self.repeat_inputs(directory / 'baseline', directory / 'baseline_repeat_2')
                self.replay(date, 'baseline_repeat_2')
                names.append('baseline_repeat_2')
                if 'adaptive_balanced' in names:
                    self.repeat_inputs(directory / 'adaptive_balanced', directory / 'adaptive_balanced_repeat')
                    self.replay(date, 'adaptive_balanced_repeat')
                    names.append('adaptive_balanced_repeat')
                self.analyse(date, names)
                if errors:
                    self.state['dates'][date] = {'status': 'incomplete', 'errors': errors}
            except CarryInValidationError:
                raise
            except Exception as exc:
                self.state['dates'][date] = {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}
            self.update()
            self.command([sys.executable, ROOT / 'summarize_multiday_campaign.py'], CAMPAIGN / 'logs/summary.log')
        status = 'complete' if all(self.state['dates'].get(d, {}).get('status') == 'complete' for d in self.manifest['dates']) else 'finished_with_failures'
        self.update(status=status, current='已结束，请查看 summary/RESULTS_ZH.md')
        self.command([sys.executable, ROOT / 'summarize_multiday_campaign.py'], CAMPAIGN / 'logs/summary.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        path = CAMPAIGN / 'STATUS.md'
        print(path.read_text() if path.exists() else 'Not started')
        return
    if args.launch:
        CAMPAIGN.mkdir(parents=True, exist_ok=True)
        with (CAMPAIGN / 'launcher.log').open('a') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'status_file': str(CAMPAIGN / 'STATUS.md')}))
        return
    with (CAMPAIGN / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This campaign is already running')
        wake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        runner = CampaignRunner()
        try:
            runner.run()
        except Exception as exc:
            runner.update(status='blocked', error=str(exc), current='需要检查 launcher.log 后恢复')
            traceback.print_exc()
            raise
        finally:
            wake.terminate()
            wake.wait()


if __name__ == '__main__':
    main()
