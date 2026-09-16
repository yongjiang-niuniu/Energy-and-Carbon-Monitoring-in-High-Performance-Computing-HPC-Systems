#!/usr/bin/env python3
"""Run isolated replays concurrently, with a single writer for campaign state."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import time
import traceback

import pandas as pd

from multiday_common import CAMPAIGN, REPO, ROOT, assert_profile_match, save_json, sha256
from run_multiday_campaign import CampaignRunner, now
from process_status import record_process_status


def task_graph(manifest):
    tasks = []
    for index, date in enumerate(manifest['dates']):
        base, first, last = [f'{date}/{name}' for name in ['baseline', 'baseline_repeat_1', 'baseline_repeat_2']]
        tasks += [dict(key=base, dependencies=[], source=None),
                  dict(key=first, dependencies=[base], source=base)]
        specs = deepcopy(manifest['policies'])
        random.Random(manifest['seed'] + index).shuffle(specs)
        policy_keys = []
        for spec in specs:
            key = f'{date}/{spec["name"]}'
            tasks.append(dict(key=key, dependencies=[base, first], source=None, spec=spec))
            policy_keys.append(key)
        tasks += [dict(key=last, dependencies=policy_keys, source=base),
                  dict(key=f'{date}/adaptive_balanced_repeat', dependencies=[last, f'{date}/adaptive_balanced'],
                       source=f'{date}/adaptive_balanced')]
    return tasks


def ready(task, records):
    return all(records.get(key, {}).get('status') == 'complete' for key in task['dependencies'])


def container_name(key):
    date, name = key.split('/')
    return 'stanage-multiday-' + date.replace('-', '') + '-' + name.replace('_', '-')


def docker_prefix(context):
    return ['docker', '--context', context]


def choose_backend(task, record, assignments, backends):
    preferred = record.get('docker_context') if record.get('status') == 'running' else task.get('context')
    if record.get('status') == 'running' and not preferred:
        preferred = backends[0]['context']
    for backend in backends:
        context = backend['context']
        if preferred and preferred != context:
            continue
        if list(assignments.values()).count(context) < backend['slots']:
            return context
    return None


def inspect_container(name, context):
    result = subprocess.run(docker_prefix(context) + ['inspect', name], capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


class ReplayWorker(CampaignRunner):
    def __init__(self, task, record, events, manifest, context='colima'):
        self.task = task
        self.key = task['key']
        self.events = events
        self.manifest = manifest
        self.context = context
        self.state = {'runs': {self.key: deepcopy(record)} if record else {}, 'dates': {}}

    def docker_command(self):
        return docker_prefix(self.context)

    def verify_container_inputs(self, directory):
        names = ['workload_profile.csv', 'sim.events', 'users.sim', 'slurm.conf', 'gres.conf', 'sim.conf']
        command = self.docker_command() + ['run', '--rm', '--entrypoint', 'sha256sum',
                                          '-v', f'{directory}:/opt/slurm-sim/etc:ro', self.manifest['image_id']]
        command += ['/opt/slurm-sim/etc/' + name for name in names]
        rows = subprocess.check_output(command, text=True, timeout=30).splitlines()
        mounted = {Path(row.split(maxsplit=1)[1]).name: row.split(maxsplit=1)[0] for row in rows}
        if mounted != {name: sha256(directory / name) for name in names}:
            raise RuntimeError(f'Container mount differs from host inputs: {self.context}/{directory.name}')

    def update(self, **values):
        self.state.update(values, updated_utc=now())
        if self.key in self.state['runs']:
            self.state['runs'][self.key].update(docker_context=self.context,
                                              diagnostic=bool(self.task.get('diagnostic')))
        self.events.put({'key': self.key, 'record': deepcopy(self.state['runs'].get(self.key)),
                         'step': self.state.get('current', self.key),
                         'step_elapsed_seconds': self.state.get('step_elapsed_seconds', 0),
                         'updated_utc': self.state['updated_utc']})

    def adopt(self, record):
        directory = CAMPAIGN / 'days' / self.key
        self.verify_container_inputs(directory)
        name = container_name(self.key)
        info = inspect_container(name, self.context)
        if info is not None:
            mounts = {item['Source'] for item in info['Mounts'] if item['Destination'] == '/opt/slurm-sim/etc'}
            if info['Image'] != self.manifest['image_id'] or str(directory) not in mounts:
                raise RuntimeError('Refusing to adopt a container with different inputs/image')
        started = datetime.fromisoformat(record['started_utc'])
        self.state['runs'][self.key] = {**record, 'execution_mode': 'adopted_live_replay'}
        while info is not None and info['State']['Running']:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            self.update(current=self.key, step_elapsed_seconds=elapsed)
            if elapsed > self.manifest['timeout_seconds'] + 180:
                subprocess.run(self.docker_command() + ['stop', '-t', '1', name], check=False)
                raise TimeoutError('Adopted replay exceeded the original timeout')
            time.sleep(10)
            info = inspect_container(name, self.context)
        log = directory.parent / 'logs' / (directory.name + '_parallel_adopt.log')
        docker_exit = info['State'].get('ExitCode') if info is not None else None
        process_status = record_process_status(directory, docker_exit)
        self.command([sys.executable, ROOT / 'analyze_simulation.py', directory], log, 300)
        validation = json.loads((directory / 'simulation_validation.json').read_text())
        if not validation.get('completion_pass', validation.get('strict_pass')):
            raise RuntimeError('Adopted replay did not complete all jobs')
        self.check_carry_in(directory)
        names = ['workload_profile.csv', 'sim.events', 'users.sim', 'job_results.csv', 'simulation_validation.json', 'process_status.json']
        self.state['runs'][self.key] = {
            'status': 'complete', 'attempts': record['attempts'], 'finished_utc': now(),
            'wall_seconds': (datetime.now(timezone.utc) - started).total_seconds(),
            'jobs': validation['expected_jobs'], 'execution_mode': 'adopted_live_replay',
            'status_scope': 'job_completion_only', 'process_status': process_status,
            'timing_pass': validation.get('timing_pass'),
            'artifact_hashes': {n: sha256(directory / n) for n in names},
        }
        self.update()

    def execute(self):
        self.check_frozen()
        date, name = self.key.split('/')
        directory = CAMPAIGN / 'days' / self.key
        record = self.state['runs'].get(self.key, {})
        if record.get('status') == 'running':
            self.adopt(record)
        else:
            if self.task.get('source'):
                self.repeat_inputs(CAMPAIGN / 'days' / self.task['source'], directory)
            elif self.task.get('spec'):
                self.generate(date, self.task['spec'])
            assert_profile_match(pd.read_csv(directory.parent / 'baseline/workload_profile.csv'),
                                 pd.read_csv(directory / 'workload_profile.csv'))
            self.replay(date, name)
            self.state['runs'][self.key]['execution_mode'] = 'parallel'
            self.update()
        return deepcopy(self.state['runs'][self.key])


class ParallelRunner(CampaignRunner):
    def __init__(self):
        super().__init__()
        self.execution = json.loads((CAMPAIGN / 'parallel_execution.json').read_text())
        self.events = queue.Queue()
        self.active = {}
        self.assignments = {}
        self.steps = {}
        self.tasks = task_graph(self.manifest)
        self.formal_keys = {task['key'] for task in self.tasks}
        self.backends = self.execution.get('backends', [{'context': 'colima', 'slots': self.execution['parallelism']}])
        if sum(backend['slots'] for backend in self.backends) != self.execution['parallelism']:
            raise ValueError('Backend slots must match total parallelism')
        secondary = self.execution.get('secondary_bridge_control')
        if secondary:
            date = secondary.split('/')[0]
            self.tasks.append(dict(key=secondary, dependencies=[f'{date}/baseline', f'{date}/baseline_repeat_1'],
                                   source=f'{date}/baseline', diagnostic=True, context=self.execution['secondary_context']))
        self.state['diagnostic_run_keys'] = [task['key'] for task in self.tasks if task.get('diagnostic')]
        bridge = self.execution['bridge_control']
        for task in self.tasks:
            if task['key'] == bridge:
                date = bridge.split('/')[0]
                task['dependencies'] = [f'{date}/baseline', f'{date}/baseline_repeat_1']
        # Adopt live work first, then measure a baseline in the parallel environment.
        self.tasks.sort(key=lambda t: (0 if self.state['runs'].get(t['key'], {}).get('status') == 'running'
                                      else 1 if t['key'] in {bridge, secondary} else 2))
        self.state.update(execution_mode='parallel', parallelism=self.execution['parallelism'],
                          execution_backends=self.backends,
                          parallel_execution_record=str(CAMPAIGN / 'parallel_execution.json'))

    def docker_command(self):
        return docker_prefix(self.backends[0]['context'])

    def drain(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            key = event['key']
            if event['record']:
                self.state['runs'][key] = event['record']
            self.steps[key] = {k: v for k, v in event.items() if k != 'record'}

    def update(self, **values):
        self.drain()
        self.state.update(values, updated_utc=now())
        self.state['active_runs'] = [{**self.steps.get(key, {'key': key, 'step': '准备输入'}),
                                      'docker_context': self.assignments[key]} for key in self.active]
        complete = sum(self.state['runs'].get(key, {}).get('status') == 'complete' for key in self.formal_keys)
        diagnostics = self.state['diagnostic_run_keys']
        diagnostic_complete = sum(self.state['runs'].get(key, {}).get('status') == 'complete' for key in diagnostics)
        durations = [item['wall_seconds'] for item in self.state['runs'].values() if item.get('status') == 'complete']
        remaining = self.manifest['planned_simulator_runs'] - complete + len(diagnostics) - diagnostic_complete
        ideal = remaining * sum(durations) / len(durations) / self.execution['parallelism'] / 3600 if durations else None
        self.state['estimated_remaining_parallel_hours'] = ideal
        save_json(CAMPAIGN / 'state.json', self.state)
        lines = ['# 多日实验进度', '', f"状态：{self.state['status']}；更新时间（UTC）：{self.state['updated_utc']}", '',
                 f"完整回放：{complete}/{self.manifest['planned_simulator_runs']}；最多{self.execution['parallelism']}组并行。", '']
        if diagnostics:
            lines += [f'额外环境核对：{diagnostic_complete}/{len(diagnostics)}，不计入55次策略对照。', '']
        for key in self.active:
            step = self.steps.get(key, {})
            lines += [f"- {key}（{self.assignments[key]}）：{step.get('step', '准备输入')}，当前步骤{step.get('step_elapsed_seconds', 0)/60:.1f}分钟"]
        if ideal is not None:
            lines += ['', f'按近期单次耗时和并行槽位粗估剩余：{ideal:.1f}小时；依赖、分析、资源竞争会使实际耗时增加。']
        if self.state.get('error'):
            lines += ['', '需处理：' + self.state['error']]
        lines += ['', '保持电脑开机、接通电源并保持Colima运行。GPU不参与这些回放。', '',
                  '当前在运行的任务已接管，不会因切换并行而重新开始。并行条件下的基线控制用于检查执行方式造成的波动。']
        (CAMPAIGN / 'STATUS.md').write_text('\n'.join(lines) + '\n')

    def bridge_check(self, key=None, output=None):
        key = key or self.execution['bridge_control']
        output = output or CAMPAIGN / 'parallel_checks'
        path = output / 'baseline_bridge.json'
        if path.exists():
            return json.loads(path.read_text())['passed']
        if self.state['runs'].get(key, {}).get('status') != 'complete':
            return None
        date, name = key.split('/')
        original = CAMPAIGN / self.execution['serial_control_archive']
        directories = [CAMPAIGN / 'days' / date / n for n in ['baseline', 'baseline_repeat_1', name]] + [original]
        command = [sys.executable, ROOT / 'calculate_carbon.py']
        for directory in directories:
            command += ['--scenario', directory]
        command += ['--carbon', CAMPAIGN / 'days' / date / 'carbon.csv',
                    '--horizon-start-utc', date + 'T00:00:00Z', '--output', output / 'carbon']
        self.command(command, output / 'bridge.log', 300)
        table = pd.read_csv(output / 'carbon/scenario_comparison.csv').set_index('scenario')
        serial = table.loc[['baseline', 'baseline_repeat_1', original.name]]
        parallel = table.loc[name]
        checks = []
        for column, floor in [('dynamic_carbon_kg_capacity_weighted', None),
                              ('total_user_wait_p95_s', 60), ('scheduler_wait_p95_s', 60)]:
            values = serial[column]
            if floor is None:
                delta = abs(100 * (parallel[column] - values.median()) / values.median())
                tolerance = max(.05, 3 * 100 * (values.max() - values.min()) / values.median())
                unit = 'percent'
            else:
                delta = abs(parallel[column] - values.median())
                tolerance = max(floor, 3 * (values.max() - values.min()))
                unit = 'seconds'
            checks.append({'metric': column, 'absolute_difference': delta, 'tolerance': tolerance,
                           'unit': unit, 'passed': bool(delta <= tolerance)})
        result = {'created_utc': now(), 'checks': checks, 'passed': all(c['passed'] for c in checks),
                  'scope': 'Execution-mode guard, not a new policy-effect acceptance rule.'}
        save_json(path, result)
        return result['passed']

    def run(self):
        self.check_frozen()
        for date in self.manifest['dates']:
            directory = CAMPAIGN / 'days' / date
            prepared = json.loads((directory / 'prepared.json').read_text())
            if sha256(directory / 'baseline/workload_profile.csv') != prepared['profile_sha256'] or sha256(directory / 'carbon.csv') != prepared['carbon_sha256']:
                raise RuntimeError('Prepared input changed')
        for key, record in list(self.state['runs'].items()):
            if record['status'] == 'complete':
                for name, digest in record['artifact_hashes'].items():
                    if sha256(CAMPAIGN / 'days' / key / name) != digest:
                        raise RuntimeError(f'Completed artifact changed: {key}/{name}')
                self.check_carry_in(CAMPAIGN / 'days' / key)
        failed = False
        with ThreadPoolExecutor(max_workers=self.execution['parallelism']) as pool:
            while True:
                self.drain()
                for key, future in list(self.active.items()):
                    if not future.done():
                        continue
                    self.drain()
                    try:
                        self.state['runs'][key] = future.result()
                    except Exception as exc:
                        self.state['runs'][key] = {**self.state['runs'].get(key, {}), 'status': 'failed', 'error': str(exc)}
                        self.state['error'] = f'{key}: {exc}'
                        failed = True
                    del self.active[key]
                    self.assignments.pop(key, None)
                if not failed and self.bridge_check() is False:
                    failed = True
                    self.state['error'] = 'Parallel baseline guard failed; inspect parallel_checks/baseline_bridge.json before accepting this execution mode.'
                secondary = self.execution.get('secondary_bridge_control')
                if not failed and secondary and self.bridge_check(secondary, CAMPAIGN / 'parallel_checks/secondary') is False:
                    failed = True
                    self.state['error'] = 'Secondary-backend baseline guard failed; inspect parallel_checks/secondary/baseline_bridge.json.'
                for date in self.manifest['dates']:
                    keys = [t['key'] for t in self.tasks if t['key'].startswith(date + '/') and not t.get('diagnostic')]
                    if self.state['dates'].get(date, {}).get('status') == 'complete':
                        continue
                    if all(self.state['runs'].get(k, {}).get('status') == 'complete' for k in keys):
                        names = ['baseline'] + [k.split('/')[1] for k in keys if not k.endswith('/baseline')]
                        self.analyse(date, names)
                        self.command([sys.executable, ROOT / 'summarize_multiday_campaign.py'], CAMPAIGN / 'logs/summary.log')
                if not failed:
                    for task in self.tasks:
                        key = task['key']
                        if len(self.active) >= self.execution['parallelism']:
                            break
                        if key in self.active or self.state['runs'].get(key, {}).get('status') == 'complete':
                            continue
                        if not ready(task, self.state['runs']):
                            continue
                        context = choose_backend(task, self.state['runs'].get(key, {}), self.assignments, self.backends)
                        if context is None:
                            continue
                        worker = ReplayWorker(task, self.state['runs'].get(key), self.events, self.manifest, context)
                        self.assignments[key] = context
                        self.active[key] = pool.submit(worker.execute)
                self.update(status='draining_after_error' if failed else 'running',
                            current='；'.join(self.active) if self.active else '汇总')
                if not self.active:
                    break
                time.sleep(5)
        complete = all(self.state['dates'].get(date, {}).get('status') == 'complete' for date in self.manifest['dates'])
        self.update(status='complete' if complete and not failed else 'blocked', current='后台批次已结束')
        self.command([sys.executable, ROOT / 'summarize_multiday_campaign.py'], CAMPAIGN / 'logs/summary.log')

    def stop_secondary_if_finished(self):
        if self.state['status'] != 'complete' or not self.execution.get('stop_secondary_after_success'):
            return
        context = self.execution['secondary_context']
        try:
            live = subprocess.check_output(docker_prefix(context) + ['ps', '-q'], text=True).strip()
            if not live:
                result = subprocess.run(['colima', 'stop', self.execution['secondary_profile']],
                                        capture_output=True, text=True, timeout=120)
                self.update(secondary_environment_stopped=result.returncode == 0)
        except Exception as exc:
            self.update(secondary_cleanup_warning=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        with (CAMPAIGN / 'parallel_launcher.log').open('a') as output:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve())], cwd=REPO,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'mode': 'parallel'}))
        return
    with (CAMPAIGN / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        runner = ParallelRunner()
        try:
            runner.run()
            runner.stop_secondary_if_finished()
        except Exception as exc:
            runner.update(status='blocked', error=str(exc))
            traceback.print_exc()
            raise
        finally:
            wake.terminate()
            wake.wait()


if __name__ == '__main__':
    main()
