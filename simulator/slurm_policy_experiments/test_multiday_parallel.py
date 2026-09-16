import unittest
import queue
import tempfile
from pathlib import Path
from unittest.mock import patch

from run_multiday_parallel import ready, task_graph, container_name, choose_backend, ReplayWorker
from multiday_common import sha256


class ParallelTests(unittest.TestCase):
    def backends(self):
        return [{'context': 'colima', 'slots': 3}, {'context': 'colima-stanage-extra', 'slots': 2}]

    def test_backend_slot_limits(self):
        assignments = {str(i): 'colima' for i in range(3)}
        self.assertEqual(choose_backend({}, {}, assignments, self.backends()), 'colima-stanage-extra')
        assignments.update(extra1='colima-stanage-extra', extra2='colima-stanage-extra')
        self.assertIsNone(choose_backend({}, {}, assignments, self.backends()))

    def test_adoption_does_not_move_a_live_container(self):
        assignments = {str(i): 'colima' for i in range(3)}
        self.assertIsNone(choose_backend({}, {'status': 'running'}, assignments, self.backends()))
        record = {'status': 'running', 'docker_context': 'colima-stanage-extra'}
        self.assertEqual(choose_backend({}, record, assignments, self.backends()), 'colima-stanage-extra')

    def test_secondary_control_stays_on_secondary_backend(self):
        task = {'context': 'colima-stanage-extra'}
        self.assertEqual(choose_backend(task, {}, {}, self.backends()), 'colima-stanage-extra')

    def test_worker_explicit_context_and_diagnostic_metadata(self):
        events = queue.Queue()
        task = {'key': '2025-11-13/probe', 'diagnostic': True}
        worker = ReplayWorker(task, {'status': 'running'}, events, {}, 'colima-stanage-extra')
        self.assertEqual(worker.docker_command(), ['docker', '--context', 'colima-stanage-extra'])
        worker.update(current=task['key'])
        event = events.get_nowait()
        self.assertEqual(event['record']['docker_context'], 'colima-stanage-extra')
        self.assertTrue(event['record']['diagnostic'])

    def test_mount_check_requires_matching_file_contents(self):
        worker = ReplayWorker({'key': '2025-11-13/test'}, None, queue.Queue(), {'image_id': 'pinned'}, 'colima-stanage-extra')
        names = ['workload_profile.csv', 'sim.events', 'users.sim', 'slurm.conf', 'gres.conf', 'sim.conf']
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in names:
                (directory / name).write_text(name)
            valid = '\n'.join(f'{sha256(directory / name)}  /opt/slurm-sim/etc/{name}' for name in names)
            with patch('run_multiday_parallel.subprocess.check_output', return_value=valid):
                worker.verify_container_inputs(directory)
            for invalid in ['', valid.replace(sha256(directory / names[0]), '0' * 64)]:
                with patch('run_multiday_parallel.subprocess.check_output', return_value=invalid):
                    with self.assertRaisesRegex(RuntimeError, 'mount differs'):
                        worker.verify_container_inputs(directory)

    def manifest(self):
        return {'dates': ['2025-11-13', '2025-11-15'], 'seed': 42,
                'policies': [{'name': 'adaptive_balanced'}, {'name': 'history_only'}]}

    def test_graph_has_unique_keys_and_only_known_dependencies(self):
        tasks = task_graph(self.manifest())
        keys = [t['key'] for t in tasks]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(set(t['dependencies']).issubset(keys) for t in tasks))

    def test_policy_waits_for_completed_baseline_and_first_control(self):
        task = next(t for t in task_graph(self.manifest()) if t['key'] == '2025-11-13/history_only')
        records = {'2025-11-13/baseline': {'status': 'complete'}}
        self.assertFalse(ready(task, records))
        records['2025-11-13/baseline_repeat_1'] = {'status': 'running'}
        self.assertFalse(ready(task, records))
        records['2025-11-13/baseline_repeat_1']['status'] = 'complete'
        self.assertTrue(ready(task, records))

    def test_dates_are_independent_and_container_names_are_unique(self):
        tasks = task_graph(self.manifest())
        for task in tasks:
            self.assertTrue(all(d.split('/')[0] == task['key'].split('/')[0] for d in task['dependencies']))
        self.assertEqual(len({container_name(t['key']) for t in tasks}), len(tasks))

    def test_last_control_depends_on_all_policies(self):
        task = next(t for t in task_graph(self.manifest()) if t['key'] == '2025-11-13/baseline_repeat_2')
        self.assertEqual(set(task['dependencies']), {'2025-11-13/adaptive_balanced', '2025-11-13/history_only'})


if __name__ == '__main__':
    unittest.main()
