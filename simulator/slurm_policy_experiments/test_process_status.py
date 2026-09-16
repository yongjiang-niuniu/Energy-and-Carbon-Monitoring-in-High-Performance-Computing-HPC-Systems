import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from process_status import record_process_status, simulator_command
from run_multiday_campaign import CampaignRunner


class ProcessStatusTests(unittest.TestCase):
    def test_shell_keeps_zero_crash_and_timeout_status(self):
        for code in (0, 139, 137):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                command = simulator_command(5)
                command = command.replace('cd /opt/slurm-sim/etc', f'cd "{folder}"')
                command = command.replace('stdbuf -oL -eL timeout -s KILL 5 slurmctld -D -i', f'(exit {code})')
                process = subprocess.run(['bash', '-c', command], check=False)
                self.assertEqual(process.returncode, code)
                record = record_process_status(Path(folder), process.returncode)
                self.assertEqual(record['simulator_exit_code'], code)
                self.assertEqual(record['process_exit_status'], 'exited_zero' if code == 0 else 'abnormal_exit')
                self.assertEqual(json.loads((Path(folder) / 'process_status.json').read_text()), record)

    def test_absent_simulator_status_is_unknown_even_if_docker_returned_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            record = record_process_status(Path(folder), 0)
            self.assertIsNone(record['simulator_exit_code'])
            self.assertEqual(record['process_exit_status'], 'simulator_exit_unknown')

    def test_docker_failure_is_not_overridden_by_zero_simulator_status(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'simulator_exit_code.txt').write_text('0\n')
            self.assertEqual(record_process_status(Path(folder), 125)['process_exit_status'], 'abnormal_exit')

    def test_invalid_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'simulator_exit_code.txt').write_text('not a status')
            with self.assertRaises(ValueError):
                record_process_status(Path(folder))
        with self.assertRaises(ValueError):
            simulator_command(0)

    def test_campaign_records_nonzero_without_silently_replacing_it(self):
        runner = CampaignRunner.__new__(CampaignRunner)
        runner.update = lambda **kwargs: None
        with tempfile.TemporaryDirectory() as folder, patch('run_multiday_campaign.time.sleep'):
            command = [sys.executable, '-c', 'raise SystemExit(7)']
            runner.command(command, Path(folder) / 'process.log', 5, allow_nonzero=True)
            self.assertEqual(runner.last_command_returncode, 7)
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                runner.command(command, Path(folder) / 'strict.log', 5)


if __name__ == '__main__':
    unittest.main()
