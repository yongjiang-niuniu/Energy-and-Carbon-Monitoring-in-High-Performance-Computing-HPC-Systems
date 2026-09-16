import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from bounded import Budget


class BudgetTests(unittest.TestCase):
    def test_restart_does_not_reset_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            first=Budget(directory)
            second=Budget(directory)
            self.assertEqual(first.record,second.record)

    def test_expired_budget_does_not_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            budget=Budget(directory)
            budget.record["deadline_epoch"]=time.time()-1
            with self.assertRaises(TimeoutError):
                budget.run("must_not_run",[sys.executable,"-c","raise RuntimeError()"])
            self.assertFalse((Path(directory)/"must_not_run.log").exists())

    def test_closed_experiment_does_not_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            budget=Budget(directory)
            (Path(directory)/"finished.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                budget.run("closed",[sys.executable,"-c","pass"])

    def test_timeout_is_recorded_and_process_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            budget=Budget(directory)
            code=budget.run("timeout",[sys.executable,"-c","import time; time.sleep(10)"],.05)
            self.assertEqual(code,124)
            record=json.loads((Path(directory)/"phases.jsonl").read_text())
            self.assertTrue(record["timed_out"])
            self.assertLess(record["elapsed_seconds"],5)
