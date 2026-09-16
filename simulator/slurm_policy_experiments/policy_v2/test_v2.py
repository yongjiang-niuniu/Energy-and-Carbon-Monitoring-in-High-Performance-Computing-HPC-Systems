import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feedback import Controller, Signal
from selection import choose


def config():
    return {"carbon_points": [[0,200],[300,50],[3000,200]], "carbon_end_s": 10000,
            "tick_s": 60, "max_delay_s": 600, "runtime_delay_ratio": 1,
            "minimum_saving_pct": 5, "user_budget_s": 300,
            "jobs": {"sim_000001": {"pool":"cpu","user":"u1","fraction":.1,
                     "predicted_runtime_s":300,"prediction_reliable":True,"flexible":True}}}


class SelectionTests(unittest.TestCase):
    def test_feasible_runner_up(self):
        best, audit = choose([{"utility":10,"gain":0,"release":10}, {"utility":9,"gain":5,"release":20}],
                             [("gain",lambda x:x["gain"]>=5)])
        self.assertEqual(best["release"],20)
        self.assertEqual(audit[0]["rejection_reason"],"gain")

    def test_tie_earlier_and_nonfinite(self):
        best, audit = choose([{"utility":1,"release":20}, {"utility":1,"release":10},
                              {"utility":math.nan,"release":5}], [])
        self.assertEqual(best["release"],10)
        self.assertIn("non_finite",audit[-1]["rejection_reason"])

    def test_every_gate_audited(self):
        best, rows = choose([{"utility":1,"release":1}],[("a",lambda x:False),("b",lambda x:False)])
        self.assertIsNone(best)
        self.assertEqual(rows[0]["rejection_reason"],"a;b")


class FeedbackTests(unittest.TestCase):
    def test_carbon_integral_and_coverage(self):
        s=Signal([[0,200],[100,100]],300)
        self.assertEqual(s.mean(50,100),150)
        self.assertIsNone(s.mean(250,100))

    def test_hold_then_release(self):
        c=Controller(config())
        c.step({"now_s":0,"arrivals":[["sim_000001",0]]})
        target=c.held["sim_000001"]["release_s"]
        result=c.step({"now_s":target,"arrivals":[]})
        self.assertEqual(result["release"],["sim_000001"])
        self.assertEqual(c.spent["u1"],target)
        self.assertEqual(c.reserved["u1"],0)

    def test_budget_rank_not_input_order(self):
        cfg=config()
        cfg["user_budget_s"]=60
        cfg["jobs"]["sim_000002"]=dict(cfg["jobs"]["sim_000001"],fraction=.5)
        c=Controller(cfg)
        c.step({"now_s":0,"arrivals":[["sim_000001",0],["sim_000002",0]]})
        self.assertIn("sim_000002",c.held)
        self.assertIn("sim_000001",c.released)
        self.assertLessEqual(c.reserved["u1"],300)

    def test_feedback_cancels_and_spent_wait_not_refunded(self):
        cfg=config()
        cfg["jobs"]["sim_000002"]=dict(cfg["jobs"]["sim_000001"],user="u2",fraction=1,
                                       predicted_runtime_s=3600,flexible=False)
        c=Controller(cfg)
        c.step({"now_s":0,"arrivals":[["sim_000001",0],["sim_000002",0]]})
        c.step({"now_s":30,"states":[{"id":"sim_000002","state":"running","start_s":0}]})
        self.assertNotIn("sim_000001",c.held)
        self.assertEqual(c.spent["u1"],30)
        self.assertEqual(c.reserved["u1"],0)
        self.assertEqual(c.records[-1]["reason"],"feedback_cancel_benefit_lost")

    def test_future_metadata_does_not_change_early_decision(self):
        cfg=config(); altered=copy.deepcopy(cfg)
        altered["jobs"]["sim_999999"]=dict(cfg["jobs"]["sim_000001"],predicted_runtime_s=999999)
        a,b=Controller(cfg),Controller(altered)
        req={"now_s":0,"arrivals":[["sim_000001",0]]}
        self.assertEqual(a.step(req),b.step(req))
        self.assertEqual(a.records,b.records)

    def test_future_snapshot_rejected(self):
        c=Controller(config())
        with self.assertRaises(ValueError):
            c.step({"now_s":0,"states":[{"id":"sim_000001","state":"running","start_s":0}]})

    def test_no_duplicate_submission(self):
        c=Controller(dict(config(),baseline=True))
        c.step({"now_s":0,"arrivals":[["sim_000001",0]]})
        with self.assertRaises(ValueError):
            c.step({"now_s":1,"arrivals":[["sim_000001",0]]})

    def test_grid_anchored_to_arrival_not_ipc_time(self):
        cfg=config();cfg["user_budget_s"]=60
        c=Controller(cfg)
        c.step({"now_s":1.8,"arrivals":[["sim_000001",0]]})
        self.assertEqual(c.held["sim_000001"]["release_s"],60)
        self.assertEqual(c.reserved["u1"],60)

    def test_budget_can_choose_runner_up(self):
        cfg=config();cfg["user_budget_s"]=240;cfg["runtime_delay_ratio"]=4
        cfg["jobs"]["sim_000001"]["predicted_runtime_s"]=120
        c=Controller(cfg)
        c.step({"now_s":0,"arrivals":[["sim_000001",0]]})
        self.assertEqual(c.held["sim_000001"]["release_s"],240)

    def test_no_carbon_coverage_abstains(self):
        c=Controller(config())
        out=c.step({"now_s":9990,"arrivals":[["sim_000001",9990]]})
        self.assertEqual(out["release"],["sim_000001"])

    def test_protected_and_unreliable(self):
        for flag in ("flexible","prediction_reliable"):
            cfg=config();cfg["jobs"]["sim_000001"][flag]=False
            c=Controller(cfg)
            self.assertEqual(c.step({"now_s":0,"arrivals":[["sim_000001",0]]})["release"],["sim_000001"])


if __name__ == "__main__":
    unittest.main()
