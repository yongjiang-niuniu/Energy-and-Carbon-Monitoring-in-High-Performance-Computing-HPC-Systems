"""Run the existing policy tests against the isolated corrected implementation."""
from pathlib import Path
import importlib.util
import sys
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import legacy_fixed
spec=importlib.util.spec_from_file_location("_v2_original_policy_tests",
    Path(__file__).resolve().parents[1]/"test_generate_policy_scenario.py")
original_tests=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original_tests)
original_tests.MODULE = legacy_fixed
PolicyScenarioTests = original_tests.PolicyScenarioTests


class FeasibleFirstIntegrationTests(unittest.TestCase):
    def fixture(self):
        epoch=pd.Timestamp("2025-11-18T00:00:00Z")
        times=[0,900,1800,3600,4500]
        carbon=pd.DataFrame({"from_utc":[epoch+pd.Timedelta(seconds=s) for s in times[:-1]],
                             "to_utc":[epoch+pd.Timedelta(seconds=s) for s in times[1:]],
                             "intensity_gco2_per_kwh":[100,96,100,94]})
        profile=pd.DataFrame([{"sim_job_id":"sim_000001","slurm_job_id_expected":1,
                              "release_dt_s":0,"eligible_dt_s":0,"runtime_s":900,
                              "partition":"sheffield","cpus":1,"scheduled_gpus":0,
                              "flexibility_score":0.0,"is_evaluation":True}])
        options=dict(flex_fraction=1,max_delay_hours=1,wait_budget_ratio=0,min_carbon_saving_pct=5,
                     wait_penalty=.04,congestion_penalty=0,high_quantile=.75,cap_fraction=.75)
        return profile,carbon,epoch,options

    def test_actual_adaptive_uses_feasible_runner_up(self):
        import importlib.util
        source=Path(__file__).resolve().parents[1]/"generate_policy_scenario.py"
        spec=importlib.util.spec_from_file_location("unchanged_legacy",source)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        profile,carbon,epoch,options=self.fixture()
        before,_=old.apply_adaptive(profile,carbon,epoch,**options)
        after,_=legacy_fixed.apply_adaptive(profile,carbon,epoch,**options)
        self.assertEqual(before.release_dt_s.iloc[0],0)
        self.assertEqual(after.release_dt_s.iloc[0],3600)
        self.assertTrue(after.attrs["v2_candidate_audit"])

    def test_baseline_remains_valid_competitor(self):
        profile,carbon,epoch,options=self.fixture()
        options["wait_penalty"]=10
        result,_=legacy_fixed.apply_adaptive(profile,carbon,epoch,**options)
        self.assertEqual(result.release_dt_s.iloc[0],0)


class ReportingTests(unittest.TestCase):
    def test_missing_explicit_reason_is_reconstructed_boolean(self):
        from reporting import decision_report
        profile=pd.DataFrame([{"sim_job_id":"sim_000001","release_dt_s":0,"eligible_dt_s":0,
                               "low_impact_rejection_reason":float("nan")}])
        jobs,candidates=decision_report(profile)
        self.assertEqual(jobs.reason.iloc[0],"rule_no_move_details_not_recorded")
        self.assertTrue(jobs.reason_is_reconstructed.iloc[0])

    def test_completion_does_not_imply_timing_or_repeatability(self):
        from reporting import validation_panels
        data=pd.DataFrame([{"sim_submit_log_utc":"x","sim_start_log_utc":"x","sim_end_log_utc":"x",
                            "terminal_status":"completed","sim_observed_runtime_s":999,"runtime_s":1}])
        panels=validation_panels(data)
        self.assertEqual(panels["completion"]["status"],"pass")
        self.assertEqual(panels["timing_fidelity"]["status"],"warning")
        self.assertEqual(panels["repeatability"]["status"],"not_assessed")
