import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analyze_multiday_outcomes import (
    MODELS, aggregate_daily, align_jobs, analyze, decompose_carbon,
    paired_jobs, runtime_intensity, wait_summary,
)


def intervals(energy=(1.0, 1.0), intensity=(100.0, 200.0)):
    starts = pd.date_range("2025-11-13", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame({"from_utc": starts, "to_utc": starts + pd.Timedelta(hours=1),
                          "duration_h": [1.0, 1.0], "intensity_gco2_per_kwh": intensity})
    for model in MODELS:
        frame[f"dynamic_energy_kwh_{model}"] = energy
        frame[f"dynamic_carbon_kg_{model}"] = np.array(energy) * np.array(intensity) / 1000
        frame[f"whole_carbon_kg_{model}"] = (np.array(energy) + 140) * np.array(intensity) / 1000
    return frame


def jobs():
    return pd.DataFrame({
        "sim_job_id": ["a", "b", "c"], "terminal_status": ["completed"] * 3,
        "runtime_s": [300] * 3, "nodes": [1] * 3, "cpus": [1] * 3,
        "scheduled_gpus": [0] * 3, "memory_per_node_mib": [100] * 3,
        "eligible_dt_s": [0, 0, 0], "dependency_delay_s": [0] * 3,
        "is_evaluation": [True, True, False], "partition": ["sheffield"] * 3,
        "user_id": ["u1", "u2", "u3"], "carry_in_type": ["", "", "running"],
        "sim_total_user_wait_s": [0.0, 100.0, 0.0], "sim_policy_delay_s": [0.0] * 3,
        "event_epoch_utc": ["2025-11-13T00:00:00Z"] * 3,
        "release_dt_s": [0.0] * 3, "sim_scheduler_wait_s": [0.0, 100.0, 0.0],
        "sim_start_offset_s": [0.0, 100.0, 0.0],
    })


class OutcomeTests(unittest.TestCase):
    def test_timing_only_increase_and_idle_cancel(self):
        row, _ = decompose_carbon(intervals((1, 0)), intervals((0, 1)), MODELS[0])
        self.assertAlmostEqual(row["energy_amount_term_kg"], 0)
        self.assertAlmostEqual(row["carbon_timing_term_kg"], .1)
        self.assertAlmostEqual(row["carbon_increase_kg"], .1)
        self.assertAlmostEqual(row["fixed_idle_difference_kg"], 0)

    def test_energy_only_at_constant_intensity(self):
        row, _ = decompose_carbon(intervals((1, 1), (100, 100)), intervals((2, 2), (100, 100)), MODELS[0])
        self.assertAlmostEqual(row["energy_amount_term_kg"], .2)
        self.assertAlmostEqual(row["carbon_timing_term_kg"], 0)

    def test_grid_and_idle_changes_rejected(self):
        base, policy = intervals(), intervals()
        policy.loc[1, "from_utc"] += pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "boundaries"):
            decompose_carbon(base, policy, MODELS[0])
        policy = intervals()
        policy.loc[0, "whole_carbon_kg_capacity_weighted"] += 1
        with self.assertRaisesRegex(ValueError, "idle"):
            decompose_carbon(base, policy, MODELS[0])

    def test_align_by_id_not_row_order_and_reject_bad_inputs(self):
        base = jobs()
        a, b = align_jobs(base, base.iloc[::-1])
        self.assertEqual(list(a.index), list(b.index))
        for bad in (base.iloc[:2], pd.concat([base, base.iloc[:1]])):
            with self.assertRaises(ValueError):
                align_jobs(base, bad)
        bad = base.copy()
        bad.loc[0, "runtime_s"] += 1
        with self.assertRaisesRegex(ValueError, "runtime_s"):
            align_jobs(base, bad)
        bad = base.copy()
        bad.loc[0, "terminal_status"] = "running"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            align_jobs(base, bad)

    def test_wait_deltas_not_difference_of_p95(self):
        base, policy = jobs(), jobs()
        policy.loc[:1, "sim_total_user_wait_s"] = [100, 0]
        frame = paired_jobs(base, policy, intervals())
        summary = wait_summary(frame.loc[frame.is_evaluation])
        self.assertAlmostEqual(summary["mean_actual_added_wait_s"], 0)
        self.assertAlmostEqual(summary["mean_positive_added_wait_s"], 50)
        self.assertAlmostEqual(summary["paired_added_wait_p95_s"], 90)
        self.assertAlmostEqual(policy.iloc[:2].sim_total_user_wait_s.quantile(.95) -
                               base.iloc[:2].sim_total_user_wait_s.quantile(.95), 0)

    def test_full_runtime_intensity_and_horizon_rejection(self):
        frame = jobs().iloc[:1].copy()
        frame.loc[0, "runtime_s"] = 3600
        frame.loc[0, "release_dt_s"] = 1800
        self.assertAlmostEqual(runtime_intensity(frame, intervals())[0], 150)
        frame.loc[0, "release_dt_s"] = 7200
        with self.assertRaisesRegex(ValueError, "outside"):
            runtime_intensity(frame, intervals())

    def test_equal_day_and_pooled_job_mean_differ(self):
        frame = pd.DataFrame({
            "date": ["a", "b"], "evaluation_jobs": [10, 100], "saving_kg": [1, 1],
            "saving_g_per_eval_job": [100, 10], "mean_actual_added_wait_s": [60, 120],
            "mean_positive_added_wait_s": [60, 120], "mean_intentional_delay_s": [60, 120],
            "reduction_pct": [10, 1], "baseline_carbon_kg": [10, 100],
        })
        result = aggregate_daily(frame, 5)
        self.assertFalse(result["complete"])
        self.assertAlmostEqual(result["mean_daily_saving_g_per_eval_job"], 55)
        self.assertAlmostEqual(result["pooled_saving_g_per_eval_job"], 2000 / 110)
        self.assertAlmostEqual(result["mean_daily_actual_added_wait_s"], 90)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_daily(pd.concat([frame, frame]), 5)

    def test_complete_day_pipeline_and_no_state_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            date = "2025-11-13"
            names = ["baseline", "baseline_repeat_1", "baseline_repeat_2", "adaptive_balanced", "adaptive_balanced_repeat"]
            manifest = {"dates": [date], "policies": [{"name": "adaptive_balanced", "information": "test"}]}
            state = {"status": "complete", "dates": {date: {"status": "complete"}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "state.json").write_text(json.dumps(state))
            comparison = root / "days" / date / "comparison"
            comparison.mkdir(parents=True)
            (comparison / "experiment_logic_audit.json").write_text(json.dumps({"policies": [{"pass": True}]}))
            rows = []
            for name in names:
                folder = comparison.parent / name
                folder.mkdir()
                jobs().to_csv(folder / "job_results.csv", index=False)
                table = intervals()
                table.to_csv(comparison / f"{name}_carbon_intervals.csv", index=False)
                rows.append({"scenario": name, **{f"{scope}_carbon_kg_{model}": table[f"{scope}_carbon_kg_{model}"].sum()
                                                  for scope in ("dynamic", "whole") for model in MODELS}})
            pd.DataFrame(rows).to_csv(comparison / "scenario_comparison.csv", index=False)
            original = (root / "state.json").read_bytes()
            analyze(root, root / "out")
            self.assertEqual(original, (root / "state.json").read_bytes())
            result = pd.read_csv(root / "out/daily_normalized_outcomes.csv")
            self.assertEqual(len(result), 6)
            self.assertTrue(result.saving_g_per_eval_job.eq(0).all())
            self.assertEqual(len(pd.read_csv(root / "out/carbon_increase_decomposition.csv")), 18)
            self.assertTrue(json.loads((root / "out/analysis_status.json").read_text())["all_dates_analyzed"])


if __name__ == "__main__":
    unittest.main()
