import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("build_history_quantile_model.py")
SPEC = importlib.util.spec_from_file_location("build_history_quantile_model", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HistoryQuantileModelTests(unittest.TestCase):
    def test_interval_average_preserves_partial_bin_energy(self):
        output = np.zeros(2)
        MODULE.add_interval_average(
            output,
            np.array([15 * 60.0]),
            np.array([45 * 60.0]),
            np.array([1.0]),
            30 * 60.0,
        )
        np.testing.assert_allclose(output, [0.5, 0.5])

    def test_nodelist_expansion_uses_known_stanage_ranges(self):
        self.assertEqual(
            MODULE.expand_nodelist("node[001-002,205]"),
            ("node001", "node002", "node205"),
        )
        self.assertEqual(
            MODULE.expand_nodelist("gpu[01,21,31]"),
            ("gpu01", "gpu21", "gpu31"),
        )

    def test_unique_node_load_does_not_double_count_shared_node(self):
        month_start = pd.Timestamp("2025-01-01T00:00Z")
        activity = pd.DataFrame(
            [
                {
                    "start": month_start,
                    "end": month_start + pd.Timedelta(minutes=30),
                    "nodelist": "node001",
                },
                {
                    "start": month_start + pd.Timedelta(minutes=10),
                    "end": month_start + pd.Timedelta(minutes=20),
                    "nodelist": "node001",
                },
            ]
        )
        load, observed = MODULE.unique_node_interval_load(
            activity,
            month_start,
            month_start + pd.Timedelta(hours=1),
            bins=2,
            step_s=1800,
        )
        self.assertEqual(observed, 1)
        np.testing.assert_allclose(load, [1 / MODULE.TOTAL_NODES, 0])

    def test_runtime_prediction_falls_back_when_resource_group_is_sparse(self):
        model = {
            "metadata": {
                "minimum_group_count": 50,
                "maximum_training_runtime_hours": 12,
            },
            "runtime": pd.DataFrame(
                [
                    {
                        "level": "resource",
                        "count": 10,
                        "partition": "sheffield",
                        "cpu_bucket": "17-64",
                        "node_bucket": "1",
                        "gpu_bucket": "0",
                        "runtime_s_q50": 100,
                    },
                    {
                        "level": "partition",
                        "count": 1000,
                        "partition": "sheffield",
                        "runtime_s_q50": 1800,
                    },
                    {"level": "global", "count": 2000, "runtime_s_q50": 2400},
                ]
            ),
        }
        profile = pd.DataFrame(
            [{"partition": "sheffield", "cpus": 32, "nodes": 1, "scheduled_gpus": 0}]
        )
        prediction = MODULE.predict_runtime(profile, model, 0.50)
        self.assertEqual(prediction.loc[0, "predicted_runtime_s"], 1800)
        self.assertEqual(prediction.loc[0, "runtime_model_level"], "partition")

    def test_history_load_uses_utc_weekday_and_half_hour_slot(self):
        carbon = pd.DataFrame(
            {
                "from_utc": pd.to_datetime(["2025-06-10T20:00Z"]),
                "to_utc": pd.to_datetime(["2025-06-10T20:30Z"]),
                "intensity_gco2_per_kwh": [200],
            }
        )
        row = {"weekday_utc": 1, "half_hour_slot_utc": 40}
        for name in ["cpu", "a100", "h100", "h100_nvl", "node"]:
            row[f"load_{name}_q75"] = 0.25
        model = {"load": pd.DataFrame([row])}
        forecast = MODULE.history_load_forecast(carbon, model, 0.75)
        self.assertEqual(forecast.loc[0, "load_cpu"], 0.25)
        self.assertEqual(forecast.loc[0, "load_node"], 0.25)


if __name__ == "__main__":
    unittest.main()
