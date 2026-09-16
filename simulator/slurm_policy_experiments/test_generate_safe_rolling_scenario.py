from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "safe", ROOT / "generate_safe_rolling_scenario.py"
)
SAFE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SAFE)


def history_model() -> dict[str, object]:
    runtime = pd.DataFrame(
        [
            {
                "level": "partition",
                "partition": "sheffield",
                "count": 1000,
                "runtime_s_q50": 3000,
                "runtime_s_q75": 3600,
                "runtime_s_q90": 3600,
            },
            {
                "level": "global",
                "count": 1000,
                "runtime_s_q50": 3000,
                "runtime_s_q75": 3600,
                "runtime_s_q90": 3600,
            },
        ]
    )
    wait = pd.DataFrame(
        [
            {
                "level": "partition",
                "partition": "sheffield",
                "count": 1000,
                "wait_s_q50": 0,
                "wait_s_q75": 0,
                "wait_s_q90": 0,
            },
            {
                "level": "global",
                "count": 1000,
                "wait_s_q50": 0,
                "wait_s_q75": 0,
                "wait_s_q90": 0,
            },
        ]
    )
    load_rows = []
    for weekday in range(7):
        for slot in range(48):
            row = {"weekday_utc": weekday, "half_hour_slot_utc": slot}
            for metric in ["load_cpu", "load_a100", "load_h100", "load_h100_nvl", "load_node"]:
                for label in ["q50", "q75", "q90"]:
                    row[f"{metric}_{label}"] = 0.0
            load_rows.append(row)
    return {
        "metadata": {
            "minimum_group_count": 50,
            "maximum_training_runtime_hours": 96,
            "training_months": ["january"],
            "holdout_month": "november",
            "temporal_split_pass": True,
        },
        "runtime": runtime,
        "wait": wait,
        "load": pd.DataFrame(load_rows),
    }


def test_safe_policy_delays_evaluation_but_not_carry_in() -> None:
    epoch = pd.Timestamp("2025-11-12T00:00:00Z")
    carbon = pd.DataFrame(
        {
            "from_utc": pd.date_range(epoch, periods=12, freq="30min"),
            "to_utc": pd.date_range(epoch + pd.Timedelta(minutes=30), periods=12, freq="30min"),
            "intensity_gco2_per_kwh": [300, 300, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100],
        }
    )
    profile = pd.DataFrame(
        [
            {
                "sim_job_id": "sim_1",
                "slurm_job_id_expected": 1,
                "source_submit_utc": epoch.isoformat(),
                "partition": "sheffield",
                "cpus": 64,
                "nodes": 1,
                "scheduled_gpus": 0,
                "runtime_s": 3600,
                "eligible_dt_s": 0,
                "release_dt_s": 0,
                "flexibility_score": 0.1,
                "is_warmup": False,
                "source_timelimit_min": 120,
                "user_id": "user_001",
            },
            {
                "sim_job_id": "sim_2",
                "slurm_job_id_expected": 2,
                "source_submit_utc": epoch.isoformat(),
                "partition": "sheffield",
                "cpus": 64,
                "nodes": 1,
                "scheduled_gpus": 0,
                "runtime_s": 3600,
                "eligible_dt_s": 0,
                "release_dt_s": 0,
                "flexibility_score": 0.1,
                "is_warmup": True,
                "source_timelimit_min": 120,
                "user_id": "user_002",
            },
        ]
    )

    result, _, candidates = SAFE.apply_safe_rolling_balanced(
        profile,
        history_model(),
        carbon,
        epoch,
        0.3,
        4,
        2,
        2,
        0.1,
        0.25,
        0.75,
        0.05,
        0.1,
        1.75,
        12,
        1.5,
        4,
        2,
    )

    assert result.loc[0, "release_dt_s"] == 3600
    assert result.loc[0, "safe_decision"] == "delay"
    assert result.loc[1, "release_dt_s"] == 0
    assert "carry_in" in result.loc[1, "safe_rejection_reason"]
    assert candidates["accepted"].sum() == 1
