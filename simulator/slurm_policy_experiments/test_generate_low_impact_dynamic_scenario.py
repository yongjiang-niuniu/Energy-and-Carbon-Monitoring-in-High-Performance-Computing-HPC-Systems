from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "low_impact", ROOT / "generate_low_impact_dynamic_scenario.py"
)
LOW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LOW)


def history_model() -> dict[str, object]:
    runtime = pd.DataFrame(
        [
            {
                "level": "partition",
                "partition": "sheffield",
                "count": 1000,
                "runtime_s_q10": 1800,
                "runtime_s_q25": 3600,
                "runtime_s_q50": 3600,
                "runtime_s_q75": 3600,
                "runtime_s_q90": 3600,
            },
            {
                "level": "global",
                "count": 1000,
                "runtime_s_q10": 1800,
                "runtime_s_q25": 3600,
                "runtime_s_q50": 3600,
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
            for metric in [
                "load_cpu",
                "load_a100",
                "load_h100",
                "load_h100_nvl",
                "load_node",
            ]:
                for label in ["q10", "q25", "q50", "q75", "q90"]:
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


def profile(epoch: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for index, warmup in enumerate([False, False, True], start=1):
        rows.append(
            {
                "sim_job_id": f"sim_{index}",
                "slurm_job_id_expected": index,
                "source_submit_utc": epoch.isoformat(),
                "partition": "sheffield",
                "cpus": 64,
                "nodes": 1,
                "scheduled_gpus": 0,
                "runtime_s": 3600,
                "eligible_dt_s": 0,
                "release_dt_s": 0,
                "flexibility_score": 0.1,
                "is_warmup": warmup,
                "source_timelimit_min": 120,
                "user_id": "user_001" if index < 3 else "user_002",
            }
        )
    return pd.DataFrame(rows)


def test_policy_enforces_user_budget_and_never_delays_carry_in() -> None:
    epoch = pd.Timestamp("2025-11-12T00:00:00Z")
    carbon = pd.DataFrame(
        {
            "from_utc": pd.date_range(epoch, periods=12, freq="30min"),
            "to_utc": pd.date_range(
                epoch + pd.Timedelta(minutes=30), periods=12, freq="30min"
            ),
            "intensity_gco2_per_kwh": [300, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100],
        }
    )
    result, metadata, candidates = LOW.apply_low_impact_dynamic(
        profile(epoch),
        history_model(),
        carbon,
        epoch,
        0.3,
        30,
        0.5,
        1.0,
        0.001,
        0.0,
        0.0,
        0.90,
        20.0,
        12.0,
        1.5,
        30,
        2,
    )

    assert result.loc[0, "release_dt_s"] == 1800
    assert result.loc[0, "low_impact_decision"] == "delay"
    assert result.loc[1, "release_dt_s"] == 0
    assert "waiting_budget_exhausted" in result.loc[1, "low_impact_rejection_reason"]
    assert result.loc[2, "release_dt_s"] == 0
    assert "carry_in" in result.loc[2, "low_impact_rejection_reason"]
    assert candidates["accepted"].sum() == 1
    assert metadata["maximum_user_policy_delay_minutes"] == 30


def test_policy_delay_budget_uses_configured_runtime_quantile() -> None:
    epoch = pd.Timestamp("2025-11-12T00:00:00Z")
    model = history_model()
    runtime = model["runtime"]
    assert isinstance(runtime, pd.DataFrame)
    runtime.loc[:, "runtime_s_q25"] = 600
    carbon = pd.DataFrame(
        {
            "from_utc": pd.date_range(epoch, periods=8, freq="30min"),
            "to_utc": pd.date_range(
                epoch + pd.Timedelta(minutes=30), periods=8, freq="30min"
            ),
            "intensity_gco2_per_kwh": [300, 50, 50, 50, 100, 100, 100, 100],
        }
    )
    result, _, _ = LOW.apply_low_impact_dynamic(
        profile(epoch).iloc[:1],
        model,
        carbon,
        epoch,
        0.3,
        30,
        0.5,
        1.0,
        0.001,
        0.0,
        0.0,
        0.90,
        20.0,
        12.0,
        1.5,
        30,
        2,
    )

    assert result.loc[0, "allowed_wait_budget_s"] == 300
    assert result.loc[0, "release_dt_s"] == 300


def test_candidate_grid_includes_exact_deadline() -> None:
    epoch = pd.Timestamp("2025-11-12T00:07:00Z")
    carbon = pd.DataFrame(
        {
            "from_utc": pd.date_range("2025-11-12T00:00:00Z", periods=8, freq="30min"),
            "to_utc": pd.date_range("2025-11-12T00:30:00Z", periods=8, freq="30min"),
            "intensity_gco2_per_kwh": [300, 300, 50, 50, 100, 100, 100, 100],
        }
    )
    deadline = epoch + pd.Timedelta(minutes=45)
    candidates = LOW.bounded_candidate_release_times(
        carbon, epoch, deadline, runtime_s=1800, step_minutes=5
    )
    assert deadline in candidates
    assert epoch + pd.Timedelta(minutes=5) in candidates


def test_holdout_observed_runtime_cannot_change_policy_decisions() -> None:
    epoch = pd.Timestamp("2025-11-12T00:00:00Z")
    carbon = pd.DataFrame(
        {
            "from_utc": pd.date_range(epoch, periods=12, freq="30min"),
            "to_utc": pd.date_range(
                epoch + pd.Timedelta(minutes=30), periods=12, freq="30min"
            ),
            "intensity_gco2_per_kwh": [300, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100],
        }
    )
    original = profile(epoch).iloc[:2].copy()
    changed = original.copy()
    changed["runtime_s"] = [60, 100000]
    args = (
        history_model(),
        carbon,
        epoch,
        0.3,
        30,
        0.5,
        1.0,
        0.001,
        0.0,
        0.0,
        0.90,
        20.0,
        12.0,
        1.5,
        30,
        2,
    )
    original_decisions, _, _ = LOW.apply_low_impact_dynamic(original, *args)
    changed_decisions, _, _ = LOW.apply_low_impact_dynamic(changed, *args)
    columns = ["release_dt_s", "low_impact_decision", "allowed_wait_budget_s"]
    pd.testing.assert_frame_equal(
        original_decisions[columns], changed_decisions[columns]
    )
