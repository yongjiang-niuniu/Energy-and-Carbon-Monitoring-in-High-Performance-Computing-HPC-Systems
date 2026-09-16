#!/usr/bin/env python3
"""Evaluate an earlier-month quantile model on a later held-out month."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_SPEC = importlib.util.spec_from_file_location(
    "build_history_quantile_model", ROOT / "build_history_quantile_model.py"
)
MODEL = importlib.util.module_from_spec(MODEL_SPEC)
assert MODEL_SPEC.loader is not None
MODEL_SPEC.loader.exec_module(MODEL)

GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_empirical_workload", ROOT / "generate_empirical_workload.py"
)
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
assert GENERATOR_SPEC.loader is not None
GENERATOR_SPEC.loader.exec_module(GENERATOR)


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    error = frame["predicted"] - frame["actual"]
    absolute = error.abs()
    denominator = frame["actual"].clip(lower=60.0)
    return {
        "observations": len(frame),
        "actual_median_s": float(frame["actual"].median()),
        "predicted_median_s": float(frame["predicted"].median()),
        "mean_absolute_error_s": float(absolute.mean()),
        "median_absolute_error_s": float(absolute.median()),
        "median_absolute_percentage_error_pct": float((absolute / denominator).median() * 100),
        "mean_bias_s": float(error.mean()),
        "observed_at_or_below_prediction_pct": float(
            frame["actual"].le(frame["predicted"]).mean() * 100
        ),
    }


def pinball_loss(actual: pd.Series, predicted: pd.Series, quantile: float) -> float:
    error = actual - predicted
    return float(np.maximum(quantile * error, (quantile - 1) * error).mean())


def runtime_holdout(
    rows: pd.DataFrame,
    model: dict[str, object],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    profile = rows.rename(columns={"gpus": "scheduled_gpus"})
    summaries: list[dict[str, object]] = []
    q50_predictions: pd.DataFrame | None = None
    for label, quantile in MODEL.QUANTILES.items():
        predicted = MODEL.predict_runtime(profile, model, quantile)
        metrics = error_metrics(rows["runtime_s"], predicted["predicted_runtime_s"])
        summaries.append({"target": "runtime", "quantile": quantile, **metrics})
        if label == "q50":
            q50_predictions = predicted
    assert q50_predictions is not None
    by_partition = []
    for partition, indices in rows.groupby("partition").groups.items():
        metrics = error_metrics(
            rows.loc[indices, "runtime_s"],
            q50_predictions.loc[indices, "predicted_runtime_s"],
        )
        by_partition.append({"partition": partition, **metrics})
    return summaries, pd.DataFrame(by_partition)


def wait_holdout(
    rows: pd.DataFrame,
    model: dict[str, object],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    valid = rows.dropna(subset=["wait_s", "eligible_utc"])
    for _, quantile in MODEL.QUANTILES.items():
        cache: dict[tuple[str, int, int], float] = {}
        values: list[float] = []
        for row in valid.itertuples(index=False):
            timestamp = pd.Timestamp(row.eligible_utc)
            key = (
                str(row.partition),
                int(timestamp.weekday()),
                int(timestamp.hour * 2 + timestamp.minute // 30),
            )
            if key not in cache:
                cache[key] = MODEL.predict_wait(
                    str(row.partition), timestamp, model, quantile
                )[0]
            values.append(cache[key])
        predicted = pd.Series(
            values,
            index=valid.index,
        )
        metrics = error_metrics(valid["wait_s"], predicted)
        summaries.append({"target": "queue_wait", "quantile": quantile, **metrics})
    return summaries


def load_holdout(
    actual: pd.DataFrame,
    model: dict[str, object],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    history = model["load"]
    assert isinstance(history, pd.DataFrame)
    joined = actual.merge(
        history,
        on=["weekday_utc", "half_hour_slot_utc"],
        how="left",
        validate="many_to_one",
    )
    summaries: list[dict[str, object]] = []
    metrics = [f"load_{name}" for name in MODEL.CLASS_CAPACITY] + ["load_node"]
    for metric in metrics:
        for _, quantile in MODEL.QUANTILES.items():
            label = MODEL.quantile_label(quantile)
            prediction = joined[f"{metric}_{label}"]
            error = prediction - joined[metric]
            summaries.append(
                {
                    "target": metric,
                    "quantile": quantile,
                    "observations": len(joined),
                    "actual_mean_fraction": float(joined[metric].mean()),
                    "predicted_mean_fraction": float(prediction.mean()),
                    "mean_absolute_error_fraction": float(error.abs().mean()),
                    "root_mean_squared_error_fraction": float(np.sqrt((error**2).mean())),
                    "mean_bias_fraction": float(error.mean()),
                    "observed_at_or_below_prediction_pct": float(
                        joined[metric].le(prediction).mean() * 100
                    ),
                    "pinball_loss": pinball_loss(joined[metric], prediction, quantile),
                }
            )
    interval_output = joined[[
        "from_utc", "weekday_utc", "half_hour_slot_utc", *metrics,
        *[f"{metric}_q50" for metric in metrics],
        *[f"{metric}_q75" for metric in metrics],
        *[f"{metric}_q90" for metric in metrics],
    ]].copy()
    return summaries, interval_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--holdout-month", choices=MODEL.MONTHS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MODEL.load_model(args.model_dir)
    metadata = model["metadata"]
    assert isinstance(metadata, dict)
    if metadata["holdout_month"] != args.holdout_month:
        raise ValueError(
            f"Model reserves {metadata['holdout_month']}, not {args.holdout_month}"
        )
    raw = GENERATOR.read_month(args.zip_path, args.holdout_month)
    cohort, cohort_audit = GENERATOR.prepare_cohort(
        raw, args.holdout_month, float(metadata["maximum_training_runtime_hours"])
    )
    runtime_rows = MODEL.runtime_training_rows(cohort)
    activity = MODEL.clean_activity_rows(raw)
    actual_load = MODEL.month_load_intervals(
        activity,
        MODEL.MONTHS.index(args.holdout_month) + 1,
        int(metadata["interval_minutes"]),
    )

    runtime_summary, runtime_by_partition = runtime_holdout(runtime_rows, model)
    wait_summary = wait_holdout(runtime_rows, model)
    load_summary, interval_output = load_holdout(actual_load, model)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runtime_summary + wait_summary).to_csv(
        args.output / "runtime_wait_accuracy.csv", index=False
    )
    runtime_by_partition.to_csv(
        args.output / "runtime_q50_accuracy_by_partition.csv", index=False
    )
    pd.DataFrame(load_summary).to_csv(args.output / "load_accuracy.csv", index=False)
    interval_output.to_csv(args.output / "holdout_load_intervals.csv", index=False)
    summary = {
        "model_dir": str(args.model_dir),
        "training_months": metadata["training_months"],
        "holdout_month": args.holdout_month,
        "temporal_split_pass": all(
            MODEL.MONTHS.index(month) < MODEL.MONTHS.index(args.holdout_month)
            for month in metadata["training_months"]
        ),
        "policy_leakage_boundary": "Holdout observations are read by this evaluator only, after model construction; generate_policy_scenario reads aggregate model files only",
        "holdout_cohort_audit": cohort_audit,
        "runtime_wait_summary": runtime_summary + wait_summary,
        "load_summary": load_summary,
    }
    (args.output / "holdout_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "temporal_split_pass": summary["temporal_split_pass"],
        "holdout_runtime_rows": len(runtime_rows),
        "holdout_load_intervals": len(actual_load),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
