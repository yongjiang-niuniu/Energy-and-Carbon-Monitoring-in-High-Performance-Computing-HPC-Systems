#!/usr/bin/env python3
"""Evaluate prior-month quantile remapping on the next untouched month."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "evaluate_history_holdout", ROOT / "evaluate_history_holdout.py"
)
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
assert EVALUATOR_SPEC.loader is not None
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)
MODEL = EVALUATOR.MODEL
GENERATOR = EVALUATOR.GENERATOR
MONTHS = MODEL.MONTHS
AVAILABLE_QUANTILES = sorted(MODEL.QUANTILES.values())
LOAD_TARGETS = ["load_cpu", "load_a100", "load_h100", "load_node"]


def selected_quantile(
    coverage_by_quantile: dict[float, float],
    desired_coverage_pct: float = 75.0,
) -> float:
    """Choose the lowest available quantile that met coverage on calibration data."""
    ordered = sorted(coverage_by_quantile)
    eligible = [
        quantile
        for quantile in ordered
        if coverage_by_quantile[quantile] >= desired_coverage_pct
    ]
    return eligible[0] if eligible else ordered[-1]


def target_metric(frame: pd.DataFrame, target: str, quantile: float) -> pd.Series:
    rows = frame.loc[
        frame["target"].eq(target)
        & pd.to_numeric(frame["quantile"], errors="coerce").sub(quantile).abs().lt(1e-9)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one {target} q{quantile} row, found {len(rows)}")
    return rows.iloc[0]


def coverage_map(frame: pd.DataFrame, target: str) -> dict[float, float]:
    return {
        quantile: float(
            target_metric(frame, target, quantile)[
                "observed_at_or_below_prediction_pct"
            ]
        )
        for quantile in AVAILABLE_QUANTILES
    }


def load_worst_coverage_map(frame: pd.DataFrame) -> dict[float, float]:
    return {
        quantile: min(
            float(
                target_metric(frame, target, quantile)[
                    "observed_at_or_below_prediction_pct"
                ]
            )
            for target in LOAD_TARGETS
        )
        for quantile in AVAILABLE_QUANTILES
    }


def evaluate_month(
    zip_path: Path,
    month: str,
    model: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = model["metadata"]
    assert isinstance(metadata, dict)
    raw = GENERATOR.read_month(zip_path, month)
    cohort, _ = GENERATOR.prepare_cohort(
        raw, month, float(metadata["maximum_training_runtime_hours"])
    )
    runtime_rows = MODEL.runtime_training_rows(cohort)
    activity = MODEL.clean_activity_rows(raw)
    actual_load = MODEL.month_load_intervals(
        activity, MONTHS.index(month) + 1, int(metadata["interval_minutes"])
    )
    runtime_summary, _ = EVALUATOR.runtime_holdout(runtime_rows, model)
    wait_summary = EVALUATOR.wait_holdout(runtime_rows, model)
    load_summary, _ = EVALUATOR.load_holdout(actual_load, model)
    return pd.DataFrame(runtime_summary + wait_summary), pd.DataFrame(load_summary)


def calibrated_row(
    target_month: str,
    calibration_month: str,
    training_months: list[str],
    calibration_runtime_wait: pd.DataFrame,
    calibration_load: pd.DataFrame,
    target_runtime_wait: pd.DataFrame,
    target_load: pd.DataFrame,
) -> dict[str, object]:
    runtime_map = coverage_map(calibration_runtime_wait, "runtime")
    wait_map = coverage_map(calibration_runtime_wait, "queue_wait")
    load_map = load_worst_coverage_map(calibration_load)
    runtime_q = selected_quantile(runtime_map)
    wait_q = selected_quantile(wait_map)
    load_q = selected_quantile(load_map)
    nominal_runtime = target_metric(target_runtime_wait, "runtime", 0.75)
    calibrated_runtime = target_metric(target_runtime_wait, "runtime", runtime_q)
    nominal_wait = target_metric(target_runtime_wait, "queue_wait", 0.75)
    calibrated_wait = target_metric(target_runtime_wait, "queue_wait", wait_q)

    target_load_nominal = [
        float(target_metric(target_load, target, 0.75)["observed_at_or_below_prediction_pct"])
        for target in LOAD_TARGETS
    ]
    target_load_calibrated = [
        float(target_metric(target_load, target, load_q)["observed_at_or_below_prediction_pct"])
        for target in LOAD_TARGETS
    ]
    return {
        "target_month": target_month,
        "calibration_month": calibration_month,
        "training_months": "|".join(training_months),
        "runtime_selected_quantile": runtime_q,
        "runtime_calibration_coverage_pct": runtime_map[runtime_q],
        "runtime_target_nominal_q75_coverage_pct": float(
            nominal_runtime["observed_at_or_below_prediction_pct"]
        ),
        "runtime_target_selected_coverage_pct": float(
            calibrated_runtime["observed_at_or_below_prediction_pct"]
        ),
        "runtime_nominal_coverage_error_pp": abs(
            float(nominal_runtime["observed_at_or_below_prediction_pct"]) - 75.0
        ),
        "runtime_selected_coverage_error_pp": abs(
            float(calibrated_runtime["observed_at_or_below_prediction_pct"]) - 75.0
        ),
        "runtime_target_nominal_median_ape_pct": float(
            nominal_runtime["median_absolute_percentage_error_pct"]
        ),
        "runtime_target_selected_median_ape_pct": float(
            calibrated_runtime["median_absolute_percentage_error_pct"]
        ),
        "wait_selected_quantile": wait_q,
        "wait_calibration_coverage_pct": wait_map[wait_q],
        "wait_target_nominal_q75_coverage_pct": float(
            nominal_wait["observed_at_or_below_prediction_pct"]
        ),
        "wait_target_selected_coverage_pct": float(
            calibrated_wait["observed_at_or_below_prediction_pct"]
        ),
        "wait_nominal_coverage_error_pp": abs(
            float(nominal_wait["observed_at_or_below_prediction_pct"]) - 75.0
        ),
        "wait_selected_coverage_error_pp": abs(
            float(calibrated_wait["observed_at_or_below_prediction_pct"]) - 75.0
        ),
        "load_selected_quantile": load_q,
        "load_calibration_worst_coverage_pct": load_map[load_q],
        "load_target_nominal_mean_coverage_pct": sum(target_load_nominal) / len(target_load_nominal),
        "load_target_selected_mean_coverage_pct": sum(target_load_calibrated) / len(target_load_calibrated),
        "load_target_nominal_worst_coverage_pct": min(target_load_nominal),
        "load_target_selected_worst_coverage_pct": min(target_load_calibrated),
    }


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    x = range(len(summary))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for axis, target, title in [
        (axes[0], "runtime", "Prior-month runtime remapping"),
        (axes[1], "wait", "Prior-month queue-wait remapping"),
    ]:
        axis.plot(
            x,
            summary[f"{target}_target_nominal_q75_coverage_pct"],
            marker="o",
            linewidth=2,
            color="#2f6f9f",
            label="fixed q75",
        )
        axis.plot(
            x,
            summary[f"{target}_target_selected_coverage_pct"],
            marker="s",
            linewidth=2,
            color="#c05a45",
            label="prior-month selected",
        )
        axis.axhline(75, color="#555555", linestyle="--", linewidth=1.2)
        axis.set_xticks(list(x), summary["target_month"].str.capitalize())
        axis.set_ylim(0, 100)
        axis.set_ylabel("Target-month coverage (%)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--first-target", choices=MONTHS, default="april")
    parser.add_argument("--last-target", choices=MONTHS, default="july")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = MONTHS.index(args.first_target)
    last = MONTHS.index(args.last_target)
    if first < 3 or first > last:
        raise ValueError("Targets must start in April or later and remain ordered")

    rows: list[dict[str, object]] = []
    target_details: list[dict[str, object]] = []
    for target_month in MONTHS[first:last + 1]:
        calibration_month = MONTHS[MONTHS.index(target_month) - 1]
        model_dir = args.model_root / calibration_month
        evaluation_dir = args.evaluation_root / calibration_month
        model = MODEL.load_model(model_dir)
        metadata = model["metadata"]
        assert isinstance(metadata, dict)
        if metadata["holdout_month"] != calibration_month:
            raise ValueError(f"{model_dir} is not reserved for {calibration_month}")
        if not all(
            MONTHS.index(month) < MONTHS.index(calibration_month)
            for month in metadata["training_months"]
        ) or MONTHS.index(calibration_month) >= MONTHS.index(target_month):
            raise ValueError("Training, calibration and target months are not ordered")

        calibration_runtime_wait = pd.read_csv(
            evaluation_dir / "runtime_wait_accuracy.csv"
        )
        calibration_load = pd.read_csv(evaluation_dir / "load_accuracy.csv")
        target_runtime_wait, target_load = evaluate_month(
            args.zip_path, target_month, model
        )
        rows.append(
            calibrated_row(
                target_month,
                calibration_month,
                list(metadata["training_months"]),
                calibration_runtime_wait,
                calibration_load,
                target_runtime_wait,
                target_load,
            )
        )
        for source, frame in [
            ("runtime_wait", target_runtime_wait),
            ("load", target_load),
        ]:
            for record in frame.to_dict(orient="records"):
                target_details.append(
                    {
                        "target_month": target_month,
                        "calibration_month": calibration_month,
                        "source": source,
                        **record,
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "prior_month_calibration_summary.csv", index=False)
    plot_summary(summary, args.output / "prior_month_calibration.png")
    pd.DataFrame(target_details).to_csv(
        args.output / "prior_month_target_metrics.csv", index=False
    )
    result = {
        "targets": len(summary),
        "temporal_order_pass": True,
        "runtime_selected_quantiles": summary["runtime_selected_quantile"].value_counts().sort_index().to_dict(),
        "wait_selected_quantiles": summary["wait_selected_quantile"].value_counts().sort_index().to_dict(),
        "load_selected_quantiles": summary["load_selected_quantile"].value_counts().sort_index().to_dict(),
        "mean_runtime_nominal_coverage_error_pp": float(summary["runtime_nominal_coverage_error_pp"].mean()),
        "mean_runtime_selected_coverage_error_pp": float(summary["runtime_selected_coverage_error_pp"].mean()),
        "mean_wait_nominal_coverage_error_pp": float(summary["wait_nominal_coverage_error_pp"].mean()),
        "mean_wait_selected_coverage_error_pp": float(summary["wait_selected_coverage_error_pp"].mean()),
        "method": "Lowest q50/q75/q90 meeting 75% coverage on the prior calibration month; q90 fallback",
        "claim_boundary": "Coverage remapping, not conformal prediction or a production guarantee",
    }
    (args.output / "prior_month_calibration_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
