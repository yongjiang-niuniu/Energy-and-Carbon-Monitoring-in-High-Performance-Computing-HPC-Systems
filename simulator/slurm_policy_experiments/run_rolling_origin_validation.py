#!/usr/bin/env python3
"""Build and evaluate deterministic history models across temporal origins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def training_months_for_holdout(
    holdout_month: str,
    mode: str,
    recent_window_months: int,
    minimum_training_months: int,
) -> list[str]:
    holdout_index = MONTHS.index(holdout_month)
    if holdout_index < minimum_training_months:
        raise ValueError(
            f"{holdout_month} has fewer than {minimum_training_months} earlier months"
        )
    if mode == "expanding":
        start = 0
    elif mode == "recent":
        start = max(0, holdout_index - recent_window_months)
    else:
        raise ValueError(f"Unsupported rolling-origin mode: {mode}")
    training = MONTHS[start:holdout_index]
    if len(training) < minimum_training_months:
        raise ValueError("Rolling split did not retain enough training months")
    return training


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT.parents[1], check=True)


def model_is_reusable(model_dir: Path, training: list[str], holdout: str) -> bool:
    metadata_path = model_dir / "model_metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("training_months") == training
        and metadata.get("holdout_month") == holdout
        and metadata.get("temporal_split_pass") is True
    )


def evaluation_is_reusable(evaluation_dir: Path, holdout: str) -> bool:
    summary_path = evaluation_dir / "holdout_evaluation_summary.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("holdout_month") == holdout and bool(
        summary.get("temporal_split_pass")
    )


def metric_row(
    frame: pd.DataFrame,
    target: str,
    quantile: float = 0.75,
) -> pd.Series:
    rows = frame.loc[
        frame["target"].eq(target)
        & pd.to_numeric(frame["quantile"], errors="coerce").sub(quantile).abs().lt(1e-9)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one {target} q{quantile} row, found {len(rows)}")
    return rows.iloc[0]


def summarize_origin(
    mode: str,
    holdout: str,
    training: list[str],
    model_dir: Path,
    evaluation_dir: Path,
) -> dict[str, object]:
    runtime_wait = pd.read_csv(evaluation_dir / "runtime_wait_accuracy.csv")
    load = pd.read_csv(evaluation_dir / "load_accuracy.csv")
    runtime = metric_row(runtime_wait, "runtime")
    wait = metric_row(runtime_wait, "queue_wait")
    output: dict[str, object] = {
        "mode": mode,
        "holdout_month": holdout,
        "training_months": "|".join(training),
        "training_month_count": len(training),
        "model_dir": str(model_dir),
        "evaluation_dir": str(evaluation_dir),
        "runtime_q75_coverage_pct": float(runtime["observed_at_or_below_prediction_pct"]),
        "runtime_q75_coverage_error_pp": abs(
            float(runtime["observed_at_or_below_prediction_pct"]) - 75.0
        ),
        "runtime_q75_median_ape_pct": float(runtime["median_absolute_percentage_error_pct"]),
        "runtime_q75_mean_bias_s": float(runtime["mean_bias_s"]),
        "wait_q75_coverage_pct": float(wait["observed_at_or_below_prediction_pct"]),
        "wait_q75_coverage_error_pp": abs(
            float(wait["observed_at_or_below_prediction_pct"]) - 75.0
        ),
    }
    for target in ["load_cpu", "load_a100", "load_h100", "load_node"]:
        row = metric_row(load, target)
        short = target.removeprefix("load_")
        output[f"{short}_q75_coverage_pct"] = float(
            row["observed_at_or_below_prediction_pct"]
        )
        output[f"{short}_q75_mae"] = float(row["mean_absolute_error_fraction"])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--mode", choices=["expanding", "recent"], required=True)
    parser.add_argument("--first-holdout", choices=MONTHS, default="march")
    parser.add_argument("--last-holdout", choices=MONTHS, default="july")
    parser.add_argument("--recent-window-months", type=int, default=3)
    parser.add_argument("--minimum-training-months", type=int, default=2)
    parser.add_argument("--minimum-group-count", type=int, default=50)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recent_window_months < args.minimum_training_months:
        raise ValueError("recent-window-months must cover minimum-training-months")
    first = MONTHS.index(args.first_holdout)
    last = MONTHS.index(args.last_holdout)
    if first > last:
        raise ValueError("first-holdout must not be later than last-holdout")

    rows: list[dict[str, object]] = []
    for holdout in MONTHS[first:last + 1]:
        training = training_months_for_holdout(
            holdout,
            args.mode,
            args.recent_window_months,
            args.minimum_training_months,
        )
        model_dir = args.model_root / args.mode / holdout
        evaluation_dir = args.output / args.mode / holdout
        if not (args.reuse and model_is_reusable(model_dir, training, holdout)):
            run([
                sys.executable,
                str(ROOT / "build_history_quantile_model.py"),
                "--zip", str(args.zip_path),
                "--train-months", *training,
                "--holdout-month", holdout,
                "--interval-minutes", "30",
                "--max-runtime-hours", "12",
                "--minimum-group-count", str(args.minimum_group_count),
                "--output", str(model_dir),
            ])
        if not (args.reuse and evaluation_is_reusable(evaluation_dir, holdout)):
            run([
                sys.executable,
                str(ROOT / "evaluate_history_holdout.py"),
                "--zip", str(args.zip_path),
                "--model-dir", str(model_dir),
                "--holdout-month", holdout,
                "--output", str(evaluation_dir),
            ])
        rows.append(
            summarize_origin(
                args.mode, holdout, training, model_dir, evaluation_dir
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / f"{args.mode}_rolling_origin_summary.csv", index=False)
    result = {
        "mode": args.mode,
        "first_holdout": args.first_holdout,
        "last_holdout": args.last_holdout,
        "origins": len(summary),
        "mean_runtime_q75_coverage_pct": float(summary["runtime_q75_coverage_pct"].mean()),
        "mean_runtime_q75_coverage_error_pp": float(
            summary["runtime_q75_coverage_error_pp"].mean()
        ),
        "mean_wait_q75_coverage_pct": float(summary["wait_q75_coverage_pct"].mean()),
        "output": str(args.output),
    }
    (args.output / f"{args.mode}_rolling_origin_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
