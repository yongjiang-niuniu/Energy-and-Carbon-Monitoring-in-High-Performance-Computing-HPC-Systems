#!/usr/bin/env python3
"""Compare expanding and recent rolling-origin history models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compare_modes(expanding: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    if expanding["holdout_month"].duplicated().any() or recent["holdout_month"].duplicated().any():
        raise ValueError("Each mode must contain one row per holdout month")
    merged = expanding.merge(
        recent,
        on="holdout_month",
        suffixes=("_expanding", "_recent"),
        validate="one_to_one",
    )
    if len(merged) != len(expanding) or len(merged) != len(recent):
        raise ValueError("Rolling-origin modes do not cover the same holdout months")
    for metric in [
        "runtime_q75_coverage_pct",
        "runtime_q75_coverage_error_pp",
        "runtime_q75_median_ape_pct",
        "wait_q75_coverage_pct",
        "wait_q75_coverage_error_pp",
    ]:
        merged[f"{metric}_recent_minus_expanding"] = (
            merged[f"{metric}_recent"] - merged[f"{metric}_expanding"]
        )
    load_targets = ["cpu", "a100", "h100", "node"]
    for mode in ["expanding", "recent"]:
        merged[f"mean_load_q75_coverage_error_pp_{mode}"] = merged[
            [f"{target}_q75_coverage_pct_{mode}" for target in load_targets]
        ].sub(75.0).abs().mean(axis=1)
    merged["mean_load_q75_coverage_error_pp_recent_minus_expanding"] = (
        merged["mean_load_q75_coverage_error_pp_recent"]
        - merged["mean_load_q75_coverage_error_pp_expanding"]
    )
    return merged


def plot_comparison(frame: pd.DataFrame, output: Path) -> None:
    x = range(len(frame))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for mode, color, marker in [
        ("expanding", "#2f6f9f", "o"),
        ("recent", "#c05a45", "s"),
    ]:
        axes[0].plot(
            x,
            frame[f"runtime_q75_coverage_pct_{mode}"],
            color=color,
            marker=marker,
            linewidth=2,
            label=mode,
        )
        axes[1].plot(
            x,
            frame[f"wait_q75_coverage_pct_{mode}"],
            color=color,
            marker=marker,
            linewidth=2,
            label=mode,
        )
    for axis, title in zip(axes, ["Runtime q75 coverage", "Queue-wait q75 coverage"]):
        axis.axhline(75, color="#555555", linestyle="--", linewidth=1.2, label="nominal 75%")
        axis.set_xticks(list(x), frame["holdout_month"].str.capitalize())
        axis.set_ylim(0, 100)
        axis.set_ylabel("Held-out coverage (%)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanding", type=Path, required=True)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = compare_modes(pd.read_csv(args.expanding), pd.read_csv(args.recent))
    args.output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output / "rolling_origin_mode_comparison.csv", index=False)
    plot_comparison(comparison, args.output / "rolling_origin_mode_comparison.png")
    summary = {
        "origins": len(comparison),
        "expanding_mean_runtime_q75_coverage_error_pp": float(
            comparison["runtime_q75_coverage_error_pp_expanding"].mean()
        ),
        "recent_mean_runtime_q75_coverage_error_pp": float(
            comparison["runtime_q75_coverage_error_pp_recent"].mean()
        ),
        "expanding_mean_wait_q75_coverage_error_pp": float(
            comparison["wait_q75_coverage_error_pp_expanding"].mean()
        ),
        "recent_mean_wait_q75_coverage_error_pp": float(
            comparison["wait_q75_coverage_error_pp_recent"].mean()
        ),
        "recent_runtime_coverage_error_wins": int(
            comparison["runtime_q75_coverage_error_pp_recent_minus_expanding"].lt(0).sum()
        ),
        "recent_wait_coverage_error_wins": int(
            comparison["wait_q75_coverage_error_pp_recent_minus_expanding"].lt(0).sum()
        ),
        "interpretation": "Lower absolute coverage error is better; no selection is made from MAPE alone",
    }
    (args.output / "rolling_origin_mode_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
