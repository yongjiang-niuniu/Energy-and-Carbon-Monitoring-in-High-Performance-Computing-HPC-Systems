#!/usr/bin/env python3
"""Plot held-out quantile coverage for the history-only predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_wait = pd.read_csv(args.evaluation / "runtime_wait_accuracy.csv")
    load = pd.read_csv(args.evaluation / "load_accuracy.csv")
    summary = json.loads(
        (args.evaluation / "holdout_evaluation_summary.json").read_text(encoding="utf-8")
    )
    holdout_label = str(summary["holdout_month"]).capitalize()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for target, color, marker in [
        ("runtime", "#2f6f9f", "o"),
        ("queue_wait", "#c05a45", "s"),
    ]:
        rows = runtime_wait.loc[runtime_wait["target"].eq(target)].sort_values("quantile")
        axes[0].plot(
            rows["quantile"] * 100,
            rows["observed_at_or_below_prediction_pct"],
            marker=marker,
            linewidth=2,
            color=color,
            label=target.replace("_", " "),
        )
    axes[0].plot([50, 90], [50, 90], linestyle="--", color="#555555", label="ideal")
    axes[0].set_xlabel("Nominal quantile (%)")
    axes[0].set_ylabel("Held-out coverage (%)")
    axes[0].set_title("Runtime and queue-wait calibration")
    axes[0].set_xticks([50, 75, 90])
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    q75 = load.loc[
        load["quantile"].eq(0.75)
        & load["target"].isin(["load_cpu", "load_a100", "load_h100", "load_node"])
    ].copy()
    q75["label"] = q75["target"].str.replace("load_", "", regex=False).str.upper()
    axes[1].bar(
        q75["label"],
        q75["observed_at_or_below_prediction_pct"],
        color=["#2f6f9f", "#d08b32", "#6f7f52", "#8b5a8c"],
    )
    axes[1].axhline(75, linestyle="--", color="#555555", linewidth=1.2)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Held-out coverage (%)")
    axes[1].set_title(f"{holdout_label} load coverage at historical q75")
    axes[1].grid(axis="y", alpha=0.2)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
