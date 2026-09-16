#!/usr/bin/env python3
"""Compare an empty-start pilot with the same policy under real carry-in load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ["capacity_weighted", "node_request_upper"]
MODEL_LABELS = ["Capacity-weighted", "Node-request upper proxy"]
CONTEXT_LABELS = ["No carry-in", "2 h real carry-in"]
COLORS = ["#2457A7", "#11866F"]


def context_row(path: Path, context: str, scenario: str) -> dict[str, object]:
    comparison = pd.read_csv(path)
    baseline = comparison.loc[comparison["scenario"].eq("baseline")]
    policy = comparison.loc[comparison["scenario"].eq(scenario)]
    if len(baseline) != 1 or len(policy) != 1:
        raise ValueError(f"Expected baseline and {scenario} in {path}")
    baseline = baseline.iloc[0]
    policy = policy.iloc[0]
    row: dict[str, object] = {
        "context": context,
        "jobs": int(policy["jobs"]),
        "evaluation_jobs": int(policy["evaluation_jobs"]),
        "warmup_jobs": int(policy["warmup_jobs"]),
        "jobs_delayed": int(policy["jobs_with_policy_delay"]),
        "total_user_wait_p95_h": float(policy["total_user_wait_p95_s"]) / 3600,
        "scheduler_wait_p95_s": float(policy["scheduler_wait_p95_s"]),
    }
    for model in MODELS:
        base_energy = float(baseline[f"dynamic_energy_kwh_{model}"])
        policy_energy = float(policy[f"dynamic_energy_kwh_{model}"])
        row[f"baseline_mean_utilization_{model}"] = float(
            baseline[f"mean_utilization_{model}"]
        )
        row[f"baseline_peak_utilization_{model}"] = float(
            baseline[f"peak_utilization_{model}"]
        )
        row[f"dynamic_carbon_reduction_pct_{model}"] = float(
            policy[f"dynamic_carbon_reduction_pct_{model}"]
        )
        row[f"dynamic_carbon_reduction_kg_{model}"] = float(
            policy[f"dynamic_carbon_reduction_kg_{model}"]
        )
        row[f"dynamic_energy_change_pct_{model}"] = (
            (policy_energy - base_energy) / base_energy * 100 if base_energy else 0.0
        )
    return row


def plot_contexts(comparison: pd.DataFrame, output: Path) -> None:
    x = np.arange(len(MODELS))
    width = 0.34
    fig, (ax_carbon, ax_util) = plt.subplots(1, 2, figsize=(12, 5.5))

    for index, row in comparison.iterrows():
        reduction = [
            row[f"dynamic_carbon_reduction_pct_{model}"] for model in MODELS
        ]
        bars = ax_carbon.bar(
            x + (index - 0.5) * width,
            reduction,
            width,
            label=CONTEXT_LABELS[index],
            color=COLORS[index],
        )
        ax_carbon.bar_label(bars, fmt="%.3f%%", padding=4, fontsize=9)

        utilization = [
            row[f"baseline_mean_utilization_{model}"] * 100 for model in MODELS
        ]
        bars = ax_util.bar(
            x + (index - 0.5) * width,
            utilization,
            width,
            label=CONTEXT_LABELS[index],
            color=COLORS[index],
        )
        ax_util.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)

    for axis in [ax_carbon, ax_util]:
        axis.set_xticks(x, MODEL_LABELS)
        axis.axhline(0, color="#AAB2BC", linewidth=1)
        axis.grid(axis="y", color="#E6E9ED")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    ax_carbon.set_ylabel("Controllable carbon reduction (%)")
    ax_carbon.set_title("B1 result changes with background load", loc="left", weight="bold")
    ax_util.set_ylabel("Mean baseline utilization proxy (%)")
    ax_util.set_title("Carry-in jobs remove the empty-cluster assumption", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(output / "cold_start_vs_warmup.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--scenario", default="b1_shift_2h")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        context_row(args.cold, "cold_start", args.scenario),
        context_row(args.warm, "real_carry_in", args.scenario),
    ]
    comparison = pd.DataFrame(rows)
    cold = comparison.iloc[0]
    warm = comparison.iloc[1]
    cold_reduction = cold["dynamic_carbon_reduction_pct_capacity_weighted"]
    warm_reduction = warm["dynamic_carbon_reduction_pct_capacity_weighted"]
    relative_share_retained = (
        warm_reduction / cold_reduction * 100 if cold_reduction else 0.0
    )
    cold_absolute = cold["dynamic_carbon_reduction_kg_capacity_weighted"]
    warm_absolute = warm["dynamic_carbon_reduction_kg_capacity_weighted"]
    absolute_retained = warm_absolute / cold_absolute * 100 if cold_absolute else 0.0
    summary = {
        "scenario": args.scenario,
        "primary_model": "capacity_weighted",
        "cold_start_reduction_pct": cold_reduction,
        "real_carry_in_reduction_pct": warm_reduction,
        "relative_reduction_share_retained_pct": relative_share_retained,
        "cold_start_absolute_reduction_kg": cold_absolute,
        "real_carry_in_absolute_reduction_kg": warm_absolute,
        "absolute_reduction_retained_pct": absolute_retained,
        "interpretation": (
            "The same movable jobs produce almost the same absolute reduction, but carry-in jobs enlarge the total dynamic-carbon denominator. The empty-start pilot therefore overstates the policy's percentage share of cluster emissions."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output / "cold_start_vs_warmup.csv", index=False)
    (args.output / "cold_start_vs_warmup.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_contexts(comparison, args.output)
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
