#!/usr/bin/env python3
"""Summarize carbon reduction per hour of policy-imposed waiting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_scenario(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("scenario must use NAME=SCENARIO_DIR")
    return name, Path(path)


def policy_wait(profile: pd.DataFrame) -> tuple[int, float]:
    rows = profile
    if "is_warmup" in profile:
        rows = profile.loc[~profile["is_warmup"].fillna(False).astype(bool)]
    delay = pd.to_numeric(rows["policy_delay_s"], errors="coerce").fillna(0.0)
    return int((delay > 0).sum()), float(delay.sum() / 3600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--scenario", action="append", type=parse_scenario, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = pd.read_csv(args.comparison).set_index("scenario")
    rows = []
    for name, path in args.scenario:
        if name not in comparison.index:
            raise ValueError(f"Scenario {name!r} is absent from the comparison")
        metrics = comparison.loc[name]
        delayed_jobs, wait_hours = policy_wait(pd.read_csv(path / "workload_profile.csv"))
        capacity_kg = float(
            metrics["dynamic_carbon_reduction_kg_capacity_weighted"]
        )
        node_kg = float(metrics["dynamic_carbon_reduction_kg_node_request_upper"])
        rows.append(
            {
                "scenario": name,
                "jobs_delayed": delayed_jobs,
                "total_policy_wait_hours": wait_hours,
                "p95_wait_change_s": float(metrics["total_wait_p95_change_s"]),
                "capacity_reduction_pct": float(
                    metrics["dynamic_carbon_reduction_pct_capacity_weighted"]
                ),
                "node_reduction_pct": float(
                    metrics["dynamic_carbon_reduction_pct_node_request_upper"]
                ),
                "capacity_reduction_kg": capacity_kg,
                "node_reduction_kg": node_kg,
                "capacity_kg_per_policy_wait_hour": (
                    capacity_kg / wait_hours if wait_hours else 0.0
                ),
                "node_kg_per_policy_wait_hour": (
                    node_kg / wait_hours if wait_hours else 0.0
                ),
            }
        )

    output = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "policy_value_comparison.csv", index=False)

    labels = output["scenario"].str.replace("_", " ")
    colors = [
        "#26734d" if node_value > 0 else "#b23a48"
        for node_value in output["node_reduction_pct"]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(
        output["total_policy_wait_hours"],
        output["capacity_reduction_pct"],
        c=colors,
        s=70,
    )
    for x, y, label in zip(
        output["total_policy_wait_hours"],
        output["capacity_reduction_pct"],
        labels,
    ):
        axes[0].annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_xlabel("Total policy-imposed wait (hours)")
    axes[0].set_ylabel("Capacity-model dynamic carbon reduction (%)")
    axes[0].set_title("Absolute carbon versus waiting cost")
    axes[0].grid(alpha=0.2)

    positions = np.arange(len(output))
    width = 0.36
    axes[1].bar(
        positions - width / 2,
        output["capacity_kg_per_policy_wait_hour"],
        width,
        label="Capacity model",
        color="#2f6f9f",
    )
    axes[1].bar(
        positions + width / 2,
        output["node_kg_per_policy_wait_hour"],
        width,
        label="Node proxy",
        color="#d08b32",
    )
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xticks(positions, labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("kgCO2e reduction per policy-wait hour")
    axes[1].set_title("Carbon value of imposed waiting")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(args.output / "policy_value_tradeoff.png", dpi=180)
    plt.close(figure)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
