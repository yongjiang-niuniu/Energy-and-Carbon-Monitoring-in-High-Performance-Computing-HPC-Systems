#!/usr/bin/env python3
"""Create report-ready figures for the pilot policy comparison."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {
    "baseline": "#2457A7",
    "b1_shift_2h": "#11866F",
    "b1_shift_6h": "#11866F",
    "b2_cap_50": "#D39128",
    "b1_b2_combined": "#6A6F7A",
    "adaptive_user_first": "#4C78A8",
    "adaptive_balanced": "#2A9D8F",
    "adaptive_carbon_first": "#E76F51",
    "time_hybrid_08_18": "#D39128",
    "b2_q75_cap50": "#D39128",
    "dynamic_forecast_strict_oracle": "#845EC2",
    "dynamic_forecast_balanced_oracle": "#5B4B8A",
    "history_only": "#4C78A8",
    "forecast_consensus_80pct": "#E76F51",
    "safe_rolling_balanced": "#2A9D8F",
    "no_policy_repeat": "#98A2B3",
    "low_impact_dynamic_q25": "#11866F",
}
LABELS = {
    "baseline": "Baseline",
    "b1_shift_2h": "B1: 2 h shift",
    "b1_shift_6h": "B1: 6 h shift",
    "b2_cap_50": "B2: 50% gate",
    "b1_b2_combined": "B1 + B2",
    "adaptive_user_first": "Adaptive: user-first",
    "adaptive_balanced": "Adaptive: balanced",
    "adaptive_carbon_first": "Adaptive: carbon-first",
    "time_hybrid_08_18": "Hybrid: B2 day / B1 night",
    "b2_q75_cap50": "B2: Q75 / 50% cap",
    "dynamic_forecast_strict_oracle": "Dynamic strict (oracle)",
    "dynamic_forecast_balanced_oracle": "Dynamic balanced (oracle)",
    "history_only": "History-only",
    "forecast_consensus_80pct": "Forecast consensus (80%)",
    "safe_rolling_balanced": "Safe rolling balanced",
    "no_policy_repeat": "No-policy repeat",
    "low_impact_dynamic_q25": "Low-impact dynamic",
}
NUMBER_OFFSETS = [
    (0, 12),
    (12, 0),
    (0, -12),
    (-12, 0),
    (10, 10),
    (10, -10),
    (-10, -10),
    (-10, 10),
]


def scenario_label(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    match = re.fullmatch(r"b1_shift_(\d+(?:\.\d+)?)h", name)
    if match:
        return f"B1: {match.group(1)} h shift"
    match = re.fullmatch(r"b2_cap_(\d+(?:\.\d+)?)", name)
    if match:
        return f"B2: {match.group(1)}% gate"
    match = re.fullmatch(r"b1_runtime_(\d+(?:\.\d+)?)h", name)
    if match:
        return f"Runtime B1: {match.group(1)} h"
    return name.replace("_", " ")


def scenario_color(name: str) -> str:
    if name in COLORS:
        return COLORS[name]
    if name.startswith("b1_shift_"):
        return "#11866F"
    if name.startswith("b2_cap_"):
        return "#D39128"
    if name.startswith("b1_runtime_"):
        return "#6F4E9C"
    if name.startswith("adaptive_"):
        return COLORS.get(name, "#2A9D8F")
    if name.startswith("dynamic_forecast_"):
        return COLORS.get(name, "#845EC2")
    return "#6A6F7A"


def numbered_legend(policies: pd.DataFrame) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=scenario_color(name),
            markeredgecolor="white",
            markersize=9,
            label=f"{index}. {scenario_label(name)}",
        )
        for index, name in enumerate(policies["scenario"], start=1)
    ]


def policy_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison.loc[~comparison["scenario"].eq("baseline")]


def scenario_directory(scenario_root: Path, name: str) -> Path:
    direct = scenario_root / name
    if direct.exists():
        return direct
    nested_policy = scenario_root / "policies" / name
    if nested_policy.exists():
        return nested_policy
    return direct


def workload_epoch(profile: pd.DataFrame) -> pd.Timestamp:
    if "event_epoch_utc" in profile:
        epochs = pd.to_datetime(profile["event_epoch_utc"], utc=True, errors="coerce").dropna().unique()
        if len(epochs) != 1:
            raise ValueError("Profile must contain exactly one event_epoch_utc")
        return pd.Timestamp(epochs[0])
    scope = profile
    if "is_warmup" in profile:
        warmup = profile["is_warmup"].fillna(False).astype(bool)
        if warmup.any():
            scope = profile.loc[warmup]
    submit = pd.to_datetime(scope["source_submit_utc"], utc=True)
    offset = pd.to_timedelta(pd.to_numeric(scope["submit_dt_s"]), unit="s")
    return (submit - offset).median()


def plot_carbon_and_releases(
    scenario_root: Path,
    scenario_names: list[str],
    carbon_path: Path,
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
    output: Path,
) -> None:
    carbon = pd.read_csv(carbon_path, parse_dates=["from_utc", "to_utc"])
    carbon["from_utc"] = pd.to_datetime(carbon["from_utc"], utc=True)
    profiles = {}
    for name in scenario_names:
        profile_path = scenario_directory(scenario_root, name) / "workload_profile.csv"
        if not profile_path.exists():
            continue
        profile = pd.read_csv(profile_path)
        epoch = workload_epoch(profile)
        profile["release_utc"] = epoch + pd.to_timedelta(profile["release_dt_s"], unit="s")
        profiles[name] = profile

    start = horizon_start.floor("30min")
    latest_release = max(
        pd.to_datetime(profile["release_utc"], utc=True).max()
        for profile in profiles.values()
    )
    shown_end = min(horizon_end.ceil("30min"), latest_release.ceil("30min") + pd.Timedelta(minutes=30))
    carbon = carbon[(carbon["from_utc"] >= start) & (carbon["from_utc"] < shown_end)]
    bins = pd.date_range(start, shown_end, freq="30min")

    fig, (ax_ci, ax_jobs) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]}
    )
    ax_ci.plot(
        carbon["from_utc"], carbon["intensity_gco2_per_kwh"],
        color="#273444", linewidth=2.2
    )
    ax_ci.fill_between(
        carbon["from_utc"], carbon["intensity_gco2_per_kwh"],
        color="#DDE8F2", alpha=0.8
    )
    ax_ci.set_ylabel("Carbon intensity\n(gCO$_2$e/kWh)")
    ax_ci.set_title("Carbon signal and job admission times during the decision window", loc="left", fontsize=16, weight="bold")
    ax_ci.grid(axis="y", color="#E6E9ED")

    centers = bins[:-1] + pd.Timedelta(minutes=15)
    for name, profile in profiles.items():
        counts, _ = np.histogram(profile["release_utc"].astype("int64"), bins=bins.astype("int64"))
        ax_jobs.step(
            centers, counts, where="mid", label=scenario_label(name),
            color=scenario_color(name), linewidth=2
        )
    ax_jobs.set_ylabel("Jobs admitted\nper 30 min")
    ax_jobs.set_xlabel("UTC")
    ax_jobs.grid(axis="y", color="#E6E9ED")
    ax_jobs.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    for axis in [ax_ci, ax_jobs]:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output / "01_carbon_signal_and_releases.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(comparison: pd.DataFrame, output: Path) -> None:
    policies = policy_rows(comparison)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for index, row in enumerate(policies.itertuples(index=False), start=1):
        x = row.total_wait_p95_change_s / 3600
        y = row.dynamic_carbon_reduction_pct_capacity_weighted
        ax.scatter(
            x,
            y,
            s=150,
            color=scenario_color(row.scenario),
            edgecolor="white",
            linewidth=1.5,
        )
        ax.annotate(
            str(index),
            (x, y),
            xytext=NUMBER_OFFSETS[(index - 1) % len(NUMBER_OFFSETS)],
            textcoords="offset points",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            weight="bold",
            bbox={
                "boxstyle": "circle,pad=0.20",
                "facecolor": scenario_color(row.scenario),
                "edgecolor": "white",
                "linewidth": 0.8,
            },
        )
    ax.axhline(0, color="#AAB2BC", linewidth=1)
    ax.set_xlabel("Increase in P95 total wait (hours)")
    ax.set_ylabel("Controllable carbon reduction (%)")
    ax.set_title("Carbon reduction versus waiting-time cost", loc="left", fontsize=16, weight="bold")
    ax.legend(
        handles=numbered_legend(policies),
        frameon=False,
        fontsize=8.5,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    ax.grid(color="#E6E9ED")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    fig.savefig(output / "02_carbon_wait_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wait_percentile_tradeoff(comparison: pd.DataFrame, output: Path) -> None:
    policies = policy_rows(comparison)
    baseline = comparison.loc[comparison["scenario"].eq("baseline")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    panels = [
        ("total_user_wait_p50_s", "Increase in P50 total wait (seconds)", 1.0),
        ("total_user_wait_p95_s", "Increase in P95 total wait (hours)", 3600.0),
    ]
    for axis, (column, xlabel, divisor) in zip(axes, panels):
        baseline_wait = float(baseline[column])
        for index, row in enumerate(policies.itertuples(index=False), start=1):
            x = (float(getattr(row, column)) - baseline_wait) / divisor
            y = row.dynamic_carbon_reduction_pct_capacity_weighted
            axis.scatter(
                x,
                y,
                s=125,
                color=scenario_color(row.scenario),
                edgecolor="white",
                linewidth=1.3,
            )
            axis.annotate(
                str(index),
                (x, y),
                xytext=NUMBER_OFFSETS[(index - 1) % len(NUMBER_OFFSETS)],
                textcoords="offset points",
                ha="center",
                va="center",
                color="white",
                fontsize=7,
                weight="bold",
                bbox={
                    "boxstyle": "circle,pad=0.18",
                    "facecolor": scenario_color(row.scenario),
                    "edgecolor": "white",
                    "linewidth": 0.7,
                },
            )
        axis.axhline(0, color="#AAB2BC", linewidth=1)
        axis.set_xlabel(xlabel)
        axis.grid(color="#E6E9ED")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Controllable carbon reduction (%)")
    axes[1].legend(
        handles=numbered_legend(policies),
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.suptitle(
        "Carbon reduction versus median and tail waiting cost",
        x=0.06,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.80, 0.94))
    fig.savefig(
        output / "05_carbon_wait_tradeoff_p50_p95.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_model_sensitivity(comparison: pd.DataFrame, output: Path) -> None:
    policies = policy_rows(comparison)
    y = np.arange(len(policies))
    width = 0.34
    central = policies["dynamic_carbon_reduction_pct_capacity_weighted"].to_numpy()
    upper = policies["dynamic_carbon_reduction_pct_node_request_upper"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, max(6.2, len(policies) * 0.55)))
    bars1 = ax.barh(y - width / 2, central, width, label="Capacity-weighted", color="#2457A7")
    bars2 = ax.barh(y + width / 2, upper, width, label="Node-request upper proxy", color="#65AFA1")
    ax.bar_label(bars1, fmt="%.3f%%", padding=4, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f%%", padding=4, fontsize=8)
    ax.set_yticks(y, [scenario_label(name) for name in policies["scenario"]])
    ax.invert_yaxis()
    ax.set_xlabel("Controllable carbon reduction (%)")
    ax.set_title("Result depends on the utilization proxy", loc="left", fontsize=16, weight="bold")
    ax.legend(frameon=False)
    ax.axvline(0, color="#AAB2BC", linewidth=1)
    ax.grid(axis="x", color="#E6E9ED")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "03_utilization_model_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_validation(validation_path: Path, output: Path) -> None:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    labels = ["P50 wait", "P95 wait"]
    source = [
        validation["source_eligible_wait_p50_s"],
        validation["source_eligible_wait_p95_s"],
    ]
    simulated = [
        validation["sim_scheduler_wait_p50_s"],
        validation["sim_scheduler_wait_p95_s"],
    ]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bars1 = ax.bar(x - width / 2, source, width, label="Real trace", color="#2457A7")
    bars2 = ax.bar(x + width / 2, simulated, width, label="24-hour replay", color="#D39128")
    ax.bar_label(bars1, fmt="%.1f s", padding=4, fontsize=10)
    ax.bar_label(bars2, fmt="%.2f s", padding=4, fontsize=10)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Eligible-to-start wait (seconds, log scale)")
    ax.set_title("Execution check: the replay is not a calibrated queue model", loc="left", fontsize=16, weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", which="both", color="#E6E9ED")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "04_baseline_queue_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--carbon", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(args.comparison)
    horizon_start = pd.to_datetime(
        comparison["common_horizon_start_utc"].iloc[0], utc=True
    )
    horizon_end = pd.to_datetime(
        comparison["common_horizon_end_utc"].iloc[0], utc=True
    )
    plot_carbon_and_releases(
        args.scenario_root,
        comparison["scenario"].tolist(),
        args.carbon,
        horizon_start,
        horizon_end,
        args.output,
    )
    plot_tradeoff(comparison, args.output)
    plot_model_sensitivity(comparison, args.output)
    plot_validation(args.baseline_validation, args.output)
    plot_wait_percentile_tradeoff(comparison, args.output)
    print(f"Wrote five figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
