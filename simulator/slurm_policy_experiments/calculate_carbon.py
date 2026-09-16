#!/usr/bin/env python3
"""Estimate scenario energy and carbon with the documented E1 power model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


CLASS_CONFIG = {
    "cpu": {"nodes": 174, "cpus": 174 * 64, "gpus": 0},
    "a100": {"nodes": 18, "cpus": 18 * 48, "gpus": 18 * 4},
    "h100": {"nodes": 6, "cpus": 6 * 48, "gpus": 6 * 2},
    "h100_nvl": {"nodes": 4, "cpus": 4 * 96, "gpus": 4 * 4},
}
TOTAL_NODES = sum(item["nodes"] for item in CLASS_CONFIG.values())
UTILIZATION_MODELS = [
    "capacity_weighted",
    "allocated_node_distinct",
    "node_request_upper",
]
PARTITION_CLASS = {
    "interactive": "cpu",
    "sheffield": "cpu",
    "sheffield-8xlong": "cpu",
    "gpu": "a100",
    "gpu-h100": "h100",
    "gpu-h100-nvl": "h100_nvl",
}


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


def model_start_offsets(results: pd.DataFrame) -> pd.Series:
    """Retain the study's re-anchoring convention; negative log waits are clipped."""
    release = pd.to_numeric(results["release_dt_s"], errors="coerce")
    scheduler_wait = pd.to_numeric(
        results["sim_scheduler_wait_s"], errors="coerce"
    ).clip(lower=0)
    anchored = release + scheduler_wait
    fallback = pd.to_numeric(results["sim_start_offset_s"], errors="coerce")
    return anchored.where(anchored.notna(), fallback)


def load_scenario(path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    profile = pd.read_csv(path / "workload_profile.csv")
    results = pd.read_csv(path / "job_results.csv")
    if not results["terminal_status"].eq("completed").all():
        counts = results["terminal_status"].value_counts().to_dict()
        raise ValueError(f"Scenario {path.name} is not strictly complete: {counts}")
    epoch = workload_epoch(profile)
    start_offsets = model_start_offsets(results)
    results["model_start_utc"] = epoch + pd.to_timedelta(
        start_offsets, unit="s"
    )
    # This is the study's configured-duration model, not logged completion time.
    # Completion and event timestamps can differ materially; audit them separately.
    results["model_end_utc"] = results["model_start_utc"] + pd.to_timedelta(
        pd.to_numeric(results["runtime_s"]), unit="s"
    )
    results["hardware_class"] = results["partition"].map(PARTITION_CLASS)
    if results["hardware_class"].isna().any():
        missing = sorted(results.loc[results["hardware_class"].isna(), "partition"].unique())
        raise ValueError(f"Unmapped partitions: {missing}")
    return results, epoch


def interval_overlap_seconds(
    starts: pd.Series, ends: pd.Series, interval_start: pd.Timestamp, interval_end: pd.Timestamp
) -> np.ndarray:
    start_ns = (
        pd.to_datetime(starts, utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
    )
    end_ns = (
        pd.to_datetime(ends, utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
    )
    interval_start_ns = pd.to_datetime(interval_start, utc=True).value
    interval_end_ns = pd.to_datetime(interval_end, utc=True).value
    left = np.maximum(start_ns, interval_start_ns)
    right = np.minimum(end_ns, interval_end_ns)
    return np.maximum(right - left, 0) / 1_000_000_000


def expand_nodelist(value: object) -> tuple[str, ...]:
    """Expand the Stanage node-list forms emitted by the simulator."""
    text = str(value).strip()
    direct = re.fullmatch(r"(node|gpu)(\d+)", text)
    if direct:
        candidates = [text]
    else:
        bracket = re.fullmatch(r"(node|gpu)\[([^]]+)\]", text)
        if not bracket:
            return ()
        prefix, body = bracket.groups()
        candidates = []
        for component in body.split(","):
            if "-" in component:
                first, last = component.split("-", 1)
                width = max(len(first), len(last))
                candidates.extend(
                    f"{prefix}{number:0{width}d}"
                    for number in range(int(first), int(last) + 1)
                )
            else:
                candidates.append(f"{prefix}{component}")

    valid = []
    for candidate in candidates:
        match = re.fullmatch(r"(node|gpu)(\d+)", candidate)
        if not match:
            continue
        prefix, raw_number = match.groups()
        number = int(raw_number)
        if prefix == "node" and (
            1 <= number <= 150 or 201 <= number <= 212 or 301 <= number <= 312
        ):
            valid.append(f"node{number:03d}")
        if prefix == "gpu" and (
            1 <= number <= 18 or 21 <= number <= 26 or 31 <= number <= 34
        ):
            valid.append(f"gpu{number:02d}")
    return tuple(valid)


def allocated_node_utilization_for_interval(
    jobs: pd.DataFrame,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> float:
    """Return node-seconds after merging overlapping jobs on each host."""
    if "allocated_nodelist" not in jobs:
        return math.nan

    by_node: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for row in jobs.itertuples(index=False):
        nodes = expand_nodelist(getattr(row, "allocated_nodelist", ""))
        if not nodes:
            continue
        start = max(pd.Timestamp(row.model_start_utc), interval_start)
        end = min(pd.Timestamp(row.model_end_utc), interval_end)
        if end <= start:
            continue
        for node in nodes:
            by_node.setdefault(node, []).append((start, end))

    duration_s = (interval_end - interval_start).total_seconds()
    if duration_s <= 0:
        return 0.0
    occupied_node_s = 0.0
    for intervals in by_node.values():
        intervals.sort()
        merged_start, merged_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= merged_end:
                merged_end = max(merged_end, end)
            else:
                occupied_node_s += (merged_end - merged_start).total_seconds()
                merged_start, merged_end = start, end
        occupied_node_s += (merged_end - merged_start).total_seconds()
    return min(occupied_node_s / (duration_s * TOTAL_NODES), 1.0)


def utilization_for_interval(
    jobs: pd.DataFrame, interval_start: pd.Timestamp, interval_end: pd.Timestamp
) -> dict[str, float]:
    duration_s = (interval_end - interval_start).total_seconds()
    overlap = interval_overlap_seconds(
        jobs["model_start_utc"], jobs["model_end_utc"], interval_start, interval_end
    )
    result: dict[str, float] = {}
    weighted = 0.0

    for class_name, config in CLASS_CONFIG.items():
        mask = jobs["hardware_class"].eq(class_name).to_numpy()
        if not mask.any():
            class_util = 0.0
        else:
            cpu_fraction = float(
                np.sum(overlap[mask] * pd.to_numeric(jobs.loc[mask, "cpus"]).to_numpy(float))
                / (duration_s * config["cpus"])
            )
            gpu_fraction = 0.0
            if config["gpus"]:
                gpu_fraction = float(
                    np.sum(
                        overlap[mask]
                        * pd.to_numeric(jobs.loc[mask, "scheduled_gpus"]).to_numpy(float)
                    )
                    / (duration_s * config["gpus"])
                )
            class_util = min(1.0, max(cpu_fraction, gpu_fraction))
        result[f"util_{class_name}"] = class_util
        weighted += config["nodes"] / TOTAL_NODES * class_util

    node_upper = float(
        np.sum(overlap * pd.to_numeric(jobs["nodes"]).to_numpy(float))
        / (duration_s * TOTAL_NODES)
    )
    result["util_capacity_weighted"] = min(1.0, weighted)
    result["util_allocated_node_distinct"] = allocated_node_utilization_for_interval(
        jobs, interval_start, interval_end
    )
    result["util_node_request_upper"] = min(1.0, node_upper)
    return result


def build_intervals(
    jobs: pd.DataFrame,
    carbon: pd.DataFrame,
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
    idle_power_kw: float,
    regular_power_kw: float,
) -> pd.DataFrame:
    rows = []
    dynamic_range_kw = regular_power_kw - idle_power_kw
    if dynamic_range_kw < 0:
        raise ValueError("regular power must be at least idle power")
    relevant = carbon.loc[
        (carbon["to_utc"] > horizon_start) & (carbon["from_utc"] < horizon_end)
    ]
    if relevant.empty:
        raise ValueError("Carbon data does not cover the common analysis horizon")

    # Every part of the common horizon must be priced exactly once.
    cursor = horizon_start
    for interval in relevant.sort_values("from_utc").itertuples(index=False):
        start = max(interval.from_utc, horizon_start)
        end = min(interval.to_utc, horizon_end)
        if start != cursor or end <= start:
            raise ValueError("Carbon data has a gap or overlap in the common analysis horizon")
        cursor = end
    if cursor != horizon_end:
        raise ValueError("Carbon data does not cover the end of the common analysis horizon")

    for carbon_row in relevant.itertuples(index=False):
        start = max(carbon_row.from_utc, horizon_start)
        end = min(carbon_row.to_utc, horizon_end)
        duration_h = (end - start).total_seconds() / 3600
        if duration_h <= 0:
            continue
        utilization = utilization_for_interval(jobs, start, end)
        intensity = float(carbon_row.intensity_gco2_per_kwh)
        row = {
            "from_utc": start,
            "to_utc": end,
            "duration_h": duration_h,
            "intensity_gco2_per_kwh": intensity,
            **utilization,
        }
        for model in UTILIZATION_MODELS:
            util = utilization[f"util_{model}"]
            if math.isnan(util):
                continue
            dynamic_power = dynamic_range_kw * util
            dynamic_energy = dynamic_power * duration_h
            whole_energy = (idle_power_kw + dynamic_power) * duration_h
            row[f"dynamic_power_kw_{model}"] = dynamic_power
            row[f"dynamic_energy_kwh_{model}"] = dynamic_energy
            row[f"dynamic_carbon_kg_{model}"] = dynamic_energy * intensity / 1000
            row[f"whole_energy_kwh_{model}"] = whole_energy
            row[f"whole_carbon_kg_{model}"] = whole_energy * intensity / 1000
        rows.append(row)
    return pd.DataFrame(rows)


def p(values: pd.Series, quantile: float) -> float:
    return float(pd.to_numeric(values, errors="coerce").dropna().quantile(quantile))


def jain_index(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if numeric.size == 0:
        return None
    denominator = numeric.size * float(np.square(numeric).sum())
    return float(numeric.sum() ** 2 / denominator) if denominator else 1.0


def summarize_scenario(
    name: str,
    jobs: pd.DataFrame,
    intervals: pd.DataFrame,
    horizon_start: pd.Timestamp,
    horizon_end: pd.Timestamp,
) -> dict[str, object]:
    if "is_warmup" in jobs:
        warmup_mask = jobs["is_warmup"].fillna(False).astype(bool)
        evaluation_jobs = jobs.loc[~warmup_mask]
    else:
        warmup_mask = pd.Series(False, index=jobs.index)
        evaluation_jobs = jobs
    if evaluation_jobs.empty:
        raise ValueError(f"Scenario {name} has no evaluation jobs")

    evaluation_jobs = evaluation_jobs.copy()
    if "runtime_s" in evaluation_jobs:
        runtime_s = pd.to_numeric(evaluation_jobs["runtime_s"], errors="coerce")
    else:
        runtime_s = (
            pd.to_datetime(evaluation_jobs["model_end_utc"], utc=True)
            - pd.to_datetime(evaluation_jobs["model_start_utc"], utc=True)
        ).dt.total_seconds()
    total_wait_s = pd.to_numeric(
        evaluation_jobs["sim_total_user_wait_s"], errors="coerce"
    )
    policy_delay_s = pd.to_numeric(
        evaluation_jobs["sim_policy_delay_s"], errors="coerce"
    )
    turnaround_s = runtime_s + total_wait_s
    bounded_slowdown = turnaround_s / np.maximum(runtime_s, 60.0)
    evaluation_jobs["_bounded_slowdown"] = bounded_slowdown

    if "allowed_wait_budget_s" in evaluation_jobs:
        wait_budget_s = pd.to_numeric(
            evaluation_jobs["allowed_wait_budget_s"], errors="coerce"
        ).fillna(0.0)
        wait_budget_violations = int((policy_delay_s > wait_budget_s + 1e-6).sum())
    else:
        wait_budget_violations = 0

    user_wait_p95 = None
    user_wait_max = None
    if "user_id" in evaluation_jobs:
        per_user_wait = evaluation_jobs.assign(_wait=total_wait_s).groupby("user_id")[
            "_wait"
        ].mean()
        user_wait_p95 = p(per_user_wait, 0.95)
        user_wait_max = float(per_user_wait.max())

    summary: dict[str, object] = {
        "scenario": name,
        "jobs": len(jobs),
        "evaluation_jobs": len(evaluation_jobs),
        "warmup_jobs": int(warmup_mask.sum()),
        "common_horizon_start_utc": horizon_start.isoformat(),
        "common_horizon_end_utc": horizon_end.isoformat(),
        "common_horizon_hours": (horizon_end - horizon_start).total_seconds() / 3600,
        "scheduler_wait_p50_s": p(evaluation_jobs["sim_scheduler_wait_s"], 0.50),
        "scheduler_wait_p95_s": p(evaluation_jobs["sim_scheduler_wait_s"], 0.95),
        "policy_delay_p50_s": p(evaluation_jobs["sim_policy_delay_s"], 0.50),
        "policy_delay_p95_s": p(evaluation_jobs["sim_policy_delay_s"], 0.95),
        "policy_delay_p99_s": p(evaluation_jobs["sim_policy_delay_s"], 0.99),
        "policy_delay_max_s": float(policy_delay_s.max()),
        "total_user_wait_p50_s": p(evaluation_jobs["sim_total_user_wait_s"], 0.50),
        "total_user_wait_p95_s": p(evaluation_jobs["sim_total_user_wait_s"], 0.95),
        "total_user_wait_p99_s": p(evaluation_jobs["sim_total_user_wait_s"], 0.99),
        "total_user_wait_max_s": float(total_wait_s.max()),
        "turnaround_p95_s": p(turnaround_s, 0.95),
        "bounded_slowdown_p50": p(bounded_slowdown, 0.50),
        "bounded_slowdown_p95": p(bounded_slowdown, 0.95),
        "bounded_slowdown_p99": p(bounded_slowdown, 0.99),
        "bounded_slowdown_max": float(bounded_slowdown.max()),
        "bounded_slowdown_jain_fairness": jain_index(bounded_slowdown),
        "per_user_mean_wait_p95_s": user_wait_p95,
        "per_user_mean_wait_max_s": user_wait_max,
        "wait_budget_violations": wait_budget_violations,
        "jobs_delayed_longer_than_runtime": int((policy_delay_s > runtime_s).sum()),
        "jobs_with_policy_delay": int(
            (evaluation_jobs["sim_policy_delay_s"] > 0).sum()
        ),
        "makespan_hours": (
            pd.to_datetime(jobs["model_end_utc"], utc=True).max()
            - pd.to_datetime(jobs["model_start_utc"], utc=True).min()
        ).total_seconds()
        / 3600,
    }
    available_models = [
        model
        for model in UTILIZATION_MODELS
        if f"util_{model}" in intervals and intervals[f"util_{model}"].notna().any()
    ]
    for model in available_models:
        summary[f"mean_utilization_{model}"] = float(
            np.average(intervals[f"util_{model}"], weights=intervals["duration_h"])
        )
        summary[f"peak_utilization_{model}"] = float(intervals[f"util_{model}"].max())
        for quantity in ["dynamic_energy_kwh", "dynamic_carbon_kg", "whole_energy_kwh", "whole_carbon_kg"]:
            summary[f"{quantity}_{model}"] = float(intervals[f"{quantity}_{model}"].sum())
    return summary


def load_carbon(path: Path) -> pd.DataFrame:
    carbon = pd.read_csv(path)
    carbon["from_utc"] = pd.to_datetime(carbon["from_utc"], utc=True)
    carbon["to_utc"] = pd.to_datetime(carbon["to_utc"], utc=True)
    carbon["intensity_gco2_per_kwh"] = pd.to_numeric(
        carbon["intensity_gco2_per_kwh"], errors="coerce"
    )
    carbon.dropna(subset=["from_utc", "to_utc", "intensity_gco2_per_kwh"], inplace=True)
    return carbon.sort_values("from_utc").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", type=Path, required=True)
    parser.add_argument("--carbon", type=Path, required=True)
    parser.add_argument("--idle-power-kw", type=float, default=140.0)
    parser.add_argument("--regular-power-kw", type=float, default=195.0)
    parser.add_argument("--horizon-start-utc")
    parser.add_argument("--horizon-end-utc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded: dict[str, pd.DataFrame] = {}
    epochs = []
    for scenario in args.scenario:
        jobs, epoch = load_scenario(scenario)
        loaded[scenario.name] = jobs
        epochs.append(epoch)
    if max(epochs) - min(epochs) > pd.Timedelta(seconds=1):
        raise ValueError("Scenarios do not share the same source workload epoch")

    carbon = load_carbon(args.carbon)
    horizon_start = min(
        pd.to_datetime(jobs["model_start_utc"], utc=True).min() for jobs in loaded.values()
    ).floor("30min")
    horizon_end = max(
        pd.to_datetime(jobs["model_end_utc"], utc=True).max() for jobs in loaded.values()
    ).ceil("30min")
    if args.horizon_start_utc:
        horizon_start = pd.Timestamp(args.horizon_start_utc)
        horizon_start = (
            horizon_start.tz_localize("UTC")
            if horizon_start.tzinfo is None
            else horizon_start.tz_convert("UTC")
        )
    if args.horizon_end_utc:
        horizon_end = pd.Timestamp(args.horizon_end_utc)
        horizon_end = (
            horizon_end.tz_localize("UTC")
            if horizon_end.tzinfo is None
            else horizon_end.tz_convert("UTC")
        )
    if horizon_end <= horizon_start:
        raise ValueError("analysis horizon end must be after its start")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, jobs in loaded.items():
        intervals = build_intervals(
            jobs,
            carbon,
            horizon_start,
            horizon_end,
            args.idle_power_kw,
            args.regular_power_kw,
        )
        intervals.to_csv(args.output / f"{name}_carbon_intervals.csv", index=False)
        summaries.append(summarize_scenario(name, jobs, intervals, horizon_start, horizon_end))

    comparison = pd.DataFrame(summaries)
    baseline_matches = comparison.loc[comparison["scenario"].eq("baseline")]
    if baseline_matches.empty:
        raise ValueError("One scenario directory must be named baseline")
    baseline = baseline_matches.iloc[0]
    available_models = [
        model
        for model in UTILIZATION_MODELS
        if f"dynamic_carbon_kg_{model}" in comparison
    ]
    for model in available_models:
        for scope in ["dynamic", "whole"]:
            column = f"{scope}_carbon_kg_{model}"
            comparison[f"{scope}_carbon_reduction_kg_{model}"] = baseline[column] - comparison[column]
            comparison[f"{scope}_carbon_reduction_pct_{model}"] = np.where(
                baseline[column] != 0,
                (baseline[column] - comparison[column]) / baseline[column] * 100,
                np.nan,
            )
    comparison["total_wait_p95_change_s"] = (
        comparison["total_user_wait_p95_s"] - baseline["total_user_wait_p95_s"]
    )
    comparison.to_csv(args.output / "scenario_comparison.csv", index=False)

    metadata = {
        "model": "E1 cluster-level power scenario",
        "idle_power_kw": args.idle_power_kw,
        "regular_power_kw": args.regular_power_kw,
        "dynamic_power_range_kw": args.regular_power_kw - args.idle_power_kw,
        "power_boundary": (
            "Fred confirmed that 140/195 kW covers all equipment inside the "
            "Stanage cage, including compute nodes, storage and network switches; "
            "PUE is not applied because facility overhead outside that cage is not "
            "part of the stated measurement boundary"
        ),
        "carbon_intensity": "NESO regional forecast/modelled estimate for region 5, not a site meter",
        "primary_utilization_proxy": "Capacity-normalized CPU/GPU demand, weighted by working node counts",
        "allocated_node_proxy": (
            "Distinct simulator-allocated hosts per interval; overlapping jobs sharing "
            "one host are counted once"
        ),
        "sensitivity_proxy": "Requested nodes divided by 202; an upper proxy because nodes may be shared",
        "common_horizon_start_utc": horizon_start.isoformat(),
        "common_horizon_end_utc": horizon_end.isoformat(),
        "comparison": comparison.to_dict(orient="records"),
    }
    (args.output / "carbon_model_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
