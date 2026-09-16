#!/usr/bin/env python3
"""Create carbon-aware Slurm simulator scenarios from a baseline profile."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_empirical_workload", ROOT / "generate_empirical_workload.py"
)
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
assert GENERATOR_SPEC.loader is not None
GENERATOR_SPEC.loader.exec_module(GENERATOR)

HISTORY_SPEC = importlib.util.spec_from_file_location(
    "build_history_quantile_model", ROOT / "build_history_quantile_model.py"
)
HISTORY = importlib.util.module_from_spec(HISTORY_SPEC)
assert HISTORY_SPEC.loader is not None
HISTORY_SPEC.loader.exec_module(HISTORY)


PARTITION_CLASS = {
    "interactive": "cpu",
    "sheffield": "cpu",
    "sheffield-8xlong": "cpu",
    "gpu": "a100",
    "gpu-h100": "h100",
    "gpu-h100-nvl": "h100_nvl",
}
CLASS_CAPACITY = {
    "cpu": {"cpus": 174 * 64, "gpus": 0},
    "a100": {"cpus": 18 * 48, "gpus": 18 * 4},
    "h100": {"cpus": 6 * 48, "gpus": 6 * 2},
    "h100_nvl": {"cpus": 4 * 96, "gpus": 4 * 4},
}
CLASS_NODES = {"cpu": 174, "a100": 18, "h100": 6, "h100_nvl": 4}
TOTAL_NODES = sum(CLASS_NODES.values())


def load_carbon(path: Path) -> pd.DataFrame:
    carbon = pd.read_csv(path)
    carbon["from_utc"] = pd.to_datetime(carbon["from_utc"], utc=True)
    carbon["to_utc"] = pd.to_datetime(carbon["to_utc"], utc=True)
    carbon["intensity_gco2_per_kwh"] = pd.to_numeric(
        carbon["intensity_gco2_per_kwh"], errors="coerce"
    )
    carbon.dropna(subset=["from_utc", "to_utc", "intensity_gco2_per_kwh"], inplace=True)
    carbon.sort_values("from_utc", inplace=True)
    carbon.reset_index(drop=True, inplace=True)
    return carbon


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


def carbon_row_at(carbon: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
    matches = carbon.loc[(carbon["from_utc"] <= timestamp) & (timestamp < carbon["to_utc"])]
    if matches.empty:
        raise ValueError(f"No carbon-intensity interval covers {timestamp.isoformat()}")
    return matches.iloc[-1]


def flexible_mask(profile: pd.DataFrame, flex_fraction: float) -> pd.Series:
    mask = (
        pd.to_numeric(profile["flexibility_score"], errors="coerce").fillna(1.0)
        < flex_fraction
    ) & ~profile["partition"].eq("interactive")
    if "is_warmup" in profile:
        mask &= ~profile["is_warmup"].fillna(False).astype(bool)
    return mask


def runtime_average_intensity(
    carbon: pd.DataFrame, start: pd.Timestamp, runtime_s: float
) -> float:
    end = start + pd.Timedelta(seconds=runtime_s)
    relevant = carbon.loc[(carbon["to_utc"] > start) & (carbon["from_utc"] < end)]
    weighted_intensity = 0.0
    covered_s = 0.0
    for row in relevant.itertuples(index=False):
        overlap_start = max(start, row.from_utc)
        overlap_end = min(end, row.to_utc)
        overlap_s = max((overlap_end - overlap_start).total_seconds(), 0.0)
        weighted_intensity += overlap_s * float(row.intensity_gco2_per_kwh)
        covered_s += overlap_s
    if covered_s + 1e-3 < runtime_s:
        raise ValueError(
            f"Carbon data covers {covered_s:.1f}s of a {runtime_s:.1f}s job starting at {start.isoformat()}"
        )
    return weighted_intensity / covered_s


def candidate_release_times(
    carbon: pd.DataFrame,
    eligible: pd.Timestamp,
    deadline: pd.Timestamp,
    runtime_s: float,
) -> list[pd.Timestamp]:
    carbon_end = carbon["to_utc"].max()
    candidates = [eligible]
    candidates.extend(
        timestamp
        for timestamp in carbon.loc[
            (carbon["from_utc"] > eligible) & (carbon["from_utc"] <= deadline),
            "from_utc",
        ]
        if timestamp + pd.Timedelta(seconds=runtime_s) <= carbon_end
    )
    return sorted(set(candidates))


def best_runtime_release_time(
    carbon: pd.DataFrame,
    eligible: pd.Timestamp,
    deadline: pd.Timestamp,
    runtime_s: float,
) -> tuple[pd.Timestamp, float, float]:
    options = [
        (candidate, runtime_average_intensity(carbon, candidate, runtime_s))
        for candidate in candidate_release_times(carbon, eligible, deadline, runtime_s)
    ]
    release, target_intensity = min(options, key=lambda item: (item[1], item[0]))
    before = runtime_average_intensity(carbon, eligible, runtime_s)
    return release, before, target_intensity


def job_capacity_fraction(row: pd.Series) -> tuple[str, float]:
    class_name = PARTITION_CLASS.get(str(row["partition"]))
    if class_name is None:
        raise ValueError(f"No capacity class for partition {row['partition']}")
    capacity = CLASS_CAPACITY[class_name]
    cpu_fraction = float(row["cpus"]) / capacity["cpus"]
    gpu_fraction = 0.0
    if capacity["gpus"]:
        gpu_fraction = float(row.get("scheduled_gpus", 0)) / capacity["gpus"]
    return class_name, max(cpu_fraction, gpu_fraction)


def projected_class_load(
    planned: list[tuple[float, float, str, float]],
    class_name: str,
    start_offset: float,
    runtime_s: float,
    job_fraction: float,
) -> tuple[float, float]:
    end_offset = start_offset + runtime_s
    breakpoints = {start_offset, end_offset}
    for planned_start, planned_end, planned_class, _ in planned:
        if planned_class != class_name:
            continue
        if start_offset < planned_start < end_offset:
            breakpoints.add(planned_start)
        if start_offset < planned_end < end_offset:
            breakpoints.add(planned_end)
    points = sorted(breakpoints)
    weighted_load = 0.0
    peak_load = job_fraction
    for left, right in zip(points, points[1:]):
        midpoint = (left + right) / 2
        load = job_fraction + sum(
            fraction
            for planned_start, planned_end, planned_class, fraction in planned
            if planned_class == class_name and planned_start <= midpoint < planned_end
        )
        weighted_load += load * (right - left)
        peak_load = max(peak_load, load)
    mean_load = weighted_load / runtime_s if runtime_s > 0 else peak_load
    return mean_load, peak_load


def marginal_runtime_carbon_cost(
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    planned: list[tuple[float, float, str, float, float]],
    class_name: str,
    start_offset: float,
    runtime_s: float,
    class_fraction: float,
    node_fraction: float,
) -> tuple[float, float, float, float]:
    """Estimate the job's marginal carbon under both utilization proxies."""
    end_offset = start_offset + runtime_s
    start = epoch + pd.Timedelta(seconds=start_offset)
    end = epoch + pd.Timedelta(seconds=end_offset)
    relevant_carbon = carbon.loc[
        (carbon["to_utc"] > start) & (carbon["from_utc"] < end)
    ]
    covered_s = sum(
        max(
            (
                min(end, row.to_utc) - max(start, row.from_utc)
            ).total_seconds(),
            0.0,
        )
        for row in relevant_carbon.itertuples(index=False)
    )
    if covered_s + 1e-3 < runtime_s:
        raise ValueError(
            f"Carbon data covers {covered_s:.1f}s of a {runtime_s:.1f}s job "
            f"starting at {start.isoformat()}"
        )

    overlapping = [
        item for item in planned if item[0] < end_offset and item[1] > start_offset
    ]
    breakpoints = {start_offset, end_offset}
    for row in relevant_carbon.itertuples(index=False):
        for timestamp in (row.from_utc, row.to_utc):
            offset = (timestamp - epoch).total_seconds()
            if start_offset < offset < end_offset:
                breakpoints.add(offset)
    for planned_start, planned_end, _, _, _ in overlapping:
        if start_offset < planned_start < end_offset:
            breakpoints.add(planned_start)
        if start_offset < planned_end < end_offset:
            breakpoints.add(planned_end)

    capacity_cost = 0.0
    node_cost = 0.0
    weighted_class_load = 0.0
    peak_class_load = class_fraction
    points = sorted(breakpoints)
    for left, right in zip(points, points[1:]):
        midpoint = (left + right) / 2
        timestamp = epoch + pd.Timedelta(seconds=midpoint)
        intensity = float(carbon_row_at(carbon, timestamp)["intensity_gco2_per_kwh"])
        existing_class_load = sum(
            planned_fraction
            for planned_start, planned_end, planned_class, planned_fraction, _ in overlapping
            if planned_class == class_name and planned_start <= midpoint < planned_end
        )
        existing_node_load = sum(
            planned_node_fraction
            for planned_start, planned_end, _, _, planned_node_fraction in overlapping
            if planned_start <= midpoint < planned_end
        )
        target_class_load = existing_class_load + class_fraction
        marginal_class_load = (
            min(1.0, target_class_load) - min(1.0, existing_class_load)
        )
        marginal_node_load = (
            min(1.0, existing_node_load + node_fraction)
            - min(1.0, existing_node_load)
        )
        duration_h = (right - left) / 3600
        capacity_cost += (
            CLASS_NODES[class_name]
            / TOTAL_NODES
            * marginal_class_load
            * duration_h
            * intensity
        )
        node_cost += marginal_node_load * duration_h * intensity
        weighted_class_load += target_class_load * (right - left)
        peak_class_load = max(peak_class_load, target_class_load)

    mean_class_load = (
        weighted_class_load / runtime_s if runtime_s > 0 else peak_class_load
    )
    return capacity_cost, node_cost, mean_class_load, peak_class_load


def interval_job_load(
    intervals: pd.DataFrame,
    epoch: pd.Timestamp,
    start_offset: float,
    runtime_s: float,
    fraction: float,
) -> np.ndarray:
    """Return a job's average normalized load in each carbon interval."""
    start = epoch + pd.Timedelta(seconds=start_offset)
    end = start + pd.Timedelta(seconds=runtime_s)
    left = intervals["from_utc"].where(intervals["from_utc"] > start, start)
    right = intervals["to_utc"].where(intervals["to_utc"] < end, end)
    overlap_s = (right - left).dt.total_seconds().clip(lower=0).to_numpy(float)
    duration_s = (intervals["to_utc"] - intervals["from_utc"]).dt.total_seconds()
    return fraction * overlap_s / duration_s.to_numpy(float)


def build_schedule_forecast(
    profile: pd.DataFrame,
    baseline_results: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    runtime_estimate_factor: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build interval loads from a baseline scheduler run."""
    required = {"sim_job_id", "release_dt_s", "sim_scheduler_wait_s"}
    missing = required - set(baseline_results.columns)
    if missing:
        raise ValueError(f"Schedule forecast is missing columns: {sorted(missing)}")
    if baseline_results["sim_job_id"].duplicated().any():
        raise ValueError("Schedule forecast contains duplicate sim_job_id values")

    starts = baseline_results.set_index("sim_job_id").apply(
        lambda row: float(row["release_dt_s"])
        + max(float(row["sim_scheduler_wait_s"]), 0.0),
        axis=1,
    )
    profile_ids = set(profile["sim_job_id"])
    if profile_ids != set(starts.index):
        missing_ids = sorted(profile_ids - set(starts.index))[:5]
        extra_ids = sorted(set(starts.index) - profile_ids)[:5]
        raise ValueError(
            "Schedule forecast job IDs do not match the profile "
            f"(missing={missing_ids}, extra={extra_ids})"
        )

    forecast = carbon[["from_utc", "to_utc", "intensity_gco2_per_kwh"]].copy()
    for class_name in CLASS_CAPACITY:
        forecast[f"load_{class_name}"] = 0.0
    forecast["load_node"] = 0.0

    for row in profile.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        class_name, class_fraction = job_capacity_fraction(row_series)
        start_offset = starts[str(row.sim_job_id)]
        runtime_s = float(row.runtime_s) * runtime_estimate_factor
        forecast[f"load_{class_name}"] += interval_job_load(
            forecast, epoch, start_offset, runtime_s, class_fraction
        )
        forecast["load_node"] += interval_job_load(
            forecast,
            epoch,
            start_offset,
            runtime_s,
            float(row.nodes) / TOTAL_NODES,
        )
    return forecast, starts.to_dict()


def forecast_marginal_carbon_cost(
    forecast: pd.DataFrame,
    epoch: pd.Timestamp,
    class_name: str,
    current_start_offset: float,
    candidate_start_offset: float,
    runtime_s: float,
    class_fraction: float,
    node_fraction: float,
    forecast_contains_current: bool = True,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Estimate one job's marginal cost against the full schedule forecast."""
    current_class = interval_job_load(
        forecast, epoch, current_start_offset, runtime_s, class_fraction
    )
    current_node = interval_job_load(
        forecast, epoch, current_start_offset, runtime_s, node_fraction
    )
    candidate_class = interval_job_load(
        forecast, epoch, candidate_start_offset, runtime_s, class_fraction
    )
    candidate_node = interval_job_load(
        forecast, epoch, candidate_start_offset, runtime_s, node_fraction
    )
    background_class = forecast[f"load_{class_name}"].to_numpy(float)
    background_node = forecast["load_node"].to_numpy(float)
    if forecast_contains_current:
        background_class = np.maximum(background_class - current_class, 0.0)
        background_node = np.maximum(background_node - current_node, 0.0)
    marginal_class = np.minimum(1.0, background_class + candidate_class) - np.minimum(
        1.0, background_class
    )
    marginal_node = np.minimum(1.0, background_node + candidate_node) - np.minimum(
        1.0, background_node
    )
    duration_h = (
        (forecast["to_utc"] - forecast["from_utc"]).dt.total_seconds().to_numpy(float)
        / 3600
    )
    intensity = forecast["intensity_gco2_per_kwh"].to_numpy(float)
    capacity_cost = float(
        np.sum(
            CLASS_NODES[class_name]
            / TOTAL_NODES
            * marginal_class
            * duration_h
            * intensity
        )
    )
    node_cost = float(np.sum(marginal_node * duration_h * intensity))
    active = candidate_class > 0
    candidate_load = background_class + candidate_class
    mean_load = (
        float(np.average(candidate_load[active], weights=duration_h[active]))
        if active.any()
        else class_fraction
    )
    peak_load = float(candidate_load[active].max()) if active.any() else class_fraction
    return (
        capacity_cost,
        node_cost,
        mean_load,
        peak_load,
        candidate_class - current_class,
        candidate_node - current_node,
    )


def apply_dynamic_history(
    profile: pd.DataFrame,
    history_model: dict[str, object],
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_hours: float,
    wait_budget_ratio: float,
    min_carbon_saving_pct: float,
    min_node_saving_pct: float,
    wait_penalty: float,
    congestion_penalty: float,
    cap_fraction: float,
    min_carbon_return: float,
    runtime_quantile: float,
    load_quantile: float,
    queue_wait_quantile: float,
    max_queue_wait_hours: float,
    max_runtime_uncertainty_ratio: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Make online-feasible decisions from earlier-month aggregate quantiles."""
    result = profile.copy()
    result["is_flexible"] = flexible_mask(result, flex_fraction)
    result["b1_delay_s"] = 0.0
    result["b2_delay_s"] = 0.0
    result["runtime_intensity_before"] = np.nan
    result["runtime_intensity_after"] = np.nan
    result["expected_runtime_carbon_saving_pct"] = 0.0
    result["expected_capacity_carbon_saving_pct"] = 0.0
    result["expected_node_carbon_saving_pct"] = 0.0
    result["expected_risk_adjusted_carbon_saving_proxy"] = 0.0
    result["carbon_return_per_wait_hour"] = 0.0
    result["allowed_wait_budget_s"] = 0.0
    result["projected_mean_class_load"] = np.nan
    result["projected_peak_class_load"] = np.nan
    result["dynamic_wait_ratio"] = 0.0
    result["dynamic_load_growth"] = 0.0
    result["dynamic_marginal_utility"] = 0.0
    result["predicted_baseline_start_s"] = np.nan
    result["predicted_policy_start_s"] = np.nan
    result["forecast_incremental_wait_s"] = 0.0
    result["predicted_baseline_queue_wait_s"] = 0.0
    result["predicted_policy_queue_wait_s"] = 0.0
    result["queue_wait_model_level"] = ""
    result["queue_wait_model_count"] = 0

    runtime_predictions = HISTORY.predict_runtime(
        result, history_model, runtime_quantile
    )
    runtime_q50 = HISTORY.predict_runtime(result, history_model, 0.50)
    runtime_q90 = HISTORY.predict_runtime(result, history_model, 0.90)
    result["decision_runtime_s"] = runtime_predictions["predicted_runtime_s"]
    result["decision_runtime_source"] = "earlier_month_grouped_quantile"
    result["runtime_model_level"] = runtime_predictions["runtime_model_level"]
    result["runtime_model_count"] = runtime_predictions["runtime_model_count"]
    result["runtime_uncertainty_ratio_q90_q50"] = (
        runtime_q90["predicted_runtime_s"]
        / runtime_q50["predicted_runtime_s"].clip(lower=60.0)
    )
    result["runtime_prediction_reliable"] = result[
        "runtime_uncertainty_ratio_q90_q50"
    ].le(max_runtime_uncertainty_ratio)
    forecast = HISTORY.history_load_forecast(carbon, history_model, load_quantile)
    max_queue_wait_s = max_queue_wait_hours * 3600
    carbon_end = pd.Timestamp(carbon["to_utc"].max())

    order = result.sort_values(["release_dt_s", "sim_job_id"]).index
    for index in order:
        row = result.loc[index]
        initial_offset = float(row["eligible_dt_s"])
        eligible = epoch + pd.Timedelta(seconds=initial_offset)
        partition = str(row["partition"])
        current_wait, wait_level, wait_count = HISTORY.predict_wait(
            partition, eligible, history_model, queue_wait_quantile
        )
        current_wait = min(current_wait, max_queue_wait_s)
        current_start = initial_offset + current_wait
        runtime_s = float(row["decision_runtime_s"])
        if epoch + pd.Timedelta(seconds=current_start + runtime_s) > carbon_end:
            current_wait = max(
                0.0,
                (carbon_end - epoch).total_seconds() - initial_offset - runtime_s,
            )
            current_start = initial_offset + current_wait
        class_name, class_fraction = job_capacity_fraction(row)
        node_fraction = float(row["nodes"]) / TOTAL_NODES
        before_intensity = runtime_average_intensity(
            carbon, epoch + pd.Timedelta(seconds=current_start), runtime_s
        )
        initial_capacity, initial_node, initial_mean, initial_peak, _, _ = (
            forecast_marginal_carbon_cost(
                forecast,
                epoch,
                class_name,
                current_start,
                current_start,
                runtime_s,
                class_fraction,
                node_fraction,
                forecast_contains_current=False,
            )
        )
        chosen = {
            "release": initial_offset,
            "start": current_start,
            "queue_wait": current_wait,
            "intensity": before_intensity,
            "capacity_pct": 0.0,
            "node_pct": 0.0,
            "robust_saving": 0.0,
            "return": 0.0,
            "wait_ratio": 0.0,
            "incremental_wait": 0.0,
            "load_growth": 0.0,
            "mean_load": initial_mean,
            "peak_load": initial_peak,
            "utility": 0.0,
            "class_delta": np.zeros(len(forecast)),
            "node_delta": np.zeros(len(forecast)),
        }

        if bool(row["is_flexible"]) and bool(row["runtime_prediction_reliable"]):
            global_budget_s = max_delay_hours * 3600
            runtime_budget_s = (
                runtime_s * wait_budget_ratio
                if wait_budget_ratio > 0
                else global_budget_s
            )
            budget_s = min(global_budget_s, runtime_budget_s)
            result.at[index, "allowed_wait_budget_s"] = budget_s
            deadline = eligible + pd.Timedelta(seconds=budget_s)
            options = []
            for candidate in candidate_release_times(
                carbon, eligible, deadline, runtime_s
            ):
                candidate_offset = (candidate - epoch).total_seconds()
                candidate_wait, _, _ = HISTORY.predict_wait(
                    partition, candidate, history_model, queue_wait_quantile
                )
                candidate_wait = min(candidate_wait, max_queue_wait_s)
                candidate_start = candidate_offset + candidate_wait
                if epoch + pd.Timedelta(seconds=candidate_start + runtime_s) > carbon_end:
                    continue
                incremental_wait = max(candidate_start - current_start, 0.0)
                (
                    candidate_capacity,
                    candidate_node,
                    mean_load,
                    peak_load,
                    class_delta,
                    node_delta,
                ) = forecast_marginal_carbon_cost(
                    forecast,
                    epoch,
                    class_name,
                    current_start,
                    candidate_start,
                    runtime_s,
                    class_fraction,
                    node_fraction,
                    forecast_contains_current=False,
                )
                capacity_saving = initial_capacity - candidate_capacity
                node_saving = initial_node - candidate_node
                robust_saving = min(capacity_saving, node_saving)
                capacity_pct = (
                    capacity_saving / initial_capacity * 100
                    if initial_capacity
                    else 0.0
                )
                node_pct = node_saving / initial_node * 100 if initial_node else 0.0
                wait_ratio = incremental_wait / max(runtime_s, 60.0)
                initial_over_cap = max(0.0, initial_peak - cap_fraction)
                candidate_over_cap = max(0.0, peak_load - cap_fraction)
                load_growth = max(0.0, mean_load - initial_mean) + max(
                    0.0, candidate_over_cap - initial_over_cap
                )
                delay_h = incremental_wait / 3600
                carbon_return = (
                    robust_saving / max(delay_h, 0.5)
                    if robust_saving > 0
                    else 0.0
                )
                utility = (
                    robust_saving
                    - wait_penalty * wait_ratio
                    - congestion_penalty * load_growth
                )
                options.append(
                    {
                        "release": candidate_offset,
                        "start": candidate_start,
                        "queue_wait": candidate_wait,
                        "intensity": runtime_average_intensity(
                            carbon,
                            epoch + pd.Timedelta(seconds=candidate_start),
                            runtime_s,
                        ),
                        "capacity_pct": capacity_pct,
                        "node_pct": node_pct,
                        "robust_saving": robust_saving,
                        "return": carbon_return,
                        "wait_ratio": wait_ratio,
                        "incremental_wait": incremental_wait,
                        "load_growth": load_growth,
                        "mean_load": mean_load,
                        "peak_load": peak_load,
                        "utility": utility,
                        "class_delta": class_delta,
                        "node_delta": node_delta,
                    }
                )
            if options:
                best = max(options, key=lambda item: (item["utility"], -item["release"]))
                if (
                    best["release"] > initial_offset
                    and best["capacity_pct"] >= min_carbon_saving_pct
                    and best["node_pct"] >= min_node_saving_pct
                    and best["return"] >= min_carbon_return
                    and best["utility"] > 0
                ):
                    chosen = best

        delay = max(float(chosen["release"]) - initial_offset, 0.0)
        result.at[index, "release_dt_s"] = int(round(float(chosen["release"])))
        result.at[index, "b1_delay_s"] = delay
        result.at[index, "runtime_intensity_before"] = before_intensity
        result.at[index, "runtime_intensity_after"] = chosen["intensity"]
        result.at[index, "expected_runtime_carbon_saving_pct"] = min(
            chosen["capacity_pct"], chosen["node_pct"]
        )
        result.at[index, "expected_capacity_carbon_saving_pct"] = chosen["capacity_pct"]
        result.at[index, "expected_node_carbon_saving_pct"] = chosen["node_pct"]
        result.at[index, "expected_risk_adjusted_carbon_saving_proxy"] = chosen[
            "robust_saving"
        ]
        result.at[index, "carbon_return_per_wait_hour"] = chosen["return"]
        result.at[index, "projected_mean_class_load"] = chosen["mean_load"]
        result.at[index, "projected_peak_class_load"] = chosen["peak_load"]
        result.at[index, "dynamic_wait_ratio"] = chosen["wait_ratio"]
        result.at[index, "dynamic_load_growth"] = chosen["load_growth"]
        result.at[index, "dynamic_marginal_utility"] = chosen["utility"]
        result.at[index, "predicted_baseline_start_s"] = current_start
        result.at[index, "predicted_policy_start_s"] = chosen["start"]
        result.at[index, "forecast_incremental_wait_s"] = chosen["incremental_wait"]
        result.at[index, "predicted_baseline_queue_wait_s"] = current_wait
        result.at[index, "predicted_policy_queue_wait_s"] = chosen["queue_wait"]
        result.at[index, "queue_wait_model_level"] = wait_level
        result.at[index, "queue_wait_model_count"] = wait_count
        if delay > 0:
            forecast[f"load_{class_name}"] = np.maximum(
                forecast[f"load_{class_name}"] + chosen["class_delta"], 0.0
            )
            forecast["load_node"] = np.maximum(
                forecast["load_node"] + chosen["node_delta"], 0.0
            )

    metadata = history_model["metadata"]
    assert isinstance(metadata, dict)
    return result, {
        "schedule_forecast": "earlier-month aggregate runtime, queue-wait and interval-load quantiles",
        "forecast_status": "history-only holdout policy; no held-out runtime or future job schedule is read",
        "training_months": metadata["training_months"],
        "holdout_month": metadata["holdout_month"],
        "temporal_split_pass": metadata["temporal_split_pass"],
        "runtime_quantile": runtime_quantile,
        "load_quantile": load_quantile,
        "queue_wait_quantile": queue_wait_quantile,
        "maximum_predicted_queue_wait_hours": max_queue_wait_hours,
        "maximum_runtime_uncertainty_ratio_q90_q50": max_runtime_uncertainty_ratio,
        "runtime_reliable_flexible_jobs": int(
            (result["is_flexible"] & result["runtime_prediction_reliable"]).sum()
        ),
        "admission_cap_capacity_fraction": cap_fraction,
        "wait_budget_ratio": wait_budget_ratio,
        "minimum_expected_capacity_saving_pct": min_carbon_saving_pct,
        "minimum_expected_node_saving_pct": min_node_saving_pct,
        "minimum_carbon_return_per_wait_hour": min_carbon_return,
        "wait_penalty": wait_penalty,
        "congestion_penalty": congestion_penalty,
    }


def apply_dynamic_forecast(
    profile: pd.DataFrame,
    baseline_results: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_hours: float,
    wait_budget_ratio: float,
    min_carbon_saving_pct: float,
    min_node_saving_pct: float,
    wait_penalty: float,
    congestion_penalty: float,
    cap_fraction: float,
    min_carbon_return: float,
    runtime_estimate_factor: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Use a baseline schedule forecast and require benefit in both proxies."""
    result = profile.copy()
    result["is_flexible"] = flexible_mask(result, flex_fraction)
    result["b1_delay_s"] = 0.0
    result["b2_delay_s"] = 0.0
    result["runtime_intensity_before"] = np.nan
    result["runtime_intensity_after"] = np.nan
    result["expected_runtime_carbon_saving_pct"] = 0.0
    result["expected_capacity_carbon_saving_pct"] = 0.0
    result["expected_node_carbon_saving_pct"] = 0.0
    result["expected_risk_adjusted_carbon_saving_proxy"] = 0.0
    result["carbon_return_per_wait_hour"] = 0.0
    result["allowed_wait_budget_s"] = 0.0
    result["projected_mean_class_load"] = np.nan
    result["projected_peak_class_load"] = np.nan
    result["dynamic_wait_ratio"] = 0.0
    result["dynamic_load_growth"] = 0.0
    result["dynamic_marginal_utility"] = 0.0
    result["predicted_baseline_start_s"] = np.nan
    result["predicted_policy_start_s"] = np.nan
    result["forecast_incremental_wait_s"] = 0.0
    result["decision_runtime_s"] = pd.to_numeric(result["runtime_s"]).astype(float)
    result["decision_runtime_source"] = "observed_runtime_sensitivity"

    forecast, predicted_starts = build_schedule_forecast(
        result,
        baseline_results,
        carbon,
        epoch,
        runtime_estimate_factor=runtime_estimate_factor,
    )
    order = result.sort_values(["release_dt_s", "sim_job_id"]).index
    for index in order:
        row = result.loc[index]
        job_id = str(row["sim_job_id"])
        initial_offset = float(row["eligible_dt_s"])
        current_start = float(predicted_starts[job_id])
        runtime_s = float(row["runtime_s"]) * runtime_estimate_factor
        result.at[index, "decision_runtime_s"] = runtime_s
        class_name, class_fraction = job_capacity_fraction(row)
        node_fraction = float(row["nodes"]) / TOTAL_NODES
        before_intensity = runtime_average_intensity(
            carbon, epoch + pd.Timedelta(seconds=current_start), runtime_s
        )
        initial_capacity, initial_node, initial_mean, initial_peak, _, _ = (
            forecast_marginal_carbon_cost(
                forecast,
                epoch,
                class_name,
                current_start,
                current_start,
                runtime_s,
                class_fraction,
                node_fraction,
            )
        )
        chosen = {
            "release": initial_offset,
            "start": current_start,
            "intensity": before_intensity,
            "capacity_pct": 0.0,
            "node_pct": 0.0,
            "robust_saving": 0.0,
            "return": 0.0,
            "wait_ratio": 0.0,
            "incremental_wait": 0.0,
            "load_growth": 0.0,
            "mean_load": initial_mean,
            "peak_load": initial_peak,
            "utility": 0.0,
            "class_delta": np.zeros(len(forecast)),
            "node_delta": np.zeros(len(forecast)),
        }

        if bool(row["is_flexible"]):
            global_budget_s = max_delay_hours * 3600
            runtime_budget_s = (
                runtime_s * wait_budget_ratio
                if wait_budget_ratio > 0
                else global_budget_s
            )
            budget_s = min(global_budget_s, runtime_budget_s)
            result.at[index, "allowed_wait_budget_s"] = budget_s
            eligible = epoch + pd.Timedelta(seconds=initial_offset)
            deadline = eligible + pd.Timedelta(seconds=budget_s)
            options = []
            for candidate in candidate_release_times(
                carbon, eligible, deadline, runtime_s
            ):
                candidate_offset = (candidate - epoch).total_seconds()
                candidate_start = max(candidate_offset, current_start)
                incremental_wait = max(candidate_start - current_start, 0.0)
                (
                    candidate_capacity,
                    candidate_node,
                    mean_load,
                    peak_load,
                    class_delta,
                    node_delta,
                ) = forecast_marginal_carbon_cost(
                    forecast,
                    epoch,
                    class_name,
                    current_start,
                    candidate_start,
                    runtime_s,
                    class_fraction,
                    node_fraction,
                )
                capacity_saving = initial_capacity - candidate_capacity
                node_saving = initial_node - candidate_node
                robust_saving = min(capacity_saving, node_saving)
                capacity_pct = (
                    capacity_saving / initial_capacity * 100
                    if initial_capacity
                    else 0.0
                )
                node_pct = node_saving / initial_node * 100 if initial_node else 0.0
                wait_ratio = incremental_wait / max(runtime_s, 60.0)
                initial_over_cap = max(0.0, initial_peak - cap_fraction)
                candidate_over_cap = max(0.0, peak_load - cap_fraction)
                load_growth = max(0.0, mean_load - initial_mean) + max(
                    0.0, candidate_over_cap - initial_over_cap
                )
                delay_h = incremental_wait / 3600
                carbon_return = (
                    robust_saving / max(delay_h, 0.5)
                    if robust_saving > 0
                    else 0.0
                )
                utility = (
                    robust_saving
                    - wait_penalty * wait_ratio
                    - congestion_penalty * load_growth
                )
                options.append(
                    {
                        "release": candidate_offset,
                        "start": candidate_start,
                        "intensity": runtime_average_intensity(
                            carbon,
                            epoch + pd.Timedelta(seconds=candidate_start),
                            runtime_s,
                        ),
                        "capacity_pct": capacity_pct,
                        "node_pct": node_pct,
                        "robust_saving": robust_saving,
                        "return": carbon_return,
                        "wait_ratio": wait_ratio,
                        "incremental_wait": incremental_wait,
                        "load_growth": load_growth,
                        "mean_load": mean_load,
                        "peak_load": peak_load,
                        "utility": utility,
                        "class_delta": class_delta,
                        "node_delta": node_delta,
                    }
                )
            best = max(options, key=lambda item: (item["utility"], -item["release"]))
            if (
                best["release"] > initial_offset
                and best["incremental_wait"] > 0
                and best["capacity_pct"] >= min_carbon_saving_pct
                and best["node_pct"] >= min_node_saving_pct
                and best["return"] >= min_carbon_return
                and best["utility"] > 0
            ):
                chosen = best

        delay = max(float(chosen["release"]) - initial_offset, 0.0)
        result.at[index, "release_dt_s"] = int(round(float(chosen["release"])))
        result.at[index, "b1_delay_s"] = delay
        result.at[index, "runtime_intensity_before"] = before_intensity
        result.at[index, "runtime_intensity_after"] = chosen["intensity"]
        result.at[index, "expected_runtime_carbon_saving_pct"] = min(
            chosen["capacity_pct"], chosen["node_pct"]
        )
        result.at[index, "expected_capacity_carbon_saving_pct"] = chosen[
            "capacity_pct"
        ]
        result.at[index, "expected_node_carbon_saving_pct"] = chosen["node_pct"]
        result.at[index, "expected_risk_adjusted_carbon_saving_proxy"] = chosen[
            "robust_saving"
        ]
        result.at[index, "carbon_return_per_wait_hour"] = chosen["return"]
        result.at[index, "projected_mean_class_load"] = chosen["mean_load"]
        result.at[index, "projected_peak_class_load"] = chosen["peak_load"]
        result.at[index, "dynamic_wait_ratio"] = chosen["wait_ratio"]
        result.at[index, "dynamic_load_growth"] = chosen["load_growth"]
        result.at[index, "dynamic_marginal_utility"] = chosen["utility"]
        result.at[index, "predicted_baseline_start_s"] = current_start
        result.at[index, "predicted_policy_start_s"] = chosen["start"]
        result.at[index, "forecast_incremental_wait_s"] = chosen["incremental_wait"]
        if delay > 0:
            forecast[f"load_{class_name}"] += chosen["class_delta"]
            forecast["load_node"] += chosen["node_delta"]

    return result, {
        "schedule_forecast": "baseline Slurm run, including future interval load",
        "forecast_status": "oracle upper-bound experiment; production needs a historical/load forecast",
        "admission_cap_capacity_fraction": cap_fraction,
        "wait_budget_ratio": wait_budget_ratio,
        "minimum_expected_capacity_saving_pct": min_carbon_saving_pct,
        "minimum_expected_node_saving_pct": min_node_saving_pct,
        "minimum_carbon_return_per_wait_hour": min_carbon_return,
        "runtime_estimate_factor": runtime_estimate_factor,
        "runtime_estimate_status": "multiplicative sensitivity around observed runtime",
        "wait_penalty": wait_penalty,
        "congestion_penalty": congestion_penalty,
    }


def apply_dynamic_marginal(
    profile: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_hours: float,
    wait_budget_ratio: float,
    min_carbon_saving_pct: float,
    wait_penalty: float,
    congestion_penalty: float,
    cap_fraction: float,
    min_carbon_return: float,
    sensitivity_penalty: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Choose releases from conservative marginal carbon and waiting utility."""
    result = profile.copy()
    result["is_flexible"] = flexible_mask(result, flex_fraction)
    result["b1_delay_s"] = 0.0
    result["b2_delay_s"] = 0.0
    result["runtime_intensity_before"] = np.nan
    result["runtime_intensity_after"] = np.nan
    result["expected_runtime_carbon_saving_pct"] = 0.0
    result["expected_capacity_carbon_saving_pct"] = 0.0
    result["expected_node_carbon_saving_pct"] = 0.0
    result["expected_risk_adjusted_carbon_saving_proxy"] = 0.0
    result["carbon_return_per_wait_hour"] = 0.0
    result["allowed_wait_budget_s"] = 0.0
    result["projected_mean_class_load"] = np.nan
    result["projected_peak_class_load"] = np.nan
    result["dynamic_wait_ratio"] = 0.0
    result["dynamic_load_growth"] = 0.0
    result["dynamic_marginal_utility"] = 0.0

    planned: list[tuple[float, float, str, float, float]] = []
    order = result.sort_values(["release_dt_s", "sim_job_id"]).index
    for index in order:
        row = result.loc[index]
        initial_offset = float(row["eligible_dt_s"])
        runtime_s = float(row["runtime_s"])
        class_name, class_fraction = job_capacity_fraction(row)
        node_fraction = float(row["nodes"]) / TOTAL_NODES
        before_intensity = runtime_average_intensity(
            carbon, epoch + pd.Timedelta(seconds=initial_offset), runtime_s
        )
        initial_capacity, initial_node, initial_mean, initial_peak = (
            marginal_runtime_carbon_cost(
                carbon,
                epoch,
                planned,
                class_name,
                initial_offset,
                runtime_s,
                class_fraction,
                node_fraction,
            )
        )
        chosen = {
            "offset": initial_offset,
            "intensity": before_intensity,
            "capacity_pct": 0.0,
            "node_pct": 0.0,
            "risk_adjusted_pct": 0.0,
            "risk_adjusted_saving": 0.0,
            "return": 0.0,
            "wait_ratio": 0.0,
            "load_growth": 0.0,
            "mean_load": initial_mean,
            "peak_load": initial_peak,
            "utility": 0.0,
        }

        if bool(row["is_flexible"]):
            global_budget_s = max_delay_hours * 3600
            runtime_budget_s = (
                runtime_s * wait_budget_ratio
                if wait_budget_ratio > 0
                else global_budget_s
            )
            budget_s = min(global_budget_s, runtime_budget_s)
            result.at[index, "allowed_wait_budget_s"] = budget_s
            eligible = epoch + pd.Timedelta(seconds=initial_offset)
            deadline = eligible + pd.Timedelta(seconds=budget_s)
            options = []
            for candidate in candidate_release_times(
                carbon, eligible, deadline, runtime_s
            ):
                candidate_offset = (candidate - epoch).total_seconds()
                delay_s = max(candidate_offset - initial_offset, 0.0)
                candidate_intensity = runtime_average_intensity(
                    carbon, candidate, runtime_s
                )
                candidate_capacity, candidate_node, mean_load, peak_load = (
                    marginal_runtime_carbon_cost(
                        carbon,
                        epoch,
                        planned,
                        class_name,
                        candidate_offset,
                        runtime_s,
                        class_fraction,
                        node_fraction,
                    )
                )
                capacity_saving = initial_capacity - candidate_capacity
                node_saving = initial_node - candidate_node
                risk_adjusted_saving = capacity_saving - sensitivity_penalty * max(
                    0.0, -node_saving
                )
                capacity_pct = (
                    capacity_saving / initial_capacity * 100
                    if initial_capacity
                    else 0.0
                )
                node_pct = node_saving / initial_node * 100 if initial_node else 0.0
                risk_adjusted_pct = (
                    risk_adjusted_saving / initial_capacity * 100
                    if initial_capacity
                    else 0.0
                )
                wait_ratio = delay_s / max(runtime_s, 60.0)
                initial_over_cap = max(0.0, initial_peak - cap_fraction)
                candidate_over_cap = max(0.0, peak_load - cap_fraction)
                load_growth = max(0.0, mean_load - initial_mean) + max(
                    0.0, candidate_over_cap - initial_over_cap
                )
                delay_h = delay_s / 3600
                carbon_return = (
                    risk_adjusted_saving / max(delay_h, 0.5)
                    if risk_adjusted_saving > 0
                    else 0.0
                )
                utility = (
                    risk_adjusted_saving
                    - wait_penalty * wait_ratio
                    - congestion_penalty * load_growth
                )
                options.append(
                    {
                        "offset": candidate_offset,
                        "intensity": candidate_intensity,
                        "capacity_pct": capacity_pct,
                        "node_pct": node_pct,
                        "risk_adjusted_pct": risk_adjusted_pct,
                        "risk_adjusted_saving": risk_adjusted_saving,
                        "return": carbon_return,
                        "wait_ratio": wait_ratio,
                        "load_growth": load_growth,
                        "mean_load": mean_load,
                        "peak_load": peak_load,
                        "utility": utility,
                    }
                )
            best = max(options, key=lambda item: (item["utility"], -item["offset"]))
            if (
                best["offset"] > initial_offset
                and best["risk_adjusted_pct"] >= min_carbon_saving_pct
                and best["return"] >= min_carbon_return
                and best["utility"] > 0
            ):
                chosen = best

        delay = max(float(chosen["offset"]) - initial_offset, 0.0)
        result.at[index, "release_dt_s"] = int(round(float(chosen["offset"])))
        result.at[index, "b1_delay_s"] = delay
        result.at[index, "runtime_intensity_before"] = before_intensity
        result.at[index, "runtime_intensity_after"] = chosen["intensity"]
        result.at[index, "expected_runtime_carbon_saving_pct"] = chosen[
            "risk_adjusted_pct"
        ]
        result.at[index, "expected_capacity_carbon_saving_pct"] = chosen[
            "capacity_pct"
        ]
        result.at[index, "expected_node_carbon_saving_pct"] = chosen["node_pct"]
        result.at[index, "expected_risk_adjusted_carbon_saving_proxy"] = chosen[
            "risk_adjusted_saving"
        ]
        result.at[index, "carbon_return_per_wait_hour"] = chosen["return"]
        result.at[index, "projected_mean_class_load"] = chosen["mean_load"]
        result.at[index, "projected_peak_class_load"] = chosen["peak_load"]
        result.at[index, "dynamic_wait_ratio"] = chosen["wait_ratio"]
        result.at[index, "dynamic_load_growth"] = chosen["load_growth"]
        result.at[index, "dynamic_marginal_utility"] = chosen["utility"]
        planned.append(
            (
                float(chosen["offset"]),
                float(chosen["offset"]) + runtime_s,
                class_name,
                class_fraction,
                node_fraction,
            )
        )

    return result, {
        "admission_cap_capacity_fraction": cap_fraction,
        "wait_budget_ratio": wait_budget_ratio,
        "minimum_expected_risk_adjusted_carbon_saving_pct": min_carbon_saving_pct,
        "minimum_carbon_return_per_wait_hour": min_carbon_return,
        "node_proxy_downside_penalty": sensitivity_penalty,
        "wait_penalty": wait_penalty,
        "congestion_penalty": congestion_penalty,
    }


def apply_runtime_b1(
    profile: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_hours: float,
) -> pd.DataFrame:
    result = profile.copy()
    result["is_flexible"] = flexible_mask(result, flex_fraction)
    result["b1_delay_s"] = 0.0
    result["runtime_intensity_before"] = np.nan
    result["runtime_intensity_after"] = np.nan
    result["expected_runtime_carbon_saving_pct"] = 0.0
    result["allowed_wait_budget_s"] = max_delay_hours * 3600

    for index, row in result.loc[result["is_flexible"]].iterrows():
        eligible = epoch + pd.Timedelta(seconds=float(row["eligible_dt_s"]))
        deadline = eligible + pd.Timedelta(hours=max_delay_hours)
        release, before, after = best_runtime_release_time(
            carbon, eligible, deadline, float(row["runtime_s"])
        )
        delay = max((release - eligible).total_seconds(), 0.0)
        result.at[index, "b1_delay_s"] = delay
        result.at[index, "runtime_intensity_before"] = before
        result.at[index, "runtime_intensity_after"] = after
        result.at[index, "expected_runtime_carbon_saving_pct"] = (
            (before - after) / before * 100 if before else 0.0
        )
        result.at[index, "release_dt_s"] = int(round(float(row["eligible_dt_s"]) + delay))
    return result


def apply_adaptive(
    profile: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_hours: float,
    wait_budget_ratio: float,
    min_carbon_saving_pct: float,
    wait_penalty: float,
    congestion_penalty: float,
    high_quantile: float,
    cap_fraction: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = profile.copy()
    result["is_flexible"] = flexible_mask(result, flex_fraction)
    result["b1_delay_s"] = 0.0
    result["b2_delay_s"] = 0.0
    result["runtime_intensity_before"] = np.nan
    result["runtime_intensity_after"] = np.nan
    result["expected_runtime_carbon_saving_pct"] = 0.0
    result["allowed_wait_budget_s"] = 0.0
    result["projected_mean_class_load"] = np.nan
    result["projected_peak_class_load"] = np.nan
    result["adaptive_objective"] = np.nan

    release_min = float(result["release_dt_s"].min())
    release_max = float((result["release_dt_s"] + result["runtime_s"]).max())
    horizon_start = epoch + pd.Timedelta(seconds=release_min)
    horizon_end = epoch + pd.Timedelta(seconds=release_max + max_delay_hours * 3600)
    relevant = carbon.loc[
        (carbon["to_utc"] > horizon_start) & (carbon["from_utc"] < horizon_end)
    ]
    if relevant.empty:
        raise ValueError("Carbon data does not overlap the adaptive-policy horizon")
    high_threshold = float(relevant["intensity_gco2_per_kwh"].quantile(high_quantile))

    planned: list[tuple[float, float, str, float]] = []
    order = result.sort_values(["release_dt_s", "sim_job_id"]).index
    for index in order:
        row = result.loc[index]
        initial_offset = float(row["eligible_dt_s"])
        runtime_s = float(row["runtime_s"])
        class_name, job_fraction = job_capacity_fraction(row)
        chosen_offset = initial_offset
        before = runtime_average_intensity(
            carbon, epoch + pd.Timedelta(seconds=initial_offset), runtime_s
        )
        after = before
        objective = 1.0

        if bool(row["is_flexible"]):
            global_budget_s = max_delay_hours * 3600
            runtime_budget_s = runtime_s * wait_budget_ratio if wait_budget_ratio > 0 else global_budget_s
            budget_s = min(global_budget_s, runtime_budget_s)
            result.at[index, "allowed_wait_budget_s"] = budget_s
            eligible = epoch + pd.Timedelta(seconds=initial_offset)
            deadline = eligible + pd.Timedelta(seconds=budget_s)
            options = []
            for candidate in candidate_release_times(carbon, eligible, deadline, runtime_s):
                candidate_offset = (candidate - epoch).total_seconds()
                candidate_intensity = runtime_average_intensity(carbon, candidate, runtime_s)
                delay_s = max(candidate_offset - initial_offset, 0.0)
                mean_load, peak_load = projected_class_load(
                    planned, class_name, candidate_offset, runtime_s, job_fraction
                )
                release_intensity = float(
                    carbon_row_at(carbon, candidate)["intensity_gco2_per_kwh"]
                )
                high_carbon_over_cap = (
                    max(0.0, peak_load - cap_fraction)
                    if release_intensity >= high_threshold
                    else 0.0
                )
                delay_fraction = delay_s / global_budget_s if global_budget_s else 0.0
                score = (
                    candidate_intensity / before
                    + wait_penalty * delay_fraction
                    + congestion_penalty * high_carbon_over_cap
                )
                options.append(
                    (score, candidate, candidate_intensity, mean_load, peak_load)
                )
            score, candidate, candidate_intensity, mean_load, peak_load = min(
                options, key=lambda item: (item[0], item[1])
            )
            saving_pct = (before - candidate_intensity) / before * 100 if before else 0.0
            if saving_pct >= min_carbon_saving_pct:
                chosen_offset = (candidate - epoch).total_seconds()
                after = candidate_intensity
                objective = score
            else:
                mean_load, peak_load = projected_class_load(
                    planned, class_name, initial_offset, runtime_s, job_fraction
                )
        else:
            mean_load, peak_load = projected_class_load(
                planned, class_name, initial_offset, runtime_s, job_fraction
            )

        delay = max(chosen_offset - initial_offset, 0.0)
        result.at[index, "release_dt_s"] = int(round(chosen_offset))
        result.at[index, "b1_delay_s"] = delay
        result.at[index, "runtime_intensity_before"] = before
        result.at[index, "runtime_intensity_after"] = after
        result.at[index, "expected_runtime_carbon_saving_pct"] = (
            (before - after) / before * 100 if before else 0.0
        )
        result.at[index, "projected_mean_class_load"] = mean_load
        result.at[index, "projected_peak_class_load"] = peak_load
        result.at[index, "adaptive_objective"] = objective
        planned.append((chosen_offset, chosen_offset + runtime_s, class_name, job_fraction))

    return result, {
        "high_carbon_quantile": high_quantile,
        "high_carbon_threshold_gco2_per_kwh": high_threshold,
        "admission_cap_capacity_fraction": cap_fraction,
        "wait_budget_ratio": wait_budget_ratio,
        "minimum_expected_carbon_saving_pct": min_carbon_saving_pct,
        "wait_penalty": wait_penalty,
        "congestion_penalty": congestion_penalty,
    }


def best_release_time(
    carbon: pd.DataFrame, eligible: pd.Timestamp, deadline: pd.Timestamp
) -> tuple[pd.Timestamp, float, float]:
    current = carbon_row_at(carbon, eligible)
    candidates = carbon.loc[
        (carbon["from_utc"] > eligible) & (carbon["from_utc"] <= deadline)
    ].copy()
    options = [(eligible, float(current["intensity_gco2_per_kwh"]))]
    options.extend(
        (row.from_utc, float(row.intensity_gco2_per_kwh))
        for row in candidates.itertuples(index=False)
    )
    release, target_intensity = min(options, key=lambda item: (item[1], item[0]))
    return release, float(current["intensity_gco2_per_kwh"]), target_intensity


def apply_b1(
    profile: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_hours: float,
) -> pd.DataFrame:
    result = profile.copy()
    result["is_flexible"] = flexible_mask(result, flex_fraction)
    result["b1_delay_s"] = 0.0
    result["release_intensity_before"] = np.nan
    result["release_intensity_after_b1"] = np.nan

    for index, row in result.loc[result["is_flexible"]].iterrows():
        eligible = epoch + pd.Timedelta(seconds=float(row["eligible_dt_s"]))
        deadline = eligible + pd.Timedelta(hours=max_delay_hours)
        release, before, after = best_release_time(carbon, eligible, deadline)
        delay = max((release - eligible).total_seconds(), 0.0)
        result.at[index, "b1_delay_s"] = delay
        result.at[index, "release_intensity_before"] = before
        result.at[index, "release_intensity_after_b1"] = after
        result.at[index, "release_dt_s"] = int(round(float(row["eligible_dt_s"]) + delay))
    return result


def high_carbon_peak_nodes(
    profile: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    threshold: float,
) -> float:
    release = pd.to_numeric(profile["release_dt_s"], errors="coerce").to_numpy(float)
    end = release + pd.to_numeric(profile["runtime_s"], errors="coerce").to_numpy(float)
    nodes = pd.to_numeric(profile["nodes"], errors="coerce").to_numpy(float)
    peak = 0.0
    for row in carbon.loc[carbon["intensity_gco2_per_kwh"] >= threshold].itertuples(index=False):
        offset = (row.from_utc - epoch).total_seconds()
        active = (release <= offset) & (offset < end)
        peak = max(peak, float(nodes[active].sum()))
    return peak


def next_carbon_interval(carbon: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(carbon_row_at(carbon, timestamp)["to_utc"])


def apply_b2(
    profile: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    high_quantile: float,
    cap_fraction: float,
    max_delay_hours: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    result = profile.copy()
    if "is_flexible" not in result:
        result["is_flexible"] = flexible_mask(result, flex_fraction)
    result["b2_delay_s"] = 0.0

    horizon_start = epoch + pd.Timedelta(seconds=float(result["release_dt_s"].min()))
    horizon_end = epoch + pd.Timedelta(
        seconds=float((result["release_dt_s"] + result["runtime_s"]).max())
    )
    relevant = carbon.loc[
        (carbon["to_utc"] > horizon_start) & (carbon["from_utc"] < horizon_end)
    ]
    if relevant.empty:
        raise ValueError("Carbon data does not overlap the workload horizon")
    threshold = float(relevant["intensity_gco2_per_kwh"].quantile(high_quantile))
    peak_nodes = high_carbon_peak_nodes(result, relevant, epoch, threshold)
    cap_nodes = max(1.0, peak_nodes * cap_fraction)

    scheduled: list[tuple[float, float, float]] = []
    order = result.sort_values(["release_dt_s", "sim_job_id"]).index
    for index in order:
        row = result.loc[index]
        initial_offset = float(row["release_dt_s"])
        candidate_offset = initial_offset
        deadline_offset = initial_offset + max_delay_hours * 3600
        job_nodes = float(row["nodes"])

        if bool(row["is_flexible"]):
            while candidate_offset < deadline_offset:
                candidate_time = epoch + pd.Timedelta(seconds=candidate_offset)
                carbon_row = carbon_row_at(carbon, candidate_time)
                intensity = float(carbon_row["intensity_gco2_per_kwh"])
                active_nodes = sum(
                    nodes for start, end, nodes in scheduled if start <= candidate_offset < end
                )
                if intensity < threshold or active_nodes + job_nodes <= cap_nodes:
                    break
                next_time = next_carbon_interval(carbon, candidate_time)
                candidate_offset = min(
                    deadline_offset, max(candidate_offset + 1, (next_time - epoch).total_seconds())
                )

        result.at[index, "release_dt_s"] = int(round(candidate_offset))
        result.at[index, "b2_delay_s"] = max(candidate_offset - initial_offset, 0.0)
        scheduled.append(
            (candidate_offset, candidate_offset + float(row["runtime_s"]), job_nodes)
        )

    return result, {
        "high_carbon_quantile": high_quantile,
        "high_carbon_threshold_gco2_per_kwh": threshold,
        "baseline_high_carbon_peak_active_node_requests": peak_nodes,
        "admission_cap_active_node_requests": cap_nodes,
        "cap_fraction": cap_fraction,
    }


def apply_time_hybrid(
    profile: pd.DataFrame,
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    day_start_hour_utc: int,
    day_end_hour_utc: int,
    b1_max_delay_hours: float,
    high_quantile: float,
    cap_fraction: float,
    b2_max_delay_hours: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Use B2 during the working day and runtime-aware B1 overnight."""
    if not 0 <= day_start_hour_utc <= 23:
        raise ValueError("day-start-hour-utc must be between 0 and 23")
    if not 1 <= day_end_hour_utc <= 24:
        raise ValueError("day-end-hour-utc must be between 1 and 24")
    if day_start_hour_utc >= day_end_hour_utc:
        raise ValueError("day-start-hour-utc must be earlier than day-end-hour-utc")

    result = profile.copy()
    base_flexible = flexible_mask(result, flex_fraction)
    eligible_utc = epoch + pd.to_timedelta(
        pd.to_numeric(result["eligible_dt_s"], errors="raise"), unit="s"
    )
    day_mask = eligible_utc.dt.hour.between(
        day_start_hour_utc, day_end_hour_utc - 1
    )
    night_flexible = base_flexible & ~day_mask

    result["is_flexible"] = night_flexible
    result["b1_delay_s"] = 0.0
    result["runtime_intensity_before"] = np.nan
    result["runtime_intensity_after"] = np.nan
    result["expected_runtime_carbon_saving_pct"] = 0.0
    result["allowed_wait_budget_s"] = 0.0
    for index, row in result.loc[night_flexible].iterrows():
        eligible = epoch + pd.Timedelta(seconds=float(row["eligible_dt_s"]))
        deadline = eligible + pd.Timedelta(hours=b1_max_delay_hours)
        release, before, after = best_runtime_release_time(
            carbon, eligible, deadline, float(row["runtime_s"])
        )
        delay = max((release - eligible).total_seconds(), 0.0)
        result.at[index, "release_dt_s"] = int(
            round(float(row["eligible_dt_s"]) + delay)
        )
        result.at[index, "b1_delay_s"] = delay
        result.at[index, "runtime_intensity_before"] = before
        result.at[index, "runtime_intensity_after"] = after
        result.at[index, "expected_runtime_carbon_saving_pct"] = (
            (before - after) / before * 100 if before else 0.0
        )
        result.at[index, "allowed_wait_budget_s"] = b1_max_delay_hours * 3600

    day_flexible = base_flexible & day_mask
    result.loc[day_flexible, "allowed_wait_budget_s"] = b2_max_delay_hours * 3600
    result["is_flexible"] = day_flexible
    result, b2_metadata = apply_b2(
        result,
        carbon,
        epoch,
        flex_fraction,
        high_quantile,
        cap_fraction,
        b2_max_delay_hours,
    )
    result["is_flexible"] = base_flexible
    result["hybrid_mode"] = "not_flexible"
    result.loc[day_flexible, "hybrid_mode"] = "day_b2"
    result.loc[night_flexible, "hybrid_mode"] = "night_runtime_b1"
    return result, {
        "day_start_hour_utc": day_start_hour_utc,
        "day_end_hour_utc": day_end_hour_utc,
        "day_b2_flexible_jobs": int(day_flexible.sum()),
        "night_b1_flexible_jobs": int(night_flexible.sum()),
        **b2_metadata,
    }


def finalize_profile(profile: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    result = profile.copy()
    result["policy_name"] = policy_name
    if "b1_delay_s" not in result:
        result["b1_delay_s"] = 0.0
    if "b2_delay_s" not in result:
        result["b2_delay_s"] = 0.0
    result["policy_delay_s"] = (
        pd.to_numeric(result["release_dt_s"]) - pd.to_numeric(result["eligible_dt_s"])
    ).clip(lower=0)
    result.rename(columns={"slurm_job_id_expected": "baseline_slurm_job_id"}, inplace=True)
    if "carry_in_type" in result:
        # Reconstruct T0 occupancy before admitting queued jobs, just as baseline does.
        category_order = {"running": 0, "queued": 1, "held": 2, "evaluation": 3}
        result["_carry_in_order"] = result["carry_in_type"].map(category_order)
        if result["_carry_in_order"].isna().any():
            raise ValueError("Unknown carry-in category")
        result.sort_values(
            ["release_dt_s", "_carry_in_order", "baseline_slurm_job_id"],
            kind="stable", inplace=True,
        )
        result.drop(columns="_carry_in_order", inplace=True)
    else:
        result.sort_values(["release_dt_s", "source_submit_utc", "sim_job_id"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    result.insert(1, "slurm_job_id_expected", np.arange(1, len(result) + 1, dtype=int))
    return result


def write_scenario(output: Path, profile: pd.DataFrame, summary: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output / "workload_profile.csv", index=False)
    (output / "sim.events").write_text(
        "\n".join(GENERATOR.event_line(row) for _, row in profile.iterrows()) + "\n",
        encoding="utf-8",
    )
    users = profile[["user_id"]].drop_duplicates().sort_values("user_id")
    with (output / "users.sim").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=":", lineterminator="\n")
        for index, user_id in enumerate(users["user_id"], start=1):
            writer.writerow([user_id, 10_000 + index, "users", 100])
    (output / "policy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--carbon", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=[
            "b1",
            "b2",
            "b1-b2",
            "b1-runtime",
            "adaptive",
            "dynamic-marginal",
            "dynamic-forecast",
            "dynamic-history",
            "time-hybrid",
        ],
        required=True,
    )
    parser.add_argument("--flex-fraction", type=float, default=0.30)
    parser.add_argument("--b1-max-delay-hours", type=float, default=6.0)
    parser.add_argument("--b2-max-delay-hours", type=float, default=6.0)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--cap-fraction", type=float, default=0.50)
    parser.add_argument("--wait-budget-ratio", type=float, default=1.0)
    parser.add_argument("--min-carbon-saving-pct", type=float, default=2.0)
    parser.add_argument("--wait-penalty", type=float, default=0.10)
    parser.add_argument("--congestion-penalty", type=float, default=0.25)
    parser.add_argument("--min-carbon-return", type=float, default=0.0)
    parser.add_argument("--sensitivity-penalty", type=float, default=0.50)
    parser.add_argument("--min-node-saving-pct", type=float, default=0.0)
    parser.add_argument("--schedule-forecast", type=Path)
    parser.add_argument("--runtime-estimate-factor", type=float, default=1.0)
    parser.add_argument("--history-model-dir", type=Path)
    parser.add_argument("--history-runtime-quantile", type=float, default=0.50)
    parser.add_argument("--history-load-quantile", type=float, default=0.75)
    parser.add_argument("--history-queue-wait-quantile", type=float, default=0.50)
    parser.add_argument("--history-max-queue-wait-hours", type=float, default=4.0)
    parser.add_argument("--history-max-runtime-uncertainty-ratio", type=float, default=2.0)
    parser.add_argument("--day-start-hour-utc", type=int, default=8)
    parser.add_argument("--day-end-hour-utc", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.flex_fraction <= 1:
        raise ValueError("flex-fraction must be between 0 and 1")
    if args.min_carbon_return < 0:
        raise ValueError("min-carbon-return must be non-negative")
    if args.sensitivity_penalty < 0:
        raise ValueError("sensitivity-penalty must be non-negative")
    if args.min_node_saving_pct < 0:
        raise ValueError("min-node-saving-pct must be non-negative")
    if args.runtime_estimate_factor <= 0:
        raise ValueError("runtime-estimate-factor must be positive")
    if args.history_max_queue_wait_hours < 0:
        raise ValueError("history-max-queue-wait-hours must be non-negative")
    if args.history_max_runtime_uncertainty_ratio < 1:
        raise ValueError("history-max-runtime-uncertainty-ratio must be at least 1")
    profile = pd.read_csv(args.baseline_profile)
    carbon = load_carbon(args.carbon)
    epoch = workload_epoch(profile)
    working = profile.copy()
    b2_metadata: dict[str, object] = {}

    if args.strategy in {"b1", "b1-b2"}:
        working = apply_b1(
            working, carbon, epoch, args.flex_fraction, args.b1_max_delay_hours
        )
    if args.strategy in {"b2", "b1-b2"}:
        working, b2_metadata = apply_b2(
            working,
            carbon,
            epoch,
            args.flex_fraction,
            args.high_quantile,
            args.cap_fraction,
            args.b2_max_delay_hours,
        )
    if args.strategy == "b1-runtime":
        working = apply_runtime_b1(
            working, carbon, epoch, args.flex_fraction, args.b1_max_delay_hours
        )
    if args.strategy == "adaptive":
        working, b2_metadata = apply_adaptive(
            working,
            carbon,
            epoch,
            args.flex_fraction,
            args.b1_max_delay_hours,
            args.wait_budget_ratio,
            args.min_carbon_saving_pct,
            args.wait_penalty,
            args.congestion_penalty,
            args.high_quantile,
            args.cap_fraction,
        )
    if args.strategy == "dynamic-marginal":
        working, b2_metadata = apply_dynamic_marginal(
            working,
            carbon,
            epoch,
            args.flex_fraction,
            args.b1_max_delay_hours,
            args.wait_budget_ratio,
            args.min_carbon_saving_pct,
            args.wait_penalty,
            args.congestion_penalty,
            args.cap_fraction,
            args.min_carbon_return,
            args.sensitivity_penalty,
        )
    if args.strategy == "dynamic-forecast":
        if args.schedule_forecast is None:
            raise ValueError("dynamic-forecast requires --schedule-forecast")
        baseline_results = pd.read_csv(args.schedule_forecast)
        working, b2_metadata = apply_dynamic_forecast(
            working,
            baseline_results,
            carbon,
            epoch,
            args.flex_fraction,
            args.b1_max_delay_hours,
            args.wait_budget_ratio,
            args.min_carbon_saving_pct,
            args.min_node_saving_pct,
            args.wait_penalty,
            args.congestion_penalty,
            args.cap_fraction,
            args.min_carbon_return,
            args.runtime_estimate_factor,
        )
    if args.strategy == "dynamic-history":
        if args.history_model_dir is None:
            raise ValueError("dynamic-history requires --history-model-dir")
        history_model = HISTORY.load_model(args.history_model_dir)
        working, b2_metadata = apply_dynamic_history(
            working,
            history_model,
            carbon,
            epoch,
            args.flex_fraction,
            args.b1_max_delay_hours,
            args.wait_budget_ratio,
            args.min_carbon_saving_pct,
            args.min_node_saving_pct,
            args.wait_penalty,
            args.congestion_penalty,
            args.cap_fraction,
            args.min_carbon_return,
            args.history_runtime_quantile,
            args.history_load_quantile,
            args.history_queue_wait_quantile,
            args.history_max_queue_wait_hours,
            args.history_max_runtime_uncertainty_ratio,
        )
    if args.strategy == "time-hybrid":
        working, b2_metadata = apply_time_hybrid(
            working,
            carbon,
            epoch,
            args.flex_fraction,
            args.day_start_hour_utc,
            args.day_end_hour_utc,
            args.b1_max_delay_hours,
            args.high_quantile,
            args.cap_fraction,
            args.b2_max_delay_hours,
        )

    result = finalize_profile(working, args.strategy)
    delayed = result["policy_delay_s"] > 0
    summary = {
        "strategy": args.strategy,
        "policy_interpretation": {
            "b1": "Delay a fixed flexible subset to the lowest regional carbon-intensity interval within its deadline",
            "b2": "Admission-control proxy: during high-carbon intervals, delay flexible jobs when estimated active node requests exceed the cap",
            "b1-b2": "Apply B1 first, then the B2 admission gate",
            "b1-runtime": "Delay flexible jobs to the lowest average carbon intensity over their full expected runtime",
            "adaptive": "Runtime-aware B1 with per-job waiting budgets, a minimum carbon-saving threshold and class-capacity B2 congestion pressure",
            "dynamic-marginal": "Deterministic per-arrival admission using capacity-weighted marginal carbon benefit, requested-node downside sensitivity, runtime-proportional wait cost and projected congestion growth",
            "dynamic-forecast": "Deterministic forecast-assisted admission requiring positive marginal carbon benefit in both utilization proxies, with runtime-proportional wait cost and projected congestion growth",
            "dynamic-history": "Deterministic holdout admission using grouped runtime, queue-wait and interval-load quantiles trained only on earlier months",
            "time-hybrid": "B2 admission control from 08:00 to 18:00 UTC and runtime-aware B1 overnight",
        }[args.strategy],
        "b2_limitation": (
            "This is an admission/concurrency proxy, not a hardware DVFS or measured power cap"
            if args.strategy
            in {
                "b2",
                "b1-b2",
                "adaptive",
                "dynamic-marginal",
                "dynamic-forecast",
                "dynamic-history",
                "time-hybrid",
            }
            else None
        ),
        "workload_epoch_utc": epoch.isoformat(),
        "jobs": len(result),
        "flex_fraction": args.flex_fraction,
        "flexible_jobs": int(result["is_flexible"].sum()),
        "jobs_delayed": int(delayed.sum()),
        "median_policy_delay_s_all_jobs": float(result["policy_delay_s"].median()),
        "p95_policy_delay_s_all_jobs": float(result["policy_delay_s"].quantile(0.95)),
        "median_policy_delay_s_delayed_jobs": (
            float(result.loc[delayed, "policy_delay_s"].median()) if delayed.any() else 0.0
        ),
        "max_policy_delay_s": float(result["policy_delay_s"].max()),
        "parameters": {
            "b1_max_delay_hours": args.b1_max_delay_hours,
            "b2_max_delay_hours": args.b2_max_delay_hours,
            **b2_metadata,
        },
    }
    write_scenario(args.output, result, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
