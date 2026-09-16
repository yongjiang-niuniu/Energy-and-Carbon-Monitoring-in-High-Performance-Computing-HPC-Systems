#!/usr/bin/env python3
"""Generate a history-only carbon policy with strict per-job and per-user delay caps."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
POLICY_SPEC = importlib.util.spec_from_file_location(
    "generate_policy_scenario", ROOT / "generate_policy_scenario.py"
)
POLICY = importlib.util.module_from_spec(POLICY_SPEC)
assert POLICY_SPEC.loader is not None
POLICY_SPEC.loader.exec_module(POLICY)
HISTORY = POLICY.HISTORY


def initialize_columns(profile: pd.DataFrame, flex_fraction: float) -> pd.DataFrame:
    result = profile.copy()
    result["is_flexible"] = POLICY.flexible_mask(result, flex_fraction)
    defaults: dict[str, object] = {
        "b1_delay_s": 0.0,
        "b2_delay_s": 0.0,
        "runtime_intensity_before": np.nan,
        "runtime_intensity_after": np.nan,
        "expected_runtime_carbon_saving_pct": 0.0,
        "expected_capacity_carbon_saving_pct": 0.0,
        "expected_node_carbon_saving_pct": 0.0,
        "expected_risk_adjusted_carbon_saving_proxy": 0.0,
        "carbon_return_per_wait_hour": 0.0,
        "allowed_wait_budget_s": 0.0,
        "projected_mean_class_load": np.nan,
        "projected_peak_class_load": np.nan,
        "dynamic_wait_ratio": 0.0,
        "dynamic_load_growth": 0.0,
        "dynamic_marginal_utility": 0.0,
        "predicted_baseline_start_s": np.nan,
        "predicted_policy_start_s": np.nan,
        "forecast_incremental_wait_s": 0.0,
        "low_impact_decision": "baseline",
        "low_impact_rejection_reason": "",
        "low_impact_candidate_count": 0,
    }
    for name, value in defaults.items():
        result[name] = value
    return result


def bounded_candidate_release_times(
    carbon: pd.DataFrame,
    eligible: pd.Timestamp,
    deadline: pd.Timestamp,
    runtime_s: float,
    step_minutes: int,
) -> list[pd.Timestamp]:
    """Combine carbon boundaries with a fine delay grid and the exact deadline."""
    carbon_end = pd.Timestamp(carbon["to_utc"].max())
    candidates = set(
        POLICY.candidate_release_times(carbon, eligible, deadline, runtime_s)
    )
    step = pd.Timedelta(minutes=step_minutes)
    candidate = eligible + step
    while candidate < deadline:
        if candidate + pd.Timedelta(seconds=runtime_s) <= carbon_end:
            candidates.add(candidate)
        candidate += step
    if deadline + pd.Timedelta(seconds=runtime_s) <= carbon_end:
        candidates.add(deadline)
    return sorted(candidates)


def apply_low_impact_dynamic(
    profile: pd.DataFrame,
    history_model: dict[str, object],
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_minutes: int,
    runtime_budget_ratio: float,
    min_capacity_saving_pct: float,
    min_carbon_return: float,
    wait_penalty: float,
    congestion_penalty: float,
    cap_fraction: float,
    max_runtime_uncertainty_ratio: float,
    long_gpu_hours: float,
    long_gpu_uncertainty_limit: float,
    user_delay_budget_minutes: int,
    user_delayed_job_limit: int,
    load_quantile: float = 0.75,
    queue_wait_quantile: float = 0.50,
    candidate_step_minutes: int = 5,
    delay_runtime_quantile: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Apply a deployable capacity-first policy using earlier-month aggregates only."""
    result = initialize_columns(profile, flex_fraction)
    predictions = {
        label: HISTORY.predict_runtime(result, history_model, quantile)
        for label, quantile in [
            ("q10", 0.10),
            ("q25", 0.25),
            ("q50", 0.50),
            ("q75", 0.75),
            ("q90", 0.90),
        ]
    }
    for label, prediction in predictions.items():
        result[f"predicted_runtime_s_{label}"] = prediction["predicted_runtime_s"]
    result["decision_runtime_s"] = result["predicted_runtime_s_q90"]
    result["decision_runtime_source"] = "earlier_month_q90"
    result["runtime_model_level"] = predictions["q90"]["runtime_model_level"]
    result["runtime_model_count"] = predictions["q90"]["runtime_model_count"]
    result["runtime_uncertainty_ratio_q90_q50"] = (
        result["predicted_runtime_s_q90"]
        / result["predicted_runtime_s_q50"].clip(lower=60.0)
    )

    forecast = HISTORY.history_load_forecast(carbon, history_model, load_quantile)
    carbon_end = pd.Timestamp(carbon["to_utc"].max())
    user_delay_s: dict[str, float] = {}
    user_delayed_jobs: dict[str, int] = {}
    candidate_rows: list[dict[str, object]] = []

    for index in result.sort_values(["release_dt_s", "sim_job_id"]).index:
        row = result.loc[index]
        job_id = str(row["sim_job_id"])
        user_id = str(row["user_id"])
        partition = str(row["partition"])
        eligible_offset = float(row["eligible_dt_s"])
        eligible = epoch + pd.Timedelta(seconds=eligible_offset)
        baseline_wait, wait_level, wait_count = HISTORY.predict_wait(
            partition, eligible, history_model, queue_wait_quantile
        )
        current_start = eligible_offset + max(float(baseline_wait), 0.0)
        runtime_s = float(row["predicted_runtime_s_q90"])
        class_name, class_fraction = POLICY.job_capacity_fraction(row)
        node_fraction = float(row["nodes"]) / POLICY.TOTAL_NODES

        result.at[index, "predicted_baseline_start_s"] = current_start
        result.at[index, "predicted_policy_start_s"] = current_start
        result.at[index, "queue_wait_model_level"] = wait_level
        result.at[index, "queue_wait_model_count"] = wait_count

        if epoch + pd.Timedelta(seconds=current_start + runtime_s) > carbon_end:
            result.at[index, "low_impact_decision"] = "abstain"
            result.at[index, "low_impact_rejection_reason"] = "carbon_horizon_incomplete"
            continue

        before_intensity = POLICY.runtime_average_intensity(
            carbon, epoch + pd.Timedelta(seconds=current_start), runtime_s
        )
        result.at[index, "runtime_intensity_before"] = before_intensity
        result.at[index, "runtime_intensity_after"] = before_intensity
        initial_capacity, initial_node, initial_mean, initial_peak, _, _ = (
            POLICY.forecast_marginal_carbon_cost(
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

        reasons: list[str] = []
        if not bool(row["is_flexible"]):
            reasons.append(
                "carry_in" if bool(row.get("is_warmup", False)) else "not_opted_in"
            )
        if str(row["runtime_model_level"]) == "global":
            reasons.append("runtime_outside_group_support")
        uncertainty_ratio = float(row["runtime_uncertainty_ratio_q90_q50"])
        if uncertainty_ratio > max_runtime_uncertainty_ratio:
            reasons.append("runtime_interval_too_wide")
        if (
            float(row.get("scheduled_gpus", 0)) > 0
            and runtime_s > long_gpu_hours * 3600
            and uncertainty_ratio > long_gpu_uncertainty_limit
        ):
            reasons.append("long_gpu_runtime_uncertain")
        if user_delayed_jobs.get(user_id, 0) >= user_delayed_job_limit:
            reasons.append("user_delayed_job_limit")

        remaining_user_budget_s = max(
            user_delay_budget_minutes * 60 - user_delay_s.get(user_id, 0.0), 0.0
        )
        declared_limit_s = max(float(row.get("source_timelimit_min", 0)) * 60, 0.0)
        budget_runtime_column = f"predicted_runtime_s_{HISTORY.quantile_label(delay_runtime_quantile)}"
        budget_s = min(
            max_delay_minutes * 60,
            float(row[budget_runtime_column]) * runtime_budget_ratio,
            declared_limit_s,
            remaining_user_budget_s,
        )
        result.at[index, "allowed_wait_budget_s"] = budget_s
        if budget_s <= 0:
            reasons.append("waiting_budget_exhausted")

        options: list[dict[str, object]] = []
        if not reasons:
            deadline = eligible + pd.Timedelta(seconds=budget_s)
            for candidate in bounded_candidate_release_times(
                carbon, eligible, deadline, runtime_s, candidate_step_minutes
            ):
                candidate_offset = (candidate - epoch).total_seconds()
                delay_s = max(candidate_offset - eligible_offset, 0.0)
                if delay_s <= 0:
                    continue
                candidate_wait, _, _ = HISTORY.predict_wait(
                    partition, candidate, history_model, queue_wait_quantile
                )
                candidate_start = candidate_offset + max(float(candidate_wait), 0.0)
                if epoch + pd.Timedelta(seconds=candidate_start + runtime_s) > carbon_end:
                    continue
                (
                    candidate_capacity,
                    candidate_node,
                    mean_load,
                    peak_load,
                    class_delta,
                    node_delta,
                ) = POLICY.forecast_marginal_carbon_cost(
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
                capacity_pct = (
                    capacity_saving / initial_capacity * 100 if initial_capacity else 0.0
                )
                node_pct = node_saving / initial_node * 100 if initial_node else 0.0
                incremental_wait = max(candidate_start - current_start, 0.0)
                wait_ratio = incremental_wait / max(runtime_s, 60.0)
                initial_over_cap = max(initial_peak - cap_fraction, 0.0)
                candidate_over_cap = max(peak_load - cap_fraction, 0.0)
                load_growth = max(mean_load - initial_mean, 0.0) + max(
                    candidate_over_cap - initial_over_cap, 0.0
                )
                carbon_return = capacity_saving / max(delay_s / 3600, 0.25)
                utility = (
                    capacity_saving
                    - wait_penalty * wait_ratio
                    - congestion_penalty * load_growth
                )
                candidate_reasons: list[str] = []
                if capacity_pct < min_capacity_saving_pct:
                    candidate_reasons.append("capacity_saving_below_threshold")
                if carbon_return < min_carbon_return:
                    candidate_reasons.append("carbon_return_below_threshold")
                if peak_load > max(initial_peak, cap_fraction) + 0.02:
                    candidate_reasons.append("class_congestion_growth")
                if peak_load > 1.10:
                    candidate_reasons.append("candidate_load_outside_history_support")
                if utility <= 0:
                    candidate_reasons.append("non_positive_utility")
                option = {
                    "release": candidate_offset,
                    "start": candidate_start,
                    "delay": delay_s,
                    "capacity_pct": capacity_pct,
                    "node_pct": node_pct,
                    "capacity_saving": capacity_saving,
                    "carbon_return": carbon_return,
                    "incremental_wait": incremental_wait,
                    "wait_ratio": wait_ratio,
                    "load_growth": load_growth,
                    "mean_load": mean_load,
                    "peak_load": peak_load,
                    "utility": utility,
                    "class_delta": class_delta,
                    "node_delta": node_delta,
                    "reasons": candidate_reasons,
                }
                options.append(option)
                candidate_rows.append(
                    {
                        "sim_job_id": job_id,
                        "user_id": user_id,
                        "candidate_delay_s": delay_s,
                        "predicted_incremental_wait_s": incremental_wait,
                        "capacity_saving_pct": capacity_pct,
                        "node_upper_saving_pct_diagnostic": node_pct,
                        "carbon_return_per_wait_hour": carbon_return,
                        "utility": utility,
                        "accepted": False,
                        "rejection_reason": ";".join(candidate_reasons),
                    }
                )

        result.at[index, "low_impact_candidate_count"] = len(options)
        feasible = [option for option in options if not option["reasons"]]
        if feasible:
            chosen = max(feasible, key=lambda item: (item["utility"], -item["release"]))
            delay_s = float(chosen["delay"])
            result.at[index, "release_dt_s"] = int(round(float(chosen["release"])))
            result.at[index, "b1_delay_s"] = delay_s
            result.at[index, "low_impact_decision"] = "delay"
            result.at[index, "expected_runtime_carbon_saving_pct"] = chosen[
                "capacity_pct"
            ]
            result.at[index, "expected_capacity_carbon_saving_pct"] = chosen[
                "capacity_pct"
            ]
            result.at[index, "expected_node_carbon_saving_pct"] = chosen["node_pct"]
            result.at[index, "expected_risk_adjusted_carbon_saving_proxy"] = chosen[
                "capacity_saving"
            ]
            result.at[index, "carbon_return_per_wait_hour"] = chosen["carbon_return"]
            result.at[index, "projected_mean_class_load"] = chosen["mean_load"]
            result.at[index, "projected_peak_class_load"] = chosen["peak_load"]
            result.at[index, "dynamic_wait_ratio"] = chosen["wait_ratio"]
            result.at[index, "dynamic_load_growth"] = chosen["load_growth"]
            result.at[index, "dynamic_marginal_utility"] = chosen["utility"]
            result.at[index, "predicted_policy_start_s"] = chosen["start"]
            result.at[index, "forecast_incremental_wait_s"] = chosen[
                "incremental_wait"
            ]
            result.at[index, "runtime_intensity_after"] = POLICY.runtime_average_intensity(
                carbon,
                epoch + pd.Timedelta(seconds=float(chosen["start"])),
                runtime_s,
            )
            user_delay_s[user_id] = user_delay_s.get(user_id, 0.0) + delay_s
            user_delayed_jobs[user_id] = user_delayed_jobs.get(user_id, 0) + 1
            forecast[f"load_{class_name}"] = np.maximum(
                forecast[f"load_{class_name}"] + chosen["class_delta"], 0.0
            )
            forecast["load_node"] = np.maximum(
                forecast["load_node"] + chosen["node_delta"], 0.0
            )
            for candidate_row in reversed(candidate_rows):
                if (
                    candidate_row["sim_job_id"] == job_id
                    and candidate_row["candidate_delay_s"] == delay_s
                ):
                    candidate_row["accepted"] = True
                    candidate_row["rejection_reason"] = ""
                    break
        else:
            if reasons:
                final_reason = ";".join(dict.fromkeys(reasons))
            elif options:
                final_reason = ";".join(
                    dict.fromkeys(
                        reason for option in options for reason in option["reasons"]
                    )
                )
            else:
                final_reason = "no_complete_candidate"
            result.at[index, "low_impact_decision"] = "abstain"
            result.at[index, "low_impact_rejection_reason"] = final_reason

    metadata = history_model["metadata"]
    assert isinstance(metadata, dict)
    policy_metadata = {
        "schedule_forecast": (
            "earlier-month q10/q25/q50/q75/q90 runtime, q50 wait and q75 interval load"
        ),
        "forecast_status": "history-only; no held-out runtime or future schedule is read",
        "training_months": metadata["training_months"],
        "holdout_month": metadata["holdout_month"],
        "temporal_split_pass": metadata["temporal_split_pass"],
        "primary_gate": "capacity-weighted marginal carbon",
        "node_request_upper_role": "diagnostic only because shared-node requests saturate the upper proxy",
        "maximum_delay_minutes": max_delay_minutes,
        "runtime_budget_quantile": delay_runtime_quantile,
        "runtime_budget_ratio": runtime_budget_ratio,
        "decision_runtime_quantile": 0.90,
        "load_quantile": load_quantile,
        "queue_wait_quantile": queue_wait_quantile,
        "candidate_step_minutes": candidate_step_minutes,
        "minimum_capacity_saving_pct": min_capacity_saving_pct,
        "minimum_carbon_return_per_wait_hour": min_carbon_return,
        "maximum_runtime_uncertainty_ratio_q90_q50": max_runtime_uncertainty_ratio,
        "per_user_delay_budget_minutes": user_delay_budget_minutes,
        "per_user_delayed_job_limit": user_delayed_job_limit,
        "maximum_user_policy_delay_minutes": max(user_delay_s.values(), default=0.0) / 60,
        "users_with_policy_delay": len(user_delay_s),
        "load_update_rule": "accepted moves update later q75 class and node forecasts",
    }
    return result, policy_metadata, pd.DataFrame(candidate_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--carbon", type=Path, required=True)
    parser.add_argument("--history-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flex-fraction", type=float, default=0.30)
    parser.add_argument("--max-delay-minutes", type=int, default=30)
    parser.add_argument("--runtime-budget-ratio", type=float, default=0.50)
    parser.add_argument(
        "--delay-runtime-quantile", type=float, choices=[0.10, 0.25, 0.50], default=0.25
    )
    parser.add_argument("--minimum-capacity-saving-pct", type=float, default=1.0)
    parser.add_argument("--minimum-carbon-return", type=float, default=0.001)
    parser.add_argument("--wait-penalty", type=float, default=0.01)
    parser.add_argument("--congestion-penalty", type=float, default=0.25)
    parser.add_argument("--cap-fraction", type=float, default=0.90)
    parser.add_argument("--max-runtime-uncertainty-ratio", type=float, default=20.0)
    parser.add_argument("--long-gpu-hours", type=float, default=12.0)
    parser.add_argument("--long-gpu-uncertainty-limit", type=float, default=1.50)
    parser.add_argument("--user-delay-budget-minutes", type=int, default=30)
    parser.add_argument("--user-delayed-job-limit", type=int, default=2)
    parser.add_argument("--candidate-step-minutes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_delay_minutes <= 0 or args.user_delay_budget_minutes <= 0:
        raise ValueError("delay budgets must be positive")
    if args.user_delayed_job_limit <= 0:
        raise ValueError("user-delayed-job-limit must be positive")
    if args.candidate_step_minutes <= 0:
        raise ValueError("candidate-step-minutes must be positive")
    profile = pd.read_csv(args.baseline_profile)
    carbon = POLICY.load_carbon(args.carbon)
    model = HISTORY.load_model(args.history_model_dir)
    epoch = POLICY.workload_epoch(profile)
    working, metadata, candidates = apply_low_impact_dynamic(
        profile,
        model,
        carbon,
        epoch,
        args.flex_fraction,
        args.max_delay_minutes,
        args.runtime_budget_ratio,
        args.minimum_capacity_saving_pct,
        args.minimum_carbon_return,
        args.wait_penalty,
        args.congestion_penalty,
        args.cap_fraction,
        args.max_runtime_uncertainty_ratio,
        args.long_gpu_hours,
        args.long_gpu_uncertainty_limit,
        args.user_delay_budget_minutes,
        args.user_delayed_job_limit,
        candidate_step_minutes=args.candidate_step_minutes,
        delay_runtime_quantile=args.delay_runtime_quantile,
    )
    result = POLICY.finalize_profile(working, "low-impact-dynamic")
    delayed = result["policy_delay_s"].gt(0)
    delay_quantile_label = f"q{int(round(args.delay_runtime_quantile * 100))}"
    summary = {
        "strategy": "low-impact-dynamic",
        "policy_interpretation": (
            "History-only capacity-first admission with q90 runtime costing, "
            f"{delay_quantile_label} x {args.runtime_budget_ratio:g} predicted-runtime "
            "delay caps, sequential load updates and hard per-user budgets"
        ),
        "workload_epoch_utc": epoch.isoformat(),
        "jobs": len(result),
        "flex_fraction": args.flex_fraction,
        "flexible_jobs": int(result["is_flexible"].sum()),
        "jobs_delayed": int(delayed.sum()),
        "total_policy_delay_hours": float(result["policy_delay_s"].sum()) / 3600,
        "p95_policy_delay_s_all_jobs": float(result["policy_delay_s"].quantile(0.95)),
        "max_policy_delay_s": float(result["policy_delay_s"].max()),
        "parameters": metadata,
    }
    POLICY.write_scenario(args.output, result, summary)
    candidates.to_csv(args.output / "low_impact_candidate_audit.csv", index=False)
    result.to_csv(args.output / "low_impact_job_decisions.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
