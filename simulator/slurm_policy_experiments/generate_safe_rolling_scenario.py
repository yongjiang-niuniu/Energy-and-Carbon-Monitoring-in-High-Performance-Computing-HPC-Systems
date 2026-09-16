#!/usr/bin/env python3
"""Generate the frozen history-only safe rolling balanced scenario."""

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
        "projected_peak_node_load": np.nan,
        "dynamic_wait_ratio": 0.0,
        "dynamic_load_growth": 0.0,
        "dynamic_marginal_utility": 0.0,
        "predicted_baseline_start_s": np.nan,
        "predicted_policy_start_s": np.nan,
        "forecast_incremental_wait_s": 0.0,
        "safe_decision": "baseline",
        "safe_rejection_reason": "",
        "safe_candidate_count": 0,
    }
    for name, value in defaults.items():
        result[name] = value
    return result


def projected_node_peak(
    forecast: pd.DataFrame,
    epoch: pd.Timestamp,
    start_offset: float,
    runtime_s: float,
    node_fraction: float,
) -> float:
    contribution = POLICY.interval_job_load(
        forecast, epoch, start_offset, runtime_s, node_fraction
    )
    active = contribution > 0
    if not active.any():
        return node_fraction
    return float(
        (forecast["load_node"].to_numpy(float) + contribution)[active].max()
    )


def apply_safe_rolling_balanced(
    profile: pd.DataFrame,
    history_model: dict[str, object],
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    flex_fraction: float,
    max_delay_hours: float,
    min_carbon_saving_pct: float,
    min_node_saving_pct: float,
    wait_penalty: float,
    congestion_penalty: float,
    cap_fraction: float,
    min_carbon_return: float,
    carbon_uncertainty_fraction: float,
    runtime_uncertainty_limit: float,
    long_gpu_hours: float,
    long_gpu_uncertainty_limit: float,
    user_delay_budget_hours: float,
    user_delayed_job_limit: int,
    candidate_step_minutes: int = 30,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Apply deterministic dual-proxy decisions with conservative abstention."""
    result = initialize_columns(profile, flex_fraction)
    runtime_predictions = {
        label: HISTORY.predict_runtime(result, history_model, quantile)
        for label, quantile in [("q50", 0.50), ("q75", 0.75), ("q90", 0.90)]
    }
    for label, prediction in runtime_predictions.items():
        result[f"predicted_runtime_s_{label}"] = prediction["predicted_runtime_s"]
    result["decision_runtime_s"] = result["predicted_runtime_s_q90"]
    result["decision_runtime_source"] = "earlier_month_q90"
    result["runtime_model_level"] = runtime_predictions["q90"]["runtime_model_level"]
    result["runtime_model_count"] = runtime_predictions["q90"]["runtime_model_count"]
    result["runtime_uncertainty_ratio_q90_q50"] = (
        result["predicted_runtime_s_q90"]
        / result["predicted_runtime_s_q50"].clip(lower=60.0)
    )
    result["runtime_interval_wide_diagnostic"] = result[
        "runtime_uncertainty_ratio_q90_q50"
    ].gt(runtime_uncertainty_limit)

    forecast = HISTORY.history_load_forecast(carbon, history_model, 0.90)
    forecast_low = forecast.copy()
    forecast_high = forecast.copy()
    forecast_low["intensity_gco2_per_kwh"] *= 1 - carbon_uncertainty_fraction
    forecast_high["intensity_gco2_per_kwh"] *= 1 + carbon_uncertainty_fraction
    carbon_end = pd.Timestamp(carbon["to_utc"].max())
    candidate_step_s = candidate_step_minutes * 60
    user_delay_s: dict[str, float] = {}
    user_delayed_jobs: dict[str, int] = {}
    candidate_rows: list[dict[str, object]] = []

    order = result.sort_values(["release_dt_s", "sim_job_id"]).index
    for index in order:
        row = result.loc[index]
        job_id = str(row["sim_job_id"])
        user_id = str(row["user_id"])
        partition = str(row["partition"])
        class_name, class_fraction = POLICY.job_capacity_fraction(row)
        node_fraction = float(row["nodes"]) / POLICY.TOTAL_NODES
        eligible_offset = float(row["eligible_dt_s"])
        eligible = epoch + pd.Timedelta(seconds=eligible_offset)
        waits = {
            label: HISTORY.predict_wait(partition, eligible, history_model, quantile)
            for label, quantile in [("q50", 0.50), ("q75", 0.75), ("q90", 0.90)]
        }
        for label in waits:
            wait_s, _, _ = waits[label]
            result.at[index, f"predicted_queue_wait_s_{label}"] = min(
                wait_s, max_delay_hours * 3600
            )
        q90_wait = float(result.at[index, "predicted_queue_wait_s_q90"])
        q90_wait_level = str(waits["q90"][1])
        result.at[index, "queue_wait_model_level"] = q90_wait_level
        result.at[index, "queue_wait_model_count"] = int(waits["q90"][2])

        runtime_s = float(row["predicted_runtime_s_q90"])
        current_start = eligible_offset + q90_wait
        result.at[index, "predicted_baseline_start_s"] = current_start
        result.at[index, "predicted_policy_start_s"] = current_start
        before_intensity = POLICY.runtime_average_intensity(
            carbon, epoch + pd.Timedelta(seconds=current_start), runtime_s
        )
        result.at[index, "runtime_intensity_before"] = before_intensity
        result.at[index, "runtime_intensity_after"] = before_intensity

        initial_capacity_low, initial_node_low, initial_mean, initial_peak, _, _ = (
            POLICY.forecast_marginal_carbon_cost(
                forecast_low,
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
        initial_capacity_nominal, initial_node_nominal, _, _, _, _ = (
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
        initial_node_peak = projected_node_peak(
            forecast, epoch, current_start, runtime_s, node_fraction
        )

        uncertainty_ratio = float(row["runtime_uncertainty_ratio_q90_q50"])
        reasons: list[str] = []
        if not bool(row["is_flexible"]):
            reasons.append(
                "carry_in" if bool(row.get("is_warmup", False)) else "not_opted_in"
            )
        if str(row["runtime_model_level"]) == "global":
            reasons.append("runtime_outside_group_support")
        if q90_wait_level == "global":
            reasons.append("queue_wait_outside_group_support")
        if (
            float(row.get("scheduled_gpus", 0)) > 0
            and runtime_s > long_gpu_hours * 3600
            and uncertainty_ratio > long_gpu_uncertainty_limit
        ):
            reasons.append("long_gpu_runtime_uncertain")
        if initial_peak > 1.10:
            reasons.append("baseline_load_outside_history_support")

        declared_limit_s = max(float(row.get("source_timelimit_min", 0)) * 60, 0.0)
        remaining_user_budget_s = max(
            user_delay_budget_hours * 3600 - user_delay_s.get(user_id, 0.0), 0.0
        )
        budget_s = min(
            max_delay_hours * 3600,
            float(row["predicted_runtime_s_q75"]),
            declared_limit_s,
            remaining_user_budget_s,
        )
        result.at[index, "allowed_wait_budget_s"] = budget_s
        if user_delayed_jobs.get(user_id, 0) >= user_delayed_job_limit:
            reasons.append("user_delayed_job_limit")
        if budget_s < candidate_step_s:
            reasons.append("waiting_budget_below_one_step")

        options: list[dict[str, object]] = []
        if not reasons:
            for delay_s in range(candidate_step_s, int(budget_s) + 1, candidate_step_s):
                candidate_release = eligible_offset + delay_s
                candidate_time = epoch + pd.Timedelta(seconds=candidate_release)
                candidate_wait, _, _ = HISTORY.predict_wait(
                    partition, candidate_time, history_model, 0.90
                )
                candidate_wait = min(candidate_wait, max_delay_hours * 3600)
                candidate_start = candidate_release + candidate_wait
                if epoch + pd.Timedelta(seconds=candidate_start + runtime_s) > carbon_end:
                    candidate_rows.append(
                        {
                            "sim_job_id": job_id,
                            "candidate_delay_s": delay_s,
                            "accepted": False,
                            "rejection_reason": "carbon_horizon_incomplete",
                        }
                    )
                    continue

                cap_high, node_high, mean_load, peak_load, class_delta, node_delta = (
                    POLICY.forecast_marginal_carbon_cost(
                        forecast_high,
                        epoch,
                        class_name,
                        current_start,
                        candidate_start,
                        runtime_s,
                        class_fraction,
                        node_fraction,
                        forecast_contains_current=False,
                    )
                )
                cap_nominal, node_nominal, _, _, _, _ = (
                    POLICY.forecast_marginal_carbon_cost(
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
                )
                node_peak = projected_node_peak(
                    forecast, epoch, candidate_start, runtime_s, node_fraction
                )
                cap_saving = initial_capacity_low - cap_high
                node_saving = initial_node_low - node_high
                cap_pct = cap_saving / initial_capacity_low * 100 if initial_capacity_low else 0.0
                node_pct = node_saving / initial_node_low * 100 if initial_node_low else 0.0
                nominal_cap_saving = initial_capacity_nominal - cap_nominal
                nominal_node_saving = initial_node_nominal - node_nominal
                robust_saving = min(cap_saving, node_saving)
                carbon_return = robust_saving / max(delay_s / 3600, 0.5)
                wait_ratio = delay_s / max(runtime_s, 60.0)
                load_growth = max(0.0, mean_load - initial_mean) + max(
                    0.0, peak_load - max(initial_peak, cap_fraction)
                )
                utility = (
                    robust_saving
                    - wait_penalty * wait_ratio
                    - congestion_penalty * load_growth
                )
                candidate_reasons: list[str] = []
                if cap_pct < min_carbon_saving_pct:
                    candidate_reasons.append("capacity_proxy_below_threshold")
                if node_pct < min_node_saving_pct:
                    candidate_reasons.append("node_proxy_below_threshold")
                if nominal_cap_saving > 0 and nominal_node_saving > 0 and robust_saving <= 0:
                    candidate_reasons.append("carbon_noise_sensitive")
                if peak_load > max(initial_peak, cap_fraction) + 0.02:
                    candidate_reasons.append("class_congestion_growth")
                if peak_load > 1.10:
                    candidate_reasons.append("candidate_load_outside_history_support")
                if node_peak > max(initial_node_peak, cap_fraction) + 0.02:
                    candidate_reasons.append("node_congestion_growth")
                if node_peak > 1.10:
                    candidate_reasons.append("candidate_node_load_outside_history_support")
                if carbon_return < min_carbon_return:
                    candidate_reasons.append("carbon_return_below_threshold")
                if utility <= 0:
                    candidate_reasons.append("non_positive_utility")

                option = {
                    "release": candidate_release,
                    "start": candidate_start,
                    "delay": float(delay_s),
                    "capacity_pct": cap_pct,
                    "node_pct": node_pct,
                    "robust_saving": robust_saving,
                    "return": carbon_return,
                    "wait_ratio": wait_ratio,
                    "load_growth": load_growth,
                    "mean_load": mean_load,
                    "peak_load": peak_load,
                    "node_peak": node_peak,
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
                        "predicted_start_s": candidate_start,
                        "predicted_runtime_s_q50": float(row["predicted_runtime_s_q50"]),
                        "predicted_runtime_s_q75": float(row["predicted_runtime_s_q75"]),
                        "predicted_runtime_s_q90": runtime_s,
                        "predicted_wait_s_q50": float(result.at[index, "predicted_queue_wait_s_q50"]),
                        "predicted_wait_s_q75": float(result.at[index, "predicted_queue_wait_s_q75"]),
                        "predicted_wait_s_q90": candidate_wait,
                        "capacity_saving_pct_robust": cap_pct,
                        "node_saving_pct_robust": node_pct,
                        "robust_saving_proxy": robust_saving,
                        "carbon_return_per_wait_hour": carbon_return,
                        "projected_mean_class_load_q90": mean_load,
                        "projected_peak_class_load_q90": peak_load,
                        "projected_peak_node_load_q90": node_peak,
                        "utility": utility,
                        "accepted": False,
                        "rejection_reason": ";".join(candidate_reasons),
                    }
                )

        result.at[index, "safe_candidate_count"] = len(options)
        feasible = [item for item in options if not item["reasons"]]
        if feasible:
            chosen = max(feasible, key=lambda item: (item["utility"], -item["release"]))
            delay_s = float(chosen["delay"])
            result.at[index, "release_dt_s"] = int(round(float(chosen["release"])))
            result.at[index, "b1_delay_s"] = delay_s
            result.at[index, "safe_decision"] = "delay"
            result.at[index, "expected_runtime_carbon_saving_pct"] = min(
                float(chosen["capacity_pct"]), float(chosen["node_pct"])
            )
            result.at[index, "expected_capacity_carbon_saving_pct"] = chosen["capacity_pct"]
            result.at[index, "expected_node_carbon_saving_pct"] = chosen["node_pct"]
            result.at[index, "expected_risk_adjusted_carbon_saving_proxy"] = chosen["robust_saving"]
            result.at[index, "carbon_return_per_wait_hour"] = chosen["return"]
            result.at[index, "projected_mean_class_load"] = chosen["mean_load"]
            result.at[index, "projected_peak_class_load"] = chosen["peak_load"]
            result.at[index, "projected_peak_node_load"] = chosen["node_peak"]
            result.at[index, "dynamic_wait_ratio"] = chosen["wait_ratio"]
            result.at[index, "dynamic_load_growth"] = chosen["load_growth"]
            result.at[index, "dynamic_marginal_utility"] = chosen["utility"]
            result.at[index, "predicted_policy_start_s"] = chosen["start"]
            result.at[index, "forecast_incremental_wait_s"] = max(
                float(chosen["start"]) - current_start, 0.0
            )
            result.at[index, "runtime_intensity_after"] = POLICY.runtime_average_intensity(
                carbon, epoch + pd.Timedelta(seconds=float(chosen["start"])), runtime_s
            )
            user_delay_s[user_id] = user_delay_s.get(user_id, 0.0) + delay_s
            user_delayed_jobs[user_id] = user_delayed_jobs.get(user_id, 0) + 1
            for target in (forecast, forecast_low, forecast_high):
                target[f"load_{class_name}"] = np.maximum(
                    target[f"load_{class_name}"] + chosen["class_delta"], 0.0
                )
                target["load_node"] = np.maximum(
                    target["load_node"] + chosen["node_delta"], 0.0
                )
            for candidate in reversed(candidate_rows):
                if (
                    candidate.get("sim_job_id") == job_id
                    and candidate.get("candidate_delay_s") == delay_s
                ):
                    candidate["accepted"] = True
                    candidate["rejection_reason"] = ""
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
            result.at[index, "safe_decision"] = "abstain"
            result.at[index, "safe_rejection_reason"] = final_reason

    metadata = history_model["metadata"]
    assert isinstance(metadata, dict)
    delayed = result["safe_decision"].eq("delay")
    policy_metadata = {
        "schedule_forecast": "earlier-month q50/q75/q90 runtime and wait, q90 interval load",
        "forecast_status": "history-only deployment direction; no held-out schedule or observed runtime is read",
        "training_months": metadata["training_months"],
        "holdout_month": metadata["holdout_month"],
        "temporal_split_pass": metadata["temporal_split_pass"],
        "candidate_step_minutes": candidate_step_minutes,
        "maximum_delay_hours": max_delay_hours,
        "carbon_uncertainty_fraction": carbon_uncertainty_fraction,
        "minimum_expected_capacity_saving_pct": min_carbon_saving_pct,
        "minimum_expected_node_saving_pct": min_node_saving_pct,
        "minimum_carbon_return_per_wait_hour": min_carbon_return,
        "runtime_uncertainty_diagnostic_q90_q50": runtime_uncertainty_limit,
        "runtime_uncertainty_rule": "Wide intervals are recorded, while the q90 runtime is used for conservative cost; only uncertain long GPU jobs are hard-abstained",
        "long_gpu_hours": long_gpu_hours,
        "long_gpu_uncertainty_limit_q90_q50": long_gpu_uncertainty_limit,
        "per_user_delay_budget_hours": user_delay_budget_hours,
        "per_user_delayed_job_limit": user_delayed_job_limit,
        "users_with_policy_delay": int(result.loc[delayed, "user_id"].nunique()),
        "maximum_user_policy_delay_hours": max(user_delay_s.values(), default=0.0) / 3600,
        "abstention_reason_counts": result.loc[
            result["safe_decision"].eq("abstain"), "safe_rejection_reason"
        ].value_counts().to_dict(),
        "load_update_rule": "Accepted moves modify the q90 class and node forecasts seen by later jobs",
        "deadline_limitation": "No user deadline is present; declared Slurm time limit is only a conservative delay-budget bound",
        "carbon_interval_limitation": "The frozen sensitivity band is not an archived as-issued forecast interval",
    }
    return result, policy_metadata, pd.DataFrame(candidate_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--carbon", type=Path, required=True)
    parser.add_argument("--history-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flex-fraction", type=float, default=0.30)
    parser.add_argument("--max-delay-hours", type=float, default=4.0)
    parser.add_argument("--minimum-saving-pct", type=float, default=2.0)
    parser.add_argument("--minimum-node-saving-pct", type=float, default=2.0)
    parser.add_argument("--wait-penalty", type=float, default=0.10)
    parser.add_argument("--congestion-penalty", type=float, default=0.25)
    parser.add_argument("--cap-fraction", type=float, default=0.75)
    parser.add_argument("--minimum-carbon-return", type=float, default=0.05)
    parser.add_argument("--carbon-uncertainty-fraction", type=float, default=0.10)
    parser.add_argument("--runtime-uncertainty-limit", type=float, default=1.75)
    parser.add_argument("--long-gpu-hours", type=float, default=12.0)
    parser.add_argument("--long-gpu-uncertainty-limit", type=float, default=1.50)
    parser.add_argument("--user-delay-budget-hours", type=float, default=4.0)
    parser.add_argument("--user-delayed-job-limit", type=int, default=2)
    parser.add_argument("--candidate-step-minutes", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_step_minutes <= 0 or 60 % args.candidate_step_minutes:
        raise ValueError("candidate-step-minutes must be a positive divisor of 60")
    if not 0 <= args.carbon_uncertainty_fraction < 1:
        raise ValueError("carbon-uncertainty-fraction must be in [0,1)")
    if args.user_delayed_job_limit < 1:
        raise ValueError("user-delayed-job-limit must be positive")

    profile = pd.read_csv(args.baseline_profile)
    carbon = POLICY.load_carbon(args.carbon)
    history_model = HISTORY.load_model(args.history_model_dir)
    epoch = POLICY.workload_epoch(profile)
    working, metadata, candidates = apply_safe_rolling_balanced(
        profile,
        history_model,
        carbon,
        epoch,
        args.flex_fraction,
        args.max_delay_hours,
        args.minimum_saving_pct,
        args.minimum_node_saving_pct,
        args.wait_penalty,
        args.congestion_penalty,
        args.cap_fraction,
        args.minimum_carbon_return,
        args.carbon_uncertainty_fraction,
        args.runtime_uncertainty_limit,
        args.long_gpu_hours,
        args.long_gpu_uncertainty_limit,
        args.user_delay_budget_hours,
        args.user_delayed_job_limit,
        args.candidate_step_minutes,
    )
    result = POLICY.finalize_profile(working, "safe-rolling-balanced")
    delayed = result["policy_delay_s"].gt(0)
    summary = {
        "strategy": "safe-rolling-balanced",
        "policy_interpretation": (
            "Thirty-minute rolling history-only admission with conservative dual-proxy "
            "carbon gates, uncertainty abstention, sequential load updates and user budgets"
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
    candidates.to_csv(args.output / "safe_rolling_candidate_audit.csv", index=False)
    decision_columns = [
        "sim_job_id",
        "user_id",
        "carry_in_type",
        "partition",
        "is_flexible",
        "safe_decision",
        "safe_rejection_reason",
        "safe_candidate_count",
        "eligible_dt_s",
        "release_dt_s",
        "policy_delay_s",
        "allowed_wait_budget_s",
        "predicted_runtime_s_q50",
        "predicted_runtime_s_q75",
        "predicted_runtime_s_q90",
        "runtime_uncertainty_ratio_q90_q50",
        "runtime_interval_wide_diagnostic",
        "predicted_queue_wait_s_q50",
        "predicted_queue_wait_s_q75",
        "predicted_queue_wait_s_q90",
        "expected_capacity_carbon_saving_pct",
        "expected_node_carbon_saving_pct",
        "carbon_return_per_wait_hour",
        "projected_peak_class_load",
        "projected_peak_node_load",
    ]
    result[[name for name in decision_columns if name in result]].to_csv(
        args.output / "safe_rolling_job_decisions.csv", index=False
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
