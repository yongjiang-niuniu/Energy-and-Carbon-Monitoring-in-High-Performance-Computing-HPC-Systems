#!/usr/bin/env python3
"""Build extended wait, fairness, load and forecast metrics for formal replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def percentile(values: pd.Series, quantile: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if len(numeric) else None


def concentration(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    total = float(numeric.sum())
    if total <= 0:
        return 0.0, 0.0
    shares = numeric / total
    return float(shares.max()), float(np.square(shares).sum())


def prediction_metrics(
    frame: pd.DataFrame, actual_column: str, prefix: str
) -> dict[str, object]:
    output: dict[str, object] = {}
    actual = pd.to_numeric(frame[actual_column], errors="coerce")
    for label in ["q50", "q75", "q90"]:
        column = f"predicted_{prefix}_s_{label}"
        if column not in frame:
            continue
        predicted = pd.to_numeric(frame[column], errors="coerce")
        valid = actual.notna() & predicted.notna()
        if not valid.any():
            continue
        error = predicted.loc[valid] - actual.loc[valid]
        output[f"{prefix}_{label}_mae_s"] = float(error.abs().mean())
        output[f"{prefix}_{label}_bias_s"] = float(error.mean())
        output[f"{prefix}_{label}_coverage_pct"] = float(
            actual.loc[valid].le(predicted.loc[valid]).mean() * 100
        )
    return output


def load_peaks(result_dir: Path, scenario: str) -> dict[str, float]:
    path = result_dir / f"{scenario}_carbon_intervals.csv"
    intervals = pd.read_csv(path)
    columns = [
        "util_cpu",
        "util_a100",
        "util_h100",
        "util_h100_nvl",
        "util_capacity_weighted",
        "util_node_request_upper",
    ]
    return {
        f"peak_{column}": float(pd.to_numeric(intervals[column], errors="coerce").max())
        for column in columns
        if column in intervals
    }


def scenario_metrics(
    scenario_path: Path,
    carbon_row: pd.Series,
    result_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    jobs = pd.read_csv(scenario_path / "job_results.csv")
    if "is_evaluation" in jobs:
        evaluation = jobs["is_evaluation"].fillna(False).astype(bool)
    elif "is_warmup" in jobs:
        evaluation = ~jobs["is_warmup"].fillna(False).astype(bool)
    else:
        evaluation = pd.Series(True, index=jobs.index)
    frame = jobs.loc[evaluation].copy()
    delay_s = pd.to_numeric(frame["sim_policy_delay_s"], errors="coerce").fillna(0)
    total_wait_s = pd.to_numeric(frame["sim_total_user_wait_s"], errors="coerce")
    frame["_policy_delay_s"] = delay_s
    frame["_total_wait_s"] = total_wait_s
    per_user = frame.groupby("user_id", as_index=False).agg(
        jobs=("sim_job_id", "count"),
        delayed_jobs=("_policy_delay_s", lambda values: int((values > 0).sum())),
        total_policy_delay_s=("_policy_delay_s", "sum"),
        mean_policy_delay_s=("_policy_delay_s", "mean"),
        mean_total_wait_s=("_total_wait_s", "mean"),
        p95_total_wait_s=("_total_wait_s", lambda values: float(values.quantile(0.95))),
    )
    per_user.insert(0, "scenario", scenario_path.name)
    top_share, hhi = concentration(per_user["total_policy_delay_s"])

    end_offset = pd.to_numeric(frame["sim_end_offset_s"], errors="coerce")
    eligible_offset = pd.to_numeric(frame["eligible_dt_s"], errors="coerce")
    elapsed_h = float(end_offset.max() - eligible_offset.min()) / 3600
    completion_ratio = float(frame["terminal_status"].eq("completed").mean())
    total_delay_h = float(delay_s.sum()) / 3600
    cap_reduction_kg = float(
        carbon_row["dynamic_carbon_reduction_kg_capacity_weighted"]
    )
    node_reduction_kg = float(
        carbon_row["dynamic_carbon_reduction_kg_node_request_upper"]
    )
    metrics: dict[str, object] = {
        "scenario": scenario_path.name,
        "evaluation_jobs": len(frame),
        "completion_ratio": completion_ratio,
        "throughput_jobs_per_h_from_first_eligible_to_last_completion": (
            len(frame) / elapsed_h if elapsed_h > 0 else None
        ),
        "jobs_with_policy_delay": int(delay_s.gt(0).sum()),
        "delayed_job_fraction": float(delay_s.gt(0).mean()),
        "total_policy_delay_h": total_delay_h,
        "policy_delay_p50_s": percentile(delay_s, 0.50),
        "policy_delay_p95_s": percentile(delay_s, 0.95),
        "policy_delay_p99_s": percentile(delay_s, 0.99),
        "total_wait_p50_s": percentile(total_wait_s, 0.50),
        "total_wait_p95_s": percentile(total_wait_s, 0.95),
        "total_wait_p99_s": percentile(total_wait_s, 0.99),
        "users": int(frame["user_id"].nunique()),
        "users_with_policy_delay": int(per_user["delayed_jobs"].gt(0).sum()),
        "per_user_total_policy_delay_p95_h": (
            float(per_user["total_policy_delay_s"].quantile(0.95)) / 3600
        ),
        "per_user_mean_wait_p95_s": float(per_user["mean_total_wait_s"].quantile(0.95)),
        "maximum_user_total_policy_delay_h": float(
            per_user["total_policy_delay_s"].max()
        )
        / 3600,
        "maximum_delayed_jobs_for_one_user": int(per_user["delayed_jobs"].max()),
        "top_user_policy_delay_share": top_share,
        "policy_delay_hhi": hhi,
        "carbon_reduction_per_wait_hour_kg_capacity_weighted": (
            cap_reduction_kg / total_delay_h if total_delay_h > 0 else None
        ),
        "carbon_reduction_per_wait_hour_kg_node_request_upper": (
            node_reduction_kg / total_delay_h if total_delay_h > 0 else None
        ),
        "both_dynamic_carbon_proxies_positive": bool(
            cap_reduction_kg > 0 and node_reduction_kg > 0
        ),
        "wait_budget_violations": int(
            (
                delay_s
                > pd.to_numeric(frame.get("allowed_wait_budget_s", 0), errors="coerce").fillna(0)
                + 1e-6
            ).sum()
        )
        if "allowed_wait_budget_s" in frame
        else 0,
        **load_peaks(result_dir, scenario_path.name),
    }
    metrics.update(prediction_metrics(frame, "runtime_s", "runtime"))
    metrics.update(prediction_metrics(frame, "source_eligible_wait_s", "queue_wait"))
    if "safe_decision" in frame:
        metrics["safe_decision_counts"] = frame["safe_decision"].value_counts().to_dict()
        metrics["safe_rejection_reason_counts"] = frame.loc[
            frame["safe_decision"].eq("abstain"), "safe_rejection_reason"
        ].value_counts().to_dict()
    return metrics, per_user


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--carbon-result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = pd.read_csv(args.comparison)
    summaries: list[dict[str, object]] = []
    users: list[pd.DataFrame] = []
    for scenario in args.scenario:
        row = comparison.loc[comparison["scenario"].eq(scenario.name)]
        if len(row) != 1:
            raise ValueError(f"Expected one carbon comparison row for {scenario.name}")
        metrics, per_user = scenario_metrics(
            scenario, row.iloc[0], args.carbon_result_dir
        )
        summaries.append(metrics)
        users.append(per_user)

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output / "extended_scenario_metrics.csv", index=False)
    pd.concat(users, ignore_index=True).to_csv(
        args.output / "per_user_fairness.csv", index=False
    )
    (args.output / "extended_scenario_metrics.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    (args.output / "scenario_comparison.json").write_text(
        comparison.to_json(orient="records", indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
