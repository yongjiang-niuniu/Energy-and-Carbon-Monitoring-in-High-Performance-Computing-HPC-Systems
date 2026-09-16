#!/usr/bin/env python3
"""Build reproducible completion evidence and replay-noise summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def validation_rows(scenario_root: Path, scenario_order: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenario_order:
        directory = scenario_root / scenario
        if not directory.exists():
            directory = scenario_root / "policies" / scenario
        validation_path = directory / "simulation_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "scenario": scenario,
                "expected_jobs": validation["expected_jobs"],
                "submitted_jobs": validation["submitted_jobs"],
                "started_jobs": validation["started_jobs"],
                "completed_jobs": validation["completed_jobs"],
                "early_completion_events": validation["early_completion_events"],
                "strict_pass": bool(validation["strict_pass"]),
            }
        )
    return rows


def scenario_directory(scenario_root: Path, scenario: str) -> Path:
    direct = scenario_root / scenario
    return direct if direct.exists() else scenario_root / "policies" / scenario


def write_job_decision_audits(
    scenario_root: Path, scenario_order: list[str], output: Path
) -> None:
    preferred_columns = [
        "sim_job_id",
        "user_id",
        "partition",
        "carry_in_category",
        "is_warmup",
        "is_flexible",
        "source_submit_utc",
        "source_eligible_utc",
        "submit_dt_s",
        "eligible_dt_s",
        "release_dt_s",
        "dependency_delay_s",
        "b1_delay_s",
        "b2_delay_s",
        "policy_delay_s",
        "allowed_wait_budget_s",
        "expected_runtime_carbon_saving_pct",
        "expected_capacity_carbon_saving_pct",
        "expected_node_carbon_saving_pct",
        "expected_risk_adjusted_carbon_saving_proxy",
        "predicted_baseline_start_s",
        "predicted_policy_start_s",
        "predicted_baseline_queue_wait_s",
        "predicted_policy_queue_wait_s",
        "predicted_runtime_s_q50",
        "predicted_runtime_s_q75",
        "predicted_runtime_s_q90",
        "predicted_queue_wait_s_q50",
        "predicted_queue_wait_s_q75",
        "predicted_queue_wait_s_q90",
        "decision_runtime_s",
        "decision_runtime_source",
        "safe_decision",
        "safe_rejection_reason",
        "consensus_selection_fraction",
        "consensus_rejection_reason",
    ]
    output.mkdir(parents=True, exist_ok=True)
    for scenario in scenario_order:
        profile = pd.read_csv(scenario_directory(scenario_root, scenario) / "workload_profile.csv")
        columns = [column for column in preferred_columns if column in profile.columns]
        audit = profile.loc[:, columns].copy()
        audit.insert(0, "scenario", scenario)
        audit.to_csv(output / f"{scenario}_job_decision_audit.csv", index=False)


def write_completion_figure(frame: pd.DataFrame, output: Path) -> None:
    display = frame.copy()
    display["scenario"] = display["scenario"].str.replace("_", " ")
    display["jobs"] = (
        display["completed_jobs"].astype(str)
        + " / "
        + display["expected_jobs"].astype(str)
    )
    display["early"] = display["early_completion_events"].astype(str)
    display["strict"] = display["strict_pass"].map({True: "PASS", False: "FAIL"})
    display = display[["scenario", "jobs", "early", "strict"]]

    fig, ax = plt.subplots(figsize=(12, 7.4))
    ax.axis("off")
    passed = int(frame["strict_pass"].sum())
    scenario_runs = int(frame["completed_jobs"].sum())
    fig.text(
        0.06,
        0.955,
        "24-hour Slurm replay completion evidence",
        fontsize=19,
        weight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.06,
        0.905,
        f"{passed}/{len(frame)} strict passes | {scenario_runs:,} completed scenario-job runs | 0 early completions",
        fontsize=11.5,
        color="#344054",
        ha="left",
    )
    fig.text(
        0.06,
        0.87,
        "Generated from each scenario's simulation_validation.json after the simulator exited.",
        fontsize=9.5,
        color="#667085",
        ha="left",
    )
    table = ax.table(
        cellText=display.values,
        colLabels=["Scenario", "Completed / expected", "Early", "Strict"],
        colWidths=[0.49, 0.25, 0.10, 0.12],
        cellLoc="left",
        colLoc="left",
        bbox=[0.04, 0.04, 0.92, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#1D2939")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F9FAFB" if row % 2 else "white")
            if column == 3:
                cell.get_text().set_color("#067647")
                cell.get_text().set_weight("bold")
    fig.savefig(output / "02_all_replays_complete.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_safe_policy_figure(audit: pd.DataFrame, output: Path, threshold: float) -> None:
    accepted = int(audit["accepted"].fillna(False).astype(bool).sum())
    unique_jobs = int(audit["sim_job_id"].nunique())
    max_return = float(audit["carbon_return_per_wait_hour"].max())
    reasons = (
        audit["rejection_reason"]
        .str.split(";")
        .explode()
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="candidate_options")
    )

    fig, (ax_metric, ax_reasons) = plt.subplots(
        1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [0.9, 1.45]}
    )
    fig.suptitle(
        "Safe rolling policy: abstention audit",
        x=0.06,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.06,
        0.90,
        f"{len(audit)} candidate delay options across {unique_jobs} jobs; {accepted} accepted",
        fontsize=11,
        color="#344054",
        ha="left",
    )

    ax_metric.bar(
        ["Best candidate", "Frozen threshold"],
        [max_return, threshold],
        color=["#D39128", "#2457A7"],
        width=0.62,
    )
    ax_metric.set_ylabel("Carbon return per wait hour")
    ax_metric.set_title("No candidate cleared the gate", loc="left", fontsize=12, weight="bold")
    ax_metric.bar_label(ax_metric.containers[0], fmt="%.4f", padding=4)
    ax_metric.set_ylim(0, max(max_return, threshold) * 1.25)
    ax_metric.grid(axis="y", color="#E6E9ED")
    ax_metric.spines[["top", "right"]].set_visible(False)

    reasons = reasons.sort_values("candidate_options")
    labels = reasons["reason"].str.replace("_", " ")
    bars = ax_reasons.barh(labels, reasons["candidate_options"], color="#65AFA1")
    ax_reasons.bar_label(bars, padding=4, fontsize=9)
    ax_reasons.set_xlabel("Rejected candidate options")
    ax_reasons.set_title("Recorded rejection reasons", loc="left", fontsize=12, weight="bold")
    ax_reasons.grid(axis="x", color="#E6E9ED")
    ax_reasons.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.06,
        0.02,
        "Generated from safe_rolling_candidate_audit.csv. Multiple reasons may apply to one option.",
        fontsize=9,
        color="#667085",
        ha="left",
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.86))
    fig.savefig(output / "03_safe_policy_abstention_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def replay_repeatability(comparison: pd.DataFrame) -> dict[str, object]:
    no_op = comparison.loc[comparison["jobs_with_policy_delay"].eq(0)].copy()
    repeats = no_op.loc[~no_op["scenario"].eq("baseline")]
    capacity = "dynamic_carbon_reduction_pct_capacity_weighted"
    node = "dynamic_carbon_reduction_pct_node_request_upper"
    return {
        "interpretation": (
            "Zero-delay scenario differences are replay repeatability noise, not policy effects. "
            "Effects at or below this envelope should not be interpreted as demonstrated savings."
        ),
        "zero_delay_scenarios": no_op["scenario"].tolist(),
        "repeat_scenarios_excluding_baseline": repeats["scenario"].tolist(),
        "max_abs_dynamic_difference_pct_capacity_weighted": float(repeats[capacity].abs().max()),
        "max_abs_dynamic_difference_pct_node_request_upper": float(repeats[node].abs().max()),
        "range_dynamic_difference_pct_capacity_weighted": [
            float(repeats[capacity].min()),
            float(repeats[capacity].max()),
        ],
        "range_dynamic_difference_pct_node_request_upper": [
            float(repeats[node].min()),
            float(repeats[node].max()),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--safe-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--carbon-return-threshold", type=float, default=0.05)
    args = parser.parse_args()

    comparison = pd.read_csv(args.comparison)
    rows = validation_rows(args.scenario_root, comparison["scenario"].tolist())
    validation = pd.DataFrame(rows)
    if not validation["strict_pass"].all():
        failed = validation.loc[~validation["strict_pass"], "scenario"].tolist()
        raise ValueError(f"Strict replay validation failed for: {failed}")

    evidence_dir = args.output_root / "run_evidence"
    decision_dir = args.output_root / "job_decisions"
    screenshot_dir = args.output_root / "screenshots"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(evidence_dir / "scenario_validation_summary.csv", index=False)
    (evidence_dir / "scenario_validation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    repeatability = replay_repeatability(comparison)
    (args.comparison.parent / "replay_repeatability_summary.json").write_text(
        json.dumps(repeatability, indent=2), encoding="utf-8"
    )
    write_completion_figure(validation, screenshot_dir)
    write_job_decision_audits(
        args.scenario_root, comparison["scenario"].tolist(), decision_dir
    )
    write_safe_policy_figure(
        pd.read_csv(args.safe_audit), screenshot_dir, args.carbon_return_threshold
    )
    print(json.dumps(repeatability, indent=2))


if __name__ == "__main__":
    main()
