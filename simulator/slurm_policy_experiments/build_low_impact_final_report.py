#!/usr/bin/env python3
"""Build bilingual report and evidence figure for the frozen low-impact holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICY = "low_impact_dynamic_q25"


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:,.{digits}f}"


def write_acceptance_figure(report: dict[str, object], output: Path) -> None:
    observed = report["observed"]
    checks = report["checks"]
    assert isinstance(observed, dict) and isinstance(checks, dict)
    rows = [
        (
            "Capacity carbon reduction",
            f"{fmt(float(observed['capacity_dynamic_carbon_reduction_pct']))}%",
            ">= 0.0500%",
            checks["capacity_dynamic_carbon_reduction_pct_minimum"],
        ),
        (
            "P95 total-wait change",
            f"{fmt(float(observed['total_user_wait_p95_change_s']), 1)} s",
            "<= 300 s",
            checks["total_user_wait_p95_increase_seconds_maximum"],
        ),
        (
            "Delay > actual runtime",
            str(observed["jobs_delayed_longer_than_actual_runtime"]),
            "0 jobs",
            checks["jobs_delayed_longer_than_actual_runtime"],
        ),
        (
            "Maximum user delay",
            f"{fmt(float(observed['maximum_user_policy_delay_minutes']), 1)} min",
            "<= 60 min",
            checks["maximum_user_policy_delay_minutes"],
        ),
        (
            "Top-user delay share",
            f"{fmt(100 * float(observed['maximum_single_user_share_of_policy_delay']), 1)}%",
            "<= 40%",
            checks["maximum_single_user_share_of_policy_delay"],
        ),
        (
            "All expected jobs completed",
            f"{observed['policy_jobs']} / {observed['expected_total_jobs']}",
            "exact",
            checks["all_expected_jobs_completed"],
        ),
    ]
    table_rows = [
        [label, value, limit, "PASS" if passed else "FAIL"]
        for label, value, limit, passed in rows
    ]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.axis("off")
    status = str(report["status"]).upper()
    fig.text(
        0.06,
        0.94,
        "Frozen December holdout acceptance",
        fontsize=21,
        weight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.06,
        0.875,
        f"Overall result: {status} | History-only policy | 5,494-job Slurm replay",
        fontsize=12,
        color="#344054",
        ha="left",
    )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Frozen gate", "Observed", "Limit", "Result"],
        colWidths=[0.45, 0.19, 0.19, 0.13],
        cellLoc="left",
        colLoc="left",
        bbox=[0.04, 0.08, 0.92, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row == 0:
            cell.set_facecolor("#1D2939")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F9FAFB" if row % 2 else "white")
            if column == 3:
                passed = table_rows[row - 1][3] == "PASS"
                cell.get_text().set_color("#067647" if passed else "#B42318")
                cell.get_text().set_weight("bold")
    fig.text(
        0.06,
        0.025,
        "Capacity is the primary model; requested-node and allocated-node results are diagnostics.",
        fontsize=9.5,
        color="#667085",
        ha="left",
    )
    fig.savefig(output / "frozen_acceptance_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    comparison: pd.DataFrame,
    acceptance: dict[str, object],
    policy_results: pd.DataFrame,
) -> str:
    baseline = comparison.loc[comparison["scenario"].eq("baseline")].iloc[0]
    repeat = comparison.loc[comparison["scenario"].eq("no_policy_repeat")].iloc[0]
    policy = comparison.loc[comparison["scenario"].eq(POLICY)].iloc[0]
    observed = acceptance["observed"]
    checks = acceptance["checks"]
    repeatability = acceptance["repeatability_diagnostics"]
    assert (
        isinstance(observed, dict)
        and isinstance(checks, dict)
        and isinstance(repeatability, dict)
    )

    delayed = policy_results.loc[
        policy_results["is_evaluation"].astype(bool)
        & pd.to_numeric(policy_results["sim_policy_delay_s"], errors="coerce").gt(0)
    ]
    users = delayed["user_id"].nunique()
    status = str(acceptance["status"]).upper()
    failed = [name for name, passed in checks.items() if not passed]
    failed_text = ", ".join(failed) if failed else "none"

    return f"""# Frozen low-impact dynamic holdout / 冻结的低干扰动态 holdout

## Result / 结果

Overall frozen acceptance: **{status}**.

最终冻结验收结果：**{status}**。

The history-only policy delayed {len(delayed)} of {int(policy['evaluation_jobs']):,} evaluation jobs across {users} users. Total policy delay was {fmt(float(observed['total_policy_delay_hours']), 3)} hours. Capacity-weighted dynamic carbon changed by {fmt(float(observed['capacity_dynamic_carbon_reduction_pct']))}% ({fmt(float(observed['capacity_dynamic_carbon_reduction_kg']), 5)} kgCO2e), while the whole-cage percentage changed by {fmt(float(policy['whole_carbon_reduction_pct_capacity_weighted']), 5)}% because the fixed 140 kW component dominates. P95 total user wait changed by {fmt(float(observed['total_user_wait_p95_change_s']), 1)} seconds.

History-only 策略在 {int(policy['evaluation_jobs']):,} 个评估作业中移动了 {len(delayed)} 个，涉及 {users} 个用户；总 policy delay 为 {fmt(float(observed['total_policy_delay_hours']), 3)} 小时。Capacity-weighted 动态碳变化为 {fmt(float(observed['capacity_dynamic_carbon_reduction_pct']))}%（{fmt(float(observed['capacity_dynamic_carbon_reduction_kg']), 5)} kgCO2e）；因为固定 140 kW 占主要部分，whole-cage 百分比变化为 {fmt(float(policy['whole_carbon_reduction_pct_capacity_weighted']), 5)}%。P95 总用户等待变化为 {fmt(float(observed['total_user_wait_p95_change_s']), 1)} 秒。

Failed frozen checks: `{failed_text}`.

未通过的冻结检查：`{failed_text}`。

The same-date no-policy repeat changed capacity-weighted dynamic carbon by {fmt(float(repeatability['no_policy_repeat_capacity_dynamic_carbon_reduction_pct']))}% relative to Baseline. The policy-minus-repeat difference was {fmt(float(repeatability['policy_minus_no_policy_repeat_capacity_pct_points']))} percentage points. This post-hoc diagnostic does not alter the frozen status. If the policy does not exceed this control, its apparent saving is not distinguishable from replay variation.

同日 no-policy repeat 相对 Baseline 的 capacity-weighted 动态碳变化为 {fmt(float(repeatability['no_policy_repeat_capacity_dynamic_carbon_reduction_pct']))}%；policy 减去 repeat 后的差值为 {fmt(float(repeatability['policy_minus_no_policy_repeat_capacity_pct_points']))} 个百分点。这个事后诊断不改变冻结验收状态。如果候选没有超过这个对照，表面减排就无法与回放波动区分。

## Main comparison / 主要对比

| Scenario | Jobs complete | Delayed jobs | Policy delay | P95 wait change | Capacity carbon | Requested-node diagnostic | Allocated-node diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | {int(baseline['jobs']):,} | 0 | 0 h | 0 s | 0% | 0% | 0% |
| No-policy repeat | {int(repeat['jobs']):,} | {int(repeat['jobs_with_policy_delay'])} | {fmt(float(repeat['policy_delay_p50_s']) / 3600, 3)} h | {fmt(float(repeat['total_wait_p95_change_s']), 1)} s | {fmt(float(repeat['dynamic_carbon_reduction_pct_capacity_weighted']))}% | {fmt(float(repeat['dynamic_carbon_reduction_pct_node_request_upper']))}% | {fmt(float(repeat['dynamic_carbon_reduction_pct_allocated_node_distinct']))}% |
| Low-impact dynamic | {int(policy['jobs']):,} | {int(policy['jobs_with_policy_delay'])} | {fmt(float(observed['total_policy_delay_hours']), 3)} h | {fmt(float(policy['total_wait_p95_change_s']), 1)} s | {fmt(float(policy['dynamic_carbon_reduction_pct_capacity_weighted']))}% | {fmt(float(policy['dynamic_carbon_reduction_pct_node_request_upper']))}% | {fmt(float(policy['dynamic_carbon_reduction_pct_allocated_node_distinct']))}% |

The no-policy repeat is the date-specific replay-noise control. A policy effect close to that repeat must not be presented as demonstrated saving.

No-policy repeat 是当天的回放噪声对照。如果策略效果接近这个差异，就不能写成已经证明的减排。

## Interpretation / 解释

- The capacity result is a modelled counterfactual under `P(t) = 140 + 55U(t)` kW, not measured Stanage electricity.
- Requested-node is an upper proxy that double-counts shared requests; allocated-node is post-hoc and placement-sensitive.
- A negative aggregate wait change can arise from queue and backfill reordering; it is not evidence that delaying work makes every user faster.
- The policy uses January-November aggregates only and does not read December observed runtime or future submissions.
- Passing one holdout supports a limited low-impact case study. Failing a gate is retained and does not trigger December retuning.

- Capacity 结果来自 `P(t) = 140 + 55U(t)` kW 反事实模型，不是 Stanage 实测用电。
- Requested-node 会重复计算共享请求；allocated-node 是事后且受 placement 影响的诊断。
- Aggregate wait 的负变化可能来自 queue 和 backfill 重新排序，不能说延迟作业让每个用户更快。
- 策略只读取 1-11 月汇总数据，不读取 12 月真实 runtime 或未来提交。
- 单个 holdout 通过也只支持有限的低干扰 case study；未通过就保留，不用 12 月重新调参。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--policy-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(args.comparison)
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    policy_results = pd.read_csv(args.policy_results)
    delayed = policy_results.loc[
        policy_results["is_evaluation"].astype(bool)
        & pd.to_numeric(policy_results["sim_policy_delay_s"], errors="coerce").gt(0)
    ]
    delayed.to_csv(args.output / "delayed_job_outcomes.csv", index=False)
    (args.output / "Low_Impact_Dynamic_Final_Report_ZH_EN.md").write_text(
        build_report(comparison, acceptance, policy_results), encoding="utf-8"
    )
    write_acceptance_figure(acceptance, args.output)
    print(f"Wrote final report and evidence to {args.output.resolve()}")


if __name__ == "__main__":
    main()
