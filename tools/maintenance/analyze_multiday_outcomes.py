"""Read completed formal days; add normalized outcomes and carbon diagnostics.

This supplement never changes replay inputs, campaign state or frozen criteria.
Carbon per job is a cluster-level normalization, not individual attribution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "simulator/slurm_policy_experiments"
sys.path.insert(0, str(EXP))
from calculate_carbon import model_start_offsets

MODELS = ("capacity_weighted", "allocated_node_distinct", "node_request_upper")
DEFAULT = EXP / "campaigns/multiday_20260908"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def grid_check(base, policy):
    require(len(base) == len(policy) and len(base) > 0, "Different/empty interval grids")
    for col in ("from_utc", "to_utc"):
        require(pd.to_datetime(base[col], utc=True).equals(pd.to_datetime(policy[col], utc=True)),
                f"Different interval boundaries: {col}")
    for frame in (base, policy):
        start = pd.to_datetime(frame.from_utc, utc=True)
        end = pd.to_datetime(frame.to_utc, utc=True)
        duration = (end - start).dt.total_seconds().to_numpy() / 3600
        require(np.all(duration > 0), "Invalid interval duration")
        require(np.array_equal(start.iloc[1:].to_numpy(), end.iloc[:-1].to_numpy()), "Non-contiguous grid")
        require(np.allclose(duration, frame.duration_h), "Duration does not match grid")
    for col in ("duration_h", "intensity_gco2_per_kwh"):
        require(np.isfinite(base[col]).all() and np.isfinite(policy[col]).all(), f"Missing {col}")
        require(np.allclose(base[col], policy[col], rtol=0, atol=1e-10), f"Different {col}")


def decompose_carbon(base, policy, model):
    """Positive values mean increased carbon; identity, not causal attribution."""
    grid_check(base, policy)
    intensity = base.intensity_gco2_per_kwh.to_numpy(float)
    mean_intensity = float(np.average(intensity, weights=base.duration_h))
    energy_col = f"dynamic_energy_kwh_{model}"
    carbon_col = f"dynamic_carbon_kg_{model}"
    whole_col = f"whole_carbon_kg_{model}"
    for frame in (base, policy):
        require(np.isfinite(frame[[energy_col, carbon_col, whole_col]].to_numpy()).all(), "Missing carbon data")
        require(np.allclose(frame[carbon_col], frame[energy_col] * intensity / 1000,
                            rtol=1e-9, atol=1e-9), "Energy-carbon reconciliation failed")
    delta_energy = policy[energy_col].to_numpy() - base[energy_col].to_numpy()
    delta_carbon = policy[carbon_col].to_numpy() - base[carbon_col].to_numpy()
    energy_term = float(delta_energy.sum() * mean_intensity / 1000)
    timing_term = float(np.dot(delta_energy, intensity - mean_intensity) / 1000)
    net = float(delta_carbon.sum())
    idle_residual = float((policy[whole_col] - base[whole_col]).sum() - net)
    require(abs(net - energy_term - timing_term) < 1e-7, "Decomposition failed")
    require(abs(idle_residual) < 1e-7, "Fixed idle carbon did not cancel")
    return {
        "model": model,
        "baseline_dynamic_energy_kwh": float(base[energy_col].sum()),
        "policy_dynamic_energy_kwh": float(policy[energy_col].sum()),
        "energy_change_kwh": float(delta_energy.sum()),
        "carbon_increase_kg": net,
        "energy_amount_term_kg": energy_term,
        "carbon_timing_term_kg": timing_term,
        "fixed_idle_difference_kg": idle_residual,
        "positive_interval_increases_kg": float(delta_carbon[delta_carbon > 0].sum()),
        "negative_interval_changes_kg": float(delta_carbon[delta_carbon < 0].sum()),
        "mean_grid_intensity_g_kwh": mean_intensity,
    }, pd.DataFrame({
        "from_utc": base.from_utc, "to_utc": base.to_utc,
        "intensity_gco2_per_kwh": intensity, "energy_change_kwh": delta_energy,
        "carbon_increase_kg": delta_carbon,
    })


def align_jobs(base, policy):
    for frame in (base, policy):
        require(frame.sim_job_id.notna().all() and frame.sim_job_id.is_unique, "Missing/duplicate job IDs")
        require(frame.terminal_status.eq("completed").all(), "Incomplete replay")
    require(set(base.sim_job_id) == set(policy.sim_job_id), "Different job cohorts")
    base = base.set_index("sim_job_id").sort_index()
    policy = policy.set_index("sim_job_id").sort_index()
    for col in ("runtime_s", "nodes", "cpus", "scheduled_gpus", "memory_per_node_mib",
                "eligible_dt_s", "dependency_delay_s"):
        require(np.allclose(base[col], policy[col], rtol=0, atol=1e-6, equal_nan=False), f"Changed invariant: {col}")
    for col in ("is_evaluation", "partition", "user_id", "carry_in_type"):
        require(base[col].fillna("").equals(policy[col].fillna("")), f"Changed invariant: {col}")
    for frame in (base, policy):
        require(np.isfinite(frame[["sim_total_user_wait_s", "sim_policy_delay_s"]].to_numpy()).all(),
                "Missing waiting observations")
    return base, policy


def runtime_intensity(jobs, intervals):
    """Full-runtime CI exposure; not a per-job power/carbon allocation."""
    epoch = pd.to_datetime(jobs.event_epoch_utc, utc=True)
    require(epoch.nunique() == 1, "Different job epochs")
    origin = epoch.iloc[0]
    left = (pd.to_datetime(intervals.from_utc, utc=True) - origin).dt.total_seconds().to_numpy()
    right = (pd.to_datetime(intervals.to_utc, utc=True) - origin).dt.total_seconds().to_numpy()
    starts = model_start_offsets(jobs).to_numpy(float)
    runtime = jobs.runtime_s.to_numpy(float)
    ends = starts + runtime
    require(np.isfinite(starts).all() and np.all(runtime > 0), "Invalid runtime/start")
    require(starts.min() >= left[0] - 1e-6 and ends.max() <= right[-1] + 1e-6,
            "Job execution outside common carbon horizon")
    boundaries = np.r_[left[0], right]
    integral = np.r_[0, np.cumsum((right - left) * intervals.intensity_gco2_per_kwh.to_numpy())]
    return (np.interp(ends, boundaries, integral) - np.interp(starts, boundaries, integral)) / runtime


def paired_jobs(base, policy, intervals):
    base, policy = align_jobs(base, policy)
    result = policy[["is_evaluation", "partition", "user_id", "carry_in_type", "runtime_s"]].copy()
    result["baseline_wait_s"] = base.sim_total_user_wait_s
    result["policy_wait_s"] = policy.sim_total_user_wait_s
    result["actual_added_wait_s"] = policy.sim_total_user_wait_s - base.sim_total_user_wait_s
    result["intentional_delay_s"] = policy.sim_policy_delay_s
    result["runtime_ci_change_g_kwh"] = runtime_intensity(policy, intervals) - runtime_intensity(base, intervals)
    result["baseline_model_start_s"] = model_start_offsets(base)
    result["policy_model_start_s"] = model_start_offsets(policy)
    for col in ("predicted_policy_start_s", "decision_runtime_s", "expected_capacity_carbon_saving_pct"):
        result[col] = pd.to_numeric(policy[col], errors="coerce") if col in policy else np.nan
    result["start_prediction_error_s"] = result.policy_model_start_s - result.predicted_policy_start_s
    return result.reset_index()


def wait_summary(frame):
    require(len(frame) > 0, "Empty wait cohort")
    delta = frame.actual_added_wait_s
    delayed = frame.intentional_delay_s.gt(0)
    positive = delta.clip(lower=0)
    user_positive = frame.assign(positive=positive).groupby("user_id").positive.sum()
    return {
        "jobs": len(frame), "directly_delayed_jobs": int(delayed.sum()),
        "directly_delayed_pct": float(delayed.mean() * 100),
        "baseline_mean_wait_s": float(frame.baseline_wait_s.mean()),
        "policy_mean_wait_s": float(frame.policy_wait_s.mean()),
        "mean_actual_added_wait_s": float(delta.mean()),
        "mean_positive_added_wait_s": float(positive.mean()),
        "paired_added_wait_p95_s": float(delta.quantile(.95)),
        "paired_added_wait_p99_s": float(delta.quantile(.99)),
        "max_actual_added_wait_s": float(delta.max()),
        "jobs_later_over_60s": int(delta.gt(60).sum()),
        "jobs_earlier_over_60s": int(delta.lt(-60).sum()),
        "mean_intentional_delay_s": float(frame.intentional_delay_s.mean()),
        "median_delay_among_delayed_s": float(frame.loc[delayed, "intentional_delay_s"].median()) if delayed.any() else 0,
        "positive_added_wait_h": float(positive.sum() / 3600),
        "earlier_start_benefit_h": float(-delta.clip(upper=0).sum() / 3600),
        "top_user_positive_wait_share": float(user_positive.max() / positive.sum()) if positive.sum() > 0 else 0,
        "higher_runtime_ci_jobs": int(frame.runtime_ci_change_g_kwh.gt(0).sum()),
        "mean_runtime_ci_change_g_kwh": float(frame.runtime_ci_change_g_kwh.mean()),
        "start_prediction_count": int(frame.start_prediction_error_s.notna().sum()),
        "start_prediction_mae_s": float(frame.start_prediction_error_s.abs().mean()),
    }


def aggregate_daily(frame, expected_days):
    require(frame.date.is_unique, "Duplicate dates in strategy aggregate")
    return {
        "days": len(frame), "expected_days": expected_days,
        "complete": len(frame) == expected_days,
        "mean_daily_saving_g_per_eval_job": float(frame.saving_g_per_eval_job.mean()),
        "median_daily_saving_g_per_eval_job": float(frame.saving_g_per_eval_job.median()),
        "worst_daily_saving_g_per_eval_job": float(frame.saving_g_per_eval_job.min()),
        "pooled_saving_g_per_eval_job": float(1000 * frame.saving_kg.sum() / frame.evaluation_jobs.sum()),
        "mean_daily_actual_added_wait_s": float(frame.mean_actual_added_wait_s.mean()),
        "worst_daily_mean_actual_added_wait_s": float(frame.mean_actual_added_wait_s.max()),
        "mean_daily_positive_added_wait_s": float(frame.mean_positive_added_wait_s.mean()),
        "mean_daily_intentional_delay_s": float(frame.mean_intentional_delay_s.mean()),
        "mean_daily_reduction_pct": float(frame.reduction_pct.mean()),
        "pooled_carbon_reduction_pct": float(100 * frame.saving_kg.sum() / frame.baseline_carbon_kg.sum()),
        "positive_days": int(frame.saving_kg.gt(0).sum()),
    }


def analyze(campaign, output):
    state = json.loads((campaign / "state.json").read_text())
    manifest = json.loads((campaign / "manifest.json").read_text())
    complete = [d for d in manifest["dates"] if state.get("dates", {}).get(d, {}).get("status") == "complete"]
    output.mkdir(parents=True, exist_ok=True)
    provenance, daily, components, groups, examples, time_bins = [], [], [], [], [], []

    def read_csv(path):
        raw = path.read_bytes()
        provenance.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        from io import BytesIO
        return pd.read_csv(BytesIO(raw))

    bases = ["baseline", "baseline_repeat_1", "baseline_repeat_2"]
    for date in complete:
        directory = campaign / "days" / date
        comparison_dir = directory / "comparison"
        audit = json.loads((comparison_dir / "experiment_logic_audit.json").read_text())
        require(all(p["pass"] for p in audit["policies"]), f"Invalid policy logic: {date}")
        comparison = read_csv(comparison_dir / "scenario_comparison.csv").set_index("scenario")
        names = bases + [p["name"] for p in manifest["policies"]] + ["adaptive_balanced_repeat"]
        jobs = {n: read_csv(directory / n / "job_results.csv") for n in names}
        intervals = {n: read_csv(comparison_dir / f"{n}_carbon_intervals.csv") for n in names}
        n_jobs = int(jobs["baseline"].is_evaluation.sum())
        require(n_jobs > 0, "No evaluation jobs")
        for spec in manifest["policies"]:
            name = spec["name"]
            repeats = [name, "adaptive_balanced_repeat"] if name == "adaptive_balanced" else [name]
            wait_repeats = []
            for base_name in bases:
                for policy_name in repeats:
                    context = {"date": date, "strategy": name, "baseline_run": base_name, "policy_run": policy_name}
                    for model in MODELS:
                        component, bins = decompose_carbon(intervals[base_name], intervals[policy_name], model)
                        components.append({**context, **component})
                        for _, row in bins.nlargest(5, "carbon_increase_kg").iterrows():
                            time_bins.append({**context, "model": model, **row.to_dict()})
                    paired = paired_jobs(jobs[base_name], jobs[policy_name], intervals[base_name])
                    evaluation = paired.loc[paired.is_evaluation]
                    wait_repeats.append(wait_summary(evaluation))
                    cohorts = {
                        "all_evaluation": evaluation,
                        "directly_delayed": evaluation.loc[evaluation.intentional_delay_s.gt(0)],
                        "not_directly_delayed": evaluation.loc[evaluation.intentional_delay_s.eq(0)],
                        "carry_in": paired.loc[~paired.is_evaluation],
                    }
                    cohorts.update({f"partition:{part}": group for part, group in evaluation.groupby("partition")})
                    for label, group in cohorts.items():
                        if not group.empty:
                            groups.append({**context, "cohort": label, **wait_summary(group)})
                    for label, group in cohorts.items():
                        if label not in ("directly_delayed", "not_directly_delayed", "carry_in"):
                            continue
                        for _, row in group.nlargest(5, "runtime_ci_change_g_kwh").iterrows():
                            examples.append({**context, "cohort": label, **row.to_dict()})
            wait = pd.DataFrame(wait_repeats).median(numeric_only=True).to_dict()
            for model in MODELS:
                for scope in ("dynamic", "whole"):
                    column = f"{scope}_carbon_kg_{model}"
                    base_c = float(comparison.loc[bases, column].median())
                    policy_c = float(comparison.loc[repeats, column].median())
                    base_range = float(comparison.loc[bases, column].max() - comparison.loc[bases, column].min())
                    policy_range = float(comparison.loc[repeats, column].max() - comparison.loc[repeats, column].min())
                    daily.append({
                        "date": date, "strategy": name, "information": spec["information"],
                        "model": model, "scope": scope, "evaluation_jobs": n_jobs,
                        "baseline_carbon_kg": base_c, "policy_carbon_kg": policy_c,
                        "saving_kg": base_c - policy_c,
                        "saving_g_per_eval_job": 1000 * (base_c - policy_c) / n_jobs,
                        "reduction_pct": 100 * (base_c - policy_c) / base_c,
                        "baseline_replay_range_pct": 100 * base_range / base_c,
                        "policy_repeat_range_pct": 100 * policy_range / base_c,
                        "policy_repeats": len(repeats),
                        "no_action": wait["directly_delayed_jobs"] == 0,
                        "inside_observed_baseline_range": bool(comparison.loc[bases, column].min() <= policy_c <= comparison.loc[bases, column].max()),
                        **wait,
                    })

    aggregates = []
    daily_frame = pd.DataFrame(daily)
    if not daily_frame.empty:
        for (name, model, scope), group in daily_frame.groupby(["strategy", "model", "scope"], sort=False):
            aggregates.append({"strategy": name, "model": model, "scope": scope,
                               **aggregate_daily(group, len(manifest["dates"]))})
    tables = {
        "daily_normalized_outcomes.csv": daily_frame,
        "five_day_normalized_summary.csv": pd.DataFrame(aggregates),
        "carbon_increase_decomposition.csv": pd.DataFrame(components),
        "paired_wait_and_spillover.csv": pd.DataFrame(groups),
        "runtime_exposure_examples.csv": pd.DataFrame(examples),
        "largest_interval_increases.csv": pd.DataFrame(time_bins),
    }
    for name, frame in tables.items():
        if not frame.empty:
            temporary = output / (name + ".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(output / name)

    lines = ["# 每作业平均表现与增排诊断", "",
             f"已分析 {len(complete)}/5 个完整日期；实验状态：{state['status']}。", "",
             "这是新增的描述性事后分析，不改变原来冻结的稳定性标准。", "",
             "## 读数说明", "",
             "- 每作业减排 = 共同窗口内集群估算净减排 / 当天评估作业数；不是单作业实测或碳排归因。",
             "- 五天主表对每日每作业均值等权平均；另存按作业数汇总和按基线碳排汇总，不能混称。",
             "- 新增等待按同一作业ID配对；有正有负。主动延迟、正向等待负担、受影响者和未直接延迟者单列。",
             "- 每日等待摘要为各基线与策略回放配对所得统计量的中位数；不是合并重复运行当作新作业。",
             "- P95新增等待是逐作业差值的P95，不是两个P95相减。超过60秒只是描述性计数，不是新验收门槛。",
             "- 固定140 kW在相同分析窗口内抵消；它能稀释减排百分比，不能单独造成净增排。",
             "- 增排分解为能耗总量项和碳强度时序项，两者相加等于模型净增排；这不是因果归因。",
             "- 分解恒等式对每一对实际回放成立；不同配对的各项分别取中位数后不保证仍可相加。",
             "- 作业运行区间碳强度变高，不自动代表其真实碳排增加；尚无单作业功率测量。",
             "- 不动作的策略出现小幅差值，应先检查回放波动；每种代理都有自己的基线范围，不能套用主模型的范围。",
             "- 示例文件列的是碳强度变化最大的作业，不是碳排贡献最大的作业。",
             "- 原有修复前隔离回放、额外环境诊断均不纳入。五个窗口可能重叠，不能视作独立随机样本。", ""]
    if aggregates:
        lines += ["## 当前汇总（主容量模型、负载相关部分）", "",
                  "|策略|完成日期|平均每评估作业净减排 g|平均新增总等待 分钟|平均主动延迟 分钟|",
                  "|---|---:|---:|---:|---:|"]
        for row in aggregates:
            if row["model"] == "capacity_weighted" and row["scope"] == "dynamic":
                lines.append(f"|{row['strategy']}|{row['days']}/5|{row['mean_daily_saving_g_per_eval_job']:.4f}|{row['mean_daily_actual_added_wait_s']/60:.3f}|{row['mean_daily_intentional_delay_s']/60:.3f}|")
        lines += ["", "不足5天的均值仅是已完成日期的暂时摘要，不能用来做最终策略排名。"]
    else:
        lines += ["等待第一个日期的全部正式回放、共同窗口碳排分析和审计完成，不生成占位零结果。"]
    (output / "FINDINGS_ZH.md").write_text("\n".join(lines) + "\n")
    save_json(output / "analysis_status.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "campaign_status": state["status"],
        "complete_dates_analyzed": complete, "expected_dates": manifest["dates"],
        "all_dates_analyzed": len(complete) == len(manifest["dates"]),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": provenance,
    })
    print(json.dumps({"dates_analyzed": complete, "report": str(output / "FINDINGS_ZH.md")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.campaign, args.output or args.campaign / "post_analysis")
