"""Collect final audit evidence and figures without changing the experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_multiday_outcomes import DEFAULT, EXP, MODELS, REPO, require, save_json

LABELS = {
    "b1_runtime_4h": "B1 runtime (4 h)", "b2_q75_cap50": "B2",
    "adaptive_balanced": "Adaptive Balanced", "time_hybrid_08_18": "Day/night hybrid",
    "history_only": "History-only", "dynamic_forecast_balanced_oracle": "Oracle (no actions)",
    "low_impact_dynamic_q25": "Low-impact",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    campaign = DEFAULT
    output = campaign / "post_analysis/final_review"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((campaign / "manifest.json").read_text())
    state = json.loads((campaign / "state.json").read_text())
    require(state["status"] == "complete", "Campaign is not complete")
    bases = ["baseline", "baseline_repeat_1", "baseline_repeat_2"]
    names = bases + [p["name"] for p in manifest["policies"]] + ["adaptive_balanced_repeat"]
    audits, source_checks, day_checks, no_actions, controls = [], [], [], [], []
    runtime_examples, prior_attempts = [], []
    files = ["sim.events", "users.sim", "slurm.conf", "gres.conf", "sim.conf"]
    for name, expected in json.loads((campaign / "code_hashes.json").read_text()).items():
        source_checks.append({"file": name, "matches": digest(EXP / name) == expected})
    for date in manifest["dates"]:
        directory = campaign / "days" / date
        comparison = pd.read_csv(directory / "comparison/scenario_comparison.csv").set_index("scenario")
        logic = json.loads((directory / "comparison/experiment_logic_audit.json").read_text())
        carry = json.loads((directory / "comparison/carry_in_audits.json").read_text())
        require(logic["overall_pass"], f"Failed logic audit: {date}")
        require(state["dates"][date]["status"] == "complete", f"Incomplete date: {date}")
        day_checks.append({"date": date, "logic_pass": logic["overall_pass"],
                           "formal_scenarios": len(comparison),
                           "horizon_hours": float(comparison.common_horizon_hours.iloc[0]),
                           "same_horizon": comparison.common_horizon_start_utc.nunique() == 1 and comparison.common_horizon_end_utc.nunique() == 1})
        for name in names:
            folder = directory / name
            record = state["runs"][f"{date}/{name}"]
            validation = json.loads((folder / "simulation_validation.json").read_text())
            frame = pd.read_csv(folder / "job_results.csv")
            error = (frame.sim_observed_runtime_s - frame.runtime_s).abs()
            for index in error.nlargest(5).index:
                if error.loc[index] > 60:
                    fields = ["sim_job_id", "is_evaluation", "carry_in_type", "partition",
                              "runtime_s", "sim_observed_runtime_s", "sim_start_log_utc", "sim_end_log_utc"]
                    runtime_examples.append({"date": date, "scenario": name,
                                             **frame.loc[index, fields].to_dict(),
                                             "runtime_abs_error_s": float(error.loc[index])})
            for prior in sorted((folder / "previous_attempts").glob("*/simulation_validation.json")):
                previous = json.loads(prior.read_text())
                prior_attempts.append({"date": date, "scenario": name, "directory": str(prior.parent),
                                       "validation": previous,
                                       "retained_hashes": {f.name: digest(f) for f in prior.parent.iterdir() if f.is_file()}})
            checks = {filename: digest(folder / filename) == expected
                      for filename, expected in record["artifact_hashes"].items()}
            row = {"date": date, "scenario": name, "attempts": record["attempts"],
                   "strict_complete": validation["strict_pass"],
                   "jobs": validation["expected_jobs"], "carry_in_pass": all(x["passed"] for x in carry[name]),
                   "artifact_hashes_match": all(checks.values()),
                   "hash_mismatches": ";".join(k for k, v in checks.items() if not v),
                   "runtime_within_60s_pct": 100 * validation["completed_runtime_within_60s_ratio"],
                   "runtime_error_over_60s_jobs": int(error.gt(60).sum()),
                   "runtime_error_over_1h_jobs": int(error.gt(3600).sum()),
                   "runtime_abs_error_p99_s": float(error.quantile(.99)),
                   "runtime_abs_error_max_s": float(error.max()),
                   "early_completion_events": validation["early_completion_events"]}
            audits.append(row)
            require(row["strict_complete"] and row["carry_in_pass"] and row["artifact_hashes_match"],
                    f"Final integrity check failed: {date}/{name}")
            delayed = int(frame.loc[frame.is_evaluation, "sim_policy_delay_s"].gt(0).sum())
            if delayed == 0:
                same = {file: digest(folder / file) == digest(directory / "baseline" / file) for file in files}
                no_actions.append({"date": date, "scenario": name,
                                   "identical_executable_inputs": all(same.values()), **same})
                controls.append({"date": date, "scenario": name,
                                 "capacity_carbon_kg": float(comparison.loc[name, "dynamic_carbon_kg_capacity_weighted"]),
                                 "mean_total_wait_min": float(frame.loc[frame.is_evaluation, "sim_total_user_wait_s"].mean() / 60),
                                 "p95_total_wait_min": float(comparison.loc[name, "total_user_wait_p95_s"] / 60),
                                 "same_inputs": all(same.values())})
    require(len(audits) == 55 and all(x["matches"] for x in source_checks), "Frozen source/count mismatch")
    require(all(x["same_horizon"] for x in day_checks), "Common horizon mismatch")
    pd.DataFrame(audits).to_csv(output / "final_replay_audit.csv", index=False)
    pd.DataFrame(day_checks).to_csv(output / "day_audit.csv", index=False)
    pd.DataFrame(no_actions).to_csv(output / "no_action_input_equivalence.csv", index=False)
    pd.DataFrame(controls).to_csv(output / "identical_input_outcomes.csv", index=False)
    pd.DataFrame(runtime_examples).to_csv(output / "runtime_error_examples.csv", index=False)
    save_json(output / "retained_previous_attempts.json", prior_attempts)
    daily = pd.read_csv(campaign / "post_analysis/daily_normalized_outcomes.csv")
    negatives = daily.loc[daily.scope.eq("dynamic") & daily.saving_kg.lt(0)].copy()
    negatives["interpretation"] = np.select(
        [negatives.no_action, negatives.inside_observed_baseline_range],
        ["no action; executable-input equivalence must be checked", "action taken; within observed baseline range"],
        default="action taken; outside baseline range, not causal proof")
    negatives.to_csv(output / "all_negative_model_cases.csv", index=False)
    summary = pd.read_csv(campaign / "post_analysis/five_day_normalized_summary.csv")
    strategy_summary = pd.read_csv(campaign / "summary/strategy_stability_summary.csv")
    flags = {
        "finished_utc": state["updated_utc"], "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        "formal_complete_runs": len(audits), "complete_days": len(day_checks),
        "day_logic_checks_pass": all(x["logic_pass"] for x in day_checks),
        "artifact_hashes_match": all(x["artifact_hashes_match"] for x in audits),
        "frozen_source_hashes_match": all(x["matches"] for x in source_checks),
        "source_checks": source_checks,
        "retry_runs": [x["date"] + "/" + x["scenario"] for x in audits if x["attempts"] > 1],
        "runs_with_runtime_errors_over_60s": sum(x["runtime_error_over_60s_jobs"] > 0 for x in audits),
        "worst_runtime_within_60s_pct": min(x["runtime_within_60s_pct"] for x in audits),
        "runs_with_runtime_errors_over_1h": sum(x["runtime_error_over_1h_jobs"] > 0 for x in audits),
        "largest_runtime_abs_error_s": max(x["runtime_abs_error_max_s"] for x in audits),
        "archived_prior_attempts": len(prior_attempts),
        "no_action_nonbaseline_runs": len([x for x in no_actions if x["scenario"] not in bases]),
        "all_no_action_inputs_match": all(x["identical_executable_inputs"] for x in no_actions),
        "stable_strategies": strategy_summary.loc[strategy_summary.stability_label.eq("directionally_stable_on_tested_days"), "strategy"].tolist(),
        "analysis_scope": "Descriptive supplement; no new simulation or altered primary acceptance rules",
    }
    save_json(output / "audit_summary.json", flags)
    figures(daily, summary, output)
    print(json.dumps({k: v for k, v in flags.items() if k != "source_checks"}, indent=2))


def figures(daily, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    primary = daily.loc[daily.model.eq("capacity_weighted") & daily.scope.eq("dynamic")]
    table = primary.pivot(index="strategy", columns="date", values="reduction_pct").reindex(LABELS)
    idle = primary.pivot(index="strategy", columns="date", values="no_action").reindex(LABELS)
    cmap = LinearSegmentedColormap.from_list("carbon", ["#D6A257", "#FFFFFF", "#248A92"])
    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    largest = float(np.abs(table.values).max())
    ax.imshow(table.values, cmap=cmap, vmin=-largest, vmax=largest, aspect="auto")
    for i in range(len(table)):
        for j in range(len(table.columns)):
            if idle.iloc[i, j]:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, color="#E9ECEF"))
            text = f"{table.iloc[i,j]:+.3f}%" + (" *" if idle.iloc[i,j] else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=11, color="#20262C")
    ax.set_xticks(range(5), [d[5:] for d in table.columns], fontsize=12)
    ax.set_yticks(range(7), [LABELS[n] for n in table.index], fontsize=11)
    ax.set_title("Five matched dates: estimated dynamic-carbon reduction", fontsize=14, pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.02, .02, "Positive = reduction; negative = increase. Grey * = no job delayed; observed replay difference, not a policy effect.", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(output / "daily_carbon_with_no_action.png", dpi=180)
    plt.close(fig)

    table = summary.loc[summary.model.eq("capacity_weighted") & summary.scope.eq("dynamic")].set_index("strategy").reindex(LABELS)
    y = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    savings = table.mean_daily_saving_g_per_eval_job
    axes[0].barh(y, savings, color=["#248A92" if x > 0 else "#D6A257" for x in savings])
    axes[0].axvline(0, color="#6B7280", lw=.7)
    axes[0].set_xlim(-.165, .075)
    for i, value in enumerate(savings):
        axes[0].text(value + (.003 if value >= 0 else -.003), i, f"{value:+.4f}",
                     ha="left" if value >= 0 else "right", va="center", fontsize=10)
    axes[0].set_xlabel("Mean daily saving, g CO2 per evaluation job", fontsize=10)
    axes[0].set_yticks(y, [LABELS[n] for n in table.index], fontsize=10)
    axes[0].invert_yaxis()
    wait = table.mean_daily_actual_added_wait_s / 60
    hold = table.mean_daily_intentional_delay_s / 60
    axes[1].barh(y - .17, wait, height=.3, color="#336B91", label="Observed added total wait")
    axes[1].barh(y + .17, hold, height=.3, color="#88BDB3", label="Intentional policy delay")
    for i, value in enumerate(wait):
        axes[1].text(value + 2, i - .17, f"{value:.2f}", va="center", fontsize=10)
    axes[1].set_xlim(0, 195)
    axes[1].set_xlabel("Equal-day mean, minutes per evaluation job", fontsize=10)
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Averages retain all five dates, including no-action replay variation", fontsize=14)
    fig.text(.02, .015, "Cluster carbon normalized by evaluation jobs, not individual measured emissions. Observed wait differences are not pure delay effects.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(output / "mean_carbon_and_wait.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
