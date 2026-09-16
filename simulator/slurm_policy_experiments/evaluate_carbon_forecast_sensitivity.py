#!/usr/bin/env python3
"""Screen dynamic-history decision stability under carbon forecast noise."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
POLICY_SPEC = importlib.util.spec_from_file_location(
    "generate_policy_scenario", ROOT / "generate_policy_scenario.py"
)
POLICY = importlib.util.module_from_spec(POLICY_SPEC)
assert POLICY_SPEC.loader is not None
POLICY_SPEC.loader.exec_module(POLICY)


def perturb_intensity(
    intensity: np.ndarray,
    relative_sigma: float,
    seed: int,
    autocorrelation: float = 0.8,
) -> np.ndarray:
    if relative_sigma < 0:
        raise ValueError("relative_sigma must be non-negative")
    if not 0 <= autocorrelation < 1:
        raise ValueError("autocorrelation must be in [0, 1)")
    if relative_sigma == 0:
        return intensity.astype(float).copy()
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, relative_sigma, len(intensity))
    noise = np.zeros(len(intensity), dtype=float)
    scale = np.sqrt(1.0 - autocorrelation**2)
    for index, innovation in enumerate(innovations):
        previous = noise[index - 1] if index else 0.0
        noise[index] = autocorrelation * previous + scale * innovation
    return np.maximum(intensity.astype(float) * (1.0 + noise), 1.0)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def run_history_policy(
    profile: pd.DataFrame,
    model: dict[str, object],
    carbon: pd.DataFrame,
    epoch: pd.Timestamp,
    max_runtime_uncertainty_ratio: float = 2.0,
) -> pd.DataFrame:
    result, _ = POLICY.apply_dynamic_history(
        profile,
        model,
        carbon,
        epoch,
        0.30,
        4.0,
        1.0,
        2.0,
        2.0,
        0.10,
        0.25,
        0.75,
        0.05,
        0.75,
        0.75,
        0.75,
        4.0,
        max_runtime_uncertainty_ratio,
    )
    return result


def true_runtime_improvement(
    row: pd.Series,
    true_carbon: pd.DataFrame,
    epoch: pd.Timestamp,
) -> float:
    runtime_s = float(row["runtime_s"])
    before = POLICY.runtime_average_intensity(
        true_carbon,
        epoch + pd.Timedelta(seconds=float(row["predicted_baseline_start_s"])),
        runtime_s,
    )
    after = POLICY.runtime_average_intensity(
        true_carbon,
        epoch + pd.Timedelta(seconds=float(row["predicted_policy_start_s"])),
        runtime_s,
    )
    return (before - after) / before * 100 if before else 0.0


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    noise = summary.loc[summary["relative_sigma_pct"].gt(0)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for seed, rows in noise.groupby("seed"):
        ordered = rows.sort_values("relative_sigma_pct")
        axes[0].plot(
            ordered["relative_sigma_pct"],
            ordered["jobs_delayed"],
            marker="o",
            alpha=0.7,
            label=f"seed {int(seed)}",
        )
        axes[1].plot(
            ordered["relative_sigma_pct"],
            ordered["jaccard_vs_reference"],
            marker="o",
            alpha=0.7,
        )
    axes[0].axhline(
        float(summary.loc[summary["relative_sigma_pct"].eq(0), "jobs_delayed"].iloc[0]),
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="reference",
    )
    axes[0].set_ylabel("Delayed jobs")
    axes[0].set_title("Decision count under forecast noise")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Jaccard overlap with reference")
    axes[1].set_title("Selected-job stability")
    for axis in axes:
        axis.set_xlabel("Relative forecast noise sigma (%)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--carbon", type=Path, required=True)
    parser.add_argument("--history-model-dir", type=Path, required=True)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--autocorrelation", type=float, default=0.8)
    parser.add_argument("--max-runtime-uncertainty-ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(level <= 0 for level in args.noise_levels):
        raise ValueError("noise-levels must be positive; the reference is added automatically")
    profile = pd.read_csv(args.baseline_profile)
    true_carbon = POLICY.load_carbon(args.carbon)
    model = POLICY.HISTORY.load_model(args.history_model_dir)
    epoch = POLICY.workload_epoch(profile)
    original = true_carbon["intensity_gco2_per_kwh"].to_numpy(float)

    reference = run_history_policy(
        profile,
        model,
        true_carbon.copy(),
        epoch,
        args.max_runtime_uncertainty_ratio,
    )
    reference_delayed = reference.loc[
        pd.to_numeric(reference["release_dt_s"]).gt(
            pd.to_numeric(reference["eligible_dt_s"])
        )
    ]
    reference_jobs = set(reference_delayed["sim_job_id"].astype(str))

    summaries: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    traces: list[pd.DataFrame] = []
    scenarios = [(0.0, 0)] + [
        (level, seed) for level in args.noise_levels for seed in args.seeds
    ]
    for relative_sigma, seed in scenarios:
        scenario_name = (
            "reference" if relative_sigma == 0 else f"noise_{relative_sigma:.3f}_seed_{seed}"
        )
        forecast = true_carbon.copy()
        forecast_values = perturb_intensity(
            original, relative_sigma, seed, args.autocorrelation
        )
        forecast["intensity_gco2_per_kwh"] = forecast_values
        result = reference if relative_sigma == 0 else run_history_policy(
            profile,
            model,
            forecast,
            epoch,
            args.max_runtime_uncertainty_ratio,
        )
        delayed = result.loc[
            pd.to_numeric(result["release_dt_s"]).gt(
                pd.to_numeric(result["eligible_dt_s"])
            )
        ].copy()
        selected_jobs = set(delayed["sim_job_id"].astype(str))
        true_improvements: list[float] = []
        for _, row in delayed.iterrows():
            improvement = true_runtime_improvement(row, true_carbon, epoch)
            true_improvements.append(improvement)
            decisions.append(
                {
                    "scenario": scenario_name,
                    "relative_sigma_pct": relative_sigma * 100,
                    "seed": seed,
                    "sim_job_id": row["sim_job_id"],
                    "policy_delay_s": float(row["release_dt_s"] - row["eligible_dt_s"]),
                    "predicted_capacity_saving_pct": float(row["expected_capacity_carbon_saving_pct"]),
                    "predicted_node_saving_pct": float(row["expected_node_carbon_saving_pct"]),
                    "true_carbon_runtime_intensity_improvement_pct": improvement,
                }
            )
        error = forecast_values - original
        summaries.append(
            {
                "scenario": scenario_name,
                "relative_sigma_pct": relative_sigma * 100,
                "seed": seed,
                "forecast_mae_gco2_per_kwh": float(np.mean(np.abs(error))),
                "forecast_rmse_gco2_per_kwh": float(np.sqrt(np.mean(error**2))),
                "forecast_mean_bias_gco2_per_kwh": float(np.mean(error)),
                "jobs_delayed": len(delayed),
                "total_policy_delay_h": float(
                    (delayed["release_dt_s"] - delayed["eligible_dt_s"]).sum() / 3600
                ),
                "jaccard_vs_reference": jaccard(reference_jobs, selected_jobs),
                "reference_jobs_retained": len(reference_jobs & selected_jobs),
                "new_jobs_selected": len(selected_jobs - reference_jobs),
                "true_carbon_positive_jobs": int(sum(value > 0 for value in true_improvements)),
                "true_carbon_nonpositive_jobs": int(sum(value <= 0 for value in true_improvements)),
                "mean_true_runtime_intensity_improvement_pct": (
                    float(np.mean(true_improvements)) if true_improvements else 0.0
                ),
            }
        )
        trace = forecast[["from_utc", "to_utc"]].copy()
        trace.insert(0, "scenario", scenario_name)
        trace["true_intensity_gco2_per_kwh"] = original
        trace["forecast_intensity_gco2_per_kwh"] = forecast_values
        traces.append(trace)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "carbon_forecast_sensitivity_summary.csv", index=False)
    pd.DataFrame(decisions).to_csv(
        args.output / "carbon_forecast_sensitivity_decisions.csv", index=False
    )
    pd.concat(traces, ignore_index=True).to_csv(
        args.output / "carbon_forecast_sensitivity_traces.csv", index=False
    )
    plot_summary(summary, args.output / "carbon_forecast_decision_stability.png")
    noisy = summary.loc[summary["relative_sigma_pct"].gt(0)]
    result = {
        "reference_jobs": sorted(reference_jobs),
        "noise_scenarios": len(noisy),
        "mean_jaccard_vs_reference": float(noisy["jaccard_vs_reference"].mean()),
        "minimum_jaccard_vs_reference": float(noisy["jaccard_vs_reference"].min()),
        "scenarios_with_exact_reference_set": int(noisy["jaccard_vs_reference"].eq(1.0).sum()),
        "scenarios_with_nonpositive_true_carbon_selection": int(
            noisy["true_carbon_nonpositive_jobs"].gt(0).sum()
        ),
        "scope": "Generation-only sensitivity; no Slurm replay and no claim of realised cluster carbon",
        "noise_model": {
            "relative_sigmas": args.noise_levels,
            "seeds": args.seeds,
            "autocorrelation": args.autocorrelation,
            "maximum_runtime_uncertainty_ratio": args.max_runtime_uncertainty_ratio,
            "minimum_intensity_gco2_per_kwh": 1.0,
        },
    }
    (args.output / "carbon_forecast_sensitivity_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
