#!/usr/bin/env python3
"""Build a conservative policy from forecast-noise-stable reference decisions."""

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


RESET_TO_ZERO = [
    "b1_delay_s",
    "b2_delay_s",
    "expected_runtime_carbon_saving_pct",
    "expected_capacity_carbon_saving_pct",
    "expected_node_carbon_saving_pct",
    "expected_risk_adjusted_carbon_saving_proxy",
    "carbon_return_per_wait_hour",
    "dynamic_wait_ratio",
    "dynamic_load_growth",
    "dynamic_marginal_utility",
    "forecast_incremental_wait_s",
]


def selection_frequencies(
    decisions: pd.DataFrame,
    noisy_scenario_names: set[str],
) -> tuple[pd.Series, int]:
    noisy = decisions.loc[~decisions["scenario"].eq("reference")].copy()
    unknown = set(noisy["scenario"].astype(str)) - noisy_scenario_names
    if unknown:
        raise ValueError(f"Decision rows contain unknown noise scenarios: {sorted(unknown)}")
    scenarios = len(noisy_scenario_names)
    if scenarios <= 0:
        raise ValueError("Decisions contain no noisy scenarios")
    counts = noisy.groupby("sim_job_id")["scenario"].nunique()
    return counts / scenarios, scenarios


def apply_consensus(
    profile: pd.DataFrame,
    frequencies: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    result = profile.copy()
    original_delayed = pd.to_numeric(result["release_dt_s"]).gt(
        pd.to_numeric(result["eligible_dt_s"])
    )
    result["consensus_selection_fraction"] = (
        result["sim_job_id"].map(frequencies).fillna(0.0).astype(float)
    )
    result["consensus_selected"] = original_delayed & result[
        "consensus_selection_fraction"
    ].ge(threshold)
    rejected = original_delayed & ~result["consensus_selected"]
    result["consensus_rejection_reason"] = ""
    result.loc[rejected, "consensus_rejection_reason"] = "below_forecast_stability_threshold"
    result.loc[rejected, "release_dt_s"] = result.loc[rejected, "eligible_dt_s"]
    for column in RESET_TO_ZERO:
        if column in result:
            result.loc[rejected, column] = 0.0
    if "predicted_policy_start_s" in result and "predicted_baseline_start_s" in result:
        result.loc[rejected, "predicted_policy_start_s"] = result.loc[
            rejected, "predicted_baseline_start_s"
        ]
    if "predicted_policy_queue_wait_s" in result and "predicted_baseline_queue_wait_s" in result:
        result.loc[rejected, "predicted_policy_queue_wait_s"] = result.loc[
            rejected, "predicted_baseline_queue_wait_s"
        ]
    result["policy_delay_s"] = (
        pd.to_numeric(result["release_dt_s"]) - pd.to_numeric(result["eligible_dt_s"])
    ).clip(lower=0)
    result["policy_name"] = "dynamic-history-consensus"
    result.sort_values(["release_dt_s", "source_submit_utc", "sim_job_id"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    result["slurm_job_id_expected"] = np.arange(1, len(result) + 1, dtype=int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scenario", type=Path, required=True)
    parser.add_argument("--noise-decisions", type=Path, required=True)
    parser.add_argument("--noise-summary", type=Path, required=True)
    parser.add_argument("--minimum-selection-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference_scenario / "workload_profile.csv")
    reference_summary = json.loads(
        (args.reference_scenario / "policy_summary.json").read_text(encoding="utf-8")
    )
    decisions = pd.read_csv(args.noise_decisions)
    noise_summary = pd.read_csv(args.noise_summary)
    noisy_scenario_names = set(
        noise_summary.loc[
            pd.to_numeric(noise_summary["relative_sigma_pct"], errors="coerce").gt(0),
            "scenario",
        ].astype(str)
    )
    frequencies, noisy_scenarios = selection_frequencies(
        decisions, noisy_scenario_names
    )
    result = apply_consensus(reference, frequencies, args.minimum_selection_fraction)
    delayed = result.loc[pd.to_numeric(result["policy_delay_s"]).gt(0)]
    original_delayed = reference.loc[
        pd.to_numeric(reference["release_dt_s"]).gt(
            pd.to_numeric(reference["eligible_dt_s"])
        )
    ]
    parameters = dict(reference_summary.get("parameters", {}))
    parameters.update(
        {
            "forecast_stability_gate": True,
            "minimum_noise_scenario_selection_fraction": args.minimum_selection_fraction,
            "noise_scenarios": noisy_scenarios,
            "consensus_scope": "Only reference decisions can be retained; noise-only opportunities are rejected",
        }
    )
    summary = {
        "strategy": "dynamic-history",
        "policy_variant": "forecast-stability-consensus",
        "policy_interpretation": "Retain only dynamic-history reference decisions selected in at least the configured fraction of carbon-forecast noise screens",
        "b2_limitation": reference_summary.get("b2_limitation"),
        "workload_epoch_utc": reference_summary["workload_epoch_utc"],
        "jobs": len(result),
        "flex_fraction": reference_summary["flex_fraction"],
        "flexible_jobs": int(result["is_flexible"].sum()),
        "reference_jobs_delayed": len(original_delayed),
        "jobs_delayed": len(delayed),
        "jobs_rejected_by_consensus": len(original_delayed) - len(delayed),
        "retained_sim_job_ids": sorted(delayed["sim_job_id"].astype(str)),
        "median_policy_delay_s_all_jobs": float(result["policy_delay_s"].median()),
        "p95_policy_delay_s_all_jobs": float(result["policy_delay_s"].quantile(0.95)),
        "median_policy_delay_s_delayed_jobs": (
            float(delayed["policy_delay_s"].median()) if len(delayed) else 0.0
        ),
        "max_policy_delay_s": float(result["policy_delay_s"].max()),
        "parameters": parameters,
    }
    POLICY.write_scenario(args.output, result, summary)
    frequency_output = pd.DataFrame(
        {
            "sim_job_id": frequencies.index.astype(str),
            "noise_selection_fraction": frequencies.values,
        }
    ).sort_values(["noise_selection_fraction", "sim_job_id"], ascending=[False, True])
    frequency_output.to_csv(args.output / "consensus_candidate_frequency.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
