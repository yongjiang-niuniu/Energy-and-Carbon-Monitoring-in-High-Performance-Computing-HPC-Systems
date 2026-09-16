#!/usr/bin/env python3
"""Audit consensus-set construction and realised dual-proxy acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def delayed_ids(profile: pd.DataFrame) -> set[str]:
    delayed = pd.to_numeric(profile["release_dt_s"], errors="coerce").gt(
        pd.to_numeric(profile["eligible_dt_s"], errors="coerce")
    )
    return set(profile.loc[delayed, "sim_job_id"].astype(str))


def audit_consensus_sets(
    reference: pd.DataFrame,
    consensus: pd.DataFrame,
    threshold: float,
) -> dict[str, object]:
    reference_ids = delayed_ids(reference)
    consensus_ids = delayed_ids(consensus)
    indexed = consensus.set_index("sim_job_id")
    selected_fractions = pd.to_numeric(
        indexed.loc[list(consensus_ids), "consensus_selection_fraction"],
        errors="coerce",
    ) if consensus_ids else pd.Series(dtype=float)
    rejected_ids = reference_ids - consensus_ids
    rejected_fractions = pd.to_numeric(
        indexed.loc[list(rejected_ids), "consensus_selection_fraction"],
        errors="coerce",
    ) if rejected_ids else pd.Series(dtype=float)
    checks = {
        "same_job_population": set(reference["sim_job_id"].astype(str))
        == set(consensus["sim_job_id"].astype(str)),
        "consensus_is_reference_subset": consensus_ids <= reference_ids,
        "all_retained_meet_threshold": bool(selected_fractions.ge(threshold).all()),
        "all_rejected_below_threshold": bool(rejected_fractions.lt(threshold).all()),
        "no_negative_delays": bool(
            (
                pd.to_numeric(consensus["release_dt_s"], errors="coerce")
                - pd.to_numeric(consensus["eligible_dt_s"], errors="coerce")
            ).ge(0).all()
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "reference_delayed_ids": sorted(reference_ids),
        "consensus_delayed_ids": sorted(consensus_ids),
        "rejected_ids": sorted(rejected_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scenario", type=Path, required=True)
    parser.add_argument("--consensus-scenario", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--experiment-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference_scenario / "workload_profile.csv")
    consensus = pd.read_csv(args.consensus_scenario / "workload_profile.csv")
    summary = json.loads(
        (args.consensus_scenario / "policy_summary.json").read_text(encoding="utf-8")
    )
    threshold = float(
        summary["parameters"]["minimum_noise_scenario_selection_fraction"]
    )
    set_audit = audit_consensus_sets(reference, consensus, threshold)
    experiment = json.loads(args.experiment_audit.read_text(encoding="utf-8"))
    comparison = pd.read_csv(args.comparison)
    row = comparison.loc[comparison["scenario"].eq(args.consensus_scenario.name)]
    if len(row) != 1:
        raise ValueError("Comparison does not contain exactly one consensus row")
    metrics = row.iloc[0]
    reductions = {
        "capacity_weighted": float(
            metrics["dynamic_carbon_reduction_pct_capacity_weighted"]
        ),
        "node_request_upper": float(
            metrics["dynamic_carbon_reduction_pct_node_request_upper"]
        ),
    }
    checks = {
        **set_audit["checks"],
        "experiment_structural_audit_pass": bool(experiment["overall_pass"]),
        "realised_capacity_proxy_nonnegative": reductions["capacity_weighted"] >= 0,
        "realised_node_proxy_nonnegative": reductions["node_request_upper"] >= 0,
    }
    accepted = all(checks.values())
    result = {
        "accepted_for_further_shadow_evaluation": accepted,
        "not_production_approval": True,
        "checks": checks,
        "minimum_selection_fraction": threshold,
        "reference_delayed_ids": set_audit["reference_delayed_ids"],
        "consensus_delayed_ids": set_audit["consensus_delayed_ids"],
        "rejected_ids": set_audit["rejected_ids"],
        "realised_dynamic_carbon_reduction_pct": reductions,
        "claim_boundary": "Acceptance means this exploratory replay passed structural and dual-proxy gates; it is not independent deployment validation",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "consensus_acceptance_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
