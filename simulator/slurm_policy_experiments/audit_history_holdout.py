#!/usr/bin/env python3
"""Audit temporal separation, privacy and policy invariants for history holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

DIRECT_IDENTIFIER_COLUMNS = {
    "account",
    "account_id",
    "command",
    "job",
    "job_id",
    "job_name",
    "jobid",
    "name",
    "numeric_job_id",
    "submit_line",
    "uid",
    "user",
    "user_id",
    "username",
    "workdir",
}


def find_identifier_columns(columns: list[str]) -> list[str]:
    """Return direct identifier fields without rejecting aggregate count columns."""
    return [
        column
        for column in columns
        if column.strip().lower() in DIRECT_IDENTIFIER_COLUMNS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(
        (args.model_dir / "model_metadata.json").read_text(encoding="utf-8")
    )
    holdout_index = MONTHS.index(metadata["holdout_month"])
    temporal_split_pass = all(
        MONTHS.index(month) < holdout_index for month in metadata["training_months"]
    )

    forbidden_columns: dict[str, list[str]] = {}
    for path in sorted(args.model_dir.glob("*.csv")):
        columns = pd.read_csv(path, nrows=0).columns
        forbidden = find_identifier_columns(columns.tolist())
        if forbidden:
            forbidden_columns[path.name] = forbidden

    policy_summary = json.loads(
        (args.scenario / "policy_summary.json").read_text(encoding="utf-8")
    )
    profile = pd.read_csv(args.scenario / "workload_profile.csv")
    delayed = profile.loc[pd.to_numeric(profile["policy_delay_s"], errors="coerce").gt(0)]
    delayed_reliability_pass = bool(
        delayed["runtime_prediction_reliable"].fillna(False).astype(bool).all()
    )
    delayed_runtime_budget_pass = bool(
        pd.to_numeric(delayed["policy_delay_s"], errors="coerce")
        .le(pd.to_numeric(delayed["runtime_s"], errors="coerce") + 1e-6)
        .all()
    )
    decision_source_pass = set(profile["decision_runtime_source"].dropna()) == {
        "earlier_month_grouped_quantile"
    }
    policy_parameters = policy_summary["parameters"]
    policy_split_pass = bool(
        policy_parameters["temporal_split_pass"]
        and policy_parameters["training_months"] == metadata["training_months"]
        and policy_parameters["holdout_month"] == metadata["holdout_month"]
    )

    evaluation = json.loads(
        (args.evaluation / "holdout_evaluation_summary.json").read_text(encoding="utf-8")
    )
    evaluation_split_pass = bool(evaluation["temporal_split_pass"])
    validation_path = args.scenario / "simulation_validation.json"
    simulation_strict_pass = None
    if validation_path.exists():
        simulation_strict_pass = bool(
            json.loads(validation_path.read_text(encoding="utf-8"))["strict_pass"]
        )

    checks = {
        "temporal_split_pass": temporal_split_pass,
        "aggregate_model_has_no_identifier_columns": not forbidden_columns,
        "policy_metadata_matches_model_split": policy_split_pass,
        "decision_runtime_uses_history_quantiles": decision_source_pass,
        "all_delayed_jobs_pass_uncertainty_gate": delayed_reliability_pass,
        "no_policy_delay_exceeds_observed_runtime": delayed_runtime_budget_pass,
        "separate_holdout_evaluator_split_pass": evaluation_split_pass,
        "simulation_strict_pass": simulation_strict_pass,
    }
    overall = all(value is True for value in checks.values())
    result = {
        "overall_pass": overall,
        "checks": checks,
        "training_months": metadata["training_months"],
        "holdout_month": metadata["holdout_month"],
        "delayed_jobs": len(delayed),
        "forbidden_model_columns": forbidden_columns,
        "code_boundary": "generate_policy_scenario.py imports the aggregate model builder, not evaluate_history_holdout.py",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "history_holdout_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
