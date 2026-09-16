#!/usr/bin/env python3
"""Audit runtime-information assumptions used by dynamic policy decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def evaluation_rows(profile: pd.DataFrame) -> pd.DataFrame:
    if "is_warmup" not in profile:
        return profile.copy()
    warmup = profile["is_warmup"].fillna(False).astype(bool)
    return profile.loc[~warmup].copy()


def ratio_summary(profile: pd.DataFrame) -> dict[str, float | int]:
    rows = evaluation_rows(profile)
    actual = pd.to_numeric(rows["runtime_s"], errors="raise")
    requested = pd.to_numeric(rows["source_timelimit_min"], errors="raise") * 60
    ratio = requested / actual
    return {
        "evaluation_jobs": len(rows),
        "requested_to_actual_ratio_p50": float(ratio.quantile(0.50)),
        "requested_to_actual_ratio_p95": float(ratio.quantile(0.95)),
        "requested_to_actual_ratio_mean": float(ratio.mean()),
        "requested_to_actual_ratio_max": float(ratio.max()),
        "requested_within_2x_actual_fraction": float((ratio <= 2.0).mean()),
        "requested_within_10x_actual_fraction": float((ratio <= 10.0).mean()),
    }


def parse_scenario(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("scenario must use NAME=WORKLOAD_PROFILE.csv")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--scenario", action="append", type=parse_scenario, default=[])
    parser.add_argument(
        "--reference-name",
        default="1.0x",
        help="Scenario name used as the decision-set reference",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_profile)
    rows = evaluation_rows(baseline)
    actual = pd.to_numeric(rows["runtime_s"], errors="raise")
    requested = pd.to_numeric(rows["source_timelimit_min"], errors="raise") * 60
    rows["requested_runtime_s"] = requested
    rows["requested_to_actual_ratio"] = requested / actual

    by_partition = (
        rows.groupby("partition", dropna=False)["requested_to_actual_ratio"]
        .agg(
            jobs="size",
            ratio_p50="median",
            ratio_p95=lambda values: values.quantile(0.95),
            ratio_mean="mean",
            ratio_max="max",
        )
        .reset_index()
    )

    selected_sets: dict[str, set[str]] = {}
    sensitivity_rows = []
    selected_rows = []
    for name, path in args.scenario:
        scenario = pd.read_csv(path)
        delayed = pd.to_numeric(scenario["policy_delay_s"], errors="coerce").fillna(0) > 0
        selected = scenario.loc[delayed].copy()
        selected_sets[name] = set(selected["sim_job_id"].astype(str))
        sensitivity_rows.append(
            {
                "scenario": name,
                "jobs_delayed": len(selected),
                "total_policy_wait_hours": float(
                    pd.to_numeric(selected["policy_delay_s"], errors="coerce").sum()
                    / 3600
                ),
                "decision_runtime_factor": float(
                    pd.to_numeric(
                        scenario.get("decision_runtime_s", scenario["runtime_s"]),
                        errors="coerce",
                    ).median()
                    / pd.to_numeric(scenario["runtime_s"], errors="coerce").median()
                ),
            }
        )
        for job_id in sorted(selected_sets[name]):
            selected_rows.append({"scenario": name, "sim_job_id": job_id})

    reference_name = args.reference_name if args.reference_name in selected_sets else None
    if reference_name:
        reference = selected_sets[reference_name]
        for row in sensitivity_rows:
            current = selected_sets[str(row["scenario"])]
            union = reference | current
            row["selection_jaccard_vs_reference"] = (
                len(reference & current) / len(union) if union else 1.0
            )

    args.output.mkdir(parents=True, exist_ok=True)
    by_partition.to_csv(args.output / "runtime_request_ratio_by_partition.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(
        args.output / "runtime_decision_sensitivity.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        args.output / "runtime_selected_jobs.csv", index=False
    )
    summary = {
        **ratio_summary(baseline),
        "interpretation": (
            "Observed runtime is an offline upper-bound input. Raw requested "
            "walltime is not an adequate direct replacement in this trace."
        ),
        "reference_selection_scenario": reference_name,
        "sensitivity_scenarios": sensitivity_rows,
    }
    (args.output / "runtime_information_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
