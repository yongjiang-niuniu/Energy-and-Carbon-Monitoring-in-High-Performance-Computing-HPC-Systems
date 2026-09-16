#!/usr/bin/env python3
"""Verify that real carry-in jobs are active at the evaluation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(
        (args.scenario / "warmup_summary.json").read_text(encoding="utf-8")
    )
    results = pd.read_csv(args.scenario / "job_results.csv")
    warmup = results.loc[boolean_series(results["is_warmup"])].copy()
    boundary_s = float(summary["warmup_hours"]) * 3600

    release = pd.to_numeric(warmup["release_dt_s"], errors="coerce")
    start = pd.to_numeric(warmup["sim_start_offset_s"], errors="coerce")
    end = pd.to_numeric(warmup["sim_end_offset_s"], errors="coerce")
    active = start.le(boundary_s) & end.gt(boundary_s)

    resources = (
        warmup.loc[active]
        .groupby("partition", as_index=False)
        .agg(
            active_jobs=("sim_job_id", "size"),
            requested_nodes=("nodes", "sum"),
            requested_cpus=("cpus", "sum"),
            scheduled_gpus=("scheduled_gpus", "sum"),
        )
    )
    expected = int(summary["carry_in_jobs_added"])
    checks = {
        "all_carry_in_jobs_present": len(warmup) == expected,
        "all_released_by_evaluation_start": int(release.le(boundary_s).sum())
        == expected,
        "all_started_by_evaluation_start": int(start.le(boundary_s).sum())
        == expected,
        "all_active_at_evaluation_start": int(active.sum()) == expected,
        "none_completed_before_evaluation_start": int(end.le(boundary_s).sum()) == 0,
    }
    audit = {
        "scenario": args.scenario.name,
        "evaluation_start_utc": summary["evaluation_start_utc"],
        "evaluation_boundary_offset_s": boundary_s,
        "carry_in_jobs_expected": expected,
        "carry_in_jobs_present": len(warmup),
        "carry_in_jobs_active_at_boundary": int(active.sum()),
        "active_gpu_jobs": int(
            (pd.to_numeric(warmup.loc[active, "scheduled_gpus"]) > 0).sum()
        ),
        "checks": checks,
        "pass": all(checks.values()),
        "resources_by_partition": resources.to_dict(orient="records"),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    resources.to_csv(args.output / "warmup_resources_at_evaluation_start.csv", index=False)
    (args.output / "warmup_state_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if not audit["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
