#!/usr/bin/env python3
"""Add real carry-in jobs before a generated policy scenario."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_empirical_workload", ROOT / "generate_empirical_workload.py"
)
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
assert GENERATOR_SPEC.loader is not None
GENERATOR_SPEC.loader.exec_module(GENERATOR)


def carry_in_profile(
    cohort: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    warmup_hours: float,
    max_runtime_hours: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    simulation_start = evaluation_start - pd.Timedelta(hours=warmup_hours)
    candidates = cohort.loc[
        (cohort["_start"] < evaluation_start)
        & (cohort["_end"] > evaluation_start)
    ].copy()
    candidates.sort_values(["_start", "_job_id"], inplace=True)

    user_counts = candidates["User"].value_counts()
    user_map = {
        raw: f"warmup_user_{index:03d}"
        for index, raw in enumerate(user_counts.index, start=1)
    }
    rows: list[dict[str, object]] = []
    dropped_capacity = 0
    max_runtime_s = int(max_runtime_hours * 3600)

    for _, source in candidates.iterrows():
        resources = GENERATOR.normalize_resources(source)
        if resources is None:
            dropped_capacity += 1
            continue

        model_start = max(source["_start"], simulation_start)
        remaining_original_s = (source["_end"] - model_start).total_seconds()
        runtime_s = min(max(1, int(round(remaining_original_s))), max_runtime_s)
        release_dt_s = max(
            0, int(round((model_start - simulation_start).total_seconds()))
        )
        runtime_min = int(math.ceil(runtime_s / 60))
        resources["runtime_s"] = runtime_s
        resources["timelimit_min"] = max(
            int(resources["timelimit_min"]),
            runtime_min + 60,
            runtime_min * 2,
        )

        rows.append(
            {
                "sim_job_id": f"warmup_{len(rows) + 1:06d}",
                # Keep source_submit_utc consistent with the simulator epoch.
                # Original arrival fields are not needed for carry-in validation.
                "source_submit_utc": model_start.isoformat(),
                "source_eligible_utc": model_start.isoformat(),
                "source_start_utc": source["_start"].isoformat(),
                "source_end_utc": source["_end"].isoformat(),
                "event_epoch_utc": simulation_start.isoformat(),
                "source_partition": source["Partition"],
                "qos": source["QOS"],
                "source_submit_wait_s": 0.0,
                "source_eligible_wait_s": 0.0,
                "dependency_delay_s": 0.0,
                "runtime_original_s": remaining_original_s,
                **resources,
                "submit_dt_s": release_dt_s,
                "eligible_dt_s": release_dt_s,
                "release_dt_s": release_dt_s,
                "user_id": user_map[source["User"]],
                "flexibility_score": 1.0,
                "is_flexible": False,
                "is_warmup": True,
                "b1_delay_s": 0.0,
                "b2_delay_s": 0.0,
                "policy_delay_s": 0.0,
            }
        )

    profile = pd.DataFrame(rows)
    summary = {
        "evaluation_start_utc": evaluation_start.isoformat(),
        "simulation_start_utc": simulation_start.isoformat(),
        "warmup_hours": warmup_hours,
        "carry_in_candidates": len(candidates),
        "carry_in_jobs_added": len(profile),
        "carry_in_jobs_dropped_for_capacity": dropped_capacity,
        "carry_in_users": int(profile["user_id"].nunique()) if not profile.empty else 0,
        "carry_in_gpu_jobs": int((profile["scheduled_gpus"] > 0).sum())
        if not profile.empty
        else 0,
        "carry_in_runtime_hours": float(profile["runtime_s"].sum() / 3600)
        if not profile.empty
        else 0.0,
    }
    return profile, summary


def combine_profiles(
    source_profile: pd.DataFrame,
    warmup: pd.DataFrame,
    warmup_seconds: int,
    policy_name: str,
) -> pd.DataFrame:
    main = source_profile.copy()
    if "event_epoch_utc" in main:
        source_epochs = pd.to_datetime(main["event_epoch_utc"], utc=True).dropna().unique()
        if len(source_epochs) != 1:
            raise ValueError("Source profile must contain one event_epoch_utc")
        combined_epoch = pd.Timestamp(source_epochs[0]) - pd.Timedelta(seconds=warmup_seconds)
        main["event_epoch_utc"] = combined_epoch.isoformat()
    for column in ["submit_dt_s", "eligible_dt_s", "release_dt_s"]:
        main[column] = pd.to_numeric(main[column], errors="raise") + warmup_seconds
    main["is_warmup"] = False
    if "is_flexible" not in main:
        main["is_flexible"] = (
            pd.to_numeric(main["flexibility_score"], errors="coerce") < 0.30
        ) & ~main["partition"].eq("interactive")
    for column in ["b1_delay_s", "b2_delay_s", "policy_delay_s"]:
        if column not in main:
            main[column] = 0.0
    if "policy_name" not in main:
        main["policy_name"] = policy_name

    all_columns = sorted(set(main.columns) | set(warmup.columns))
    combined = pd.concat(
        [main.reindex(columns=all_columns), warmup.reindex(columns=all_columns)],
        ignore_index=True,
    )
    combined.sort_values(
        ["release_dt_s", "is_warmup", "source_submit_utc", "sim_job_id"],
        ascending=[True, False, True, True],
        inplace=True,
    )
    combined.reset_index(drop=True, inplace=True)
    if "slurm_job_id_expected" in combined:
        combined.drop(columns=["slurm_job_id_expected"], inplace=True)
    combined.insert(
        1,
        "slurm_job_id_expected",
        np.arange(1, len(combined) + 1, dtype=int),
    )
    if combined["sim_job_id"].duplicated().any():
        raise ValueError("sim_job_id collision after adding warm-up jobs")
    return combined


def write_scenario(
    output: Path,
    profile: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output / "workload_profile.csv", index=False)
    (output / "sim.events").write_text(
        "\n".join(GENERATOR.event_line(row) for _, row in profile.iterrows()) + "\n",
        encoding="utf-8",
    )
    users = profile[["user_id"]].drop_duplicates().sort_values("user_id")
    with (output / "users.sim").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=":", lineterminator="\n")
        for index, user_id in enumerate(users["user_id"], start=1):
            writer.writerow([user_id, 10_000 + index, "users", 100])
    (output / "warmup_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scenario", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--month", default="june")
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--warmup-hours", type=float, default=2.0)
    parser.add_argument("--max-runtime-hours", type=float, default=12.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation_start = pd.Timestamp(args.evaluation_start)
    if evaluation_start.tzinfo is None:
        evaluation_start = evaluation_start.tz_localize("UTC")
    else:
        evaluation_start = evaluation_start.tz_convert("UTC")
    if args.warmup_hours <= 0:
        raise ValueError("warmup-hours must be positive")

    source_profile = pd.read_csv(args.source_scenario / "workload_profile.csv")
    raw = GENERATOR.read_month(args.zip_path, args.month)
    cohort, cohort_audit = GENERATOR.prepare_cohort(
        raw, args.month, args.max_runtime_hours
    )
    warmup, warmup_summary = carry_in_profile(
        cohort,
        evaluation_start,
        args.warmup_hours,
        args.max_runtime_hours,
    )
    combined = combine_profiles(
        source_profile,
        warmup,
        int(round(args.warmup_hours * 3600)),
        args.source_scenario.name,
    )
    summary = {
        "source_scenario": str(args.source_scenario),
        "source_main_jobs": len(source_profile),
        "combined_jobs": len(combined),
        "cohort_audit": cohort_audit,
        **warmup_summary,
    }
    write_scenario(args.output, combined, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
