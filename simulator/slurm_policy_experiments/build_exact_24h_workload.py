#!/usr/bin/env python3
"""Build an exact 24-hour evaluation workload with audited carry-in state."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import Counter
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

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
CATEGORY_ORDER = {"running": 0, "queued": 1, "held": 2, "evaluation": 3}


def load_trace(
    zip_path: Path, months: list[str], max_runtime_hours: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    month_audits: dict[str, dict[str, int]] = {}
    for month in months:
        raw = GENERATOR.read_month(zip_path, month)
        cohort, audit = GENERATOR.prepare_cohort(raw, month, max_runtime_hours)
        frames.append(cohort)
        month_audits[month] = audit

    trace = pd.concat(frames, ignore_index=True)
    before_dedup = len(trace)
    trace.sort_values(["_submit", "_job_id"], inplace=True)
    trace.drop_duplicates("_job_id", keep="last", inplace=True)
    trace.reset_index(drop=True, inplace=True)
    return trace, {
        "months_loaded": months,
        "per_month": month_audits,
        "rows_before_cross_month_deduplication": before_dedup,
        "rows_after_cross_month_deduplication": len(trace),
    }


def classify_jobs(
    trace: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, object]]:
    usable = trace.loc[
        trace["_start"].notna()
        & trace["_end"].notna()
        & trace["_end"].gt(trace["_start"])
    ].copy()

    running = usable["_start"].lt(t0) & usable["_end"].gt(t0)
    queued = (
        ~running
        & usable["_eligible"].le(t0)
        & usable["_start"].gt(t0)
    )
    held = (
        ~running
        & ~queued
        & usable["_submit"].lt(t0)
        & usable["_eligible"].gt(t0)
    )
    evaluation_raw = usable["_eligible"].ge(t0) & usable["_eligible"].lt(t1)
    evaluation = evaluation_raw & ~running & ~queued & ~held

    selected = usable.loc[running | queued | held | evaluation].copy()
    selected["carry_in_type"] = ""
    selected.loc[running, "carry_in_type"] = "running"
    selected.loc[queued, "carry_in_type"] = "queued"
    selected.loc[held, "carry_in_type"] = "held"
    selected.loc[evaluation, "carry_in_type"] = "evaluation"

    overlap = int((evaluation_raw & held).sum())
    audit = {
        "usable_completed_jobs": len(usable),
        "running_at_t0": int(running.sum()),
        "queued_at_t0": int(queued.sum()),
        "held_or_dependency_at_t0": int(held.sum()),
        "evaluation_jobs": int(evaluation.sum()),
        "eligible_window_jobs_before_carry_in_precedence": int(evaluation_raw.sum()),
        "held_and_evaluation_definition_overlap": overlap,
        "category_precedence": ["running", "queued", "held", "evaluation"],
        "precedence_reason": (
            "A job submitted before T0 is treated as carry-in even when it becomes "
            "Eligible inside the evaluation window. This prevents policy delay from "
            "being applied to state-reconstruction jobs."
        ),
        "selected_jobs_before_capacity_checks": len(selected),
    }
    return selected, audit


def build_profile(
    selected: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, seed: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    dropped_by_category: Counter[str] = Counter()

    for _, source in selected.iterrows():
        category = str(source["carry_in_type"])
        source_runtime_s = float(source["_runtime_s"])
        if category == "running":
            simulated_runtime_s = max((source["_end"] - t0).total_seconds(), 1.0)
        else:
            simulated_runtime_s = source_runtime_s

        normalized_source = source.copy()
        normalized_source["_runtime_s"] = simulated_runtime_s
        resources = GENERATOR.normalize_resources(normalized_source)
        if resources is None:
            dropped_by_category[category] += 1
            continue

        if category in {"running", "queued"}:
            release_offset_s = 0
        else:
            release_offset_s = max(
                0, int(round((source["_eligible"] - t0).total_seconds()))
            )

        source_timelimit_min = float(source["_timelimit_min"])
        rows.append(
            {
                "source_submit_utc": source["_submit"].isoformat(),
                "source_eligible_utc": source["_eligible"].isoformat(),
                "source_start_utc": source["_start"].isoformat(),
                "source_end_utc": source["_end"].isoformat(),
                "source_partition": source["Partition"],
                "qos": source["QOS"],
                "source_user_private": source["User"],
                "source_submit_wait_s": (
                    source["_start"] - source["_submit"]
                ).total_seconds(),
                "source_eligible_wait_s": (
                    source["_start"] - source["_eligible"]
                ).total_seconds(),
                "dependency_delay_s": max(
                    (source["_eligible"] - source["_submit"]).total_seconds(), 0.0
                ),
                "runtime_original_s": float(source["_runtime_original_s"]),
                "source_runtime_capped_s": source_runtime_s,
                "carry_in_type": category,
                "is_warmup": category != "evaluation",
                "is_evaluation": category == "evaluation",
                "source_submit_offset_s": (
                    source["_submit"] - t0
                ).total_seconds(),
                "source_eligible_offset_s": (
                    source["_eligible"] - t0
                ).total_seconds(),
                "source_start_offset_s": (
                    source["_start"] - t0
                ).total_seconds(),
                "source_end_offset_s": (
                    source["_end"] - t0
                ).total_seconds(),
                "deadline_proxy_source": "source_time_limit",
                "source_timelimit_allowance_s": (
                    max(source_timelimit_min * 60 - source_runtime_s, 0.0)
                    if math.isfinite(source_timelimit_min)
                    else 0.0
                ),
                **resources,
                "submit_dt_s": release_offset_s,
                "eligible_dt_s": release_offset_s,
                "release_dt_s": release_offset_s,
                "event_epoch_utc": t0.isoformat(),
            }
        )

    profile = pd.DataFrame(rows)
    if profile.empty:
        raise ValueError("All selected jobs exceeded the simulator capacity assumptions")

    user_counts = profile["source_user_private"].value_counts()
    user_map = {
        raw: f"user_{index:03d}"
        for index, raw in enumerate(user_counts.index, start=1)
    }
    profile["user_id"] = profile["source_user_private"].map(user_map)
    profile.drop(columns=["source_user_private"], inplace=True)

    rng = random.Random(seed)
    profile["flexibility_score"] = [rng.random() for _ in range(len(profile))]
    profile.loc[profile["partition"].eq("interactive"), "flexibility_score"] = 1.0
    profile.loc[profile["is_warmup"], "flexibility_score"] = 1.0
    profile["b1_delay_s"] = 0.0
    profile["b2_delay_s"] = 0.0
    profile["policy_delay_s"] = 0.0
    profile["policy_name"] = "baseline"

    profile["_category_order"] = profile["carry_in_type"].map(CATEGORY_ORDER)
    profile.sort_values(
        ["release_dt_s", "_category_order", "source_eligible_utc", "source_start_utc"],
        inplace=True,
    )
    profile.drop(columns=["_category_order"], inplace=True)
    profile.reset_index(drop=True, inplace=True)
    profile.insert(
        0, "sim_job_id", [f"sim_{index:06d}" for index in range(1, len(profile) + 1)]
    )
    profile.insert(
        1, "slurm_job_id_expected", np.arange(1, len(profile) + 1, dtype=int)
    )

    counts = profile["carry_in_type"].value_counts().to_dict()
    audit = {
        "jobs_in_profile": len(profile),
        "category_counts_after_capacity_checks": {
            name: int(counts.get(name, 0)) for name in CATEGORY_ORDER
        },
        "dropped_for_working_capacity": {
            name: int(dropped_by_category.get(name, 0)) for name in CATEGORY_ORDER
        },
        "unique_pseudonymous_users": int(profile["user_id"].nunique()),
        "evaluation_release_span_s": int(
            profile.loc[profile["is_evaluation"], "release_dt_s"].max()
        ),
        "latest_source_evaluation_end_utc": profile.loc[
            profile["is_evaluation"], "source_end_utc"
        ].max(),
        "longest_simulated_runtime_hours": float(profile["runtime_s"].max()) / 3600,
        "t0_cluster_not_empty": bool((profile["release_dt_s"] == 0).any()),
        "t0_utc": t0.isoformat(),
        "t1_utc": t1.isoformat(),
    }
    return profile, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--t0-utc", required=True)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--max-runtime-hours", type=float, default=96.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_hours < 24:
        raise ValueError("duration-hours must be at least 24")
    t0 = pd.Timestamp(args.t0_utc)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    t1 = t0 + pd.Timedelta(hours=args.duration_hours)
    last_month_index = t1.month - 1
    months = MONTHS[: last_month_index + 1]

    trace, trace_audit = load_trace(args.zip_path, months, args.max_runtime_hours)
    selected, category_audit = classify_jobs(trace, t0, t1)
    profile, profile_audit = build_profile(selected, t0, t1, args.seed)

    summary = {
        "source_archive": str(args.zip_path),
        "source_timezone": "UTC",
        "privacy": "Raw user and source job identifiers are not exported",
        "method": "Exact carry-in reconstruction plus a 24-hour Eligible evaluation window",
        "event_release_rule": {
            "running": "released first at T0 with remaining observed runtime",
            "queued": "released at T0 with full observed runtime",
            "held": "released at the source Eligible time after T0",
            "evaluation": "released at source Eligible in [T0,T1)",
        },
        "carry_in_policy_rule": "All carry-in categories are non-flexible and cannot receive strategy delay",
        "seed": args.seed,
        "duration_hours": args.duration_hours,
        "max_runtime_hours": args.max_runtime_hours,
        "trace_audit": trace_audit,
        "category_audit": category_audit,
        "profile_audit": profile_audit,
    }
    GENERATOR.write_outputs(args.output, profile, summary)
    (args.output / "exact_carry_in_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
