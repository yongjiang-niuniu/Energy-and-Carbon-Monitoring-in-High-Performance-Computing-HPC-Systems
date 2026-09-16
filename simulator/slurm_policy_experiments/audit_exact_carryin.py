#!/usr/bin/env python3
"""Audit exact carry-in categories before and after a baseline replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CATEGORIES = {"running", "queued", "held", "evaluation"}


def audit_profile(profile: pd.DataFrame) -> list[dict[str, object]]:
    t0 = pd.to_datetime(profile["event_epoch_utc"], utc=True).iloc[0]
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("unique_sim_job_id", not profile["sim_job_id"].duplicated().any(), len(profile))
    add(
        "known_categories",
        profile["carry_in_type"].notna().all()
        and set(profile["carry_in_type"].unique()).issubset(CATEGORIES),
        sorted(profile["carry_in_type"].dropna().unique()),
    )
    running = profile["carry_in_type"].eq("running")
    queued = profile["carry_in_type"].eq("queued")
    held = profile["carry_in_type"].eq("held")
    evaluation = profile["carry_in_type"].eq("evaluation")
    add("running_released_at_t0", profile.loc[running, "release_dt_s"].eq(0).all(), int(running.sum()))
    add("queued_released_at_t0", profile.loc[queued, "release_dt_s"].eq(0).all(), int(queued.sum()))
    expected_held = (
        pd.to_datetime(profile.loc[held, "source_eligible_utc"], utc=True) - t0
    ).dt.total_seconds().round().astype(int)
    add(
        "held_released_at_source_eligible",
        profile.loc[held, "release_dt_s"].astype(int).reset_index(drop=True).equals(
            expected_held.reset_index(drop=True)
        ),
        int(held.sum()),
    )
    eligible = pd.to_datetime(profile.loc[evaluation, "source_eligible_utc"], utc=True)
    add(
        "evaluation_is_exact_24h_window",
        bool(eligible.ge(t0).all() and eligible.lt(t0 + pd.Timedelta(hours=24)).all()),
        int(evaluation.sum()),
    )
    carry_in = ~evaluation
    carry_in_flexible = (
        profile.loc[carry_in, "is_flexible"].fillna(False).astype(bool)
        if "is_flexible" in profile
        else pd.to_numeric(
            profile.loc[carry_in, "flexibility_score"], errors="coerce"
        ).lt(1.0)
    )
    add(
        "carry_in_is_non_flexible",
        bool(~carry_in_flexible.any()),
        int(carry_in.sum()),
    )
    add(
        "carry_in_has_zero_policy_delay",
        profile.loc[carry_in, "policy_delay_s"].fillna(0).eq(0).all(),
        int(carry_in.sum()),
    )
    add(
        "no_negative_release_offsets",
        pd.to_numeric(profile["release_dt_s"], errors="coerce").ge(0).all(),
        float(pd.to_numeric(profile["release_dt_s"], errors="coerce").min()),
    )
    return checks


def audit_results(profile: pd.DataFrame, results: pd.DataFrame) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    result_columns = [
        name
        for name in ["sim_job_id", "terminal_status", "sim_start_offset_s"]
        if name in results
    ]
    merged = profile[["sim_job_id", "carry_in_type"]].merge(
        results[result_columns], on="sim_job_id", how="left", validate="one_to_one"
    )
    running = merged["carry_in_type"].eq("running")
    add(
        "all_profile_jobs_in_results",
        merged["terminal_status"].notna().all(),
        int(merged["terminal_status"].notna().sum()),
    )
    add(
        "all_jobs_completed",
        merged["terminal_status"].eq("completed").all(),
        merged["terminal_status"].value_counts().to_dict(),
    )
    start_offset = pd.to_numeric(merged.loc[running, "sim_start_offset_s"], errors="coerce")
    add(
        "running_carry_in_started_within_60s",
        start_offset.le(60).all(),
        {"jobs": int(running.sum()), "max_start_offset_s": float(start_offset.max())},
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--job-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = pd.read_csv(args.profile)
    checks = audit_profile(profile)
    if args.job_results:
        checks.extend(audit_results(profile, pd.read_csv(args.job_results)))
    summary = {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
