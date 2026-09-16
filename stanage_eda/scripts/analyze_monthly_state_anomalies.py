#!/usr/bin/env python3
"""Audit monthly Slurm state changes without exporting direct identifiers."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from analyse_stanage import MONTHS, base_state, unique_headers


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    seen_ids: set[int] = set()
    states_by_month: dict[str, Counter[str]] = defaultdict(Counter)
    failed_users_by_month: dict[str, Counter[str]] = defaultdict(Counter)

    with zipfile.ZipFile(args.zip_path) as archive:
        for source_month in reversed(MONTHS):
            entry = f"stanage_2025_queue_record/{source_month}.txt"
            with archive.open(entry) as handle:
                raw_headers = handle.readline().decode("utf-8").rstrip("\r\n").split("|")
            headers = unique_headers(raw_headers)
            with archive.open(entry) as handle:
                reader = pd.read_csv(
                    handle,
                    sep="|",
                    header=0,
                    names=headers,
                    usecols=["JobIDRaw", "Submit", "State", "User"],
                    dtype=str,
                    chunksize=args.chunksize,
                    keep_default_na=False,
                    na_filter=False,
                    engine="c",
                    on_bad_lines="warn",
                )
                for frame in reader:
                    numeric_ids = pd.to_numeric(frame["JobIDRaw"], errors="coerce")
                    valid = numeric_ids.notna()
                    frame = frame.loc[valid].copy()
                    ids = numeric_ids.loc[valid].astype("int64").to_numpy()
                    keep = []
                    for index, job_id in enumerate(ids):
                        value = int(job_id)
                        if value not in seen_ids:
                            seen_ids.add(value)
                            keep.append(index)
                    if not keep:
                        continue
                    frame = frame.iloc[keep].copy()
                    submit = pd.to_datetime(frame["Submit"], errors="coerce", utc=True)
                    cohort = submit.dt.year.eq(2025).fillna(False)
                    frame = frame.loc[cohort].copy()
                    submit = submit.loc[cohort]
                    if frame.empty:
                        continue
                    months = submit.dt.strftime("%Y-%m")
                    states = base_state(frame["State"])
                    users = frame["User"].replace({"": "UNKNOWN", "None": "UNKNOWN"})
                    for month, indices in months.groupby(months).groups.items():
                        month_states = states.loc[indices]
                        states_by_month[month].update(month_states.tolist())
                        failed = ~month_states.eq("COMPLETED")
                        failed_users_by_month[month].update(users.loc[indices][failed].tolist())

    state_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    for month in sorted(states_by_month):
        counts = states_by_month[month]
        total = sum(counts.values())
        for state, jobs in counts.most_common():
            state_rows.append(
                {
                    "submit_month": month,
                    "state": state,
                    "jobs": jobs,
                    "month_share_pct": 100.0 * jobs / total if total else 0.0,
                }
            )
        failed_users = failed_users_by_month[month]
        failure_jobs = sum(failed_users.values())
        ordered = sorted(failed_users.values(), reverse=True)
        concentration_rows.append(
            {
                "submit_month": month,
                "jobs": total,
                "completed_jobs": counts.get("COMPLETED", 0),
                "completion_rate_pct": 100.0 * counts.get("COMPLETED", 0) / total,
                "noncompleted_jobs": failure_jobs,
                "noncompleted_users": len(failed_users),
                "top_1_user_noncompleted_share_pct": (
                    100.0 * sum(ordered[:1]) / failure_jobs if failure_jobs else 0.0
                ),
                "top_5_users_noncompleted_share_pct": (
                    100.0 * sum(ordered[:5]) / failure_jobs if failure_jobs else 0.0
                ),
            }
        )

    if not state_rows:
        raise ValueError("No 2025 state rows were produced")
    write_rows(args.output / "monthly_state_summary.csv", state_rows)
    write_rows(args.output / "monthly_failure_concentration.csv", concentration_rows)

    focus = {
        row["submit_month"]: row
        for row in concentration_rows
        if row["submit_month"] in {"2025-07", "2025-08"}
    }
    state_focus = [
        row for row in state_rows if row["submit_month"] in {"2025-07", "2025-08"}
    ]
    summary = {
        "privacy": "Raw users are used only for aggregate concentration and are not exported",
        "deduplication": "Newest monthly record for each numeric JobIDRaw is retained",
        "july_august": focus,
        "july_august_state_breakdown": state_focus,
    }
    (args.output / "july_august_anomaly_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
