#!/usr/bin/env python3
"""Parse a Slurm simulator log and validate it against its workload profile."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP = re.compile(r"^\[(?P<timestamp>[^]]+)]")
SUBMITTED = re.compile(r"_slurm_rpc_submit_batch_job: JobId=(?P<job_id>\d+)")
ALLOCATED = re.compile(
    r"sched: Allocate JobId=(?P<job_id>\d+) NodeList=(?P<nodes>\S+) "
    r"#CPUs=(?P<cpus>\d+) Partition=(?P<partition>\S+)"
)
BACKFILLED = re.compile(
    r"_start_job: Started JobId=(?P<job_id>\d+) in (?P<partition>\S+) "
    r"on (?P<nodes>\S+)"
)
COMPLETED = re.compile(r"_job_complete: JobId=(?P<job_id>\d+) done")
TIMED_OUT = re.compile(r"Time limit exhausted for JobId=(?P<job_id>\d+)")
EPILOG = re.compile(r"job_epilog_complete for JobId=(?P<job_id>\d+)")
EARLY_COMPLETION = re.compile(r"Can not stop (?P<job_id>\d+) job, it is not running")


def parse_log(log_path: Path) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = defaultdict(dict)
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            timestamp_match = TIMESTAMP.match(line)
            if not timestamp_match:
                continue
            timestamp = pd.Timestamp(timestamp_match.group("timestamp"), tz="UTC")

            match = SUBMITTED.search(line)
            if match:
                job_id = int(match.group("job_id"))
                records[job_id].setdefault("submitted_at", timestamp)
                continue

            match = ALLOCATED.search(line) or BACKFILLED.search(line)
            if match:
                job_id = int(match.group("job_id"))
                records[job_id].setdefault("started_at", timestamp)
                records[job_id].setdefault("allocated_nodelist", match.group("nodes"))
                records[job_id].setdefault("allocated_partition", match.group("partition"))
                if "cpus" in match.groupdict():
                    records[job_id].setdefault("allocated_cpus_log", int(match.group("cpus")))
                continue

            match = COMPLETED.search(line)
            if match:
                job_id = int(match.group("job_id"))
                records[job_id].setdefault("completed_at", timestamp)
                continue

            match = TIMED_OUT.search(line)
            if match:
                job_id = int(match.group("job_id"))
                records[job_id].setdefault("timed_out_at", timestamp)
                continue

            match = EPILOG.search(line)
            if match:
                job_id = int(match.group("job_id"))
                records[job_id].setdefault("epilog_at", timestamp)
                continue

            match = EARLY_COMPLETION.search(line)
            if match:
                job_id = int(match.group("job_id"))
                records[job_id].setdefault("early_completion_line", line_number)

    return dict(records)


def terminal_status(record: dict[str, object]) -> str:
    if "completed_at" in record:
        return "completed"
    if "timed_out_at" in record:
        return "timed_out"
    if "started_at" in record:
        return "started_no_completion"
    if "submitted_at" in record:
        return "submitted_no_start"
    return "missing"


def percentile(values: pd.Series, quantile: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.quantile(quantile))


def build_results(profile: pd.DataFrame, records: dict[int, dict[str, object]]) -> pd.DataFrame:
    if profile["slurm_job_id_expected"].duplicated().any():
        raise ValueError("slurm_job_id_expected must be unique")

    rows: list[dict[str, object]] = []
    for source in profile.to_dict(orient="records"):
        job_id = int(source["slurm_job_id_expected"])
        log = records.get(job_id, {})
        submitted_at = log.get("submitted_at")
        started_at = log.get("started_at")
        completed_at = log.get("completed_at")
        timed_out_at = log.get("timed_out_at")
        end_at = completed_at or log.get("epilog_at") or timed_out_at

        scheduler_wait = np.nan
        observed_runtime = np.nan
        if submitted_at is not None and started_at is not None:
            scheduler_wait = (started_at - submitted_at).total_seconds()
        if started_at is not None and end_at is not None:
            observed_runtime = (end_at - started_at).total_seconds()

        policy_delay = max(
            float(source.get("release_dt_s", 0)) - float(source.get("eligible_dt_s", 0)),
            0.0,
        )
        dependency_delay = float(source.get("dependency_delay_s", 0) or 0)
        total_user_wait = (
            dependency_delay + policy_delay + scheduler_wait
            if np.isfinite(scheduler_wait)
            else np.nan
        )

        rows.append(
            {
                **source,
                "sim_submit_log_utc": submitted_at,
                "sim_start_log_utc": started_at,
                "sim_end_log_utc": end_at,
                "sim_scheduler_wait_s": scheduler_wait,
                "sim_policy_delay_s": policy_delay,
                "sim_total_user_wait_s": total_user_wait,
                "sim_observed_runtime_s": observed_runtime,
                "terminal_status": terminal_status(log),
                "allocated_nodelist": log.get("allocated_nodelist", ""),
                "allocated_partition_log": log.get("allocated_partition", ""),
                "early_completion_event": "early_completion_line" in log,
                "timeout_event": "timed_out_at" in log,
            }
        )

    results = pd.DataFrame(rows)
    results["negative_queue_wait"] = results["sim_scheduler_wait_s"].lt(0)
    results["negative_total_wait"] = results["sim_total_user_wait_s"].lt(0)
    submitted = pd.to_datetime(results["sim_submit_log_utc"], utc=True, errors="coerce")
    release = pd.to_numeric(results["release_dt_s"], errors="coerce")
    epoch_candidates = submitted - pd.to_timedelta(release, unit="s")
    epoch = epoch_candidates.dropna().median() if epoch_candidates.notna().any() else pd.NaT
    results["sim_epoch_utc"] = epoch
    for source_column, output_column in [
        ("sim_submit_log_utc", "sim_submit_offset_s"),
        ("sim_start_log_utc", "sim_start_offset_s"),
        ("sim_end_log_utc", "sim_end_offset_s"),
    ]:
        timestamps = pd.to_datetime(results[source_column], utc=True, errors="coerce")
        results[output_column] = (timestamps - epoch).dt.total_seconds() if pd.notna(epoch) else np.nan
    return results


def validation_summary(results: pd.DataFrame) -> dict[str, object]:
    total = len(results)
    if "is_warmup" in results:
        warmup_mask = results["is_warmup"].fillna(False).astype(bool)
        evaluation_results = results.loc[~warmup_mask]
    else:
        warmup_mask = pd.Series(False, index=results.index)
        evaluation_results = results
    if evaluation_results.empty:
        raise ValueError("Simulation contains no evaluation jobs")
    status_counts = results["terminal_status"].value_counts().to_dict()
    completed = int(status_counts.get("completed", 0))
    submitted = int(results["sim_submit_log_utc"].notna().sum())
    started = int(results["sim_start_log_utc"].notna().sum())

    runtime_error = (
        pd.to_numeric(results["sim_observed_runtime_s"], errors="coerce")
        - pd.to_numeric(results["runtime_s"], errors="coerce")
    ).abs()
    completed_mask = results["terminal_status"].eq("completed")
    completed_error = runtime_error.loc[completed_mask].dropna()
    queue_wait = pd.to_numeric(results["sim_scheduler_wait_s"], errors="coerce")
    total_wait = pd.to_numeric(results["sim_total_user_wait_s"], errors="coerce")
    observed_runtime = pd.to_numeric(results["sim_observed_runtime_s"], errors="coerce")
    timeout_events = int(results.get("timeout_event", pd.Series(False, index=results.index)).sum())
    completion_pass = total > 0 and completed == total and submitted == total and started == total and timeout_events == 0
    missing_timing = int((~np.isfinite(pd.concat([queue_wait, total_wait, observed_runtime], axis=1))).any(axis=1).sum())
    ordering_pass = (
        completion_pass and missing_timing == 0 and bool(queue_wait.ge(0).all())
        and bool(total_wait.ge(0).all()) and bool(observed_runtime.ge(0).all())
    )
    runtime_pass = completion_pass and len(completed_error) == total and bool(completed_error.le(60).all())

    summary: dict[str, object] = {
        "expected_jobs": total,
        "evaluation_jobs": len(evaluation_results),
        "warmup_jobs": int(warmup_mask.sum()),
        "submitted_jobs": submitted,
        "started_jobs": started,
        "completed_jobs": completed,
        "completion_ratio": completed / total if total else None,
        "completion_pass": completion_pass,
        "strict_pass": completion_pass,
        "strict_pass_scope": "legacy alias for completion_pass; not timing or model validity",
        "timeout_events": timeout_events,
        "negative_scheduler_wait_jobs": int(queue_wait.lt(0).sum()),
        "negative_total_wait_jobs": int(total_wait.lt(0).sum()),
        "minimum_scheduler_wait_s": float(queue_wait.min()) if np.isfinite(queue_wait.min()) else None,
        "minimum_total_wait_s": float(total_wait.min()) if np.isfinite(total_wait.min()) else None,
        "missing_or_nonfinite_timing_jobs": missing_timing,
        "event_order_pass": ordering_pass,
        "runtime_tolerance_60s_pass": runtime_pass,
        "timing_pass": ordering_pass and runtime_pass,
        "raw_wait_values_preserved": True,
        "terminal_status_counts": status_counts,
        "early_completion_events": int(results["early_completion_event"].sum()),
        "wait_metric_scope": "evaluation_jobs_only",
        "source_eligible_wait_p50_s": percentile(evaluation_results["source_eligible_wait_s"], 0.50),
        "source_eligible_wait_p95_s": percentile(evaluation_results["source_eligible_wait_s"], 0.95),
        "sim_scheduler_wait_p50_s": percentile(evaluation_results["sim_scheduler_wait_s"], 0.50),
        "sim_scheduler_wait_p95_s": percentile(evaluation_results["sim_scheduler_wait_s"], 0.95),
        "source_total_wait_p50_s": percentile(evaluation_results["source_submit_wait_s"], 0.50),
        "source_total_wait_p95_s": percentile(evaluation_results["source_submit_wait_s"], 0.95),
        "sim_total_wait_p50_s": percentile(evaluation_results["sim_total_user_wait_s"], 0.50),
        "sim_total_wait_p95_s": percentile(evaluation_results["sim_total_user_wait_s"], 0.95),
        "configured_runtime_p50_s": percentile(results["runtime_s"], 0.50),
        "configured_runtime_p95_s": percentile(results["runtime_s"], 0.95),
        "observed_runtime_p50_s": percentile(results.loc[completed_mask, "sim_observed_runtime_s"], 0.50),
        "observed_runtime_p95_s": percentile(results.loc[completed_mask, "sim_observed_runtime_s"], 0.95),
        "completed_runtime_within_60s_ratio": (
            float((completed_error <= 60).mean()) if not completed_error.empty else None
        ),
    }

    start = pd.to_numeric(results["sim_start_offset_s"], errors="coerce").min()
    end = pd.to_numeric(results.loc[completed_mask, "sim_end_offset_s"], errors="coerce").max()
    span_hours = (end - start) / 3600 if np.isfinite(start) and np.isfinite(end) and end > start else None
    summary["simulated_completion_span_h"] = float(span_hours) if span_hours else None
    summary["simulated_throughput_jobs_per_h"] = completed / span_hours if span_hours else None
    return summary


def write_outputs(scenario: Path, results: pd.DataFrame, summary: dict[str, object]) -> None:
    results.to_csv(scenario / "job_results.csv", index=False)
    (scenario / "simulation_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    flat_rows = [
        {"metric": key, "value": value}
        for key, value in summary.items()
        if not isinstance(value, dict)
    ]
    pd.DataFrame(flat_rows).to_csv(scenario / "simulation_validation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--require-valid-timing", action="store_true",
                        help="Exit 3 on timing anomalies; raw values are always retained")
    args = parser.parse_args()

    profile_path = args.scenario / "workload_profile.csv"
    log_path = args.scenario / "slurmctld.log"
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    profile = pd.read_csv(profile_path)
    records = parse_log(log_path)
    results = build_results(profile, records)
    summary = validation_summary(results)
    write_outputs(args.scenario, results, summary)
    print(json.dumps(summary, indent=2))
    if not summary["completion_pass"]:
        raise SystemExit(2)
    if not summary["timing_pass"]:
        print("Job completion passed, but timing validation failed; see simulation_validation.json.", file=sys.stderr)
        if args.require_valid_timing:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
