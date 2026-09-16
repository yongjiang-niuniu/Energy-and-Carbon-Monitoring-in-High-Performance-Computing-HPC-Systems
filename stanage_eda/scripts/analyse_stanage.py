#!/usr/bin/env python3
"""Reproducible, privacy-conscious EDA for the Stanage 2025 Slurm queue archive."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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

FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
MAX_REASONABLE_GPU_COUNT = 10_000

METRIC_FIELDS = [
    "End",
    "Start",
    "Submit",
    "Eligible",
    "User",
    "UID",
    "Account",
    "AllocCPUS",
    "AllocNodes",
    "AllocTRES",
    "AllocTRES__2",
    "AveCPU",
    "AveRSS",
    "ConsumedEnergy",
    "ConsumedEnergyRaw",
    "CPUTimeRAW",
    "ElapsedRaw",
    "ExitCode",
    "JobID",
    "JobIDRaw",
    "JobName",
    "MaxRSS",
    "NCPUS",
    "NNodes",
    "NTasks",
    "Partition",
    "Priority",
    "QOS",
    "Reason",
    "ReqCPUS",
    "ReqMem",
    "ReqNodes",
    "ReqTRES",
    "ReqTRES__2",
    "State",
    "SystemCPU",
    "TimelimitRaw",
    "TotalCPU",
    "TRESUsageInAve",
    "TRESUsageInMax",
    "TRESUsageInTot",
    "UserCPU",
    "WorkDir",
]


def unique_headers(raw_headers: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    result: list[str] = []
    for index, raw_name in enumerate(raw_headers):
        name = raw_name.strip() or f"_blank_{index + 1}"
        counts[name] += 1
        result.append(name if counts[name] == 1 else f"{name}__{counts[name]}")
    return result


def safe_float_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return numeric[np.isfinite(numeric)]


def parse_datetime(values: pd.Series) -> pd.Series:
    cleaned = values.replace({"": None, "None": None, "Unknown": None, "N/A": None})
    return pd.to_datetime(cleaned, errors="coerce", utc=True)


def base_state(values: pd.Series) -> pd.Series:
    extracted = values.fillna("").str.extract(r"^([A-Z_]+)", expand=False)
    return extracted.fillna("UNKNOWN")


def parse_tres_value(value: str, kind: str) -> float:
    if not value:
        return 0.0
    total = 0.0
    for component in value.split(","):
        if "=" not in component:
            continue
        key, raw_number = component.rsplit("=", 1)
        key = key.strip().lower()
        raw_number = raw_number.strip()
        if kind == "gpu":
            # Count GPU device TRES such as gres/gpu or gres/gpu:h100, but do
            # not count telemetry fields such as gres/gpumem or gres/gpuutil.
            if not (key == "gpu" or re.fullmatch(r"gres/gpu(?::[^,=]+)?", key)):
                continue
            try:
                number = float(raw_number)
            except ValueError:
                continue
            # Larger values cannot represent a device count for this cluster.
            if number > MAX_REASONABLE_GPU_COUNT:
                return math.nan
            total += number
        elif kind == "mem":
            if key != "mem":
                continue
            match = re.fullmatch(r"([0-9.]+)\s*([kmgtp]?)", raw_number.lower())
            if not match:
                continue
            number = float(match.group(1))
            unit = match.group(2)
            factor = {"": 1.0 / (1024 * 1024), "k": 1.0 / 1024, "m": 1.0, "g": 1024.0, "t": 1024.0**2, "p": 1024.0**3}[unit]
            total += number * factor
    return total


def series_tres(values: pd.Series, kind: str) -> np.ndarray:
    return values.fillna("").map(lambda value: parse_tres_value(str(value), kind)).to_numpy(dtype=float)


def add_arrays(target: dict[str, list[np.ndarray]], name: str, values: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size:
        target[name].append(finite.astype(float, copy=False))


def concat_arrays(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def percentile(values: np.ndarray, q: float) -> float | None:
    if not values.size:
        return None
    return float(np.nanpercentile(values, q))


def finite_sum(values: np.ndarray) -> float:
    return float(np.nansum(values[np.isfinite(values)]))


def json_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@dataclass
class GroupAccumulator:
    jobs: int = 0
    completed: int = 0
    failed: int = 0
    gpu_jobs: int = 0
    core_hours: float = 0.0
    node_hours: float = 0.0
    gpu_hours: float = 0.0
    runtime_parts: list[np.ndarray] = field(default_factory=list)
    wait_parts: list[np.ndarray] = field(default_factory=list)
    req_cpu_parts: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        states: pd.Series,
        runtime_sec: np.ndarray,
        wait_sec: np.ndarray,
        req_cpus: np.ndarray,
        core_hours: np.ndarray,
        node_hours: np.ndarray,
        req_gpus: np.ndarray,
        gpu_hours: np.ndarray,
    ) -> None:
        self.jobs += len(states)
        self.completed += int((states == "COMPLETED").sum())
        self.failed += int(states.isin(FAILURE_STATES).sum())
        self.gpu_jobs += int(np.sum(req_gpus > 0))
        self.core_hours += finite_sum(core_hours)
        self.node_hours += finite_sum(node_hours)
        self.gpu_hours += finite_sum(gpu_hours)
        finite_runtime = runtime_sec[np.isfinite(runtime_sec)]
        finite_wait = wait_sec[np.isfinite(wait_sec)]
        finite_req_cpus = req_cpus[np.isfinite(req_cpus)]
        if finite_runtime.size:
            self.runtime_parts.append(finite_runtime)
        if finite_wait.size:
            self.wait_parts.append(finite_wait)
        if finite_req_cpus.size:
            self.req_cpu_parts.append(finite_req_cpus)

    def as_row(self, label: str, label_name: str) -> dict[str, object]:
        runtime = concat_arrays(self.runtime_parts)
        wait = concat_arrays(self.wait_parts)
        req_cpu = concat_arrays(self.req_cpu_parts)
        return {
            label_name: label,
            "jobs": self.jobs,
            "share_pct": None,
            "completed_jobs": self.completed,
            "completion_rate_pct": 100.0 * self.completed / self.jobs if self.jobs else None,
            "failure_jobs": self.failed,
            "failure_rate_pct": 100.0 * self.failed / self.jobs if self.jobs else None,
            "gpu_jobs": self.gpu_jobs,
            "median_runtime_min": percentile(runtime / 60.0, 50),
            "p95_runtime_min": percentile(runtime / 60.0, 95),
            "median_submit_wait_min": percentile(wait / 60.0, 50),
            "p95_submit_wait_min": percentile(wait / 60.0, 95),
            "median_requested_cpus": percentile(req_cpu, 50),
            "core_hours": self.core_hours,
            "node_hours": self.node_hours,
            "gpu_hours": self.gpu_hours,
        }


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_number(value) for key, value in row.items()})


def gini(values: np.ndarray) -> float | None:
    if not values.size or np.sum(values) == 0:
        return None
    ordered = np.sort(values.astype(float))
    index = np.arange(1, len(ordered) + 1)
    return float((2 * np.sum(index * ordered) / (len(ordered) * np.sum(ordered))) - (len(ordered) + 1) / len(ordered))


def concentration_share(values: np.ndarray, top_fraction: float) -> float | None:
    if not values.size or np.sum(values) == 0:
        return None
    count = max(1, int(math.ceil(len(values) * top_fraction)))
    return float(np.sort(values)[-count:].sum() / values.sum())


def distribution_rows(metric_arrays: dict[str, list[np.ndarray]]) -> list[dict[str, object]]:
    units = {
        "submit_wait_sec": "minutes",
        "eligible_wait_sec": "minutes",
        "runtime_sec": "minutes",
        "turnaround_sec": "minutes",
        "slowdown": "ratio",
        "requested_cpus": "count",
        "allocated_cpus": "count",
        "allocated_nodes": "count",
        "requested_memory_mib": "MiB",
        "requested_gpus": "count",
        "timelimit_utilisation": "ratio",
        "core_hours_per_job": "core-hours",
        "interarrival_sec": "seconds",
    }
    rows: list[dict[str, object]] = []
    for metric, parts in metric_arrays.items():
        values = concat_arrays(parts)
        if metric in {"submit_wait_sec", "eligible_wait_sec", "runtime_sec", "turnaround_sec"}:
            values = values / 60.0
        rows.append(
            {
                "metric": metric,
                "unit": units[metric],
                "valid_count": int(values.size),
                "mean": float(np.mean(values)) if values.size else None,
                "p25": percentile(values, 25),
                "median": percentile(values, 50),
                "p75": percentile(values, 75),
                "p90": percentile(values, 90),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
                "maximum": float(np.max(values)) if values.size else None,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--months", nargs="+", choices=MONTHS, default=MONTHS)
    parser.add_argument("--chunksize", type=int, default=75_000)
    args = parser.parse_args()

    start_time = time.time()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    months = [month for month in MONTHS if month in set(args.months)]

    seen_ids: set[int] = set()
    raw_rows = 0
    deduplicated_rows = 0
    cohort_rows = 0
    duplicate_rows = 0
    invalid_job_ids = 0
    excluded_non_2025 = 0

    source_inventory: list[dict[str, object]] = []
    state_counts: Counter[str] = Counter()
    qos_counts: Counter[str] = Counter()
    submit_hour_counts: Counter[int] = Counter()
    submit_weekday_counts: Counter[str] = Counter()
    user_job_counts: Counter[str] = Counter()
    month_groups: dict[str, GroupAccumulator] = defaultdict(GroupAccumulator)
    partition_groups: dict[str, GroupAccumulator] = defaultdict(GroupAccumulator)
    metric_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    field_nonempty: Counter[str] = Counter()
    field_nonzero: Counter[str] = Counter()
    quality: Counter[str] = Counter()
    sample_parts: list[pd.DataFrame] = []
    submit_epoch_ns_parts: list[np.ndarray] = []

    with zipfile.ZipFile(args.zip_path) as archive:
        archive_names = set(archive.namelist())
        expected_entries = {month: f"stanage_2025_queue_record/{month}.txt" for month in months}
        missing_entries = [entry for entry in expected_entries.values() if entry not in archive_names]
        if missing_entries:
            raise FileNotFoundError(f"Archive entries missing: {missing_entries}")

        # Read newest source files first, so the last observed state of a repeated job is retained.
        for month in reversed(months):
            entry = expected_entries[month]
            info = archive.getinfo(entry)
            with archive.open(entry) as handle:
                raw_header = handle.readline().decode("utf-8").rstrip("\r\n").split("|")
            headers = unique_headers(raw_header)
            missing_columns = [name for name in METRIC_FIELDS if name not in headers]
            if missing_columns:
                raise ValueError(f"{entry} is missing required columns: {missing_columns}")

            month_rows = 0
            month_unique_ids: set[int] = set()
            month_new_ids = 0
            month_duplicate_ids = 0
            month_submitted_2025 = 0
            month_timestamp_min: dict[str, pd.Timestamp | None] = {name: None for name in ["Submit", "Eligible", "Start", "End"]}
            month_timestamp_max: dict[str, pd.Timestamp | None] = {name: None for name in ["Submit", "Eligible", "Start", "End"]}

            with archive.open(entry) as handle:
                reader = pd.read_csv(
                    handle,
                    sep="|",
                    header=0,
                    names=headers,
                    usecols=METRIC_FIELDS,
                    dtype=str,
                    chunksize=args.chunksize,
                    keep_default_na=False,
                    na_filter=False,
                    engine="c",
                    on_bad_lines="warn",
                )

                for frame in reader:
                    frame_rows = len(frame)
                    raw_rows += frame_rows
                    month_rows += frame_rows

                    for field_name in METRIC_FIELDS:
                        values = frame[field_name].astype(str)
                        nonempty_mask = ~values.isin(["", "None", "Unknown", "N/A"])
                        field_nonempty[field_name] += int(nonempty_mask.sum())
                        if field_name in {"ConsumedEnergy", "ConsumedEnergyRaw"}:
                            numeric = pd.to_numeric(values, errors="coerce").fillna(0)
                            field_nonzero[field_name] += int((numeric != 0).sum())
                        elif field_name in {"TotalCPU", "UserCPU", "SystemCPU"}:
                            zero_like = ["", "None", "Unknown", "N/A", "0", "00:00:00", "0:0"]
                            field_nonzero[field_name] += int((~values.isin(zero_like)).sum())

                    quality["alloc_tres_duplicate_mismatch"] += int((frame["AllocTRES"] != frame["AllocTRES__2"]).sum())
                    quality["req_tres_duplicate_mismatch"] += int((frame["ReqTRES"] != frame["ReqTRES__2"]).sum())

                    job_ids = pd.to_numeric(frame["JobIDRaw"], errors="coerce")
                    valid_id = job_ids.notna()
                    invalid_job_ids += int((~valid_id).sum())
                    frame = frame.loc[valid_id].copy()
                    job_id_array = job_ids.loc[valid_id].astype(np.int64).to_numpy()

                    new_mask = np.zeros(len(frame), dtype=bool)
                    for index, job_id in enumerate(job_id_array):
                        numeric_id = int(job_id)
                        month_unique_ids.add(numeric_id)
                        if numeric_id in seen_ids:
                            month_duplicate_ids += 1
                            duplicate_rows += 1
                        else:
                            seen_ids.add(numeric_id)
                            new_mask[index] = True
                            month_new_ids += 1

                    frame = frame.iloc[np.flatnonzero(new_mask)].copy()
                    job_id_array = job_id_array[new_mask]
                    deduplicated_rows += len(frame)
                    if frame.empty:
                        continue

                    times = {name: parse_datetime(frame[name]) for name in ["Submit", "Eligible", "Start", "End"]}
                    for time_name, time_values in times.items():
                        valid_times = time_values.dropna()
                        if not valid_times.empty:
                            current_min = valid_times.min()
                            current_max = valid_times.max()
                            month_timestamp_min[time_name] = current_min if month_timestamp_min[time_name] is None else min(month_timestamp_min[time_name], current_min)
                            month_timestamp_max[time_name] = current_max if month_timestamp_max[time_name] is None else max(month_timestamp_max[time_name], current_max)

                    submit = times["Submit"]
                    cohort_mask = submit.dt.year.eq(2025).fillna(False).to_numpy()
                    month_submitted_2025 += int(cohort_mask.sum())
                    excluded_non_2025 += int((~cohort_mask).sum())
                    frame = frame.iloc[np.flatnonzero(cohort_mask)].copy()
                    job_id_array = job_id_array[cohort_mask]
                    if frame.empty:
                        continue
                    cohort_rows += len(frame)
                    times = {name: values.iloc[np.flatnonzero(cohort_mask)].reset_index(drop=True) for name, values in times.items()}
                    frame.reset_index(drop=True, inplace=True)

                    state = base_state(frame["State"])
                    partition = frame["Partition"].replace({"": "UNKNOWN", "None": "UNKNOWN"})
                    qos = frame["QOS"].replace({"": "UNKNOWN", "None": "UNKNOWN"})
                    state_counts.update(state.tolist())
                    qos_counts.update(qos.tolist())
                    user_job_counts.update(frame["User"].replace({"": "UNKNOWN"}).tolist())

                    elapsed = pd.to_numeric(frame["ElapsedRaw"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    alloc_cpus = pd.to_numeric(frame["AllocCPUS"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    req_cpus = pd.to_numeric(frame["ReqCPUS"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    alloc_nodes = pd.to_numeric(frame["AllocNodes"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    req_nodes = pd.to_numeric(frame["ReqNodes"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    timelimit_min = pd.to_numeric(frame["TimelimitRaw"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    req_gpus = series_tres(frame["ReqTRES"], "gpu")
                    alloc_gpus = series_tres(frame["AllocTRES"], "gpu")
                    req_memory_mib = series_tres(frame["ReqTRES"], "mem")
                    quality["invalid_requested_gpu_tres"] += int(np.isnan(req_gpus).sum())
                    quality["invalid_allocated_gpu_tres"] += int(np.isnan(alloc_gpus).sum())

                    submit_wait = (times["Start"] - times["Submit"]).dt.total_seconds().to_numpy(dtype=float, na_value=np.nan)
                    eligible_wait = (times["Start"] - times["Eligible"]).dt.total_seconds().to_numpy(dtype=float, na_value=np.nan)
                    observed_runtime = (times["End"] - times["Start"]).dt.total_seconds().to_numpy(dtype=float, na_value=np.nan)
                    turnaround = (times["End"] - times["Submit"]).dt.total_seconds().to_numpy(dtype=float, na_value=np.nan)

                    quality["missing_submit"] += int(times["Submit"].isna().sum())
                    quality["missing_eligible"] += int(times["Eligible"].isna().sum())
                    quality["missing_start"] += int(times["Start"].isna().sum())
                    quality["missing_end"] += int(times["End"].isna().sum())
                    quality["negative_submit_wait"] += int(np.sum(np.isfinite(submit_wait) & (submit_wait < 0)))
                    quality["negative_eligible_wait"] += int(np.sum(np.isfinite(eligible_wait) & (eligible_wait < 0)))
                    quality["end_before_start"] += int(np.sum(np.isfinite(observed_runtime) & (observed_runtime < 0)))
                    quality["elapsed_mismatch_gt_5s"] += int(
                        np.sum(np.isfinite(observed_runtime) & np.isfinite(elapsed) & (np.abs(observed_runtime - elapsed) > 5))
                    )
                    quality["requested_allocated_cpu_mismatch"] += int(
                        np.sum(np.isfinite(req_cpus) & np.isfinite(alloc_cpus) & (req_cpus != alloc_cpus))
                    )
                    quality["requested_allocated_node_mismatch"] += int(
                        np.sum(np.isfinite(req_nodes) & np.isfinite(alloc_nodes) & (req_nodes != alloc_nodes))
                    )
                    quality["array_or_variant_job_ids"] += int((frame["JobID"] != frame["JobIDRaw"]).sum())

                    valid_submit_wait = np.where(submit_wait >= 0, submit_wait, np.nan)
                    valid_eligible_wait = np.where(eligible_wait >= 0, eligible_wait, np.nan)
                    started_mask = times["Start"].notna().to_numpy()
                    valid_runtime = np.where(started_mask & (elapsed >= 0), elapsed, np.nan)
                    valid_turnaround = np.where(turnaround >= 0, turnaround, np.nan)
                    slowdown = np.divide(
                        valid_turnaround,
                        valid_runtime,
                        out=np.full_like(valid_turnaround, np.nan),
                        where=np.isfinite(valid_runtime) & (valid_runtime > 0),
                    )
                    core_hours = alloc_cpus * valid_runtime / 3600.0
                    node_hours = alloc_nodes * valid_runtime / 3600.0
                    gpu_hours = alloc_gpus * valid_runtime / 3600.0
                    timelimit_utilisation = np.divide(
                        valid_runtime,
                        timelimit_min * 60.0,
                        out=np.full_like(valid_runtime, np.nan),
                        where=np.isfinite(timelimit_min) & (timelimit_min > 0),
                    )

                    add_arrays(metric_arrays, "submit_wait_sec", valid_submit_wait)
                    add_arrays(metric_arrays, "eligible_wait_sec", valid_eligible_wait)
                    add_arrays(metric_arrays, "runtime_sec", valid_runtime)
                    add_arrays(metric_arrays, "turnaround_sec", valid_turnaround)
                    add_arrays(metric_arrays, "slowdown", slowdown)
                    add_arrays(metric_arrays, "requested_cpus", req_cpus)
                    add_arrays(metric_arrays, "allocated_cpus", alloc_cpus)
                    add_arrays(metric_arrays, "allocated_nodes", alloc_nodes)
                    add_arrays(metric_arrays, "requested_memory_mib", req_memory_mib)
                    add_arrays(metric_arrays, "requested_gpus", req_gpus)
                    add_arrays(metric_arrays, "timelimit_utilisation", timelimit_utilisation)
                    add_arrays(metric_arrays, "core_hours_per_job", core_hours)

                    submit_month = times["Submit"].dt.strftime("%Y-%m")
                    submit_epoch_ns_parts.append(times["Submit"].astype("int64").to_numpy())
                    submit_hour_counts.update(times["Submit"].dt.hour.dropna().astype(int).tolist())
                    submit_weekday_counts.update(times["Submit"].dt.day_name().dropna().tolist())

                    for label, indices in submit_month.groupby(submit_month).groups.items():
                        idx = np.asarray(list(indices), dtype=int)
                        month_groups[label].update(
                            state.iloc[idx], valid_runtime[idx], valid_submit_wait[idx], req_cpus[idx], core_hours[idx], node_hours[idx], req_gpus[idx], gpu_hours[idx]
                        )

                    for label, indices in partition.groupby(partition).groups.items():
                        idx = np.asarray(list(indices), dtype=int)
                        partition_groups[str(label)].update(
                            state.iloc[idx], valid_runtime[idx], valid_submit_wait[idx], req_cpus[idx], core_hours[idx], node_hours[idx], req_gpus[idx], gpu_hours[idx]
                        )

                    sample_mask = (job_id_array % 97) == 0
                    if np.any(sample_mask):
                        sample_index = np.flatnonzero(sample_mask)
                        sample_parts.append(
                            pd.DataFrame(
                                {
                                    "submit_utc": times["Submit"].iloc[sample_index].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
                                    "eligible_utc": times["Eligible"].iloc[sample_index].dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("").to_numpy(),
                                    "start_utc": times["Start"].iloc[sample_index].dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("").to_numpy(),
                                    "end_utc": times["End"].iloc[sample_index].dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("").to_numpy(),
                                    "partition": partition.iloc[sample_index].to_numpy(),
                                    "qos": qos.iloc[sample_index].to_numpy(),
                                    "state": state.iloc[sample_index].to_numpy(),
                                    "runtime_s": valid_runtime[sample_index],
                                    "submit_wait_s": valid_submit_wait[sample_index],
                                    "requested_cpus": req_cpus[sample_index],
                                    "allocated_cpus": alloc_cpus[sample_index],
                                    "requested_nodes": req_nodes[sample_index],
                                    "allocated_nodes": alloc_nodes[sample_index],
                                    "requested_memory_mib": req_memory_mib[sample_index],
                                    "requested_gpus": req_gpus[sample_index],
                                    "allocated_gpus": alloc_gpus[sample_index],
                                    "timelimit_min": timelimit_min[sample_index],
                                }
                            )
                        )

            source_inventory.append(
                {
                    "source_month": month,
                    "archive_entry": entry,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "raw_rows": month_rows,
                    "unique_job_ids_in_file": len(month_unique_ids),
                    "rows_retained_after_global_dedup": month_new_ids,
                    "rows_also_present_in_later_files": month_duplicate_ids,
                    "retained_jobs_submitted_in_2025": month_submitted_2025,
                    **{
                        f"min_{name.lower()}_utc": value.isoformat() if value is not None else None
                        for name, value in month_timestamp_min.items()
                    },
                    **{
                        f"max_{name.lower()}_utc": value.isoformat() if value is not None else None
                        for name, value in month_timestamp_max.items()
                    },
                }
            )
            print(f"Processed {month}: {month_rows:,} rows, {month_new_ids:,} retained", flush=True)

    total_group_jobs = sum(group.jobs for group in month_groups.values())
    monthly_rows = [month_groups[key].as_row(key, "submit_month") for key in sorted(month_groups)]
    for row in monthly_rows:
        row["share_pct"] = 100.0 * int(row["jobs"]) / total_group_jobs if total_group_jobs else None

    partition_rows = [partition_groups[key].as_row(key, "partition") for key in partition_groups]
    for row in partition_rows:
        row["share_pct"] = 100.0 * int(row["jobs"]) / cohort_rows if cohort_rows else None
    partition_rows.sort(key=lambda row: int(row["jobs"]), reverse=True)

    state_rows = [
        {
            "state": state,
            "jobs": count,
            "share_pct": 100.0 * count / cohort_rows if cohort_rows else None,
            "category": "completed" if state == "COMPLETED" else ("failure" if state in FAILURE_STATES else "other"),
        }
        for state, count in state_counts.most_common()
    ]
    qos_rows = [
        {"qos": qos, "jobs": count, "share_pct": 100.0 * count / cohort_rows if cohort_rows else None}
        for qos, count in qos_counts.most_common()
    ]

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hour_rows = [
        {"utc_hour": hour, "submitted_jobs": submit_hour_counts.get(hour, 0), "share_pct": 100.0 * submit_hour_counts.get(hour, 0) / cohort_rows if cohort_rows else None}
        for hour in range(24)
    ]
    weekday_rows = [
        {"weekday": day, "submitted_jobs": submit_weekday_counts.get(day, 0), "share_pct": 100.0 * submit_weekday_counts.get(day, 0) / cohort_rows if cohort_rows else None}
        for day in weekday_order
    ]

    if submit_epoch_ns_parts:
        submit_epoch_ns = np.sort(np.concatenate(submit_epoch_ns_parts))
        interarrival_sec = np.diff(submit_epoch_ns).astype(float) / 1_000_000_000.0
        interarrival_sec = interarrival_sec[interarrival_sec >= 0]
        add_arrays(metric_arrays, "interarrival_sec", interarrival_sec)
        quality["simultaneous_submit_intervals"] = int(np.sum(interarrival_sec == 0))

    distribution = distribution_rows(metric_arrays)
    user_values = np.array(list(user_job_counts.values()), dtype=float)
    user_summary_rows = [
        {"metric": "unique_users", "value": len(user_values), "unit": "users"},
        {"metric": "median_jobs_per_user", "value": percentile(user_values, 50), "unit": "jobs"},
        {"metric": "p90_jobs_per_user", "value": percentile(user_values, 90), "unit": "jobs"},
        {"metric": "p99_jobs_per_user", "value": percentile(user_values, 99), "unit": "jobs"},
        {"metric": "user_job_count_gini", "value": gini(user_values), "unit": "ratio"},
        {"metric": "top_1pct_user_job_share", "value": concentration_share(user_values, 0.01), "unit": "ratio"},
        {"metric": "top_5pct_user_job_share", "value": concentration_share(user_values, 0.05), "unit": "ratio"},
        {"metric": "top_10pct_user_job_share", "value": concentration_share(user_values, 0.10), "unit": "ratio"},
    ]

    user_bucket_specs = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-50", 11, 50),
        ("51-100", 51, 100),
        ("101-500", 101, 500),
        ("501-1000", 501, 1000),
        ("1001+", 1001, math.inf),
    ]
    user_bucket_rows = []
    for label, low, high in user_bucket_specs:
        mask = (user_values >= low) & (user_values <= high)
        user_bucket_rows.append(
            {
                "jobs_per_user_bucket": label,
                "users": int(mask.sum()),
                "jobs": int(user_values[mask].sum()),
                "job_share_pct": 100.0 * user_values[mask].sum() / user_values.sum() if user_values.sum() else None,
            }
        )

    quality_rows = [
        {"check": "raw_rows", "count": raw_rows, "denominator": raw_rows, "rate_pct": 100.0},
        {"check": "globally_unique_job_ids", "count": deduplicated_rows, "denominator": raw_rows, "rate_pct": 100.0 * deduplicated_rows / raw_rows if raw_rows else None},
        {"check": "duplicate_rows_across_or_within_files", "count": duplicate_rows, "denominator": raw_rows, "rate_pct": 100.0 * duplicate_rows / raw_rows if raw_rows else None},
        {"check": "invalid_job_ids", "count": invalid_job_ids, "denominator": raw_rows, "rate_pct": 100.0 * invalid_job_ids / raw_rows if raw_rows else None},
        {"check": "retained_jobs_submitted_in_2025", "count": cohort_rows, "denominator": deduplicated_rows, "rate_pct": 100.0 * cohort_rows / deduplicated_rows if deduplicated_rows else None},
        {"check": "retained_jobs_outside_2025_or_missing_submit", "count": excluded_non_2025, "denominator": deduplicated_rows, "rate_pct": 100.0 * excluded_non_2025 / deduplicated_rows if deduplicated_rows else None},
    ]
    for name, count in sorted(quality.items()):
        if name.endswith("duplicate_mismatch"):
            denominator = raw_rows
        elif name == "simultaneous_submit_intervals":
            denominator = max(cohort_rows - 1, 0)
        else:
            denominator = cohort_rows
        quality_rows.append(
            {
                "check": name,
                "count": count,
                "denominator": denominator,
                "rate_pct": 100.0 * count / denominator if denominator else None,
            }
        )

    field_rows = []
    for field_name in METRIC_FIELDS:
        row = {
            "field": field_name,
            "nonempty_rows": field_nonempty[field_name],
            "nonempty_rate_pct": 100.0 * field_nonempty[field_name] / raw_rows if raw_rows else None,
            "nonzero_rows": field_nonzero.get(field_name, None),
            "privacy_class": "direct_identifier" if field_name in {"User", "UID", "JobName", "WorkDir"} else "analysis_field",
        }
        field_rows.append(row)

    power_rows = [
        {"scenario": "regular_use_cluster_power", "power_kw": 195.0, "energy_per_hour_kwh": 195.0, "energy_per_day_mwh": 4.68, "status": "Fred approximate value"},
        {"scenario": "idle_cluster_power", "power_kw": 140.0, "energy_per_hour_kwh": 140.0, "energy_per_day_mwh": 3.36, "status": "Fred approximate value"},
        {"scenario": "increment_above_idle", "power_kw": 55.0, "energy_per_hour_kwh": 55.0, "energy_per_day_mwh": 1.32, "status": "derived difference, not measured job power"},
    ]

    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    if not sample.empty:
        sample.sort_values(["submit_utc", "partition"], inplace=True, na_position="last")
        sample.reset_index(drop=True, inplace=True)
        sample.insert(0, "sim_job_id", [f"sim_{index:06d}" for index in range(1, len(sample) + 1)])
        sample.to_csv(output_dir / "simulator_workload_sample.csv", index=False)

    source_inventory.sort(key=lambda row: MONTHS.index(str(row["source_month"])))
    write_csv(output_dir / "source_month_inventory.csv", source_inventory)
    write_csv(output_dir / "monthly_submit_summary.csv", monthly_rows)
    write_csv(output_dir / "partition_summary.csv", partition_rows)
    write_csv(output_dir / "state_summary.csv", state_rows)
    write_csv(output_dir / "qos_summary.csv", qos_rows)
    write_csv(output_dir / "submission_hour_utc.csv", hour_rows)
    write_csv(output_dir / "submission_weekday_utc.csv", weekday_rows)
    write_csv(output_dir / "distribution_summary.csv", distribution)
    write_csv(output_dir / "data_quality_summary.csv", quality_rows)
    write_csv(output_dir / "field_availability.csv", field_rows)
    write_csv(output_dir / "user_workload_summary.csv", user_summary_rows)
    write_csv(output_dir / "user_workload_buckets.csv", user_bucket_rows)
    write_csv(output_dir / "energy_power_assumptions.csv", power_rows)

    distributions_by_name = {row["metric"]: row for row in distribution}
    summary = {
        "analysis_scope": {
            "archive": str(args.zip_path),
            "months": months,
            "timezone_interpretation": "UTC, following Fred's response",
            "deduplication": "Processed source months newest-to-oldest and kept the latest occurrence of each numeric JobIDRaw",
            "cohort": "Globally deduplicated jobs with Submit timestamp in calendar year 2025",
            "privacy": "Direct identifiers were read only for aggregate counts and were excluded from all row-level outputs",
        },
        "counts": {
            "raw_rows": raw_rows,
            "globally_unique_job_ids": deduplicated_rows,
            "duplicate_rows": duplicate_rows,
            "cohort_jobs_submitted_in_2025": cohort_rows,
            "unique_users": len(user_values),
            "simulator_sample_rows": len(sample),
        },
        "outcomes": {
            "completed_jobs": state_counts.get("COMPLETED", 0),
            "completion_rate_pct": 100.0 * state_counts.get("COMPLETED", 0) / cohort_rows if cohort_rows else None,
            "failure_jobs": sum(state_counts.get(state, 0) for state in FAILURE_STATES),
            "failure_rate_pct": 100.0 * sum(state_counts.get(state, 0) for state in FAILURE_STATES) / cohort_rows if cohort_rows else None,
        },
        "selected_distributions": {
            "median_submit_wait_min": distributions_by_name.get("submit_wait_sec", {}).get("median"),
            "p95_submit_wait_min": distributions_by_name.get("submit_wait_sec", {}).get("p95"),
            "median_runtime_min": distributions_by_name.get("runtime_sec", {}).get("median"),
            "p95_runtime_min": distributions_by_name.get("runtime_sec", {}).get("p95"),
            "median_requested_cpus": distributions_by_name.get("requested_cpus", {}).get("median"),
            "p95_requested_cpus": distributions_by_name.get("requested_cpus", {}).get("p95"),
        },
        "energy_evidence": {
            "nonzero_consumed_energy_rows": field_nonzero.get("ConsumedEnergy", 0),
            "nonzero_consumed_energy_raw_rows": field_nonzero.get("ConsumedEnergyRaw", 0),
            "nonzero_total_cpu_rows": field_nonzero.get("TotalCPU", 0),
            "nonzero_user_cpu_rows": field_nonzero.get("UserCPU", 0),
            "nonzero_system_cpu_rows": field_nonzero.get("SystemCPU", 0),
            "regular_use_power_kw": 195.0,
            "idle_power_kw": 140.0,
            "increment_above_idle_kw": 55.0,
            "pue_available": False,
        },
        "runtime_seconds": time.time() - start_time,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
