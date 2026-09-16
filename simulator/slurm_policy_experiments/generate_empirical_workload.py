#!/usr/bin/env python3
"""Create a privacy-safe Slurm simulator workload from a real monthly export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PARTITION_MAP = {
    "sheffield": "sheffield",
    "sheffield-8xlong": "sheffield-8xlong",
    "interactive": "interactive",
    "gpu": "gpu",
    "gpu-h100": "gpu-h100",
    "gpu-h100-nvl": "gpu-h100-nvl",
    "hp-cpu": "sheffield",
    "hp-a100": "gpu",
    "hp-h100": "gpu-h100",
    "hp-h100-nvl": "gpu-h100-nvl",
    "test-gpu-h100": "gpu-h100-nvl",
}

PARTITION_CAPACITY = {
    "interactive": {"nodes": 2, "cpus_per_node": 64, "gpus_per_node": 0, "memory_mib": 257_024},
    "sheffield": {"nodes": 174, "cpus_per_node": 64, "gpus_per_node": 0, "memory_mib": 2_062_336},
    "sheffield-8xlong": {"nodes": 174, "cpus_per_node": 64, "gpus_per_node": 0, "memory_mib": 2_062_336},
    "gpu": {"nodes": 18, "cpus_per_node": 48, "gpus_per_node": 4, "memory_mib": 515_088},
    "gpu-h100": {"nodes": 6, "cpus_per_node": 48, "gpus_per_node": 2, "memory_mib": 515_088},
    "gpu-h100-nvl": {"nodes": 4, "cpus_per_node": 96, "gpus_per_node": 4, "memory_mib": 515_040},
}

PARTITION_MAX_TIME_MIN = {
    "interactive": 8 * 60,
    "sheffield": 4 * 24 * 60,
    "sheffield-8xlong": 8 * 24 * 60,
    "gpu": 4 * 24 * 60,
    "gpu-h100": 4 * 24 * 60,
    "gpu-h100-nvl": 4 * 24 * 60,
}

GPU_TYPE = {
    "gpu": "a100",
    "gpu-h100": "h100",
    "gpu-h100-nvl": "h100nvl",
}

USE_COLUMNS = [
    "End",
    "Start",
    "Submit",
    "Eligible",
    "User",
    "JobIDRaw",
    "NodeList",
    "AllocCPUS",
    "AllocNodes",
    "AllocTRES",
    "ElapsedRaw",
    "Partition",
    "QOS",
    "ReqCPUS",
    "ReqNodes",
    "ReqTRES",
    "State",
    "TimelimitRaw",
]


def unique_headers(raw_headers: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    result: list[str] = []
    for index, raw_name in enumerate(raw_headers):
        name = raw_name.strip() or f"_blank_{index + 1}"
        counts[name] += 1
        result.append(name if counts[name] == 1 else f"{name}__{counts[name]}")
    return result


def parse_tres_value(value: str, kind: str) -> float:
    if not value:
        return 0.0
    total = 0.0
    for component in str(value).split(","):
        if "=" not in component:
            continue
        key, raw_number = component.rsplit("=", 1)
        key = key.strip().lower()
        raw_number = raw_number.strip().lower()
        if kind == "gpu":
            if not (key == "gpu" or re.fullmatch(r"gres/gpu(?::[^,=]+)?", key)):
                continue
            try:
                number = float(raw_number)
            except ValueError:
                continue
            if number > 10_000:
                return math.nan
            total += number
        elif kind == "mem" and key == "mem":
            match = re.fullmatch(r"([0-9.]+)\s*([kmgtp]?)", raw_number)
            if not match:
                continue
            number = float(match.group(1))
            factor = {
                "": 1.0 / (1024 * 1024),
                "k": 1.0 / 1024,
                "m": 1.0,
                "g": 1024.0,
                "t": 1024.0**2,
                "p": 1024.0**3,
            }[match.group(2)]
            total += number * factor
    return total


def base_state(value: object) -> str:
    match = re.match(r"^([A-Z_]+)", str(value or ""))
    return match.group(1) if match else "UNKNOWN"


def read_month(zip_path: Path, month: str) -> pd.DataFrame:
    entry = f"stanage_2025_queue_record/{month}.txt"
    with zipfile.ZipFile(zip_path) as archive:
        if entry not in archive.namelist():
            raise FileNotFoundError(f"{entry} is not present in {zip_path}")
        with archive.open(entry) as handle:
            raw_header = handle.readline().decode("utf-8").rstrip("\r\n").split("|")
        headers = unique_headers(raw_header)
        missing = [name for name in USE_COLUMNS if name not in headers]
        if missing:
            raise ValueError(f"Missing source columns: {missing}")
        with archive.open(entry) as handle:
            frame = pd.read_csv(
                handle,
                sep="|",
                header=0,
                names=headers,
                usecols=USE_COLUMNS,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                engine="c",
                on_bad_lines="warn",
            )
    return frame


def prepare_cohort(frame: pd.DataFrame, month: str, max_runtime_hours: float) -> tuple[pd.DataFrame, dict[str, int]]:
    audit: Counter[str] = Counter()
    audit["raw_month_rows"] = len(frame)

    frame = frame.copy()
    frame["_job_id"] = pd.to_numeric(frame["JobIDRaw"], errors="coerce")
    frame = frame.loc[frame["_job_id"].notna()].copy()
    frame.sort_index(inplace=True)
    frame.drop_duplicates("_job_id", keep="last", inplace=True)
    audit["unique_numeric_job_ids"] = len(frame)

    for name in ["Submit", "Eligible", "Start", "End"]:
        frame[f"_{name.lower()}"] = pd.to_datetime(
            frame[name].replace({"": None, "None": None, "Unknown": None}),
            errors="coerce",
            utc=True,
        )

    month_number = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }[month]
    submit_mask = frame["_submit"].dt.year.eq(2025) & frame["_submit"].dt.month.eq(month_number)
    frame = frame.loc[submit_mask].copy()
    audit["submitted_in_named_month"] = len(frame)

    frame["_state"] = frame["State"].map(base_state)
    frame = frame.loc[frame["_state"].eq("COMPLETED")].copy()
    audit["completed_jobs"] = len(frame)

    frame["_runtime_s"] = pd.to_numeric(frame["ElapsedRaw"], errors="coerce")
    frame["_requested_cpus"] = pd.to_numeric(frame["ReqCPUS"], errors="coerce")
    frame["_allocated_cpus"] = pd.to_numeric(frame["AllocCPUS"], errors="coerce")
    frame["_requested_nodes"] = pd.to_numeric(frame["ReqNodes"], errors="coerce")
    frame["_allocated_nodes"] = pd.to_numeric(frame["AllocNodes"], errors="coerce")
    frame["_timelimit_min"] = pd.to_numeric(frame["TimelimitRaw"], errors="coerce")
    frame["_requested_gpus"] = frame["ReqTRES"].map(lambda value: parse_tres_value(value, "gpu"))
    frame["_allocated_gpus"] = frame["AllocTRES"].map(lambda value: parse_tres_value(value, "gpu"))
    frame["_requested_memory_mib"] = frame["ReqTRES"].map(lambda value: parse_tres_value(value, "mem"))
    frame["_mapped_partition"] = frame["Partition"].map(PARTITION_MAP)

    valid = (
        frame["_submit"].notna()
        & frame["_eligible"].notna()
        & frame["_runtime_s"].gt(0)
        & frame["_requested_cpus"].gt(0)
        & frame["_requested_nodes"].gt(0)
        & frame["_mapped_partition"].notna()
    )
    frame = frame.loc[valid].copy()
    audit["valid_completed_supported_jobs"] = len(frame)

    eligible_before_submit = frame["_eligible"] < frame["_submit"]
    audit["eligible_before_submit_repaired"] = int(eligible_before_submit.sum())
    frame.loc[eligible_before_submit, "_eligible"] = frame.loc[eligible_before_submit, "_submit"]

    frame["_runtime_original_s"] = frame["_runtime_s"]
    runtime_cap = int(max_runtime_hours * 3600)
    frame["_runtime_s"] = frame["_runtime_s"].clip(lower=1, upper=runtime_cap)
    audit["runtime_values_capped"] = int((frame["_runtime_original_s"] > runtime_cap).sum())

    frame.sort_values(["_submit", "_job_id"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame, dict(audit)


def candidate_score(window: pd.DataFrame, target_jobs: int, reference: dict[str, float]) -> float:
    count = len(window)
    if count == 0:
        return math.inf
    gpu_share = float((window["_requested_gpus"] > 0).mean())
    runtime_median = max(float(window["_runtime_s"].median()), 1.0)
    runtime_p95 = max(float(window["_runtime_s"].quantile(0.95)), 1.0)
    cpu_median = max(float(window["_requested_cpus"].median()), 1.0)
    cpu_p95 = max(float(window["_requested_cpus"].quantile(0.95)), 1.0)
    core_hours_per_job = max(
        float((window["_requested_cpus"] * window["_runtime_s"] / 3600.0).mean()),
        1e-6,
    )
    ordered_submit = window["_submit"].sort_values()
    simultaneous_share = float(ordered_submit.diff().dt.total_seconds().eq(0).mean())
    top_user_share = float(window["User"].value_counts(normalize=True).iloc[0])
    return (
        2.0 * abs(math.log(count / target_jobs))
        + abs(gpu_share - reference["gpu_share"]) / max(reference["gpu_share"], 0.02)
        + 0.5 * abs(math.log(runtime_median / reference["runtime_median_s"]))
        + 0.75 * abs(math.log(runtime_p95 / reference["runtime_p95_s"]))
        + 0.5 * abs(math.log(cpu_median / reference["cpu_median"]))
        + 0.75 * abs(math.log(cpu_p95 / reference["cpu_p95"]))
        + 0.5 * abs(math.log(core_hours_per_job / reference["core_hours_per_job"]))
        + abs(simultaneous_share - reference["simultaneous_share"])
        + abs(top_user_share - reference["top_user_share"])
    )


def choose_representative_window(frame: pd.DataFrame, target_jobs: int) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered_submit = frame["_submit"].sort_values()
    reference = {
        "gpu_share": float((frame["_requested_gpus"] > 0).mean()),
        "runtime_median_s": max(float(frame["_runtime_s"].median()), 1.0),
        "runtime_p95_s": max(float(frame["_runtime_s"].quantile(0.95)), 1.0),
        "cpu_median": max(float(frame["_requested_cpus"].median()), 1.0),
        "cpu_p95": max(float(frame["_requested_cpus"].quantile(0.95)), 1.0),
        "core_hours_per_job": max(
            float((frame["_requested_cpus"] * frame["_runtime_s"] / 3600.0).mean()),
            1e-6,
        ),
        "simultaneous_share": float(ordered_submit.diff().dt.total_seconds().eq(0).mean()),
        "top_user_share": float(frame["User"].value_counts(normalize=True).iloc[0]),
    }
    first = frame["_submit"].min().floor("h")
    last = frame["_submit"].max().ceil("h")
    starts = pd.date_range(first, last, freq="1h", tz="UTC")
    durations = [1, 2, 4, 6, 12]
    candidates: list[tuple[float, pd.Timestamp, int, int]] = []

    for start in starts:
        for duration_hours in durations:
            end = start + pd.Timedelta(hours=duration_hours)
            window = frame.loc[(frame["_submit"] >= start) & (frame["_submit"] < end)]
            if len(window) < max(20, int(target_jobs * 0.45)):
                continue
            if len(window) > int(target_jobs * 2.0):
                continue
            score = candidate_score(window, target_jobs, reference)
            candidates.append((score, start, duration_hours, len(window)))

    if not candidates:
        raise ValueError("No representative source window could be selected")
    score, start, duration_hours, source_count = min(candidates, key=lambda item: item[0])
    end = start + pd.Timedelta(hours=duration_hours)
    window = frame.loc[(frame["_submit"] >= start) & (frame["_submit"] < end)].copy()

    if len(window) > target_jobs:
        positions = np.linspace(0, len(window) - 1, target_jobs).round().astype(int)
        positions = np.unique(positions)
        window = window.iloc[positions].copy()

    selection = {
        "source_window_start_utc": start.isoformat(),
        "source_window_end_utc": end.isoformat(),
        "source_window_hours": duration_hours,
        "source_window_jobs_before_systematic_sample": source_count,
        "selected_jobs": len(window),
        "selection_score": score,
        "monthly_reference": reference,
    }
    return window, selection


def choose_fixed_window(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    duration_hours: float,
    target_jobs: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if duration_hours <= 0:
        raise ValueError("window duration must be positive")
    end = start + pd.Timedelta(hours=duration_hours)
    window = frame.loc[(frame["_submit"] >= start) & (frame["_submit"] < end)].copy()
    source_count = len(window)
    if source_count == 0:
        raise ValueError(f"No supported completed jobs in fixed window {start} to {end}")
    if len(window) > target_jobs:
        positions = np.linspace(0, len(window) - 1, target_jobs).round().astype(int)
        window = window.iloc[np.unique(positions)].copy()
    selection = {
        "selection_method": "fixed_utc_window",
        "source_window_start_utc": start.isoformat(),
        "source_window_end_utc": end.isoformat(),
        "source_window_hours": duration_hours,
        "source_window_jobs_before_systematic_sample": source_count,
        "selected_jobs": len(window),
    }
    return window, selection


def normalize_resources(row: pd.Series) -> dict[str, object] | None:
    partition = str(row["_mapped_partition"])
    capacity = PARTITION_CAPACITY[partition]
    nodes = max(1, int(math.ceil(float(row["_requested_nodes"]))))
    cpus = max(1, int(math.ceil(float(row["_requested_cpus"]))))
    requested_gpus = max(
        0,
        int(math.ceil(max(float(row["_requested_gpus"]), float(row["_allocated_gpus"])))),
    )

    nodes = max(nodes, math.ceil(cpus / capacity["cpus_per_node"]))
    if requested_gpus:
        if capacity["gpus_per_node"] == 0:
            return None
        nodes = max(nodes, math.ceil(requested_gpus / capacity["gpus_per_node"]))
    if nodes > capacity["nodes"]:
        return None

    cpus = min(cpus, nodes * capacity["cpus_per_node"])
    gpu_per_node = 0
    scheduled_gpus = 0
    if requested_gpus:
        gpu_per_node = math.ceil(requested_gpus / nodes)
        if gpu_per_node > capacity["gpus_per_node"]:
            return None
        scheduled_gpus = gpu_per_node * nodes

    total_memory = max(0.0, float(row["_requested_memory_mib"]))
    memory_per_node = int(math.ceil(total_memory / nodes)) if total_memory else 0
    if memory_per_node > capacity["memory_mib"]:
        return None

    requested_runtime_s = max(1, int(round(float(row["_runtime_s"]))))
    partition_max_time_min = PARTITION_MAX_TIME_MIN[partition]
    # Keep at least one minute between simulated completion and Partition MaxTime.
    partition_runtime_cap_s = (partition_max_time_min - 1) * 60
    runtime_s = min(requested_runtime_s, partition_runtime_cap_s)
    source_timelimit = float(row["_timelimit_min"])
    if not math.isfinite(source_timelimit) or source_timelimit <= 0:
        source_timelimit = 1
    runtime_min = int(math.ceil(runtime_s / 60.0))
    # These source jobs completed successfully. Keep the source limit for
    # traceability, but give the simulator enough headroom that coarse virtual
    # clock jumps do not turn a completion event into a false timeout.
    timelimit_min = max(
        int(math.ceil(source_timelimit)),
        runtime_min + 60,
        runtime_min * 2,
    )
    timelimit_min = min(timelimit_min, partition_max_time_min)

    return {
        "partition": partition,
        "nodes": nodes,
        "cpus": cpus,
        "memory_per_node_mib": memory_per_node,
        "requested_gpus": requested_gpus,
        "scheduled_gpus": scheduled_gpus,
        "gpu_per_node": gpu_per_node,
        "gpu_type": GPU_TYPE.get(partition, ""),
        "runtime_s": runtime_s,
        "runtime_partition_capped": requested_runtime_s > runtime_s,
        "source_timelimit_min": int(math.ceil(source_timelimit)),
        "timelimit_min": timelimit_min,
    }


def build_profile(
    window: pd.DataFrame,
    seed: int,
    source_count_before_sampling: int | None = None,
    event_epoch: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    dropped_capacity = 0
    source_window_start = event_epoch if event_epoch is not None else window["_submit"].min()
    if source_window_start.tzinfo is None:
        source_window_start = source_window_start.tz_localize("UTC")
    else:
        source_window_start = source_window_start.tz_convert("UTC")
    source_span_s = max((window["_submit"].max() - source_window_start).total_seconds(), 1.0)

    for _, source in window.iterrows():
        resources = normalize_resources(source)
        if resources is None:
            dropped_capacity += 1
            continue
        rows.append(
            {
                "source_submit_utc": source["_submit"].isoformat(),
                "source_eligible_utc": source["_eligible"].isoformat(),
                "source_start_utc": source["_start"].isoformat() if pd.notna(source["_start"]) else "",
                "source_end_utc": source["_end"].isoformat() if pd.notna(source["_end"]) else "",
                "source_partition": source["Partition"],
                "qos": source["QOS"],
                "source_user_private": source["User"],
                "source_submit_wait_s": (
                    (source["_start"] - source["_submit"]).total_seconds()
                    if pd.notna(source["_start"])
                    else math.nan
                ),
                "source_eligible_wait_s": (
                    (source["_start"] - source["_eligible"]).total_seconds()
                    if pd.notna(source["_start"])
                    else math.nan
                ),
                "dependency_delay_s": max((source["_eligible"] - source["_submit"]).total_seconds(), 0.0),
                "runtime_original_s": float(source["_runtime_original_s"]),
                **resources,
            }
        )

    profile = pd.DataFrame(rows)
    if profile.empty:
        raise ValueError("Every selected job exceeded the working simulator capacity")

    # If systematic sampling removed jobs, compress the event clock by the same
    # ratio so average offered load remains close to the source window.
    source_count = source_count_before_sampling or len(window)
    compression = min(1.0, len(profile) / source_count)
    submit_offset = (
        pd.to_datetime(profile["source_submit_utc"], utc=True) - source_window_start
    ).dt.total_seconds()
    eligible_offset = (
        pd.to_datetime(profile["source_eligible_utc"], utc=True) - source_window_start
    ).dt.total_seconds()
    profile["submit_dt_s"] = np.maximum(0, np.rint(submit_offset * compression)).astype(int)
    profile["eligible_dt_s"] = np.maximum(
        profile["submit_dt_s"], np.rint(eligible_offset * compression).astype(int)
    )
    profile["release_dt_s"] = profile["eligible_dt_s"]
    profile["event_epoch_utc"] = source_window_start.isoformat()

    user_counts = profile["source_user_private"].value_counts()
    user_map = {raw: f"user_{index:03d}" for index, raw in enumerate(user_counts.index, start=1)}
    profile["user_id"] = profile["source_user_private"].map(user_map)
    profile.drop(columns=["source_user_private"], inplace=True)

    rng = random.Random(seed)
    profile["flexibility_score"] = [rng.random() for _ in range(len(profile))]
    profile.loc[profile["partition"].eq("interactive"), "flexibility_score"] = 1.0
    profile.sort_values(["release_dt_s", "source_submit_utc"], inplace=True)
    profile.reset_index(drop=True, inplace=True)
    profile.insert(0, "sim_job_id", [f"sim_{index:06d}" for index in range(1, len(profile) + 1)])
    profile.insert(1, "slurm_job_id_expected", np.arange(1, len(profile) + 1, dtype=int))

    audit = {
        "selected_before_capacity_checks": source_count,
        "dropped_for_working_capacity": dropped_capacity,
        "jobs_in_profile": len(profile),
        "runtime_values_partition_capped": int(
            profile["runtime_partition_capped"].fillna(False).astype(bool).sum()
        ),
        "unique_pseudonymous_users": profile["user_id"].nunique(),
        "event_time_compression": compression,
        "source_span_s": source_span_s,
        "event_submit_span_s": int(profile["submit_dt_s"].max()),
        "event_release_span_s": int(profile["release_dt_s"].max()),
    }
    return profile, audit


def event_line(row: pd.Series) -> str:
    options = [
        f"--uid={row['user_id']}",
        f"-J jobid_{int(row['slurm_job_id_expected'])}",
        f"-p {row['partition']}",
        f"-N {int(row['nodes'])}",
        f"-n {int(row['cpus'])}",
    ]
    if int(row["memory_per_node_mib"]) > 0:
        options.append(f"--mem={int(row['memory_per_node_mib'])}M")
    if int(row["gpu_per_node"]) > 0:
        options.append(f"--gres=gpu:{row['gpu_type']}:{int(row['gpu_per_node'])}")
    options.extend(
        [
            f"-t {int(row['timelimit_min'])}",
            f"-sim-walltime {int(row['runtime_s'])}",
            "pseudo.job",
            f"-sleep {int(row['runtime_s'])}",
        ]
    )
    return f"-e submit_batch_job -dt {int(row['release_dt_s'])} | " + " ".join(options)


def write_outputs(output_dir: Path, profile: pd.DataFrame, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_dir / "workload_profile.csv", index=False)
    (output_dir / "sim.events").write_text(
        "\n".join(event_line(row) for _, row in profile.iterrows()) + "\n",
        encoding="utf-8",
    )

    users = profile[["user_id"]].drop_duplicates().sort_values("user_id")
    with (output_dir / "users.sim").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=":", lineterminator="\n")
        for index, user_id in enumerate(users["user_id"], start=1):
            writer.writerow([user_id, 10_000 + index, "users", 100])

    (output_dir / "workload_generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    validation = pd.DataFrame(
        [
            {"metric": "jobs", "value": len(profile), "unit": "jobs"},
            {"metric": "gpu_job_share", "value": float((profile["scheduled_gpus"] > 0).mean()), "unit": "ratio"},
            {"metric": "median_runtime", "value": float(profile["runtime_s"].median()), "unit": "seconds"},
            {"metric": "p95_runtime", "value": float(profile["runtime_s"].quantile(0.95)), "unit": "seconds"},
            {"metric": "median_requested_cpus", "value": float(profile["cpus"].median()), "unit": "cores"},
            {"metric": "p95_requested_cpus", "value": float(profile["cpus"].quantile(0.95)), "unit": "cores"},
            {"metric": "median_dependency_delay", "value": float(profile["dependency_delay_s"].median()), "unit": "seconds"},
            {"metric": "p95_dependency_delay", "value": float(profile["dependency_delay_s"].quantile(0.95)), "unit": "seconds"},
            {"metric": "event_release_span", "value": float(profile["release_dt_s"].max()), "unit": "seconds"},
        ]
    )
    validation.to_csv(output_dir / "workload_validation_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--month", choices=[
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ], default="june")
    parser.add_argument("--target-jobs", type=int, default=200)
    parser.add_argument("--max-runtime-hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-start-utc")
    parser.add_argument("--window-duration-hours", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = read_month(args.zip_path, args.month)
    cohort, cohort_audit = prepare_cohort(raw, args.month, args.max_runtime_hours)
    if bool(args.window_start_utc) != bool(args.window_duration_hours):
        raise ValueError(
            "window-start-utc and window-duration-hours must be supplied together"
        )
    if args.window_start_utc:
        window, selection = choose_fixed_window(
            cohort,
            pd.Timestamp(args.window_start_utc),
            args.window_duration_hours,
            args.target_jobs,
        )
    else:
        window, selection = choose_representative_window(cohort, args.target_jobs)
    event_epoch = (
        pd.Timestamp(selection["source_window_start_utc"])
        if selection.get("selection_method") == "fixed_utc_window"
        else None
    )
    profile, profile_audit = build_profile(
        window,
        args.seed,
        int(selection["source_window_jobs_before_systematic_sample"]),
        event_epoch,
    )

    summary = {
        "source_archive": str(args.zip_path),
        "source_month": args.month,
        "source_timezone": "UTC",
        "privacy": "Raw user IDs are replaced with window-local pseudonyms and are not exported",
        "cohort": "Completed jobs with valid Submit, Eligible, runtime, and supported resources",
        "scheduler_release_rule": "Events are injected at Eligible, not Submit; dependency/hold delay remains separate",
        "hardware_status": "202 observed node names; current hardware specifications are published; 2025 range-to-class mapping is inferred from node names and partitions",
        "seed": args.seed,
        "max_runtime_hours": args.max_runtime_hours,
        "cohort_audit": cohort_audit,
        "window_selection": selection,
        "profile_audit": profile_audit,
    }
    write_outputs(args.output, profile, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
