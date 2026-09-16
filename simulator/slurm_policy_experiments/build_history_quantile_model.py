#!/usr/bin/env python3
"""Build privacy-safe runtime, queue-wait and load quantiles from earlier months."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

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
PARTITION_CLASS = {
    "interactive": "cpu",
    "sheffield": "cpu",
    "sheffield-8xlong": "cpu",
    "gpu": "a100",
    "gpu-h100": "h100",
    "gpu-h100-nvl": "h100_nvl",
}
CLASS_CAPACITY = {
    "cpu": {"cpus": 174 * 64, "gpus": 0},
    "a100": {"cpus": 18 * 48, "gpus": 18 * 4},
    "h100": {"cpus": 6 * 48, "gpus": 6 * 2},
    "h100_nvl": {"cpus": 4 * 96, "gpus": 4 * 4},
}
TOTAL_NODES = 174 + 18 + 6 + 4
QUANTILES = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}


def quantile_label(value: float) -> str:
    for label, expected in QUANTILES.items():
        if abs(value - expected) < 1e-9:
            return label
    raise ValueError(f"Unsupported quantile {value}; choose one of {sorted(QUANTILES.values())}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bucket_label(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
    return pd.cut(numeric, bins=edges, labels=labels, include_lowest=True).astype(str)


def add_resource_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cpu_bucket"] = bucket_label(
        result["cpus"], [-1, 1, 4, 16, 64, 256, np.inf],
        ["1", "2-4", "5-16", "17-64", "65-256", "257+"],
    )
    result["node_bucket"] = bucket_label(
        result["nodes"], [-1, 1, 2, 4, 8, 16, np.inf],
        ["1", "2", "3-4", "5-8", "9-16", "17+"],
    )
    result["gpu_bucket"] = bucket_label(
        result["gpus"], [-1, 0, 1, 2, 4, 8, np.inf],
        ["0", "1", "2", "3-4", "5-8", "9+"],
    )
    return result


def load_model(model_dir: Path) -> dict[str, object]:
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_type") != "deterministic_history_quantiles":
        raise ValueError(f"Unsupported history model in {model_dir}")
    if not metadata.get("temporal_split_pass"):
        raise ValueError("History model did not pass its temporal split check")
    return {
        "metadata": metadata,
        "runtime": pd.read_csv(model_dir / "runtime_quantiles.csv"),
        "wait": pd.read_csv(model_dir / "wait_quantiles.csv"),
        "load": pd.read_csv(model_dir / "load_quantiles.csv"),
    }


def _first_eligible_group(
    table: pd.DataFrame,
    levels: list[tuple[str, dict[str, object]]],
    minimum_count: int,
) -> pd.Series:
    for level, filters in levels:
        matches = table.loc[table["level"].eq(level)]
        if matches.empty or any(name not in matches.columns for name in filters):
            continue
        for name, value in filters.items():
            matches = matches.loc[matches[name].astype(str).eq(str(value))]
        eligible = matches.loc[pd.to_numeric(matches["count"], errors="coerce") >= minimum_count]
        if not eligible.empty:
            return eligible.iloc[0]
    global_rows = table.loc[table["level"].eq("global")]
    if global_rows.empty:
        raise ValueError("History model has no global fallback")
    return global_rows.iloc[0]


def predict_runtime(
    profile: pd.DataFrame,
    model: dict[str, object],
    quantile: float,
) -> pd.DataFrame:
    label = quantile_label(quantile)
    table = model["runtime"]
    metadata = model["metadata"]
    assert isinstance(table, pd.DataFrame) and isinstance(metadata, dict)
    features = pd.DataFrame(
        {
            "partition": profile["partition"].astype(str),
            "cpus": pd.to_numeric(profile["cpus"], errors="coerce"),
            "nodes": pd.to_numeric(profile["nodes"], errors="coerce"),
            "gpus": pd.to_numeric(profile["scheduled_gpus"], errors="coerce").fillna(0),
        },
        index=profile.index,
    )
    features = add_resource_buckets(features)
    minimum_count = int(metadata["minimum_group_count"])
    output = pd.DataFrame(index=features.index)
    output["predicted_runtime_s"] = np.nan
    output["runtime_model_level"] = ""
    output["runtime_model_count"] = 0
    for level, keys in [
        ("resource", ["partition", "cpu_bucket", "node_bucket", "gpu_bucket"]),
        ("partition_gpu", ["partition", "gpu_bucket"]),
        ("partition", ["partition"]),
    ]:
        candidates = table.loc[
            table["level"].eq(level)
            & pd.to_numeric(table["count"], errors="coerce").ge(minimum_count)
        ].copy()
        if candidates.empty:
            continue
        left = features[keys].copy()
        for key in keys:
            left[key] = left[key].astype(str)
            candidates[key] = candidates[key].astype(str)
        joined = left.merge(
            candidates[[*keys, "count", f"runtime_s_{label}"]],
            on=keys,
            how="left",
            validate="many_to_one",
        )
        joined.index = features.index
        fill = output["predicted_runtime_s"].isna() & joined[f"runtime_s_{label}"].notna()
        output.loc[fill, "predicted_runtime_s"] = joined.loc[fill, f"runtime_s_{label}"]
        output.loc[fill, "runtime_model_level"] = level
        output.loc[fill, "runtime_model_count"] = joined.loc[fill, "count"].astype(int)
    global_rows = table.loc[table["level"].eq("global")]
    if global_rows.empty:
        raise ValueError("History model has no global runtime fallback")
    missing = output["predicted_runtime_s"].isna()
    output.loc[missing, "predicted_runtime_s"] = float(global_rows.iloc[0][f"runtime_s_{label}"])
    output.loc[missing, "runtime_model_level"] = "global"
    output.loc[missing, "runtime_model_count"] = int(global_rows.iloc[0]["count"])
    maximum_s = float(metadata["maximum_training_runtime_hours"]) * 3600
    output["predicted_runtime_s"] = output["predicted_runtime_s"].clip(60.0, maximum_s)
    return output


def predict_wait(
    partition: str,
    timestamp: pd.Timestamp,
    model: dict[str, object],
    quantile: float,
) -> tuple[float, str, int]:
    label = quantile_label(quantile)
    table = model["wait"]
    metadata = model["metadata"]
    assert isinstance(table, pd.DataFrame) and isinstance(metadata, dict)
    utc = pd.Timestamp(timestamp)
    utc = utc.tz_localize("UTC") if utc.tzinfo is None else utc.tz_convert("UTC")
    weekday = int(utc.weekday())
    slot = int(utc.hour * 2 + utc.minute // 30)
    selected = _first_eligible_group(
        table,
        [
            (
                "partition_slot",
                {
                    "partition": partition,
                    "weekday_utc": weekday,
                    "half_hour_slot_utc": slot,
                },
            ),
            ("partition_weekday", {"partition": partition, "weekday_utc": weekday}),
            ("partition", {"partition": partition}),
        ],
        int(metadata["minimum_group_count"]),
    )
    return max(float(selected[f"wait_s_{label}"]), 0.0), str(selected["level"]), int(selected["count"])


def history_load_forecast(
    carbon: pd.DataFrame,
    model: dict[str, object],
    quantile: float,
) -> pd.DataFrame:
    label = quantile_label(quantile)
    load = model["load"]
    assert isinstance(load, pd.DataFrame)
    lookup = load.set_index(["weekday_utc", "half_hour_slot_utc"])
    forecast = carbon[["from_utc", "to_utc", "intensity_gco2_per_kwh"]].copy()
    timestamps = pd.to_datetime(forecast["from_utc"], utc=True)
    for class_name in CLASS_CAPACITY:
        forecast[f"load_{class_name}"] = [
            float(lookup.loc[(int(ts.weekday()), int(ts.hour * 2 + ts.minute // 30)), f"load_{class_name}_{label}"])
            for ts in timestamps
        ]
    forecast["load_node"] = [
        float(lookup.loc[(int(ts.weekday()), int(ts.hour * 2 + ts.minute // 30)), f"load_node_{label}"])
        for ts in timestamps
    ]
    return forecast


def runtime_training_rows(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "partition": cohort["_mapped_partition"].astype(str),
            "cpus": pd.to_numeric(cohort["_requested_cpus"], errors="coerce"),
            "nodes": pd.to_numeric(cohort["_requested_nodes"], errors="coerce"),
            "gpus": pd.concat(
                [cohort["_requested_gpus"], cohort["_allocated_gpus"]], axis=1
            ).max(axis=1),
            "runtime_s": pd.to_numeric(cohort["_runtime_s"], errors="coerce"),
            "wait_s": (
                cohort["_start"] - cohort["_eligible"]
            ).dt.total_seconds().clip(lower=0),
            "eligible_utc": cohort["_eligible"],
        }
    ).dropna(subset=["runtime_s", "eligible_utc"])
    rows = add_resource_buckets(rows)
    rows["weekday_utc"] = rows["eligible_utc"].dt.weekday.astype(int)
    rows["half_hour_slot_utc"] = (
        rows["eligible_utc"].dt.hour * 2 + rows["eligible_utc"].dt.minute // 30
    ).astype(int)
    return rows


def quantile_rows(
    frame: pd.DataFrame,
    value: str,
    levels: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for level_name, keys in levels:
        groups = [((), frame)] if not keys else frame.groupby(keys, observed=True)
        for raw_key, group in groups:
            key_values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            row: dict[str, object] = {"level": level_name, "count": len(group)}
            row.update(dict(zip(keys, key_values)))
            numeric = pd.to_numeric(group[value], errors="coerce").dropna()
            if numeric.empty:
                continue
            for label, quantile in QUANTILES.items():
                row[f"{value}_{label}"] = float(numeric.quantile(quantile))
            output.append(row)
    return pd.DataFrame(output)


def clean_activity_rows(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["_job_id"] = pd.to_numeric(frame["JobIDRaw"], errors="coerce")
    frame = frame.loc[frame["_job_id"].notna()].copy()
    frame.drop_duplicates("_job_id", keep="last", inplace=True)
    frame["start"] = pd.to_datetime(
        frame["Start"], errors="coerce", utc=True, format="mixed"
    )
    frame["end"] = pd.to_datetime(
        frame["End"], errors="coerce", utc=True, format="mixed"
    )
    frame["partition"] = frame["Partition"].map(GENERATOR.PARTITION_MAP)
    frame["class"] = frame["partition"].map(PARTITION_CLASS)
    requested_cpus = pd.to_numeric(frame["ReqCPUS"], errors="coerce").astype(float)
    allocated_cpus = pd.to_numeric(frame["AllocCPUS"], errors="coerce").astype(float)
    frame["cpus"] = allocated_cpus.where(allocated_cpus.gt(0), requested_cpus)
    requested_nodes = pd.to_numeric(frame["ReqNodes"], errors="coerce").astype(float)
    allocated_nodes = pd.to_numeric(frame["AllocNodes"], errors="coerce").astype(float)
    frame["nodes"] = allocated_nodes.where(allocated_nodes.gt(0), requested_nodes)
    frame["requested_gpus"] = frame["ReqTRES"].map(
        lambda value: GENERATOR.parse_tres_value(value, "gpu")
    )
    frame["allocated_gpus"] = frame["AllocTRES"].map(
        lambda value: GENERATOR.parse_tres_value(value, "gpu")
    )
    frame["gpus"] = frame["allocated_gpus"].where(
        frame["allocated_gpus"].gt(0), frame["requested_gpus"]
    )
    frame["nodelist"] = frame["NodeList"].astype(str)
    valid = (
        frame["start"].notna()
        & frame["end"].notna()
        & frame["end"].gt(frame["start"])
        & frame["class"].notna()
        & frame["cpus"].gt(0)
        & frame["nodes"].gt(0)
        & frame["gpus"].notna()
    )
    frame = frame.loc[valid, [
        "start", "end", "partition", "class", "cpus", "nodes", "gpus", "nodelist"
    ]].copy()
    frame["class_fraction"] = 0.0
    for class_name, capacity in CLASS_CAPACITY.items():
        mask = frame["class"].eq(class_name)
        cpu_fraction = frame.loc[mask, "cpus"] / capacity["cpus"]
        if capacity["gpus"]:
            gpu_fraction = frame.loc[mask, "gpus"] / capacity["gpus"]
            frame.loc[mask, "class_fraction"] = np.maximum(cpu_fraction, gpu_fraction)
        else:
            frame.loc[mask, "class_fraction"] = cpu_fraction
    frame["class_fraction"] = frame["class_fraction"].clip(lower=0, upper=1)
    frame["node_fraction"] = (frame["nodes"] / TOTAL_NODES).clip(lower=0, upper=1)
    return frame


def expand_nodelist(value: str) -> tuple[str, ...]:
    text = str(value).strip()
    direct = re.fullmatch(r"(node|gpu)(\d+)", text)
    if direct:
        candidates = [text]
    else:
        bracket = re.fullmatch(r"(node|gpu)\[([^]]+)\]", text)
        if not bracket:
            return ()
        prefix, body = bracket.groups()
        candidates = []
        for component in body.split(","):
            if "-" in component:
                first, last = component.split("-", 1)
                width = max(len(first), len(last))
                candidates.extend(
                    f"{prefix}{number:0{width}d}"
                    for number in range(int(first), int(last) + 1)
                )
            else:
                candidates.append(f"{prefix}{component}")
    valid = []
    for candidate in candidates:
        match = re.fullmatch(r"(node|gpu)(\d+)", candidate)
        if not match:
            continue
        prefix, raw_number = match.groups()
        number = int(raw_number)
        if prefix == "node" and (
            1 <= number <= 150 or 201 <= number <= 212 or 301 <= number <= 312
        ):
            valid.append(f"node{number:03d}")
        if prefix == "gpu" and (
            1 <= number <= 18 or 21 <= number <= 26 or 31 <= number <= 34
        ):
            valid.append(f"gpu{number:02d}")
    return tuple(valid)


def unique_node_interval_load(
    activity: pd.DataFrame,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    bins: int,
    step_s: float,
) -> tuple[np.ndarray, int]:
    by_node: dict[str, list[tuple[float, float]]] = defaultdict(list)
    cache: dict[str, tuple[str, ...]] = {}
    for row in activity.itertuples(index=False):
        nodes = cache.setdefault(row.nodelist, expand_nodelist(row.nodelist))
        if not nodes or row.end <= month_start or row.start >= month_end:
            continue
        start_s = max((row.start - month_start).total_seconds(), 0.0)
        end_s = min((row.end - month_start).total_seconds(), (month_end - month_start).total_seconds())
        for node in nodes:
            by_node[node].append((start_s, end_s))

    merged: list[tuple[float, float]] = []
    for intervals in by_node.values():
        intervals.sort()
        current_start, current_end = intervals[0]
        for start_s, end_s in intervals[1:]:
            if start_s <= current_end:
                current_end = max(current_end, end_s)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start_s, end_s
        merged.append((current_start, current_end))

    values = np.zeros(bins, dtype=float)
    if merged:
        add_interval_average(
            values,
            np.array([item[0] for item in merged], dtype=float),
            np.array([item[1] for item in merged], dtype=float),
            np.full(len(merged), 1.0 / TOTAL_NODES),
            step_s,
        )
    return values, len(by_node)


def add_interval_average(
    output: np.ndarray,
    starts_s: np.ndarray,
    ends_s: np.ndarray,
    weights: np.ndarray,
    step_s: float,
) -> None:
    if not len(starts_s):
        return
    count = len(output)
    start_index = np.floor(starts_s / step_s).astype(int).clip(0, count - 1)
    end_index = (np.ceil(ends_s / step_s).astype(int) - 1).clip(0, count - 1)
    same = start_index == end_index
    np.add.at(
        output,
        start_index[same],
        weights[same] * (ends_s[same] - starts_s[same]) / step_s,
    )
    multi = ~same
    if not multi.any():
        return
    start_multi = start_index[multi]
    end_multi = end_index[multi]
    weight_multi = weights[multi]
    np.add.at(
        output,
        start_multi,
        weight_multi * (((start_multi + 1) * step_s) - starts_s[multi]) / step_s,
    )
    np.add.at(
        output,
        end_multi,
        weight_multi * (ends_s[multi] - end_multi * step_s) / step_s,
    )
    difference = np.zeros(count + 1, dtype=float)
    np.add.at(difference, start_multi + 1, weight_multi)
    np.add.at(difference, end_multi, -weight_multi)
    output += np.cumsum(difference[:-1])


def month_load_intervals(
    activity: pd.DataFrame,
    month_number: int,
    interval_minutes: int,
) -> pd.DataFrame:
    start = pd.Timestamp(year=2025, month=month_number, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    step_s = float(interval_minutes * 60)
    timestamps = pd.date_range(start, end, freq=f"{interval_minutes}min", inclusive="left")
    output = pd.DataFrame({"from_utc": timestamps})
    overlap = activity.loc[(activity["end"] > start) & (activity["start"] < end)].copy()
    clipped_start = overlap["start"].clip(lower=start)
    clipped_end = overlap["end"].clip(upper=end)
    starts_s = (clipped_start - start).dt.total_seconds().to_numpy(float)
    ends_s = (clipped_end - start).dt.total_seconds().to_numpy(float)
    for class_name in CLASS_CAPACITY:
        values = np.zeros(len(timestamps), dtype=float)
        mask = overlap["class"].eq(class_name).to_numpy()
        add_interval_average(
            values,
            starts_s[mask],
            ends_s[mask],
            overlap.loc[mask, "class_fraction"].to_numpy(float),
            step_s,
        )
        output[f"load_{class_name}"] = values
    node_values, observed_nodes = unique_node_interval_load(
        overlap, start, end, len(timestamps), step_s
    )
    output["load_node"] = node_values
    output["observed_nodes_in_month"] = observed_nodes
    output["weekday_utc"] = output["from_utc"].dt.weekday.astype(int)
    output["half_hour_slot_utc"] = (
        output["from_utc"].dt.hour * 2
        + output["from_utc"].dt.minute // interval_minutes
    ).astype(int)
    return output


def load_quantile_rows(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"load_{name}" for name in CLASS_CAPACITY] + ["load_node"]
    output: list[dict[str, object]] = []
    for (weekday, slot), group in frame.groupby(
        ["weekday_utc", "half_hour_slot_utc"], observed=True
    ):
        row: dict[str, object] = {
            "weekday_utc": int(weekday),
            "half_hour_slot_utc": int(slot),
            "sample_intervals": len(group),
        }
        for metric in metrics:
            for label, quantile in QUANTILES.items():
                row[f"{metric}_{label}"] = float(group[metric].quantile(quantile))
        output.append(row)
    return pd.DataFrame(output).sort_values(
        ["weekday_utc", "half_hour_slot_utc"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--train-months", nargs="+", choices=MONTHS, required=True)
    parser.add_argument("--holdout-month", choices=MONTHS, required=True)
    parser.add_argument("--interval-minutes", type=int, default=30)
    parser.add_argument("--max-runtime-hours", type=float, default=12.0)
    parser.add_argument("--minimum-group-count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_indices = [MONTHS.index(month) for month in args.train_months]
    if len(set(args.train_months)) != len(args.train_months):
        raise ValueError("train-months contains duplicates")
    if max(train_indices) >= MONTHS.index(args.holdout_month):
        raise ValueError("Every training month must be earlier than the holdout month")
    if args.interval_minutes <= 0 or 60 % args.interval_minutes:
        raise ValueError("interval-minutes must be a positive divisor of 60")

    runtime_parts: list[pd.DataFrame] = []
    load_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for month in args.train_months:
        raw = GENERATOR.read_month(args.zip_path, month)
        cohort, audit = GENERATOR.prepare_cohort(raw, month, args.max_runtime_hours)
        runtime_rows = runtime_training_rows(cohort)
        activity = clean_activity_rows(raw)
        load = month_load_intervals(
            activity, MONTHS.index(month) + 1, args.interval_minutes
        )
        runtime_parts.append(runtime_rows)
        load_parts.append(load)
        summaries.append(
            {
                "month": month,
                **audit,
                "runtime_training_rows": len(runtime_rows),
                "activity_rows": len(activity),
                "load_intervals": len(load),
                "observed_nodes_in_month": int(load["observed_nodes_in_month"].max()),
            }
        )
        del raw, cohort, runtime_rows, activity, load

    runtime_data = pd.concat(runtime_parts, ignore_index=True)
    load_data = pd.concat(load_parts, ignore_index=True)
    runtime_quantiles = quantile_rows(
        runtime_data,
        "runtime_s",
        [
            ("resource", ["partition", "cpu_bucket", "node_bucket", "gpu_bucket"]),
            ("partition_gpu", ["partition", "gpu_bucket"]),
            ("partition", ["partition"]),
            ("global", []),
        ],
    )
    wait_quantiles = quantile_rows(
        runtime_data.dropna(subset=["wait_s"]),
        "wait_s",
        [
            ("partition_slot", ["partition", "weekday_utc", "half_hour_slot_utc"]),
            ("partition_weekday", ["partition", "weekday_utc"]),
            ("partition", ["partition"]),
            ("global", []),
        ],
    )
    load_quantiles = load_quantile_rows(load_data)

    args.output.mkdir(parents=True, exist_ok=True)
    runtime_quantiles.to_csv(args.output / "runtime_quantiles.csv", index=False)
    wait_quantiles.to_csv(args.output / "wait_quantiles.csv", index=False)
    load_quantiles.to_csv(args.output / "load_quantiles.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "training_month_summary.csv", index=False)
    metadata = {
        "model_type": "deterministic_history_quantiles",
        "training_months": args.train_months,
        "holdout_month": args.holdout_month,
        "temporal_split_pass": True,
        "source_archive_sha256": file_sha256(args.zip_path),
        "source_archive_not_exported": True,
        "privacy": "Only aggregate quantiles and counts are exported; no user or job identifiers are retained",
        "interval_minutes": args.interval_minutes,
        "maximum_training_runtime_hours": args.max_runtime_hours,
        "minimum_group_count": args.minimum_group_count,
        "quantiles": QUANTILES,
        "runtime_training_rows": len(runtime_data),
        "load_training_intervals": len(load_data),
        "runtime_group_rows": len(runtime_quantiles),
        "wait_group_rows": len(wait_quantiles),
        "load_group_rows": len(load_quantiles),
        "class_load_method": "30-minute average allocated CPU/GPU fraction within each hardware class",
        "node_load_method": "30-minute average fraction of distinct allocated NodeList hosts; shared nodes are counted once",
        "policy_access_boundary": "The policy reads these earlier-month aggregates only; holdout observations are evaluated separately",
    }
    (args.output / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
