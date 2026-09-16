#!/usr/bin/env python3
"""Build an aggregate Stanage node inventory from the Slurm accounting archive.

The accounting export identifies allocated hosts through ``NodeList`` but does
not record hardware model or power telemetry. This script therefore separates
observed facts from current hardware specifications published by Sheffield and
the 2025 class mapping inferred from node numbering and partition names.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


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
MISSING_NODE_VALUES = {"", "none", "none assigned", "unknown", "n/a"}


def split_top_level(value: str) -> list[str]:
    """Split a Slurm hostlist on commas outside square brackets."""
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for character in value:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        if character == "," and depth == 0:
            if current:
                result.append("".join(current))
                current = []
        else:
            current.append(character)
    if current:
        result.append("".join(current))
    return result


def expand_hostlist(value: str) -> list[str]:
    """Expand the simple numeric ranges used by Stanage NodeList values."""
    expanded: list[str] = []
    for token in split_top_level(value):
        token = token.strip()
        match = re.fullmatch(r"([^\[]+)\[([^\]]+)\](.*)", token)
        if not match:
            if token:
                expanded.append(token)
            continue
        prefix, body, suffix = match.groups()
        for item in body.split(","):
            if "-" not in item:
                expanded.append(f"{prefix}{item}{suffix}")
                continue
            lower, upper = item.split("-", 1)
            if not lower.isdigit() or not upper.isdigit():
                expanded.append(token)
                continue
            width = max(len(lower), len(upper))
            expanded.extend(
                f"{prefix}{number:0{width}d}{suffix}"
                for number in range(int(lower), int(upper) + 1)
            )
    return expanded


def inferred_group(node: str) -> tuple[str, str, str]:
    """Return group, expected hardware, and evidence status for a node name."""
    match = re.fullmatch(r"node(\d+)", node)
    if match:
        number = int(match.group(1))
        if 1 <= number <= 150:
            return (
                "general_cpu",
                "Dell R650; 2x32-core Intel Xeon Platinum 8358; 256 GB RAM (current official specification)",
                "node range observed; current specification published; 2025 range-to-class mapping inferred",
            )
        if 201 <= number <= 212:
            return (
                "large_memory_cpu",
                "Dell R650; 2x32-core Intel Xeon Platinum 8358; 1 TB RAM (current official specification)",
                "node range observed; current specification published; 2025 range-to-class mapping inferred",
            )
        if 301 <= number <= 312:
            return (
                "very_large_memory_cpu",
                "Dell R650; 2x32-core Intel Xeon Platinum 8358; 2 TB RAM (current official specification)",
                "node range observed; current specification published; 2025 range-to-class mapping inferred",
            )
    match = re.fullmatch(r"gpu(\d+)", node)
    if match:
        number = int(match.group(1))
        if 1 <= number <= 18:
            return (
                "gpu_01_18",
                "Dell XE8545; 2x24-core AMD EPYC 7413; 512 GB RAM; 4x NVIDIA A100 80 GB (current official specification)",
                "node and partition observed; current specification published; 2025 range-to-class mapping inferred",
            )
        if 21 <= number <= 26:
            return (
                "gpu_21_26",
                "Dell R7525; 2x24-core AMD EPYC 7413; 512 GB RAM; 2x NVIDIA H100 80 GB (current official specification)",
                "node and partition observed; current specification published; 2025 range-to-class mapping inferred",
            )
        if 31 <= number <= 34:
            return (
                "gpu_31_34",
                "Dell R760xa; 2x48-core Intel Xeon Platinum 8468; 512 GB RAM; 4x NVIDIA H100 NVL 94 GB (current official specification)",
                "node and partition observed; current specification published; 2025 range-to-class mapping inferred",
            )
    return (
        "unclassified",
        "unknown hardware class",
        "node observed; classification requires Fred confirmation",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    node_mentions: Counter[str] = Counter()
    node_partitions: dict[str, set[str]] = defaultdict(set)
    node_months: dict[str, set[str]] = defaultdict(set)
    raw_rows = 0
    rows_without_assigned_node = 0

    with zipfile.ZipFile(args.zip_path) as archive:
        for month in MONTHS:
            entry = f"stanage_2025_queue_record/{month}.txt"
            with archive.open(entry) as raw_handle:
                handle = io.TextIOWrapper(
                    raw_handle,
                    encoding="utf-8",
                    errors="replace",
                    newline="",
                )
                reader = csv.reader(handle, delimiter="|")
                header = next(reader)
                node_index = header.index("NodeList")
                partition_index = header.index("Partition")
                for row in reader:
                    raw_rows += 1
                    if len(row) <= max(node_index, partition_index):
                        rows_without_assigned_node += 1
                        continue
                    expression = row[node_index].strip()
                    if expression.lower() in MISSING_NODE_VALUES:
                        rows_without_assigned_node += 1
                        continue
                    partition = row[partition_index].strip() or "UNKNOWN"
                    for node in expand_hostlist(expression):
                        node_mentions[node] += 1
                        node_partitions[node].add(partition)
                        node_months[node].add(month)

    node_rows: list[dict[str, object]] = []
    for node in sorted(
        node_mentions,
        key=lambda item: (
            re.sub(r"\d+$", "", item),
            int(re.search(r"\d+$", item).group()) if re.search(r"\d+$", item) else -1,
        ),
    ):
        group, expected_hardware, evidence = inferred_group(node)
        node_rows.append(
            {
                "node": node,
                "inferred_group": group,
                "observed_partitions": ";".join(sorted(node_partitions[node])),
                "source_months_seen": len(node_months[node]),
                "raw_allocation_mentions": node_mentions[node],
                "expected_hardware": expected_hardware,
                "evidence_status": evidence,
            }
        )

    group_nodes: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in node_rows:
        group_nodes[str(row["inferred_group"])].append(row)
    group_rows: list[dict[str, object]] = []
    for group, rows in sorted(group_nodes.items()):
        group_rows.append(
            {
                "inferred_group": group,
                "observed_nodes": len(rows),
                "observed_partitions": ";".join(
                    sorted(
                        {
                            partition
                            for row in rows
                            for partition in str(row["observed_partitions"]).split(";")
                            if partition
                        }
                    )
                ),
                "expected_hardware": rows[0]["expected_hardware"],
                "evidence_status": rows[0]["evidence_status"],
            }
        )

    write_csv(args.output / "node_inventory.csv", node_rows)
    write_csv(args.output / "node_group_summary.csv", group_rows)
    summary_rows = [
        {"metric": "raw_archive_rows", "value": raw_rows},
        {"metric": "rows_without_assigned_node", "value": rows_without_assigned_node},
        {"metric": "unique_observed_nodes", "value": len(node_rows)},
    ]
    write_csv(args.output / "node_inventory_audit.csv", summary_rows)

    print(f"Observed {len(node_rows)} unique nodes across {raw_rows:,} raw rows")
    for row in group_rows:
        print(f"{row['inferred_group']}: {row['observed_nodes']} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
