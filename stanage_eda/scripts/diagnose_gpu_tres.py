#!/usr/bin/env python3
"""Inspect aggregate GPU TRES components without printing job identifiers."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import zipfile

import pandas as pd

from analyse_stanage import MONTHS, unique_headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=150_000)
    args = parser.parse_args()

    key_counts: Counter[str] = Counter()
    key_max: dict[str, float] = {}
    device_value_counts: Counter[str] = Counter()
    large_components: Counter[tuple[str, str]] = Counter()

    with zipfile.ZipFile(args.zip_path) as archive:
        for month in MONTHS:
            entry = f"stanage_2025_queue_record/{month}.txt"
            with archive.open(entry) as handle:
                headers = unique_headers(handle.readline().decode("utf-8").rstrip("\r\n").split("|"))
            with archive.open(entry) as handle:
                chunks = pd.read_csv(
                    handle,
                    sep="|",
                    header=0,
                    names=headers,
                    usecols=["ReqTRES"],
                    dtype=str,
                    chunksize=args.chunksize,
                    keep_default_na=False,
                    na_filter=False,
                    engine="c",
                )
                for frame in chunks:
                    for value in frame["ReqTRES"]:
                        for component in value.split(","):
                            if "=" not in component:
                                continue
                            key, raw = component.rsplit("=", 1)
                            key = key.strip().lower()
                            if key != "gpu" and not key.startswith("gres/gpu"):
                                continue
                            key_counts[key] += 1
                            try:
                                numeric = float(raw)
                            except ValueError:
                                continue
                            key_max[key] = max(key_max.get(key, numeric), numeric)
                            if key in {"gpu", "gres/gpu"} or key.startswith("gres/gpu:"):
                                device_value_counts[raw] += 1
                                if numeric > 100:
                                    large_components[(key, raw)] += 1

    print("GPU-related TRES keys")
    for key, occurrences in key_counts.most_common():
        print(f"{key}: occurrences={occurrences:,}, maximum={key_max.get(key)}")
    print("\nDevice-count components above 100")
    for (key, raw), occurrences in large_components.most_common(30):
        print(f"{key}={raw}: occurrences={occurrences:,}")
    print("\nMost common device-count values")
    for raw, occurrences in device_value_counts.most_common(30):
        print(f"{raw}: occurrences={occurrences:,}")


if __name__ == "__main__":
    main()
