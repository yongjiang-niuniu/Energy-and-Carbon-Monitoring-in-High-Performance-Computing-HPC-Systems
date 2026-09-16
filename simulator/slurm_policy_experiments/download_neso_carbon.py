#!/usr/bin/env python3
"""Download regional 30-minute carbon-intensity estimates from NESO."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


API_ROOT = "https://api.carbonintensity.org.uk"


def build_url(start: str, end: str, region_id: int) -> str:
    start_encoded = urllib.parse.quote(start, safe=":-")
    end_encoded = urllib.parse.quote(end, safe=":-")
    return f"{API_ROOT}/regional/intensity/{start_encoded}/{end_encoded}/regionid/{region_id}"


def normalize_payload(payload: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    region = payload.get("data", {})
    if not isinstance(region, dict) or "data" not in region:
        raise ValueError("Unexpected NESO regional API response")
    rows = []
    for item in region["data"]:
        intensity = item.get("intensity", {})
        generation = {entry["fuel"]: entry["perc"] for entry in item.get("generationmix", [])}
        rows.append(
            {
                "from_utc": item.get("from"),
                "to_utc": item.get("to"),
                "intensity_gco2_per_kwh": intensity.get("forecast"),
                "intensity_index": intensity.get("index"),
                **{f"generation_{fuel}_pct": value for fuel, value in generation.items()},
            }
        )
    frame = pd.DataFrame(rows)
    frame["from_utc"] = pd.to_datetime(frame["from_utc"], utc=True)
    frame["to_utc"] = pd.to_datetime(frame["to_utc"], utc=True)
    frame["intensity_gco2_per_kwh"] = pd.to_numeric(
        frame["intensity_gco2_per_kwh"], errors="coerce"
    )
    frame.dropna(subset=["from_utc", "to_utc", "intensity_gco2_per_kwh"], inplace=True)
    frame.drop_duplicates("from_utc", keep="last", inplace=True)
    frame.sort_values("from_utc", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    metadata = {
        "region_id": region.get("regionid"),
        "dno_region": region.get("dnoregion"),
        "short_name": region.get("shortname"),
        "value_type": "regional forecast/modelled carbon intensity",
        "unit": "gCO2e/kWh",
        "timezone": "UTC",
        "interval": "30 minutes",
    }
    return frame, metadata


def download(start: str, end: str, region_id: int) -> tuple[pd.DataFrame, dict[str, object]]:
    url = build_url(start, end, region_id)
    request = urllib.request.Request(url, headers={"User-Agent": "stanage-carbon-msc/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    frame, metadata = normalize_payload(payload)
    metadata["api_url"] = url
    metadata["requested_from"] = start
    metadata["requested_to"] = end
    metadata["rows"] = len(frame)
    return frame, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", required=True, help="ISO-8601 UTC timestamp")
    parser.add_argument("--to", dest="end", required=True, help="ISO-8601 UTC timestamp")
    parser.add_argument("--region-id", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame, metadata = download(args.start, args.end, args.region_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
