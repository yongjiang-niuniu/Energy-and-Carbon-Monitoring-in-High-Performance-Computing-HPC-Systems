#!/usr/bin/env python3
"""Build tabular and visual completion evidence for selected Slurm replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_final_evidence as evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-root", type=Path, required=True)
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = evidence.validation_rows(args.scenario_root, args.scenario)
    frame = pd.DataFrame(rows)
    if not frame["strict_pass"].all():
        failed = frame.loc[~frame["strict_pass"], "scenario"].tolist()
        raise ValueError(f"Strict replay validation failed for: {failed}")

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "scenario_validation_summary.csv", index=False)
    (args.output / "scenario_validation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    evidence.write_completion_figure(frame, args.output)
    print(f"Wrote replay completion evidence to {args.output.resolve()}")


if __name__ == "__main__":
    main()
