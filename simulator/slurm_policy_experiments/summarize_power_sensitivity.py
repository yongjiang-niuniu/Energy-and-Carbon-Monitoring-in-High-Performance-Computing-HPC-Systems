#!/usr/bin/env python3
"""Combine calculate_carbon outputs from named power-model scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must use LABEL=RESULT_DIRECTORY")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("case must use LABEL=RESULT_DIRECTORY")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        help="Policy to include; repeat for several policies. Defaults to every non-baseline scenario.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for label, result_dir in args.case:
        comparison = pd.read_csv(result_dir / "scenario_comparison.csv")
        summary_path = result_dir / "carbon_model_summary.json"
        model = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = comparison.loc[~comparison["scenario"].eq("baseline")]
        if args.policy:
            missing = sorted(set(args.policy) - set(selected["scenario"]))
            if missing:
                raise ValueError(f"Policies not found in {result_dir}: {missing}")
            selected = selected.loc[selected["scenario"].isin(args.policy)]
        baseline = comparison.loc[comparison["scenario"].eq("baseline")].iloc[0]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "power_scenario": label,
                    "idle_power_kw": model["idle_power_kw"],
                    "regular_power_kw": model["regular_power_kw"],
                    "dynamic_range_kw": model["dynamic_power_range_kw"],
                    "policy": row["scenario"],
                    "baseline_dynamic_carbon_kg_capacity_weighted": baseline[
                        "dynamic_carbon_kg_capacity_weighted"
                    ],
                    "baseline_whole_carbon_kg_capacity_weighted": baseline[
                        "whole_carbon_kg_capacity_weighted"
                    ],
                    "dynamic_carbon_reduction_kg_capacity_weighted": row[
                        "dynamic_carbon_reduction_kg_capacity_weighted"
                    ],
                    "dynamic_carbon_reduction_pct_capacity_weighted": row[
                        "dynamic_carbon_reduction_pct_capacity_weighted"
                    ],
                    "whole_carbon_reduction_kg_capacity_weighted": row[
                        "whole_carbon_reduction_kg_capacity_weighted"
                    ],
                    "whole_carbon_reduction_pct_capacity_weighted": row[
                        "whole_carbon_reduction_pct_capacity_weighted"
                    ],
                    "dynamic_carbon_reduction_kg_node_request_upper": row[
                        "dynamic_carbon_reduction_kg_node_request_upper"
                    ],
                    "dynamic_carbon_reduction_pct_node_request_upper": row[
                        "dynamic_carbon_reduction_pct_node_request_upper"
                    ],
                    "whole_carbon_reduction_kg_node_request_upper": row[
                        "whole_carbon_reduction_kg_node_request_upper"
                    ],
                    "whole_carbon_reduction_pct_node_request_upper": row[
                        "whole_carbon_reduction_pct_node_request_upper"
                    ],
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "power_sensitivity_summary.csv", index=False)
    (args.output / "power_sensitivity_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
