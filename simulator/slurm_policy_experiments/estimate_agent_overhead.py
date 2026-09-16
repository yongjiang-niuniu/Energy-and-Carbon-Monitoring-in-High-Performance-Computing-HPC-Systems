#!/usr/bin/env python3
"""Create a transparent scenario estimate for per-job LLM scheduling overhead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ENERGY_SCENARIOS_J_PER_TOKEN = {
    "efficient_text_conversation": 0.151,
    "same_batch_reasoning": 0.312,
    "large_reasoning_example": 0.400,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon", type=Path, required=True)
    parser.add_argument("--annual-jobs", type=int, default=4_098_801)
    parser.add_argument("--holdout-jobs", type=int, default=3_211)
    parser.add_argument("--holdout-flexible-jobs", type=int, default=944)
    parser.add_argument("--annual-flexible-fraction", type=float, default=0.30)
    parser.add_argument("--tokens", nargs="+", type=int, default=[250, 500, 1000, 7000])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    carbon = pd.read_csv(args.carbon)
    intensity = pd.to_numeric(carbon["intensity_gco2_per_kwh"], errors="coerce").dropna()
    mean_intensity = float(intensity.mean())
    rows = []
    for workload, decisions in {
        "june_24h_holdout": args.holdout_jobs,
        "june_24h_flexible_only": args.holdout_flexible_jobs,
        "full_2025_archive": args.annual_jobs,
        "full_2025_flexible_only": round(
            args.annual_jobs * args.annual_flexible_fraction
        ),
    }.items():
        for tokens in args.tokens:
            for scenario, joules_per_token in ENERGY_SCENARIOS_J_PER_TOKEN.items():
                energy_kwh = decisions * tokens * joules_per_token / 3_600_000
                rows.append(
                    {
                        "workload": workload,
                        "one_agent_call_per_job": decisions,
                        "output_token_equivalent_per_call": tokens,
                        "energy_scenario": scenario,
                        "joules_per_output_token": joules_per_token,
                        "estimated_it_energy_kwh": energy_kwh,
                        "illustrative_carbon_intensity_gco2e_per_kwh": mean_intensity,
                        "estimated_operational_carbon_kgco2e": energy_kwh * mean_intensity / 1000,
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "agent_overhead_scenarios.csv", index=False)
    summary = {
        "status": "illustrative scenario, not a measurement of a named hosted model",
        "assumption": "one inference call per job; token counts are output-token-equivalent scenarios",
        "energy_source": "ML.ENERGY v3 examples: 0.151 and 0.312 J/output-token for Qwen 3 32B on B200; approximately 0.4 J/output-token for a larger reasoning example",
        "energy_source_url": "https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/",
        "benchmark_paper_url": "https://proceedings.neurips.cc/paper_files/paper/2025/file/9dc510e3d7b0b3b2a58ffed7a3ad6b0f-Paper-Datasets_and_Benchmarks_Track.pdf",
        "carbon_intensity_source": str(args.carbon),
        "carbon_intensity_mean_gco2e_per_kwh": mean_intensity,
        "exclusions": [
            "input-token-specific energy",
            "datacentre PUE",
            "network and storage",
            "embodied carbon",
            "retries or repeated queue scans",
        ],
        "interpretation": "Any online LLM decision adds positive inference energy. The estimate does not prove that this overhead would exceed scheduling savings, but no LLM is needed for the deterministic policy logic.",
    }
    (args.output / "agent_overhead_method.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
