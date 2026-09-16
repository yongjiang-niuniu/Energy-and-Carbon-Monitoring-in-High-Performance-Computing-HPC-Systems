#!/usr/bin/env python3
"""Run a reproducible matrix of carbon-aware Slurm policy scenarios."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def run(command: list[str]) -> float:
    started = time.monotonic()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return time.monotonic() - started


def policy_command(
    baseline: Path,
    carbon: Path,
    output: Path,
    spec: dict[str, object],
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "generate_policy_scenario.py"),
        "--baseline-profile",
        str(baseline / "workload_profile.csv"),
        "--carbon",
        str(carbon),
        "--strategy",
        str(spec["strategy"]),
        "--output",
        str(output),
    ]
    parameters = spec.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("scenario parameters must be an object")
    for key, value in parameters.items():
        option = "--" + str(key).replace("_", "-")
        command.extend([option, str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    baseline = resolve_repo_path(manifest["baseline"])
    carbon = resolve_repo_path(manifest["carbon"])
    scenario_root = resolve_repo_path(manifest["scenario_root"])
    result_root = resolve_repo_path(manifest["result_root"])
    timeout_s = int(manifest.get("timeout_seconds", 1200))
    scenarios = manifest.get("scenarios", [])
    if not scenarios:
        raise ValueError("manifest must contain at least one policy scenario")

    baseline_validation = json.loads(
        (baseline / "simulation_validation.json").read_text(encoding="utf-8")
    )
    if not baseline_validation.get("strict_pass"):
        raise ValueError(f"Baseline has not passed strict validation: {baseline}")

    scenario_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, object]] = []
    policy_paths: list[Path] = []

    for spec in scenarios:
        name = str(spec["name"])
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
            raise ValueError(f"Unsafe scenario name: {name}")
        output = scenario_root / name
        policy_paths.append(output)
        generation_s = run(policy_command(baseline, carbon, output, spec))
        simulation_s = run([str(ROOT / "run_scenario.sh"), str(output), str(timeout_s)])
        analysis_s = run([sys.executable, str(ROOT / "analyze_simulation.py"), str(output)])
        run_records.append(
            {
                "scenario": name,
                "strategy": spec["strategy"],
                "parameters": spec.get("parameters", {}),
                "generation_wall_s": generation_s,
                "simulation_wall_s": simulation_s,
                "analysis_wall_s": analysis_s,
            }
        )

    carbon_command = [
        sys.executable,
        str(ROOT / "calculate_carbon.py"),
        "--scenario",
        str(baseline),
    ]
    for policy_path in policy_paths:
        carbon_command.extend(["--scenario", str(policy_path)])
    carbon_command.extend(["--carbon", str(carbon), "--output", str(result_root)])
    if manifest.get("horizon_start_utc"):
        carbon_command.extend(
            ["--horizon-start-utc", str(manifest["horizon_start_utc"])]
        )
    if manifest.get("horizon_end_utc"):
        carbon_command.extend(
            ["--horizon-end-utc", str(manifest["horizon_end_utc"])]
        )
    carbon_s = run(carbon_command)

    audit_command = [
        sys.executable,
        str(ROOT / "validate_experiment.py"),
        "--baseline",
        str(baseline),
    ]
    for policy_path in policy_paths:
        audit_command.extend(["--policy", str(policy_path)])
    audit_command.extend(
        [
            "--comparison",
            str(result_root / "scenario_comparison.csv"),
            "--output",
            str(result_root),
        ]
    )
    audit_s = run(audit_command)

    summary = {
        "manifest": str(args.manifest.resolve()),
        "baseline": str(baseline),
        "carbon": str(carbon),
        "strict_baseline_pass": True,
        "policy_runs": run_records,
        "carbon_calculation_wall_s": carbon_s,
        "audit_wall_s": audit_s,
        "total_wall_s": sum(
            record["generation_wall_s"]
            + record["simulation_wall_s"]
            + record["analysis_wall_s"]
            for record in run_records
        )
        + carbon_s
        + audit_s,
    }
    (result_root / "matrix_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
