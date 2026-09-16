#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/stanage_eda/.venv/bin/python}"
SCENARIO="${1:?usage: run_and_analyze_scenario.sh SCENARIO_DIR [TIMEOUT_SECONDS] [RUN_LOG]}"
TIMEOUT_SECONDS="${2:-21600}"
RUN_LOG="${3:-$SCENARIO/replay_console.log}"

mkdir -p "$(dirname "$RUN_LOG")"
{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'scenario=%s\n' "$SCENARIO"
    "$ROOT/run_scenario.sh" "$SCENARIO" "$TIMEOUT_SECONDS"
    "$PYTHON" "$ROOT/analyze_simulation.py" "$SCENARIO"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$RUN_LOG"
