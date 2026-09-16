#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCENARIO="${1:?usage: run_scenario.sh SCENARIO_DIR [TIMEOUT_SECONDS]}"
TIMEOUT_SECONDS="${2:-900}"
IMAGE="${SLURM_SIM_IMAGE:-slurm-sim-scheduler-fix:latest}"
SOURCE_REPO="${SLURM_SIM_SOURCE:-$ROOT/../slurm_simulator_source_legacy_desktop}"

SCENARIO="$(cd "$SCENARIO" && pwd)"
python3 "$ROOT/prepare_scenario.py" "$SCENARIO"

if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

if ! colima status >/dev/null 2>&1; then
    colima start --cpu 4 --memory 8 --disk 60 --vm-type vz --mount-type virtiofs
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$SOURCE_REPO/Dockerfile.local-sim" "$SOURCE_REPO"
fi

rm -f "$SCENARIO/slurmctld.log" "$SCENARIO/sched.log" "$SCENARIO/core" \
    "$SCENARIO/simulator_exit_code.txt" "$SCENARIO/process_status.json"
simulator_command="$(python3 "$ROOT/process_status.py" command "$TIMEOUT_SECONDS")"

set +e
docker run --rm --hostname node001 \
    -v "$SCENARIO":/opt/slurm-sim/etc \
    "$IMAGE" \
    bash -c "$simulator_command"
docker_status=$?
set -e

python3 "$ROOT/process_status.py" record "$SCENARIO" --docker-exit-code "$docker_status"

rm -f "$SCENARIO/core"

expected="$(awk -F, 'NR > 1 {count++} END {print count + 0}' "$SCENARIO/workload_profile.csv")"
submitted="$({ grep -E '_slurm_rpc_submit_batch_job: JobId=[0-9]+' "$SCENARIO/slurmctld.log" 2>/dev/null || true; } | sed -E 's/.*JobId=([0-9]+).*/\1/' | sort -n -u | wc -l | tr -d ' ')"
started="$({ grep -E 'sched: Allocate JobId=[0-9]+|_start_job: Started JobId=[0-9]+' "$SCENARIO/slurmctld.log" 2>/dev/null || true; } | sed -E 's/.*JobId=([0-9]+).*/\1/' | sort -n -u | wc -l | tr -d ' ')"
completed="$({ grep -E '_job_complete: JobId=[0-9]+ done' "$SCENARIO/slurmctld.log" 2>/dev/null || true; } | sed -E 's/.*JobId=([0-9]+).*/\1/' | sort -n -u | wc -l | tr -d ' ')"
timed_out="$({ grep -E 'Time limit exhausted for JobId=[0-9]+' "$SCENARIO/slurmctld.log" 2>/dev/null || true; } | sed -E 's/.*JobId=([0-9]+).*/\1/' | sort -n -u | wc -l | tr -d ' ')"

echo "Jobs: expected=$expected submitted=$submitted started=$started completed=$completed timed_out=$timed_out"
if grep -q 'All done' "$SCENARIO/slurmctld.log" 2>/dev/null &&
   [ "$submitted" -eq "$expected" ] &&
   [ "$started" -eq "$expected" ] &&
   [ "$completed" -eq "$expected" ] &&
   [ "$timed_out" -eq 0 ]; then
    echo "Job-completion check passed; process and timing validity are separate checks"
else
    echo "Simulation did not pass the strict completion check (docker_exit=$docker_status)"
    exit 2
fi
python3 "$ROOT/analyze_simulation.py" "$SCENARIO" --require-valid-timing
if [ "$docker_status" -ne 0 ]; then
    echo "Simulator/Docker exited abnormally ($docker_status); completion does not make this a clean exit"
    exit 3
fi
