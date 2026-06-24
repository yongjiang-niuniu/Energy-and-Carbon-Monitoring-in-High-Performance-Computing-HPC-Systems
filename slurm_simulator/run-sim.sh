#!/usr/bin/env bash
#
# One-shot launcher for the local Slurm simulator demo.
#
# What it does:
#   1. Makes sure the Colima Linux VM (Docker backend) is running.
#   2. Builds the simulator image (only rebuilds when sources change).
#   3. Runs slurmctld in simulator mode against demo-sim/ and prints the
#      job scheduling timeline (submission -> start -> completion).
#
# The simulator replays the jobs listed in demo-sim/sim.events and exits on
# its own once every job has finished ("All done.").
#
set -euo pipefail

cd "$(dirname "$0")"

# Make brew-installed colima/docker visible in non-login shells.
if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

echo "==> Ensuring Colima VM is running..."
if ! colima status >/dev/null 2>&1; then
    colima start --cpu 4 --memory 8 --disk 60 --vm-type vz --mount-type virtiofs
fi

echo "==> Building simulator image (slurm-sim-local:latest)..."
docker build -t slurm-sim-local:latest -f Dockerfile.local-sim . >/dev/null

echo "==> Running the simulation..."
rm -f demo-sim/slurmctld.log demo-sim/sched.log
docker run --rm --hostname linux1 \
    -v "$PWD/demo-sim":/opt/slurm-sim/etc \
    slurm-sim-local:latest \
    bash -c 'cd /opt/slurm-sim/etc && stdbuf -oL -eL timeout -s KILL 60 slurmctld -D -vvv -i >/dev/null 2>&1; true'

echo
echo "==================== Job scheduling timeline ===================="
grep -E "submit_batch_job: JobId|_start_job: Started JobId|_job_complete: JobId=[0-9]+ done|All done\." \
    demo-sim/slurmctld.log || true
echo "================================================================"
echo
echo "Full controller log: demo-sim/slurmctld.log"
