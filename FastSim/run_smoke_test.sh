#!/usr/bin/env bash
# FastSim smoke test: generate minimal sample data and run one simulation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Generating sample Slurm dump..."
python3 sample_data/generate_sample.py

echo "==> Installing Python dependencies (if needed)..."
python3 -m pip install -q pandas numpy pyyaml dill matplotlib 2>/dev/null || \
  python3 -m pip install -q pandas numpy pyyaml dill

echo "==> Running FastSim smoke test..."
cd scheduler
python3 main.py "../configs/stanage/smoke_test_conf.yaml"

echo "==> Smoke test passed."
