#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/stanage_eda/.venv/bin/python}"
SCENARIOS="$ROOT/scenarios/low_impact_dynamic_dec08_20260902"
RESULTS="$ROOT/results/low_impact_dynamic_dec08_20260902"
CARBON="$ROOT/data/neso_south_yorkshire_2025-12-07_to_16.csv"
MANIFEST="$ROOT/manifests/low_impact_dynamic_dec08_20260902.json"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"

mkdir -p "$RESULTS/run_logs"

(
    cd "$REPO_ROOT"
    shasum -a 256 -c \
        "$ROOT/manifests/low_impact_dynamic_dec08_20260902.sha256"
)

if [ ! -d "$SCENARIOS/no_policy_repeat" ]; then
    mkdir -p "$SCENARIOS/no_policy_repeat"
    rsync -a "$SCENARIOS/baseline/" "$SCENARIOS/no_policy_repeat/"
fi

for name in baseline no_policy_repeat low_impact_dynamic_q25; do
    "$ROOT/run_and_analyze_scenario.sh" \
        "$SCENARIOS/$name" "$TIMEOUT_SECONDS" "$RESULTS/run_logs/$name.log"
done

"$PYTHON" "$ROOT/audit_exact_carryin.py" \
    --profile "$SCENARIOS/baseline/workload_profile.csv" \
    --job-results "$SCENARIOS/baseline/job_results.csv" \
    --output "$RESULTS/post_run_carry_in_audit"

"$PYTHON" "$ROOT/calculate_carbon.py" \
    --scenario "$SCENARIOS/baseline" \
    --scenario "$SCENARIOS/no_policy_repeat" \
    --scenario "$SCENARIOS/low_impact_dynamic_q25" \
    --carbon "$CARBON" \
    --horizon-start-utc 2025-12-08T00:00:00Z \
    --idle-power-kw 140 --regular-power-kw 195 \
    --output "$RESULTS/final_comparison"

for power_case in 40_95 40_195; do
    idle_power="${power_case%%_*}"
    regular_power="${power_case##*_}"
    "$PYTHON" "$ROOT/calculate_carbon.py" \
        --scenario "$SCENARIOS/baseline" \
        --scenario "$SCENARIOS/no_policy_repeat" \
        --scenario "$SCENARIOS/low_impact_dynamic_q25" \
        --carbon "$CARBON" \
        --horizon-start-utc 2025-12-08T00:00:00Z \
        --idle-power-kw "$idle_power" --regular-power-kw "$regular_power" \
        --output "$RESULTS/power_sensitivity_$power_case"
done

"$PYTHON" "$ROOT/summarize_power_sensitivity.py" \
    --case "140_195=$RESULTS/final_comparison" \
    --case "40_95=$RESULTS/power_sensitivity_40_95" \
    --case "40_195=$RESULTS/power_sensitivity_40_195" \
    --policy low_impact_dynamic_q25 \
    --output "$RESULTS/power_sensitivity"

"$PYTHON" "$ROOT/validate_experiment.py" \
    --baseline "$SCENARIOS/baseline" \
    --policy "$SCENARIOS/no_policy_repeat" \
    --policy "$SCENARIOS/low_impact_dynamic_q25" \
    --comparison "$RESULTS/final_comparison/scenario_comparison.csv" \
    --output "$RESULTS/final_comparison"

"$PYTHON" "$ROOT/evaluate_low_impact_acceptance.py" \
    --comparison "$RESULTS/final_comparison/scenario_comparison.csv" \
    --baseline-results "$SCENARIOS/baseline/job_results.csv" \
    --policy-results "$SCENARIOS/low_impact_dynamic_q25/job_results.csv" \
    --manifest "$MANIFEST" \
    --policy-name low_impact_dynamic_q25 \
    --output "$RESULTS/final_acceptance.json"

MPLCONFIGDIR="$REPO_ROOT/.tmp/matplotlib" \
"$PYTHON" "$ROOT/plot_results.py" \
    --scenario-root "$SCENARIOS" \
    --comparison "$RESULTS/final_comparison/scenario_comparison.csv" \
    --carbon "$CARBON" \
    --baseline-validation "$SCENARIOS/baseline/simulation_validation.json" \
    --output "$RESULTS/figures"

MPLCONFIGDIR="$REPO_ROOT/.tmp/matplotlib" \
"$PYTHON" "$ROOT/build_low_impact_final_report.py" \
    --comparison "$RESULTS/final_comparison/scenario_comparison.csv" \
    --acceptance "$RESULTS/final_acceptance.json" \
    --policy-results "$SCENARIOS/low_impact_dynamic_q25/job_results.csv" \
    --output "$RESULTS/final_report"

printf 'Final holdout outputs: %s\n' "$RESULTS"
