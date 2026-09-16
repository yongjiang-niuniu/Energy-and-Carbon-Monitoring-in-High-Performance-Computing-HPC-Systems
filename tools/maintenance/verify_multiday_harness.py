"""Exercise the real simulator and report writer outside the formal campaign."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'simulator/slurm_policy_experiments'
sys.path.insert(0, str(EXP))
import generate_empirical_workload as generator
import summarize_multiday_campaign as summary
from multiday_common import assert_profile_match, model_for_date

WORK = REPO / 'work_logs/maintenance_20260908/harness_check'
WORK.mkdir(parents=True, exist_ok=True)
old = EXP / 'scenarios/untouched_24h_20260902/baseline'
profile = pd.read_csv(old / 'workload_profile.csv').head(8).copy()
profile['runtime_s'] = 60
profile['timelimit_min'] = 10
profile['release_dt_s'] = range(0, 80, 10)
profile['eligible_dt_s'] = profile['release_dt_s']
profile['submit_dt_s'] = profile['release_dt_s']
profile['is_warmup'] = False
profile['is_evaluation'] = True
profile['carry_in_type'] = 'evaluation'
profile['policy_delay_s'] = 0
profile['flexibility_score'] = [0.1, 0.2, 0.4, 0.5, 0.1, 0.2, 0.4, 0.5]
generator.write_outputs(WORK / 'smoke/baseline', profile, {'scope': 'Harness smoke only; shortened artificial runtimes; excluded from research results'})
with (WORK / 'smoke.log').open('w') as log:
    subprocess.run([str(EXP / 'run_scenario.sh'), str(WORK / 'smoke/baseline'), '120'], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(EXP / 'analyze_simulation.py'), str(WORK / 'smoke/baseline')], stdout=log, stderr=subprocess.STDOUT, check=True)
validation = json.loads((WORK / 'smoke/baseline/simulation_validation.json').read_text())
assert validation['strict_pass'] and validation['completed_jobs'] == 8

manifest = json.loads((EXP / 'campaigns/multiday_20260908/manifest.json').read_text())
for spec in manifest['policies']:
    target = WORK / 'smoke' / spec['name']
    command = [sys.executable, str(EXP / spec['generator']), '--baseline-profile', str(WORK / 'smoke/baseline/workload_profile.csv'),
               '--carbon', str(EXP / 'data/neso_south_yorkshire_2025-11-11_to_25.csv'), '--output', str(target)]
    if spec.get('strategy'):
        command += ['--strategy', spec['strategy']]
    for key, value in spec['parameters'].items():
        command += ['--' + key.replace('_', '-'), str(value)]
    if spec.get('uses_history'):
        command += ['--history-model-dir', str(model_for_date('2025-11-13'))]
    if spec.get('uses_baseline_schedule'):
        command += ['--schedule-forecast', str(WORK / 'smoke/baseline/job_results.csv')]
    with (WORK / 'generator_checks.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=REPO)
    assert_profile_match(profile, pd.read_csv(target / 'workload_profile.csv'))

# Use copies of the old December outputs to check schema compatibility, not as new observations.
fixture = WORK / 'report_fixture'
fixture.mkdir(exist_ok=True)
manifest = json.loads((EXP / 'campaigns/multiday_20260908/manifest.json').read_text())
manifest['dates'] = ['2025-12-08']
fixture.joinpath('manifest.json').write_text(json.dumps(manifest))
directory = fixture / 'days/2025-12-08'
directory.mkdir(parents=True, exist_ok=True)
comparison = pd.read_csv(EXP / 'results/low_impact_dynamic_dec08_20260902/final_comparison/scenario_comparison.csv')
comparison = comparison.loc[comparison['scenario'].ne('no_policy_repeat')]
copies = []
for name in ['baseline_repeat_1', 'baseline_repeat_2']:
    copied = comparison.loc[comparison['scenario'].eq('baseline')].copy()
    copied['scenario'] = name
    copies.append(copied)
target = directory / 'comparison'
target.mkdir(exist_ok=True)
pd.concat([comparison, *copies]).to_csv(target / 'scenario_comparison.csv', index=False)
for spec in manifest['policies']:
    source = EXP / 'scenarios/low_impact_dynamic_dec08_20260902' / spec['name'] / 'job_results.csv'
    if source.exists():
        (directory / spec['name']).mkdir(exist_ok=True)
        shutil.copy2(source, directory / spec['name'] / 'job_results.csv')
summary.CAMPAIGN = fixture
summary.main()
result = json.loads((fixture / 'summary/summary.json').read_text())
assert len(result['strategies']) == 1
assert all(x['stability_label'] == 'incomplete_or_invalid' for x in result['strategies'])
assert (fixture / 'summary/daily_carbon_stability.png').exists()
print(json.dumps({'simulator_smoke_completed_jobs': 8, 'policy_generators_checked': 7, 'report_writer_checked_strategies': 1, 'scope': 'technical check only; excluded from formal campaign'}))
