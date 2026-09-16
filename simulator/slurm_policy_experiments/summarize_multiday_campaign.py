#!/usr/bin/env python3
"""Summarize paired daily carbon, service costs and empirical replay variation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from multiday_common import CAMPAIGN, save_json

MODELS = ['capacity_weighted', 'node_request_upper', 'allocated_node_distinct']


def reduction(base, policy):
    return 100 * (base - policy) / base if base > 0 else math.nan


def aggregate_days(frame: pd.DataFrame, expected_days: int, criteria: dict) -> dict:
    values = frame['reduction_pct_capacity_weighted']
    noise_wins = int(frame['exceeds_noise_and_floor'].sum())
    leave_one_out = [values.drop(i).mean() for i in values.index] if len(values) > 1 else []
    pooled = reduction(frame['baseline_carbon_kg_capacity_weighted'].sum(), frame['policy_carbon_kg_capacity_weighted'].sum())
    full = len(frame) == expected_days and bool(frame['valid'].all())
    stable = full and noise_wins >= criteria['positive_days_required'] and values.min() >= -criteria['worst_day_carbon_increase_limit_pct'] and pooled > 0 and min(leave_one_out) > 0
    if not full:
        label = 'incomplete_or_invalid'
    elif frame['delayed_jobs'].sum() == 0:
        label = 'no_action'
    elif stable:
        label = 'directionally_stable_on_tested_days'
    elif (values > 0).any() and (values < 0).any():
        label = 'mixed_direction'
    elif noise_wins == 0:
        label = 'no_repeated_benefit_above_noise'
    else:
        label = 'positive_but_not_stable_by_frozen_rule'
    return {
        'days': len(frame), 'expected_days': expected_days, 'valid_days': int(frame['valid'].sum()),
        'mean_reduction_pct': values.mean(), 'median_reduction_pct': values.median(),
        'sd_reduction_pp': values.std(ddof=1), 'iqr_reduction_pp': values.quantile(.75) - values.quantile(.25),
        'worst_reduction_pct': values.min(), 'best_reduction_pct': values.max(),
        'positive_days': int(values.gt(0).sum()), 'days_above_noise_and_floor': noise_wins,
        'pooled_carbon_weighted_reduction_pct': pooled,
        'sum_paired_reduction_kg': frame['reduction_kg_capacity_weighted'].sum(),
        'leave_one_out_min_mean_pct': min(leave_one_out) if leave_one_out else math.nan,
        'mean_whole_cage_reduction_pct': frame['whole_reduction_pct'].mean(),
        'pooled_whole_cage_reduction_pct': reduction(frame['baseline_whole_carbon_kg'].sum(), frame['policy_whole_carbon_kg'].sum()),
        'total_added_policy_wait_h': frame['policy_delay_h'].sum(),
        'mean_added_policy_wait_per_job_s': 3600 * frame['policy_delay_h'].sum() / frame['evaluation_jobs'].sum(),
        'median_daily_p95_wait_change_s': frame['p95_wait_change_s'].median(),
        'worst_daily_p95_wait_change_s': frame['p95_wait_change_s'].max(),
        'low_impact_days': int(frame['low_impact_screen'].sum()),
        'max_top_user_delay_share': frame['top_user_delay_share'].max(),
        'days_all_carbon_models_positive': int(frame['all_models_positive'].sum()),
        'stability_label': label,
    }


def extra_metrics(path: Path):
    jobs = pd.read_csv(path / 'job_results.csv')
    jobs = jobs.loc[jobs['is_evaluation']].copy()
    delay = jobs['sim_policy_delay_s'].fillna(0)
    users = jobs.assign(delay=delay).groupby('user_id').agg(total=('delay', 'sum'), count=('delay', lambda x: (x > 0).sum()))
    return {
        'policy_delay_h': delay.sum() / 3600,
        'delayed_jobs': int(delay.gt(0).sum()),
        'evaluation_jobs': len(jobs),
        'top_user_delay_share': users['total'].max() / delay.sum() if delay.sum() else 0,
        'max_user_delay_minutes': users['total'].max() / 60,
        'max_user_delayed_jobs': users['count'].max(),
        'delayed_beyond_runtime': int(delay.gt(jobs['runtime_s'] + 1e-6).sum()),
    }


def main():
    manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
    state_path = CAMPAIGN / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'not_started', 'dates': {}}
    rows, controls, counts = [], [], []
    output = CAMPAIGN / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    criteria = manifest['stability']
    for date in manifest['dates']:
        directory = CAMPAIGN / 'days' / date
        if (directory / 'prepared.json').exists():
            counts.append(json.loads((directory / 'prepared.json').read_text()))
        path = directory / 'comparison/scenario_comparison.csv'
        if not path.exists():
            continue
        comparison = pd.read_csv(path).set_index('scenario')
        bases = comparison.loc[comparison.index.str.startswith('baseline')]
        if len(bases) != manifest['baseline_runs_per_day']:
            continue
        baseline = bases.median(numeric_only=True)
        noise = {}
        for model in MODELS:
            column = 'dynamic_carbon_kg_' + model
            noise[model] = 100 * (bases[column].max() - bases[column].min()) / baseline[column] if baseline[column] > 0 else math.inf
        controls.append({'date': date, 'baseline_repeats': len(bases), **{f'noise_pct_{m}': noise[m] for m in MODELS}, 'p95_wait_range_s': bases['total_user_wait_p95_s'].max() - bases['total_user_wait_p95_s'].min()})
        logic_path = directory / 'comparison/experiment_logic_audit.json'
        logic = json.loads(logic_path.read_text()) if logic_path.exists() else {}
        audits = {x['scenario']: x['pass'] for x in logic.get('policies', [])}
        carries = json.loads((directory / 'comparison/carry_in_audits.json').read_text()) if (directory / 'comparison/carry_in_audits.json').exists() else {}
        control_valid = all(all(x['passed'] for x in carries.get(n, [{'passed': False}])) for n in bases.index)
        for spec in manifest['policies']:
            name = spec['name']
            if name not in comparison.index:
                continue
            selected = [name]
            if name == 'adaptive_balanced' and 'adaptive_balanced_repeat' in comparison.index:
                selected.append('adaptive_balanced_repeat')
            repeats = comparison.loc[selected]
            policy = repeats.median(numeric_only=True)
            extra = extra_metrics(directory / name)
            row = {'date': date, 'strategy': name, 'information': spec['information'], **extra}
            row['valid'] = control_valid and all(audits.get(n, False) and all(x['passed'] for x in carries.get(n, [{'passed': False}])) for n in selected)
            row['policy_repeats'] = len(repeats)
            for model in MODELS:
                col = 'dynamic_carbon_kg_' + model
                row['baseline_carbon_kg_' + model] = baseline[col]
                row['policy_carbon_kg_' + model] = policy[col]
                row['reduction_kg_' + model] = baseline[col] - policy[col]
                row['reduction_pct_' + model] = reduction(baseline[col], policy[col])
                row['conservative_reduction_pct_' + model] = reduction(bases[col].min(), repeats[col].max())
            cap = 'dynamic_carbon_kg_capacity_weighted'
            adaptive_range = 100 * (repeats[cap].max() - repeats[cap].min()) / baseline[cap]
            row['noise_threshold_pct'] = max(criteria['material_saving_floor_pct'], noise['capacity_weighted'], adaptive_range)
            row['exceeds_noise_and_floor'] = row['reduction_pct_capacity_weighted'] > row['noise_threshold_pct'] and row['conservative_reduction_pct_capacity_weighted'] > 0
            whole = 'whole_carbon_kg_capacity_weighted'
            row.update(baseline_whole_carbon_kg=baseline[whole], policy_whole_carbon_kg=policy[whole], whole_reduction_pct=reduction(baseline[whole], policy[whole]))
            row['p95_wait_change_s'] = policy['total_user_wait_p95_s'] - baseline['total_user_wait_p95_s']
            row['p99_wait_change_s'] = policy['total_user_wait_p99_s'] - baseline['total_user_wait_p99_s']
            row['p95_scheduler_wait_change_s'] = policy['scheduler_wait_p95_s'] - baseline['scheduler_wait_p95_s']
            row['p95_slowdown_change'] = policy['bounded_slowdown_p95'] - baseline['bounded_slowdown_p95']
            row['all_models_positive'] = all(row['reduction_pct_' + model] > 0 for model in MODELS)
            row['low_impact_screen'] = (
                row['p95_wait_change_s'] <= criteria['service_p95_increase_limit_s']
                and row['delayed_beyond_runtime'] <= criteria['service_delayed_beyond_runtime_limit']
                and row['top_user_delay_share'] <= criteria['fairness_top_user_delay_share_limit']
                and row['max_user_delay_minutes'] <= criteria['fairness_max_user_delay_minutes']
                and row['max_user_delayed_jobs'] <= criteria['fairness_max_delayed_jobs_per_user']
            )
            rows.append(row)
    if counts:
        pd.DataFrame(counts).to_csv(output / 'daily_workload_counts.csv', index=False)
    records = []
    frame = pd.DataFrame(rows)
    if rows:
        frame.to_csv(output / 'daily_strategy_results.csv', index=False)
        pd.DataFrame(controls).to_csv(output / 'baseline_repeat_noise.csv', index=False)
        for name, group in frame.groupby('strategy', sort=False):
            records.append({'strategy': name, **aggregate_days(group, len(manifest['dates']), criteria)})
        pd.DataFrame(records).to_csv(output / 'strategy_stability_summary.csv', index=False)
        plots(frame, pd.DataFrame(records), output)
    missing = [{'date': d, 'strategy': p['name']} for d in manifest['dates'] for p in manifest['policies'] if not any(r['date'] == d and r['strategy'] == p['name'] for r in rows)]
    save_json(output / 'summary.json', {'status': state.get('status'), 'strategies': records, 'missing_comparisons': missing, 'criteria': criteria})
    lines = ['# 多日策略比较', '', f"批次状态：{state.get('status')}。已汇总 {len(rows)}/35 个日期与策略组合。", '',
             '每个日期使用相同作业、资源、初始队列和可延迟作业集合。日期之间保留实际作业数量差异。基线使用三次回放的中位数，Adaptive Balanced使用两次回放的中位数，其余策略每日期一次。', '',
             '本轮是统一回测，碳强度来自历史区域模型序列。History-only仅限制运行时长和负载的信息来源，不代表使用了当时真实发布的碳预测。Oracle策略单独标注。', '']
    if records:
        lines += ['|策略|完整日期|平均减排%|中位数%|最差日%|超过波动与门槛天数|判定|', '|---|---:|---:|---:|---:|---:|---|']
        for r in records:
            lines.append(f"|{r['strategy']}|{r['valid_days']}/5|{r['mean_reduction_pct']:.4f}|{r['median_reduction_pct']:.4f}|{r['worst_reduction_pct']:.4f}|{r['days_above_noise_and_floor']}/5|{r['stability_label']}|")
    lines += ['', '上表减排为容量模型的负载相关部分；完整机柜估算的百分比和绝对kg也保存在CSV中。三个功率代理的结果全部保留。', '',
              '稳定性筛选：5天均完整有效、至少4天减排超过该日基线波动及0.05%门槛，最差日增排不超过0.1%，汇总减排为正，并且去掉任何一天后的平均减排仍为正。门槛是本项目事先声明的筛选规则。', '',
              '用户代价单独报告：P95/P99等待、调度器等待、累计策略延迟、超过运行时长的延迟、单用户最大延迟和负担集中程度。较稳定的减排不自动意味着用户代价可接受。', '',
              '5个日期的初始作业和运行尾部可能重叠，因此不能将这些天当作独立随机样本来宣称统计显著性。均值、离散程度、最差值和删去一天后的结果用于描述稳健程度。累计kg是这些配对窗口的求和，不是年度总减排。', '',
              f'尚缺 {len(missing)} 个组合。失败或未完成日期会保留在状态文件中，验收时应检查它们。']
    (output / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n')


def plots(frame, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = frame.pivot(index='strategy', columns='date', values='reduction_pct_capacity_weighted')
    fig, ax = plt.subplots(figsize=(11, 5.5))
    maximum = max(.05, float(np.nanmax(np.abs(table.values))))
    image = ax.imshow(table.values, cmap='RdYlGn', vmin=-maximum, vmax=maximum, aspect='auto')
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=20)
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f'{value:.3f}%', ha='center', va='center', fontsize=10)
    ax.set_title('Daily dynamic-carbon reduction: capacity model')
    fig.colorbar(image, ax=ax, label='Reduction (%)')
    fig.tight_layout()
    fig.savefig(output / 'daily_carbon_stability.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, group in frame.groupby('strategy'):
        ax.scatter(group['p95_wait_change_s'] / 60, group['reduction_pct_capacity_weighted'], label=name, s=40)
    ax.axhline(0, color='grey', linewidth=.8)
    ax.axvline(0, color='grey', linewidth=.8)
    ax.set(xlabel='Change in P95 total wait (minutes)', ylabel='Dynamic carbon reduction (%)', title='Carbon and waiting across matched days')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(output / 'carbon_wait_tradeoff.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
