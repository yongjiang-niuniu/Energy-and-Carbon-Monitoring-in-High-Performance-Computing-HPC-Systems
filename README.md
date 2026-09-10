# Energy and Carbon Monitoring in HPC Systems

A research workflow for connecting **HPC scheduling events to per-job energy and carbon estimates**. It creates synthetic workloads, replays them through a Slurm simulator configured around Stanage-style resources, and joins scheduling logs with CPU/GPU power assumptions to produce analysis tables.

**中文概述：** 这个项目研究如何从高性能计算任务的调度记录估算能耗与碳排放，包含合成工作负载、Slurm 模拟配置、任务级分析和研究笔记。当前仓库保存研究阶段的模拟材料，能耗为模型估计值，毕业论文最终稿尚未作为正式提交归档。

**Dissertation reading copy:** the [10 September 2026 working draft](dissertation/README.md) contains the later trace-driven study's results and conclusions. This repository's code remains the earlier synthetic-workload snapshot described below; it is not the implementation version used for all experiments in that manuscript. The PDF is an Overleaf working draft, not a verified official university submission.

## Project at a glance

| Aspect | Details |
| --- | --- |
| Project type | HPC research and simulation archive |
| Research context | University of Sheffield, COM6906 dissertation work in progress |
| Technologies | Python standard library, Slurm simulator, shell, Docker/Colima |
| Inputs | Synthetic job profiles, cluster configuration, simulator controller logs |
| Outputs | Per-job scheduling, energy and carbon CSVs; aggregate research summaries |
| Preserved demonstration | 140-job simulation export dated 30 July 2026 |
| Dissertation | 92-page Overleaf working draft, 10 September 2026; later study, separate from the archived code version |

## What it does

- Generates mixed CPU, GPU and memory-intensive job profiles with matching simulator submission events.
- Describes a Stanage-style collection of partitions and resource types for scheduling experiments.
- Parses job submission, start and completion events from a simulator controller log.
- Estimates energy from allocated resources, runtime and power coefficients, then applies PUE and carbon intensity.
- Exports task-level results and summaries that can be examined alongside the workload and modelling assumptions.

```text
synthetic job profiles + replay events
                  ↓
        Slurm scheduling simulation
                  ↓
       controller log + job profiles
                  ↓
      scheduling, energy and carbon CSVs
```

## Repository guide

| Location | Role |
| --- | --- |
| [stanage-sim/](slurm_simulator/stanage-sim/) | Cluster configuration, users, replay events and workload profiles |
| [generate_workload.py](slurm_simulator/workload-gen/generate_workload.py) | Generate paired synthetic profiles and replay events |
| [analyze_run.py](slurm_simulator/workload-gen/analyze_run.py) | Join logs to profiles and calculate job-level estimates |
| [run-stanage.sh](slurm_simulator/run-stanage.sh) | Existing simulator launcher with workload, verification and realistic modes |
| [Dockerfile.local-sim](slurm_simulator/Dockerfile.local-sim) | Linux simulator image build definition |
| [results/](slurm_simulator/results/) | Preserved small simulation exports |
| [Research notes](md文件/) | Architecture, workload and carbon-model discussion |
| [Literature index](论文/文献索引.md) | Reading list and literature context |
| [Documentation index](docs/README.md) | Suggested reading order and reproduction guidance |
| [Dissertation reading copy](dissertation/README.md) | Later manuscript, version boundary and PDF integrity information |

Project configuration and analysis live inside a tree that also contains the underlying Slurm distribution. The complete bundled scheduler is not original dissertation code.

## Getting started

Clone the project and inspect the lightweight analysis interface:

```bash
git clone https://github.com/yongjiang-niuniu/Energy-and-Carbon-Monitoring-in-High-Performance-Computing-HPC-Systems.git hpc-energy-carbon
cd hpc-energy-carbon
python3 slurm_simulator/workload-gen/generate_workload.py --help
python3 slurm_simulator/workload-gen/analyze_run.py --help
```

The generator and analyzer use Python's standard library. Start by reading the [saved summary](slurm_simulator/results/realistic_20260730/run_summary_realistic_20260730.txt) and [per-job CSV](slurm_simulator/results/realistic_20260730/job_results.csv); this does not require a simulator or container runtime.

For a new simulation, follow [model and reproduction notes](docs/MODEL_AND_REPRODUCTION.md). The existing launcher expects macOS, Colima and Docker. Its image build currently references an uncommitted `demo-sim/` directory, so a fresh checkout needs that configuration restored or an explicitly validated build adaptation before the complete launcher can run. The guide explains the expected inputs, run modes and analysis command.

## Model and design

Each synthetic job profile supplies its resource allocation and power coefficients. The analyzer associates a completed simulated job with the matching profile and calculates:

```text
power_W    = allocated_CPU_cores × W_per_core + allocated_GPUs × W_per_GPU
energy_kWh = power_W × runtime_hours / 1000 × PUE
carbon_kg  = energy_kWh × carbon_intensity_g_per_kWh / 1000
```

The profile and controller log must belong to the same run. Jobs without a parsed completion time do not contribute an energy estimate. See the [model guide](docs/MODEL_AND_REPRODUCTION.md) for coefficients, assumptions, workload generation and interpretation of partial runs.

## Results and verification

The preserved 30 July 2026 demonstration reports:

| Measure | Historical saved value |
| --- | --- |
| Generated / completed jobs | 140 / 140 |
| Timeout jobs | 0 |
| Facility-adjusted energy estimate | 132.7 kWh |
| Carbon estimate | 19.9 kg CO2 |
| PUE / carbon intensity | 1.20 / 150 gCO2 per kWh |

These figures are existing simulation outputs. Documentation maintenance checks the committed file layout and command interfaces; it does not claim a new simulator run or physical energy measurements. Runtime caps and the workload generator intentionally avoid timeout outcomes, so the completion count is a property of this demonstration.

## Limitations and next steps

- The model uses allocated resources and fixed power coefficients rather than measured utilization or power-meter readings.
- Idle energy and memory, network and storage power are not modelled separately; PUE and carbon intensity are constant within a run.
- The complete controller log for the saved export is not included, and the container build has the missing configuration prerequisite described above.
- The configuration is a research snapshot, with no claim to reproduce every property of the live Stanage cluster.
- Future runs should preserve their matching profiles, complete logs, environment and coefficients before comparing scheduling or carbon results.

## Attribution and provenance

Maintained by **Yongjiang Liu**. The bundled Slurm distribution retains its [README](slurm_simulator/README.rst), [COPYING](slurm_simulator/COPYING) and [AUTHORS](slurm_simulator/AUTHORS). Literature retains its original authorship and publication rights. This project description grants no new license over those works.

The archive preserves the existing research snapshot and its history. It is not the verified final Blackboard dissertation submission. See the [academic project portfolio](https://github.com/yongjiang-niuniu/academic-project-portfolio) for the wider collection.
