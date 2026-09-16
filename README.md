# Carbon-Aware HPC Scheduling — Stanage Case Study

**[Read the submitted dissertation](Dissertation.pdf)** · **[LaTeX source](latex/main.tex)** · **[Reproduction notes](docs/REPRODUCIBILITY.md)**

MSc Advanced Computer Science dissertation by **Yongjiang Liu**, University of Sheffield. The 99-page dissertation and final code package were submitted on **16 September 2026**. This repository preserves that completed submission; the paper and submitted program files are unchanged.

**中文概述：** 谢菲尔德大学高级计算机科学硕士毕业项目，研究 HPC 任务的碳感知调度及其等待时间、公平性和模型不确定性。论文和代码均已于 2026 年 9 月 16 日提交。根目录保留唯一最终论文，`latex/` 存放对应源码，旧论文草稿和早期演示代码已从当前版本移除。

## What the project does

- Analyses anonymised 2025 Stanage Slurm accounting records and constructs trace-driven workloads with running/queued carry-in state.
- Compares admission policies using the UBC CR Slurm Simulator and regional NESO carbon-intensity series.
- Evaluates modelled carbon, waiting time, bounded slowdown and per-user delay, alongside power-model and timing sensitivity.
- Includes a separate bounded feedback-policy engineering extension, documented in the dissertation.

The main study contains **55 completed replays over five dates and seven representative policies**. No policy meets all frozen stability conditions. Carbon values are counterfactual model estimates, not measured electricity savings; the whole-cage power scenario is not per-job meter data. The 64-job engineering pilot is separate from the main comparison.

## Repository layout

| Path | Contents |
| --- | --- |
| [Dissertation.pdf](Dissertation.pdf) | Exact submitted 99-page dissertation |
| [latex/](latex) | Final paper's main file, chapters, references and figures |
| [simulator/](simulator) | Policy/replay/analysis programs and required upstream Slurm Simulator build source |
| [stanage_eda/](stanage_eda) | Accounting-data analysis, notebook and tests |
| [tools/](tools) | Submitted maintenance and validation programs |
| [docs/](docs) | Reproduction limits and submission integrity |

The internal code paths are preserved because the submitted launchers and tests depend on them. The larger vendored simulator tree is third-party build source, not a collection of obsolete project versions.

## Install and check

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Python 3.10+ is required. Building the Linux simulator additionally requires Docker:

```bash
make build-simulator
```

The simulator image is not needed for the unit tests. Data-dependent analyses require authorised Stanage accounting data and carbon series; the code submission intentionally excludes raw accounting records, credentials, generated workloads and full experiment logs. See [reproduction notes](docs/REPRODUCIBILITY.md) before attempting a replay.

## Paper source and provenance

Compile `latex/main.tex` with a LaTeX distribution providing the packages declared there, BibTeX and the IEEEtran bibliography style, for example `cd latex && latexmk -pdf main.tex`. The published PDF is the exact submitted file, not a replacement compilation.

The uploaded code is the extracted final submission package: **2,660 files**, verified against its archived ZIP. The original source-package manifest remains in [submission_manifest.json](submission_manifest.json). [Submission integrity](docs/submission.json) records the final PDF and package checksums.

The Slurm Simulator and bundled libraries retain their original licenses and notices. Their implementation is not claimed as original dissertation code. The submitted snapshot includes export-specific build-file adjustments and a code-only EDA notebook, as recorded in the original manifest.
