# Reproduction notes

The submitted code is preserved without algorithm changes. This is a source-code release, not a self-contained data/results bundle.

## Inputs and environment

1. Use authorised anonymised Stanage accounting data with `stanage_eda/scripts/analyse_stanage.py` and `simulator/slurm_policy_experiments/build_exact_24h_workload.py`.
2. Obtain the regional series with `simulator/slurm_policy_experiments/download_neso_carbon.py`.
3. Regenerate history-based tables with `build_history_quantile_model.py` when needed; trained tables are not part of the submission.
4. Configure the Linux Docker simulator. Historical launchers contain local path and Colima-context assumptions; inspect them before execution. In particular, `prepare_multiday_campaign.py` and the submitted manifest identify local dependencies.

Use each script's `--help` to inspect its inputs. Unit tests use synthetic fixtures and do not establish the dissertation's benchmark results. No research experiments were rerun during publication.

## Verification

The university-submitted PDF and code-package hashes were checked against local submission receipts. The paper is 99 pages, SHA-256 `28835c679be4785e67ed86f8f383c5b3102db9017e1c5cb1beb2465733320b97`. The code ZIP contains 2,660 files, SHA-256 `d666512c07e9b79f14ecb1fa29b4a78d8e35b7b694efe7a25728d51eae659d7b`.

Pre-submission checks on 16 September passed 159 tests and 5 subtests, parsed 98 Python files and checked 11 notebook code cells. The publication check also passed **159 tests and 5 subtests** after extraction (`python -m pytest -q -p no:cacheprovider`). Neither check rebuilds the full simulator or reruns the 55 experiments.

The paper source retains the final prose, figures and references byte for byte. Redundant authoring logs under the ZIP's `supporting/` directory are omitted. Earlier drafts are removed from the current repository tree; Git history remains available.

[Back to the project](../README.md)
