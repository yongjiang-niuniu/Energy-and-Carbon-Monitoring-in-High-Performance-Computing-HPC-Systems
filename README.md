# Energy and Carbon Monitoring in HPC Systems

This repository stores the dissertation project materials for monitoring and estimating energy use and carbon emissions in HPC systems, with Stanage HPC as the main case study.

## Folder Structure

- `md文件/`: project notes, weekly research notes, workload simulation notes, and carbon model notes.
- `论文/`: literature papers and paper indexes. Slurm simulator literature is under `论文/simulator/`.
- `slurm_simulator/`: the Slurm simulator source code, Stanage-style simulator configuration, synthetic workload generation, and analysis scripts.
- `slurm_simulator/results/`: small exported run results for discussion, slides, or dissertation writing.

## Current Simulator Result

The latest realistic Stanage-style workload run is stored in:

- `slurm_simulator/results/realistic_20260730/job_results.csv`
- `slurm_simulator/results/realistic_20260730/run_summary_realistic_20260730.txt`

This run used 140 synthetic mixed HPC jobs and completed all jobs without timeout. The estimated total energy was 132.7 kWh, with 19.9 kg CO2 using PUE 1.20 and carbon intensity 150 gCO2/kWh.

## Notes

FastSim has been removed from this repository. Large local logs, build outputs, local PDF extraction cache, private supervisor notes, and email attachments are ignored by Git.
