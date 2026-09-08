# Energy and Carbon Monitoring in HPC Systems

A dissertation project archive exploring how HPC job scheduling information can support estimates of energy use and carbon emissions. The repository combines a Slurm simulator, a Stanage-style cluster configuration, a synthetic workload generator, per-job analysis, and research notes.

The implemented workflow is:

```text
synthetic job profiles + simulator events
                  ↓
        Slurm scheduling simulation
                  ↓
   controller log + workload_profile.csv
                  ↓
   per-job energy and carbon estimates
```

The workload is synthetic and its energy/carbon values are model estimates. The archived run does not contain physical power-meter measurements from Stanage.

## Main project materials

| Path | Purpose |
| --- | --- |
| `slurm_simulator/stanage-sim/` | Cluster configuration, replay events, users, and workload profiles |
| `slurm_simulator/workload-gen/generate_workload.py` | Synthetic mixed-workload generation |
| `slurm_simulator/workload-gen/analyze_run.py` | Join scheduling logs to profiles and estimate per-job energy/carbon |
| `slurm_simulator/run-stanage.sh` | Colima/Docker launcher for workload, verification, and realistic modes |
| `slurm_simulator/Dockerfile.local-sim` | Linux simulator build environment |
| `slurm_simulator/results/` | Small exported results retained for research discussion |
| `md文件/` | Architecture, workload, carbon-model, and research notes |
| `论文/` | Literature collection and indexes |

The simulator source tree also contains the underlying Slurm code and its upstream documentation. See [scope, model, and reproduction notes](docs/MODEL_AND_REPRODUCTION.md) for the distinction between project configuration/analysis and the bundled simulator.

## Archived run: 30 July 2026

The saved summary in [realistic_20260730](slurm_simulator/results/realistic_20260730/run_summary_realistic_20260730.txt) records:

| Measure | Saved value |
| --- | --- |
| Synthetic jobs in profile | 140 |
| Finished jobs | 140 |
| Timeout jobs | 0 |
| Estimated facility-adjusted energy | 132.7 kWh |
| Estimated carbon emissions | 19.9 kg CO2 |
| PUE assumption | 1.20 |
| Carbon-intensity assumption | 150 gCO2/kWh |

[Per-job results](slurm_simulator/results/realistic_20260730/job_results.csv) are included. These are the existing archived outputs; they were not regenerated during documentation maintenance. Job runtimes in this demonstration are capped and failure/timeout outcomes are intentionally avoided by the workload generator.

## Inspect or reproduce the workflow

The two analysis/generation scripts use Python's standard library. Their options can be inspected without launching the simulator:

```bash
python3 slurm_simulator/workload-gen/generate_workload.py --help
python3 slurm_simulator/workload-gen/analyze_run.py --help
```

A fresh checkout has a known build prerequisite missing: `Dockerfile.local-sim` copies `demo-sim/`, but that directory is ignored by Git and is not present in this archive. As committed, the Docker build cannot complete until that configuration is restored or the Dockerfile is deliberately adapted. The full simulator launcher should therefore not be treated as a verified one-command setup.

The [reproduction notes](docs/MODEL_AND_REPRODUCTION.md) describe the existing launcher, expected files, analysis command, and model assumptions so the missing prerequisite can be addressed explicitly.

## Interpretation

The analysis estimates power from allocated CPU cores and GPUs using profile coefficients, multiplies by observed simulated runtime and PUE, then applies a fixed carbon intensity. It does not directly measure utilization, idle energy, memory/network/storage power, or changing grid carbon intensity.

The Stanage-style configuration is a project snapshot; it is not a claim that the simulator reproduces every operational property of the live university cluster.

## Attribution and archive scope

The dissertation archive is maintained by **Yongjiang Liu**. The bundled Slurm distribution retains its [README](slurm_simulator/README.rst), [COPYING](slurm_simulator/COPYING), [AUTHORS](slurm_simulator/AUTHORS), and other upstream notices. The full Slurm source tree is not original dissertation code.

Literature files retain their authorship and publication rights; inclusion here does not grant a new license over those works. Private supervisor/email materials, credentials, build outputs, and large local logs are excluded by the existing `.gitignore`. No new project-wide license is asserted by this README.
