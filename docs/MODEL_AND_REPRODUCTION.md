# Model and reproduction notes

These notes describe the repository snapshot and its archived result. They do not report a new simulator execution.

## Project-specific workflow

The main workflow files are the Stanage-style configuration in `slurm_simulator/stanage-sim/`, the launch scripts, and the generator/analyzer under `slurm_simulator/workload-gen/`. The same tree also includes a large Slurm distribution and simulator integration. Upstream licensing and authorship remain applicable to that bundled code.

The generator produces two matching artifacts:

- `stanage-sim/sim.events.realistic`: simulator submission events.
- `stanage-sim/workload_profile.csv`: resource requests, intended runtimes, workload classes, and power coefficients.

The analyzer joins the profile's `job_sim_id` to job IDs parsed from the controller log. Preserve the matching profile and log from each run; mixing files from different runs can invalidate the join.

## Synthetic workload

The generator's default arguments are 140 jobs, a 12-hour submission window, and seed 42. Classes include short HTC jobs, MPI physics/capability jobs, A100 training/development jobs, H100 NVL jobs, and large-memory jobs.

The code uses synthetic user names, weighted job classes, sampled arrivals, and sampled resource requirements. No real user jobs are executed: the simulator replays scheduler events.

For this presentation-oriented workload, actual runtimes are clipped to 30–1,800 seconds and requested walltimes are chosen to avoid time-limit termination. The resulting 140/140 completion record demonstrates that configured scenario; it is not a measured reliability result for a real cluster or a realistic estimate of production failure rates.

To regenerate the input files, from the repository root:

```bash
python3 slurm_simulator/workload-gen/generate_workload.py --hours 12 --jobs 140 --seed 42
```

This overwrites the two files listed above. Preserve a run-specific copy when conducting new experiments.

## Runtime environment and missing input

`run-stanage.sh` is oriented toward a macOS host with Colima and the Docker CLI. It starts a Colima VM when necessary, builds an Ubuntu 24.04 image, and runs the simulator in a container. The installer at `install_linux_env.command` installs Homebrew if needed, installs Colima/Docker, and starts the VM; it changes the host environment.

The current fresh checkout cannot build that image because `Dockerfile.local-sim` contains:

```dockerfile
COPY demo-sim/ /opt/slurm-sim/etc/
```

`demo-sim/` is excluded by `.gitignore` and is absent from the committed tree. Restore the intended demo configuration or make and verify an explicit build change before attempting the launcher. This documentation update leaves the source and configuration unchanged.

After that prerequisite is resolved, the existing launcher interface is:

```bash
cd slurm_simulator
./run-stanage.sh verify
./run-stanage.sh realistic
```

`verify` selects `sim.events.verify`. `realistic` selects `sim.events.realistic`, uses `ClockScaling=3000`, and has a 2,100-second host timeout. A run may end earlier when all simulated jobs complete.

The launcher replaces previous controller/scheduler logs, temporarily edits `sim.conf`, and restores its backup on the normal execution path. It does not install a trap to guarantee restoration after every interruption. It also suppresses container exit failures, so inspect the `All done` marker and per-job states instead of treating the shell exit code as proof of success.

The existing `run-sim.sh` demo launcher also depends on the missing `demo-sim/` configuration.

## Analysis command and outputs

After a compatible controller log exists, analysis can be run independently:

```bash
python3 slurm_simulator/workload-gen/analyze_run.py \
  --log slurm_simulator/stanage-sim/slurmctld.log \
  --profile slurm_simulator/stanage-sim/workload_profile.csv \
  --out slurm_simulator/stanage-sim/job_results.csv \
  --pue 1.2 \
  --ci 150
```

The CSV includes job class, user, partition, allocated cores/GPUs, submission/start/end timestamps, wait time, runtime, energy, carbon, state, and nodes. The analyzer also prints aggregate and per-class summaries.

The archived `results/realistic_20260730/` directory contains a summary and per-job CSV. The full controller log is ignored and is not supplied with those result files, so those two outputs alone cannot reproduce the complete log-to-result transformation.

## Energy and carbon equations

For a job with a parsed start and completion time:

```text
power_W = cores × cpu_W_per_core + GPUs × GPU_W
energy_kWh = power_W × runtime_hours / 1000 × PUE
carbon_kg = energy_kWh × carbon_intensity_g_per_kWh / 1000
```

Power coefficients in the generator are assumptions based on component-power approximations: 7.8 W/core for standard and large-memory partitions, 6.0 W/core for GPU host CPUs, and GPU coefficients of 400 W (A100), 350 W (H100), or 400 W (H100 NVL). They are not measurements taken from the replayed jobs.

PUE and carbon intensity default to 1.2 and 150 gCO2/kWh and can be overridden. Energy reported by this formula already includes PUE. Carbon is calculated from that facility-adjusted energy.

Jobs without a parsed completion time do not receive an energy estimate and do not contribute to the computed energy total. A partial run's total is therefore not an estimate for the entire submitted workload.

## Limits of the archived evidence

- The model uses allocated resources and fixed power coefficients, without utilization sampling.
- CPU/GPU active power is represented; idle infrastructure and memory, storage, and network components are not modeled separately.
- PUE and carbon intensity are constants during a run.
- Synthetic workload distributions and runtime caps affect scheduling and energy results.
- Recorded zero/near-zero waits in this run do not establish scheduling performance under production load.
- The bundled simulator and current cluster configuration require their own validation before claims about real Stanage operations.

Future experiments can retain versioned input profiles, full logs in an appropriate archive, software/environment details, and the exact model coefficients alongside each exported result.
