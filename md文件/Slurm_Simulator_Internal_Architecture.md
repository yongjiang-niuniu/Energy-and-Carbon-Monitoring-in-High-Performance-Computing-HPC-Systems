# Slurm Simulator Internal Architecture

> Project: Energy and Carbon Monitoring in HPC Systems
> Target: `slurm_simulator/` (ubccr Slurm Simulator, based on Slurm 23.11 source)
> This document covers: how jobs are created, scheduled, and "executed"; how synthetic jobs are generated; and how to configure job parameters.

---

## 1. Overall architecture: real Slurm, tamed

The simulator does not reimplement Slurm. It compiles the actual Slurm source and replaces only the parts that talk to real hardware or the network.

| Real Slurm | Simulator replacement | Source file |
|---|---|---|
| Real clock (`time()`, etc.) | Accelerated virtual clock | `contribs/sim/sim_time.c` |
| Network RPC (slurmctld ↔ slurmd/sbatch) | In-process calls + event queue | `contribs/sim/slurmctld_agent.c`, `src/common/slurm_protocol_api.c` |
| Real job processes (forked by slurmd) | A timer event | `contribs/sim/sim_jobs.c` + event loop |

The scheduler itself (priority, backfill, cons_tres resource selection) is unchanged Slurm code. The simulator path is selected at compile time via `-DSLURM_SIMULATOR`.

Entry wrapper (`contribs/sim/slurmctld_controller.c`):

```c
#define main slurmctld_main
#include "../../src/slurmctld/controller.c"
#undef main

int main(int argc, char **argv)
{
    /* Read sim.conf, users.sim, sim.events; set up virtual clock; call slurmctld */
    ...
    slurmctld_main(slurmctld_argc, slurmctld_argv);
}
```

---

## 2. References and documentation

### 2.1 Core papers

| Paper | Topic |
|---|---|
| Simakov et al., PMBS 2017 (LNCS 10724) | Slurm Simulator implementation and parametric analysis; replays historical or synthetic workloads on real Slurm code |
| Simakov et al., PEARC 2018 | Multi-controller and node sharing on large clusters |
| Lucero (BSC, SLUG 2015) | Early Slurm Workload Simulator; `sim_mgr` + `test.trace` event-driven model |
| Trofinoff & Benini | Further development of Lucero's simulator |

### 2.2 Official resources

| Resource | URL | Purpose |
|---|---|---|
| Slurm Simulator homepage | https://ubccr-slurm-simulator.github.io/ | Overview, paper links |
| User Guide v1.2 | https://ubccr-slurm-simulator.github.io/docs/slurm_sim_manual_v1.2.html | Install, `sim.conf`, R toolchain |
| slurm_sim_tools (GitHub) | https://github.com/ubccr-slurm-simulator/slurm_sim_tools | R package `RSlurmSimTools`: `sim_job()` + `write_trace()` |
| Slurm official docs | https://slurm.schedmd.com/ | sbatch parameter semantics (`-N`, `-n`, `-t`, `--gres`, etc.) |

### 2.3 Additional references for this project

| Source | Used for |
|---|---|
| Feitelson Parallel Workloads Archive | Heavy-tailed runtimes, power-of-two node counts |
| F-DATA (Fugaku workload dataset) | Large MPI jobs dominate node-hours; requested vs actual runtime ratio |
| Stanage cluster docs | 1:1 replica of nodes and partitions in `slurm.conf` |
| `真实HPC_Workload模拟调研.md` | Workload statistics and FastSim comparison |

The official ubccr toolchain generates traces in R (`.trace` format). This project uses Python `generate_workload.py` to write `sim.events` directly. Same idea, different format and statistics tuned for Stanage.

---

## 3. Virtual time system (`sim_time.c`)

The simulator wraps libc time functions at link time (`--wrap`). All Slurm code sees virtual time:

```c
int64_t get_sim_utime()
{
    int64_t cur_real_utime = get_real_utime();
    /* t_sim = t_real + shift + (scale - 1) * t_real */
    int64_t cur_sim_time = cur_real_utime + *sim_timeval_shift
                         + (int64_t)((*sim_timeval_scale - 1.0) * cur_real_utime);
    return cur_sim_time;
}
```

- `scale` and `shift` live in shared memory (`SharedMemoryName` in `sim.conf`), shared across components.
- `ClockScaling=3000` means virtual time runs 3000× faster: a 12-hour submission window finishes in about 15 real minutes.
- **Warning**: keep `ClockScaling` below ~5000. `get_sim_utime()` uses int64 microseconds; higher values overflow and corrupt job IDs (the "year 2683" bug noted in `run-stanage.sh`).
- Wrapped functions: `gettimeofday`, `time`, `sleep`, `usleep`, `nanosleep`.

---

## 4. Configuration files

### 4.1 `sim.conf` (simulator-specific)

| Parameter | Meaning | Stanage example |
|---|---|---|
| `TimeStart` | Virtual start time (seconds) | `0` |
| `TimeStop` | `0` = run forever; `1` = exit after last job completes | `1` |
| `SecondsBeforeFirstJob` | Wait before first job event (seconds) | `1` |
| `ClockScaling` | Virtual/real time ratio | `3000` (realistic mode) |
| `SharedMemoryName` | Shared memory path | `/slurm_sim_stanage.shm` |
| `EventsFile` | Workload event file | `sim.events.realistic` |
| `TimeAfterAllEventsDone` | Extra run time after all events (seconds) | `2` |
| `FirstJobDelay` | Extra delay before first job (microseconds) | `0` |
| `CompJobDelay` | Delay from completion to epilog (microseconds) | `0` |
| `TimeLimitDelay` | Delay from timeout kill to completion event (microseconds) | `0` |

### 4.2 `users.sim` (virtual users)

Format: `username:uid:groupname:gid`, one user per line. The simulator wraps `getpwnam_r` / `getpwuid_r`, so `--uid=alice` resolves to uid 1001.

```
alice:1001:users:100
bob:1002:users:100
...
```

Usernames must be defined in `users.sim` before submission, or sbatch will fail.

### 4.3 `slurm.conf` + `gres.conf` (cluster topology)

Stanage 1:1 replica: 189 public nodes, 6 partitions.

| Partition | Nodes | CPU | Memory | GPU |
|---|---|---|---|---|
| `standard` | node[001-142] | 64c | 251 GB | none |
| `bigmem` | bigmem[01-12] | 64c | 1 TB | none |
| `hugemem` | hugemem[01-12] | 64c | 2 TB | none |
| `gpu` | gpua100[01-13] | 48c | 512 GB | 4× A100 |
| `gpu-h100` | gpuh100[01-06] | 48c | 512 GB | 2× H100 |
| `gpu-h100-nvl` | gpuh100nvl[01-04] | 96c | 512 GB | 4× H100 NVL |

Scheduling: `SchedulerType=sched/backfill`, `SelectType=select/cons_tres`, `SelectTypeParameters=CR_CPU`.

`gres.conf` declares GPU types and counts only. No real GPU device files are needed.

---

## 5. Event system (`sim_events.c/h`)

The simulator is a discrete-event engine (DES) with a doubly linked list sorted by virtual time.

```c
typedef enum {
    SIM_TIME_ZERO = 1001,
    SIM_TIME_INF,
    SIM_NODE_REGISTRATION,
    SIM_SUBMIT_BATCH_JOB,
    SIM_COMPLETE_BATCH_SCRIPT,
    SIM_EPILOG_COMPLETE,
    SIM_CANCEL_JOB,
    SIM_ACCOUNTING_UPDATE,
    SIM_PRIORITY_DECAY,
    SIM_SET_DB_INDEX,
} sim_event_type_t;
```

Events at the same timestamp are ordered by arrival (`<=` comparison). That is what makes runs reproducible.

On startup, a `SIM_NODE_REGISTRATION` event is inserted automatically to bring all configured nodes online.

### 5.1 `sim.events` line format

```
-e submit_batch_job -dt <seconds> | <sbatch argument string>
```

Left of `|`: event metadata. Right of `|`: a nearly complete sbatch command line.

Example:

```
-e submit_batch_job -dt 68 | --uid=alice -J jobid_2 -p gpu --gres=gpu:a100:2 -N 1 -n 12 -t 30 -sim-walltime 1841 pseudo.job -sleep 1841
```

| Field | Meaning |
|---|---|
| `-dt N` | Submit at N seconds after simulation start |
| `--uid=` | Submitting user (must exist in `users.sim`) |
| `-J jobid_N` | **Required** name `jobid_<integer>`; links to the parallel job ledger |
| `-p` | Target partition |
| `-N` | Node count |
| `-n` | Task/core count |
| `--gres=` | GPU resources (e.g. `gpu:a100:2`) |
| `-t` | Requested walltime (minutes); used by scheduler/backfill |
| `-sim-walltime` | Actual runtime (seconds); used to schedule completion |
| `pseudo.job -sleep N` | Placeholder script + redundant sleep (same value as `-sim-walltime`) |

**Hard constraints** (`sim_submit_batch_job_get_payload()` will fatal exit on violation):

1. Job name must be `jobid_<integer>`
2. Do not set `-jid` manually (Slurm JobId is assigned internally)
3. Set `-sim-walltime` and/or `pseudo.job -sleep` (values should match)
4. **No `#` comments** in the file (the parser only treats `#` as a comment at line start with no prior characters; inline `#` breaks parsing)

### 5.2 Requested walltime vs actual runtime

- `-t`: user-requested walltime; backfill uses it to plan the timeline
- `-sim-walltime`: how long the job "runs"; the simulator inserts `SIM_COMPLETE_BATCH_SCRIPT` accordingly

Separating them reproduces the common HPC pattern where requested time is much longer than actual use. If `-sim-walltime > -t`, the real timeout logic fires `REQUEST_KILL_TIMELIMIT` and the job ends in TIMEOUT state.

---

## 6. Main event loop (`slurmctld_controller.c`)

Most background threads in real slurmctld are removed. Their work is folded into a single loop:

```c
void sim_slurmctld_event_main_loop()
{
    _slurmctld_background(NULL);
    while (1) {
        now = get_sim_utime();
        sim_main_thread_sleep_till = now + 1000000;
        while (sim_main_thread_sleep_till > now)
            now = sim_events_loop();
        _slurmctld_background(NULL);
    }
}
```

Each `sim_events_loop()` iteration:

1. Pops and dispatches due events
2. Calls `schedule()` and `_attempt_backfill()` directly (no thread races, reproducible)
3. Drives the slurmdbd agent (accounting queue)
4. Checks exit condition (`TimeStop=1` stops after the last job finishes)

---

## 7. Full job lifecycle

Example: GPU job `jobid_2`.

### Step 1: Creation (virtual time t=68s)

`SIM_SUBMIT_BATCH_JOB` fires → `submit_job()` in `contribs/sim/sim_sbatch.c` (embedded sbatch):

- Parses argv into `job_desc_msg_t` (partition, nodes, GRES, time limit, uid, etc.)
- Resolves uid from fake passwd in `users.sim` (`alice → 1001`)
- Calls real `slurm_submit_batch_job()` via in-process RPC shortcut (`slurm_protocol_api.c`, `#ifdef SLURM_SIMULATOR`)
- Real Slurm creates `job_record`, enters priority queue
- Simulator records in parallel ledger (`sim_jobs.c`): `job_id, walltime=1841s`

### Step 2: Scheduling (unchanged Slurm code)

- `schedule()`: pick job by priority → `select/cons_tres` selects nodes
- If head-of-line blocks, `_attempt_backfill()` uses `-t` to fill gaps with smaller jobs
- Log: `sched: Allocate JobId=2 NodeList=gpu-node001 ...`

### Step 3: "Execution" (the core trick)

- `__wrap_agent_queue_request()` in `slurmctld_agent.c` intercepts `REQUEST_BATCH_JOB_LAUNCH`
- Looks up walltime in `sim_jobs`, inserts `SIM_COMPLETE_BATCH_SCRIPT` at `start + walltime`
- **No process is started.** `pseudo.job` is never read.

### Step 4: Completion and resource release

- `SIM_COMPLETE_BATCH_SCRIPT` → `REQUEST_COMPLETE_BATCH_SCRIPT` → `_job_complete: JobId=2 done`
- `SIM_EPILOG_COMPLETE` → `job_epilog_complete()` → nodes freed, backfill can use them

### Step 5: Timeout path

If `-sim-walltime > -t`: `_slurmctld_background()` timeout check → `REQUEST_KILL_TIMELIMIT` (intercepted by agent) → mark `requested_kill_timelimit` → schedule immediate completion → TIMEOUT state

---

## 8. How synthetic jobs are generated

### 8.1 Why placeholders can be replaced with realistic workloads

The simulator runs no real computation. The scheduler only sees a resource profile:

| Parameter | Role |
|---|---|
| Submit time (`-dt`) | When the job enters the queue |
| Partition (`-p`) | Which hardware pool |
| Nodes (`-N`) | MPI scale |
| Cores (`-n`) | CPU allocation |
| GPU (`--gres`) | GPU allocation |
| Requested walltime (`-t`) | Backfill planning |
| Actual runtime (`-sim-walltime`) | Completion event timing |

If these follow real cluster statistics, queueing, backfill, and resource occupancy are usable activity data. Energy and carbon estimates are applied on top via a power model.

### 8.2 Statistical sources

- **Feitelson Archive**: heavy-tailed runtimes (log-normal), power-of-two node counts, diurnal arrival patterns
- **F-DATA / NERSC / ARCHER2**: many short jobs by count, few large MPI jobs by node-hours; requested walltime is 1.5–10× actual, rounded to 15min/1h/4h/12h/24h slots
- **Failures/timeouts**: ~5–8% fail quickly after start; ~5% of large MPI jobs hit walltime

### 8.3 Generator: `workload-gen/generate_workload.py`

```bash
cd slurm_simulator/workload-gen
python3 generate_workload.py --hours 12 --jobs 140 --seed 42
```

Output:

- `stanage-sim/sim.events.realistic`: 140 event lines (no comments)
- `stanage-sim/workload_profile.csv`: per-job class/user/resources/power parameters

#### Eight job classes and default weights

| Class | Default weight | Partition | Typical scale | Typical duration |
|---|---|---|---|---|
| `htc_short` | 38 | standard | 1 node, 4–32 cores | 10min–3h |
| `mpi_physics` | 22 | standard | 2–32 nodes | 1–2h (capped) |
| `mpi_capability` | 2 | standard | 64–96 nodes | 1–2h |
| `ai_train_a100` | 12 | gpu | 1–2 nodes, 4 GPU/node | 1–2h |
| `ai_dev_gpu` | 13 | gpu / gpu-h100 | 1 node, 1–2 GPU | 2–90min |
| `llm_h100nvl` | 3 | gpu-h100-nvl | 1 node, 4 GPU | 2h (capped) |
| `bigmem` | 6 | bigmem | 1 node | 1–12h |
| `hugemem` | 2 | hugemem | 1 node | 1–12h |

#### Generation algorithm

1. **Arrivals**: non-homogeneous Poisson (thinning); daytime intensity (~09:00–18:00 in the window) is about 3× nighttime
2. **Runtime**: log-normal `lognormal_capped(median, sigma, cap)`; max simulated runtime 2h (`MAX_SIM_WALL_S`)
3. **Requested walltime**: `round_up_request(actual, factor)` rounds to `[15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320]` minutes, and is ≥ actual × factor
4. **User assignment**: 8 users sampled by weight (few users submit most jobs)
5. **Timeout/failure**: ~5% of `mpi_physics` jobs have actual > req (TIMEOUT); ~8% of `htc_short` fail quickly

#### Event line template

```python
"-e submit_batch_job -dt %d | --uid=%s -J jobid_%d -p %s "
"-N %d -n %d%s -t %d -sim-walltime %d pseudo.job -sleep %d\n"
# gres part: " --gres=gpu:a100:4" or empty string
```

### 8.4 Official R toolchain (for comparison)

ubccr's `RSlurmSimTools` provides `sim_job()` and `write_trace()`:

```r
sim_job(
    job_id=1001,
    submit="2016-10-01 00:01:00",
    wclimit=300L,      # maps to -t (seconds)
    duration=600L,     # maps to -sim-walltime (seconds)
    tasks=12L,
    tasks_per_node=12L
)
write_trace("dependency_test.trace", trace)
```

The R tools support more fields (dependency, account, qos, reservation). The Python generator here covers the fields needed for Stanage energy research.

### 8.5 Hand-written sim.events

For small tests, write events directly. See `stanage-sim/sim.events` (7 placeholder jobs across 6 partitions):

```
-e submit_batch_job -dt 0 | --uid=alice -J jobid_1 -p standard -N 8 -n 512 -t 60 -sim-walltime 12 pseudo.job -sleep 12
-e submit_batch_job -dt 1 | --uid=alice -J jobid_4 -p gpu -N 1 -n 12 --gres=gpu:a100:2 -t 60 -sim-walltime 10 pseudo.job -sleep 10
```

---

## 9. Job parameter quick reference

### 9.1 sbatch parameters and their effects

| Parameter | sim.events syntax | What the scheduler sees | Energy estimate use |
|---|---|---|---|
| Partition | `-p gpu` | Node pool constraint | CPU/GPU power model |
| Nodes | `-N 8` | MPI scale, node occupancy | node-hours |
| Cores/tasks | `-n 512` | CPU allocation (cons_tres) | CPU power = cores × W/core |
| GPU | `--gres=gpu:a100:2` | GRES constraint | GPU power = gpus × W/gpu |
| Time limit | `-t 480` (minutes) | Backfill timeline | Not in energy formula directly |
| Actual runtime | `-sim-walltime 7200` (seconds) | Completion event | runtime × power |
| User | `--uid=alice` | Fair-share weight | Per-user aggregation |
| Job name | `-J jobid_42` | Links to sim_jobs ledger | Aligns with workload_profile |

### 9.2 Common sbatch parameters (all supported)

The simulator embeds full sbatch parsing (`sim_sbatch.c` includes `src/sbatch/sbatch.c`). These work in `sim.events`:

- `-p / --partition`
- `-N / --nodes`
- `-n / --ntasks`
- `-c / --cpus-per-task`
- `--gres=`
- `-t / --time` (`D-HH:MM:SS` or minutes)
- `--mem`
- `-A / --account`
- `-q / --qos`
- `--dependency`
- `-D / --chdir` (default `/home/<username>`)

Simulator-specific (not in standard sbatch):

- `-sim-walltime <seconds>`: actual runtime

### 9.3 Partition and resource examples

| Scenario | Example line |
|---|---|
| Standard MPI | `-p standard -N 16 -n 1024 -t 480 -sim-walltime 7200` |
| A100 training | `-p gpu -N 2 -n 96 --gres=gpu:a100:4 -t 480 -sim-walltime 7200` |
| H100 debug | `-p gpu-h100 -N 1 -n 8 --gres=gpu:h100:1 -t 60 -sim-walltime 1393` |
| Large memory | `-p bigmem -N 1 -n 64 -t 240 -sim-walltime 3895` |
| Walltime timeout | `-t 120 -sim-walltime 4431` (actual > req × 60) |

---

## 10. Running and analyzing

### 10.1 Run the simulation

```bash
cd slurm_simulator
./run-stanage.sh              # default: 7 placeholder jobs
./run-stanage.sh realistic    # 140 synthetic jobs (ClockScaling=3000)
./run-stanage.sh verify       # fill every partition with all its nodes
```

`run-stanage.sh` runs slurmctld in Docker, mounting `stanage-sim/` as the config directory. Realistic mode temporarily changes `EventsFile` and `ClockScaling` in `sim.conf`, then restores them.

### 10.2 Analyze results

```bash
python3 workload-gen/analyze_run.py
```

Input: `slurmctld.log` + `workload_profile.csv`
Output: `stanage-sim/job_results.csv`

Per job: submit/start/end times, wait time, energy estimate, carbon emissions.

```
E_job [kWh] = (cores × W_core + gpus × W_gpu) × runtime_h / 1000 × PUE
CO2 [g]     = E_job × CI
```

Defaults: PUE=1.2, CI=150 gCO₂/kWh (typical UK grid order of magnitude).

Power parameters (TDP approximation):

| Resource | Power |
|---|---|
| Ice Lake 8358 CPU | 7.8 W/core |
| A100 GPU | 400 W |
| H100 GPU | 350 W |
| H100-NVL GPU | 400 W |

---

## 11. Three bugs we fixed

| Bug | Stage | Cause and fix |
|---|---|---|
| `restrict_uid is not set` fatal | Creation: in-process RPC | Slurm 23.11 added a security check; shortcut path did not set `r_uid`. Fixed in `slurm_protocol_api.c` with `slurm_msg_set_r_uid(msg, SLURM_AUTH_UID_ANY)` |
| Only 1 node online | Node registration | `SIM_NODE_REGISTRATION` registered localhost only; changed to iterate all configured nodes |
| `Not implemented agent request` fatal | Execution: agent intercept | Periodic `REQUEST_PING / HEALTH_CHECK / ACCT_GATHER_UPDATE` dropped (no real slurmds) |

---

## 12. Relevance to energy and carbon monitoring

The simulator produces no hardware energy readings, but it outputs all activity data needed for estimation:

- **Per job**: submit/start/end times, node list, core count, GPU count, partition, final state
- **Per node**: occupancy at any point in time (derived from job allocations)

Extrapolation:

```
E_job = Σ_resources (power_model(type, utilization) × occupancy_duration) × PUE
carbon = E_job × grid_carbon_intensity (gCO2/kWh)
```

Compared to FastSim: FastSim needs a real sacct dump and reads `ConsumedEnergyRaw` directly. This approach uses synthetic workloads plus a power model, which fits framework validation when no real measurements exist. See section 5 of `真实HPC_Workload模拟调研.md`.

---

## 13. Known limitations

1. No real computation; CPU/GPU utilization assumed at 100% (conservative upper bound)
2. Power model uses TDP; no DVFS, idle power, network, or storage overhead
3. Carbon intensity CI is a fixed value; no hourly grid variation
4. `sim.events` comment parsing is buggy; event files must not contain comments
5. Long runs may need a patch to `assoc_mgr.c` (without slurmdbd, assoc refresh fails with false `invalid account`)
6. `ClockScaling` above ~5000 causes int64 time overflow

---

## 14. File index

| File | Purpose |
|---|---|
| `contribs/sim/slurmctld_controller.c` | Main wrapper, event loop |
| `contribs/sim/sim_events.c/h` | Event queue, sim.events parsing |
| `contribs/sim/sim_jobs.c/h` | Parallel ledger (actual walltime) |
| `contribs/sim/sim_time.c/h` | Virtual clock |
| `contribs/sim/sim_sbatch.c` | Embedded sbatch |
| `contribs/sim/sim_users.c` | Virtual users (wrap getpwnam) |
| `contribs/sim/slurmctld_agent.c` | RPC intercept layer |
| `contribs/sim/sim_conf.c` | sim.conf parsing |
| `src/common/slurm_protocol_api.c` | In-process RPC shortcut |
| `workload-gen/generate_workload.py` | Synthetic workload generator |
| `workload-gen/analyze_run.py` | Results analysis + energy/carbon |
| `stanage-sim/slurm.conf, gres.conf` | Stanage topology |
| `stanage-sim/sim.conf` | Clock, event file path |
| `stanage-sim/users.sim` | Virtual users |
| `stanage-sim/sim.events` | Placeholder workload (7 jobs) |
| `stanage-sim/sim.events.realistic` | Synthetic workload (140 jobs) |
| `stanage-sim/workload_profile.csv` | Job profiles |
| `stanage-sim/job_results.csv` | Simulation results (generated after run) |
| `run-stanage.sh` | One-command build and run |
