# Slurm Simulator 内部结构详解

> 项目：Energy and Carbon Monitoring in HPC Systems
> 对象：`slurm_simulator/`（ubccr Slurm Simulator，基于 Slurm 23.11 源码改造）
> 本文回答：Job 怎么被创建、调度和"执行"？Synthetic jobs 怎么生成？各参数怎么配？

---

## 1. 总体架构：被"驯化"的真 Slurm

这个模拟器没有重写 Slurm，而是编译真实 Slurm 源码，只替换掉与真实硬件/网络交互的部分。

| 真实 Slurm | 模拟器中的替代 | 实现文件 |
|---|---|---|
| 真实时钟（`time()` 等） | 可加速的虚拟时钟 | `contribs/sim/sim_time.c` |
| 网络 RPC（slurmctld ↔ slurmd/sbatch） | 进程内函数调用 + 事件队列 | `contribs/sim/slurmctld_agent.c`、`src/common/slurm_protocol_api.c` |
| 真实作业进程（slurmd fork 出的任务） | 一个定时器事件 | `contribs/sim/sim_jobs.c` + 事件循环 |

调度器本身（priority、backfill、cons_tres 资源选择）全部是真实 Slurm 代码。编译时通过 `-DSLURM_SIMULATOR` 宏区分模拟器路径。

入口包装（`contribs/sim/slurmctld_controller.c`）：

```c
#define main slurmctld_main
#include "../../src/slurmctld/controller.c"
#undef main

int main(int argc, char **argv)
{
    /* 读 sim.conf、users.sim、sim.events，建立虚拟时钟，再调用 slurmctld */
    ...
    slurmctld_main(slurmctld_argc, slurmctld_argv);
}
```

---

## 2. 参考文献与官方文档

### 2.1 核心论文

| 文献 | 内容 |
|---|---|
| Simakov et al., PMBS 2017 (LNCS 10724) | Slurm Simulator 的实现与参数分析；基于真实 Slurm 源码，支持历史/合成负载回放 |
| Simakov et al., PEARC 2018 | 多 controller、node sharing 对大规模集群调度性能的影响 |
| Lucero (BSC, SLUG 2015) | 早期 Slurm Workload Simulator；`sim_mgr` + `test.trace` 事件驱动模型 |
| Trofinoff & Benini | 在 Lucero 基础上的进一步开发 |

### 2.2 官方资源

| 资源 | URL | 用途 |
|---|---|---|
| Slurm Simulator 主页 | https://ubccr-slurm-simulator.github.io/ | 项目概述、论文链接 |
| User Guide v1.2 | https://ubccr-slurm-simulator.github.io/docs/slurm_sim_manual_v1.2.html | 安装、`sim.conf`、R 工具链 |
| slurm_sim_tools (GitHub) | https://github.com/ubccr-slurm-simulator/slurm_sim_tools | R 包 `RSlurmSimTools`：`sim_job()` + `write_trace()` 生成 trace |
| Slurm 官方文档 | https://slurm.schedmd.com/ | sbatch 参数语义（`-N`、`-n`、`-t`、`--gres` 等） |

### 2.3 本项目额外参考

| 来源 | 用途 |
|---|---|
| Feitelson Parallel Workloads Archive | 运行时重尾分布、节点数偏好 2 的幂 |
| F-DATA (Fugaku workload dataset) | 大 MPI 作业贡献 node-hours、申请/实际时长比 |
| Stanage 集群文档 | `slurm.conf` 节点/分区 1:1 复刻 |
| `真实HPC_Workload模拟调研.md` | 负载统计特征与 FastSim 对比 |

官方 ubccr 工具链用 R 生成 trace（`.trace` 格式）；本项目用 Python `generate_workload.py` 直接写 `sim.events`，原理相同，只是格式和统计模型针对 Stanage 定制。

---

## 3. 虚拟时间系统（`sim_time.c`）

模拟器用链接期 `--wrap` 包装 libc 时间函数，Slurm 内部拿到的都是虚拟时间：

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

- `scale` 与 `shift` 存在共享内存（`sim.conf` 的 `SharedMemoryName`）里，多组件共享同一时钟。
- `ClockScaling=3000` 表示虚拟时间以 3000 倍速流逝：模拟 12 小时提交窗口，真实约需 15 分钟。
- **注意**：`ClockScaling` 不宜超过 ~5000。`get_sim_utime()` 用 int64 微秒计数，scale 过大时会溢出，出现 job ID 年份变成 2683 的 bug（`run-stanage.sh` 里有注释）。
- 被包装的函数包括 `gettimeofday / time / sleep / usleep / nanosleep`。

---

## 4. 配置文件

### 4.1 `sim.conf`（模拟器专用）

| 参数 | 含义 | Stanage 示例 |
|---|---|---|
| `TimeStart` | 模拟起始虚拟时间（秒） | `0` |
| `TimeStop` | `0`=永不停止；`1`=最后一个作业完成后退出 | `1` |
| `SecondsBeforeFirstJob` | 第一个作业事件之前的等待（秒） | `1` |
| `ClockScaling` | 虚拟/真实时间倍率 | `3000`（realistic 模式） |
| `SharedMemoryName` | 共享内存路径 | `/slurm_sim_stanage.shm` |
| `EventsFile` | 工作负载事件文件 | `sim.events.realistic` |
| `TimeAfterAllEventsDone` | 全部事件处理完后再跑多久（秒） | `2` |
| `FirstJobDelay` | 第一个作业额外延迟（微秒） | `0` |
| `CompJobDelay` | 完成事件到 epilog 的延迟（微秒） | `0` |
| `TimeLimitDelay` | 超时 kill 到完成事件的延迟（微秒） | `0` |

### 4.2 `users.sim`（虚拟用户）

格式：`username:uid:groupname:gid`，每行一个用户。模拟器 wrap 了 `getpwnam_r` / `getpwuid_r`，`--uid=alice` 会查到 uid 1001。

```
alice:1001:users:100
bob:1002:users:100
...
```

用户名必须在 `users.sim` 里预先定义，否则 sbatch 提交会失败。

### 4.3 `slurm.conf` + `gres.conf`（集群拓扑）

Stanage 1:1 复刻：189 个公开节点，6 个分区。

| 分区 | 节点 | CPU | 内存 | GPU |
|---|---|---|---|---|
| `standard` | node[001-142] | 64c | 251 GB | 无 |
| `bigmem` | bigmem[01-12] | 64c | 1 TB | 无 |
| `hugemem` | hugemem[01-12] | 64c | 2 TB | 无 |
| `gpu` | gpua100[01-13] | 48c | 512 GB | 4× A100 |
| `gpu-h100` | gpuh100[01-06] | 48c | 512 GB | 2× H100 |
| `gpu-h100-nvl` | gpuh100nvl[01-04] | 96c | 512 GB | 4× H100 NVL |

调度相关配置：`SchedulerType=sched/backfill`，`SelectType=select/cons_tres`，`SelectTypeParameters=CR_CPU`。

`gres.conf` 只声明 GPU 类型和数量，不需要真实 GPU 设备文件。

---

## 5. 事件系统（`sim_events.c/h`）

模拟器是一个离散事件仿真器（DES）：维护按虚拟时间排序的双向链表。

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

插入事件时 `<=` 比较保证同一时刻的事件按到达顺序排队，这是确定性（可复现）的基础。

启动时还会自动插入 `SIM_NODE_REGISTRATION` 事件，让所有配置节点上线。

### 5.1 `sim.events` 行格式

```
-e submit_batch_job -dt <秒> | <sbatch 参数字符串>
```

`|` 左边是事件元数据，右边是近乎完整的 sbatch 命令行。

示例：

```
-e submit_batch_job -dt 68 | --uid=alice -J jobid_2 -p gpu --gres=gpu:a100:2 -N 1 -n 12 -t 30 -sim-walltime 1841 pseudo.job -sleep 1841
```

| 字段 | 含义 |
|---|---|
| `-dt N` | 相对模拟起点 N 秒时触发提交 |
| `--uid=` | 提交用户（须在 `users.sim` 中） |
| `-J jobid_N` | **必须**命名为 `jobid_<整数>`，模拟器用它关联平行账本 |
| `-p` | 目标分区 |
| `-N` | 节点数 |
| `-n` | 任务/核数 |
| `--gres=` | GPU 资源（如 `gpu:a100:2`） |
| `-t` | 申请 walltime（分钟），调度器/backfill 用这个 |
| `-sim-walltime` | 实际运行秒数，模拟器据此安排完成事件 |
| `pseudo.job -sleep N` | 占位脚本 + 冗余 sleep 参数（与 `-sim-walltime` 值相同） |

**硬性约束**（`sim_submit_batch_job_get_payload()` 会 fatal exit）：

1. 作业名必须是 `jobid_<整数>` 格式
2. 不能手动设 `-jid`（内部自动分配 Slurm JobId）
3. `-sim-walltime` 和 `pseudo.job -sleep` 二选一或同时设（值应一致）
4. 文件**不能含 `#` 注释**（解析器只在行首无字符时才识别注释，行内 `#` 会误解析）

### 5.2 申请 walltime vs 实际运行时长

- `-t`：用户申请的 walltime，backfill 调度器用它规划时间轴
- `-sim-walltime`：作业"实际"跑多久，模拟器据此插入 `SIM_COMPLETE_BATCH_SCRIPT`

两者分离可以复现真实 HPC 里"申请时长远大于实际用时"的情况。若 `-sim-walltime > -t`，真实超时逻辑会先触发 `REQUEST_KILL_TIMELIMIT`，作业以 TIMEOUT 结束。

---

## 6. 主事件循环（`slurmctld_controller.c`）

模拟器砍掉了真实 slurmctld 的大部分后台线程，收编进单一主循环：

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

`sim_events_loop()` 每轮做四件事：

1. 弹出到期事件并分发
2. 手动调用 `schedule()` 和 `_attempt_backfill()`（消除线程竞争，保证可复现）
3. 驱动 slurmdbd agent（记账消息队列）
4. 检查退出条件（`TimeStop=1` 时最后一个作业完成后停机）

---

## 7. 一个 Job 的完整生命周期

以 GPU 作业 `jobid_2` 为例：

### ① 创建（虚拟时刻 t=68s）

`SIM_SUBMIT_BATCH_JOB` 到期 → `submit_job()`（`contribs/sim/sim_sbatch.c`，内嵌 sbatch）：

- 解析 argv，填 `job_desc_msg_t`（分区、节点数、GRES、时限、uid 等）
- uid 从 `users.sim` 的假 passwd 查（`alice → 1001`）
- 调用真实 `slurm_submit_batch_job()`；进程内 RPC 短路（`slurm_protocol_api.c` 的 `#ifdef SLURM_SIMULATOR` 分支）
- 真实 Slurm 建立 `job_record`，进入优先级队列
- 模拟器在 `sim_jobs.c` 平行账本记一笔：`job_id, walltime=1841s`

### ② 调度（真实 Slurm 代码）

- `schedule()`：按优先级取作业 → `select/cons_tres` 选节点
- 队头资源不够时 `_attempt_backfill()` 用 `-t` 构建时间轴，小作业回填进空洞
- 成功后日志：`sched: Allocate JobId=2 NodeList=gpu-node001 ...`

### ③ "执行"（模拟的核心 trick）

- `slurmctld_agent.c` 的 `__wrap_agent_queue_request()` 拦截 `REQUEST_BATCH_JOB_LAUNCH`
- 从 `sim_jobs` 账本查 walltime，插入 `SIM_COMPLETE_BATCH_SCRIPT`，`when = start + walltime`
- **没有进程被启动**。`pseudo.job` 从未被读取。

### ④ 完成与资源释放

- `SIM_COMPLETE_BATCH_SCRIPT` → `REQUEST_COMPLETE_BATCH_SCRIPT` → `_job_complete: JobId=2 done`
- `SIM_EPILOG_COMPLETE` → `job_epilog_complete()` → 节点释放，backfill 可用

### ⑤ 超时路径

若 `-sim-walltime > -t`：`_slurmctld_background()` 超时检查 → `REQUEST_KILL_TIMELIMIT`（被 agent 拦截）→ 标记 `requested_kill_timelimit` → 立即安排完成事件 → TIMEOUT 状态

---

## 8. Synthetic Jobs 如何生成

### 8.1 为什么 placeholder 可以替换成真实负载

模拟器不执行真实计算。调度器看到的只有资源画像：

| 参数 | 作用 |
|---|---|
| 提交时刻 (`-dt`) | 作业何时进队列 |
| 分区 (`-p`) | 路由到哪个硬件池 |
| 节点数 (`-N`) | MPI 规模 |
| 核数 (`-n`) | CPU 占用 |
| GPU (`--gres`) | GPU 占用 |
| 申请 walltime (`-t`) | backfill 规划 |
| 实际运行时长 (`-sim-walltime`) | 完成事件时刻 |

只要这些参数服从真实集群的统计规律，排队、回填、资源占用就是可用的活动数据；能耗/碳排放在此之上用功率模型外推。

### 8.2 统计规律来源

- **Feitelson Archive**：运行时重尾（对数正态），节点数偏好 2 的幂，到达过程带日间周期
- **F-DATA / NERSC / ARCHER2**：大量短作业贡献作业数，少量大 MPI 贡献 node-hours；申请 walltime 是实际用时的 1.5~10 倍，取整到 15min/1h/4h/12h/24h
- **失败/超时**：约 5~8% 启动后迅速失败；约 5% 大 MPI 撞 walltime

### 8.3 生成器：`workload-gen/generate_workload.py`

```bash
cd slurm_simulator/workload-gen
python3 generate_workload.py --hours 12 --jobs 140 --seed 42
```

输出：

- `stanage-sim/sim.events.realistic`：140 行事件（无注释）
- `stanage-sim/workload_profile.csv`：每作业的 class/user/资源/功率参数

#### 8 类作业及默认比例

| 类别 | 默认占比 | 分区 | 典型规模 | 典型时长 |
|---|---|---|---|---|
| `htc_short` | 38 | standard | 1 节点, 4~32 核 | 10min~3h |
| `mpi_physics` | 22 | standard | 2~32 节点 | 1~2h（cap） |
| `mpi_capability` | 2 | standard | 64~96 节点 | 1~2h |
| `ai_train_a100` | 12 | gpu | 1~2 节点, 4 GPU/节点 | 1~2h |
| `ai_dev_gpu` | 13 | gpu / gpu-h100 | 1 节点, 1~2 GPU | 2~90min |
| `llm_h100nvl` | 3 | gpu-h100-nvl | 1 节点, 4 GPU | 2~2h |
| `bigmem` | 6 | bigmem | 1 节点 | 1~12h |
| `hugemem` | 2 | hugemem | 1 节点 | 1~12h |

#### 生成算法要点

1. **到达过程**：非齐次泊松（thinning 法），白天（窗口内 09:00~18:00 段）强度约为夜间 3 倍
2. **运行时**：对数正态分布 `lognormal_capped(median, sigma, cap)`，单作业最长模拟运行 2h（`MAX_SIM_WALL_S`）
3. **申请 walltime**：`round_up_request(actual, factor)` 取整到 `[15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320]` 分钟档位，且 ≥ 实际用时 × factor
4. **用户分配**：8 个用户按权重随机（少数用户贡献更多作业）
5. **超时/失败**：`mpi_physics` 约 5% 故意让 actual > req（TIMEOUT）；`htc_short` 约 8% 短运行后失败

#### 生成的事件行模板

```python
"-e submit_batch_job -dt %d | --uid=%s -J jobid_%d -p %s "
"-N %d -n %d%s -t %d -sim-walltime %d pseudo.job -sleep %d\n"
# gres 部分: " --gres=gpu:a100:4" 或空字符串
```

### 8.4 官方 R 工具链（对照）

ubccr 的 `RSlurmSimTools` 提供 `sim_job()` 和 `write_trace()`：

```r
sim_job(
    job_id=1001,
    submit="2016-10-01 00:01:00",
    wclimit=300L,      # 对应 -t（秒）
    duration=600L,       # 对应 -sim-walltime（秒）
    tasks=12L,
    tasks_per_node=12L
)
write_trace("dependency_test.trace", trace)
```

R 工具支持更多参数（dependency、account、qos、reservation 等）。本项目的 Python 生成器目前只覆盖 Stanage 能耗研究需要的核心字段。

### 8.5 手动编写 sim.events

小规模测试可以直接手写，参考 `stanage-sim/sim.events`（7 个 placeholder 作业，覆盖 6 个分区）：

```
-e submit_batch_job -dt 0 | --uid=alice -J jobid_1 -p standard -N 8 -n 512 -t 60 -sim-walltime 12 pseudo.job -sleep 12
-e submit_batch_job -dt 1 | --uid=alice -J jobid_4 -p gpu -N 1 -n 12 --gres=gpu:a100:2 -t 60 -sim-walltime 10 pseudo.job -sleep 10
```

---

## 9. Job 参数配置速查

### 9.1 sbatch 参数与调度影响

| 参数 | sim.events 写法 | 调度器看到什么 | 能耗估算用途 |
|---|---|---|---|
| 分区 | `-p gpu` | 节点池约束 | 决定 CPU/GPU 功率模型 |
| 节点数 | `-N 8` | MPI 规模、节点占用 | node-hours 计算 |
| 核/任务数 | `-n 512` | CPU 分配（cons_tres） | CPU 功率 = cores × W/core |
| GPU | `--gres=gpu:a100:2` | GRES 约束 | GPU 功率 = gpus × W/gpu |
| 申请时限 | `-t 480`（分钟） | backfill 时间轴 | 不直接进能耗公式 |
| 实际时长 | `-sim-walltime 7200`（秒） | 完成事件时刻 | runtime 乘功率 |
| 用户 | `--uid=alice` | fair-share 权重 | 按用户聚合 |
| 作业名 | `-J jobid_42` | 关联 sim_jobs 账本 | 与 workload_profile 对齐 |

### 9.2 常用 sbatch 参数（模拟器均支持）

模拟器内嵌完整 sbatch 解析（`sim_sbatch.c` include 了 `src/sbatch/sbatch.c`），以下参数在 `sim.events` 中可用：

- `-p / --partition`：分区
- `-N / --nodes`：节点数
- `-n / --ntasks`：任务数
- `-c / --cpus-per-task`：每任务 CPU 数
- `--gres=`：通用资源（GPU 等）
- `-t / --time`：walltime（`D-HH:MM:SS` 或分钟）
- `--mem`：内存请求
- `-A / --account`：账户
- `-q / --qos`：QoS
- `--dependency`：作业依赖（官方 R 工具有示例）
- `-D / --chdir`：工作目录（默认 `/home/<username>`）

模拟器专用（Slurm 标准 sbatch 没有）：

- `-sim-walltime <秒>`：实际运行时长

### 9.3 分区与资源约束示例

| 场景 | 示例行 |
|---|---|
| 标准 MPI | `-p standard -N 16 -n 1024 -t 480 -sim-walltime 7200` |
| A100 训练 | `-p gpu -N 2 -n 96 --gres=gpu:a100:4 -t 480 -sim-walltime 7200` |
| H100 调试 | `-p gpu-h100 -N 1 -n 8 --gres=gpu:h100:1 -t 60 -sim-walltime 1393` |
| 大内存 | `-p bigmem -N 1 -n 64 -t 240 -sim-walltime 3895` |
| 撞墙超时 | `-t 120 -sim-walltime 4431`（actual > req × 60） |

---

## 10. 运行与分析

### 10.1 运行模拟

```bash
cd slurm_simulator
./run-stanage.sh              # 默认：7 个 placeholder 作业
./run-stanage.sh realistic    # 140 个合成负载（ClockScaling=3000）
./run-stanage.sh verify       # 每分区填满所有节点的验证模式
```

`run-stanage.sh` 在 Docker 容器里跑 slurmctld，挂载 `stanage-sim/` 为配置目录。realistic 模式临时改 `sim.conf` 的 `EventsFile` 和 `ClockScaling`，跑完恢复。

### 10.2 分析结果

```bash
python3 workload-gen/analyze_run.py
```

输入：`slurmctld.log` + `workload_profile.csv`
输出：`stanage-sim/job_results.csv`

每作业包含：提交/开始/结束时刻、等待时间、能耗估算、碳排放。

```
E_job [kWh] = (cores × W_core + gpus × W_gpu) × runtime_h / 1000 × PUE
CO2 [g]     = E_job × CI
```

默认 PUE=1.2，CI=150 gCO₂/kWh（英国电网量级）。

功率参数（TDP 近似）：

| 资源 | 功率 |
|---|---|
| Ice Lake 8358 CPU | 7.8 W/核 |
| A100 GPU | 400 W |
| H100 GPU | 350 W |
| H100-NVL GPU | 400 W |

---

## 11. 我们修过的三个 bug

| Bug | 所在环节 | 本质 |
|---|---|---|
| `restrict_uid is not set` fatal | ①创建：进程内 RPC 短路 | Slurm 23.11 新增安全校验，旁路没设 `r_uid`；在 `slurm_protocol_api.c` 补 `slurm_msg_set_r_uid(msg, SLURM_AUTH_UID_ANY)` |
| 只有 1 个节点上线 | 节点注册事件 | `SIM_NODE_REGISTRATION` 原来只注册本机；改为遍历全部配置节点 |
| `Not implemented agent request` fatal | ③执行：agent 拦截层 | 周期性 `REQUEST_PING / HEALTH_CHECK / ACCT_GATHER_UPDATE` 直接丢弃 |

---

## 12. 对能耗/碳排放项目的意义

模拟器本身不产生能耗数据（没有真实硬件），但产出能耗估算所需的全部活动数据：

- **每作业**：submit/start/end 时刻、节点列表、核数、GPU 数、分区、结束状态
- **每节点**：任意时刻占用情况（从作业分配反推 utilization 时间线）

外推公式：

```
E_job = Σ_资源 (功率模型(资源类型, 利用率) × 占用时长) × PUE
碳排放 = E_job × 当时的电网碳强度 (gCO2/kWh)
```

与 FastSim 对比：FastSim 需要真实 sacct dump 且直接读 `ConsumedEnergyRaw`；本方案用合成负载 + 功率模型外推，适合没有真实测量数据时的框架验证。详见 `真实HPC_Workload模拟调研.md` 第 5 节。

---

## 13. 已知限制

1. 不执行真实计算，CPU/GPU 利用率按 100% 估算（保守上界）
2. 功率模型用 TDP 近似，未考虑 DVFS、空闲功耗、网络/存储
3. 碳强度 CI 用固定值，未建模电网小时波动
4. `sim.events` 解析器对 `#` 注释支持有 bug，事件文件不能含注释
5. 长时间模拟可能需要 patch `assoc_mgr.c`（无 slurmdbd 时 assoc 刷新失败导致 false `invalid account`）
6. `ClockScaling` 超过 ~5000 会导致 int64 时间溢出

---

## 14. 关键文件速查

| 文件 | 作用 |
|---|---|
| `contribs/sim/slurmctld_controller.c` | main 包装、主事件循环 |
| `contribs/sim/sim_events.c/h` | 事件队列、sim.events 解析 |
| `contribs/sim/sim_jobs.c/h` | 平行账本（真实 walltime） |
| `contribs/sim/sim_time.c/h` | 虚拟时钟 |
| `contribs/sim/sim_sbatch.c` | 内嵌 sbatch |
| `contribs/sim/sim_users.c` | 虚拟用户（wrap getpwnam） |
| `contribs/sim/slurmctld_agent.c` | RPC 拦截层 |
| `contribs/sim/sim_conf.c` | sim.conf 解析 |
| `src/common/slurm_protocol_api.c` | 进程内 RPC 短路 |
| `workload-gen/generate_workload.py` | 合成负载生成器 |
| `workload-gen/analyze_run.py` | 结果分析 + 能耗/碳排放 |
| `stanage-sim/slurm.conf, gres.conf` | Stanage 拓扑 |
| `stanage-sim/sim.conf` | 时钟、事件文件路径 |
| `stanage-sim/users.sim` | 虚拟用户 |
| `stanage-sim/sim.events` | placeholder 负载（7 作业） |
| `stanage-sim/sim.events.realistic` | 合成负载（140 作业） |
| `stanage-sim/workload_profile.csv` | 作业画像 |
| `stanage-sim/job_results.csv` | 模拟结果（运行后生成） |
| `run-stanage.sh` | 一键构建 + 运行 |
