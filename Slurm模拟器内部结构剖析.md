# Slurm Simulator 内部结构详解 —— Job 是如何被创建、调度和执行的

> 项目：Energy and Carbon Monitoring in HPC Systems
> 对象：`slurm_simulator/`（ubccr Slurm Simulator，基于真实 Slurm 23.11 源码改造）
> 本笔记回答三个问题：**Job 怎么被创建？怎么被调度？怎么被"执行"？**

---

## 1. 总体架构：它不是"仿制品"，而是被"驯化"的真 Slurm

这个模拟器最重要的设计决策：**它不重写 Slurm，而是直接编译真实的 Slurm 源码，只替换掉与"真实世界"打交道的部分**。

被替换/拦截的三样东西：

| 真实 Slurm | 模拟器中的替代 | 实现文件 |
|---|---|---|
| 真实时钟（`time()` 等） | 可加速的虚拟时钟 | `contribs/sim/sim_time.c` |
| 网络 RPC（slurmctld ↔ slurmd/sbatch） | 进程内函数调用 + 事件队列 | `contribs/sim/slurmctld_agent.c`、`src/common/slurm_protocol_api.c` |
| 真实作业进程（slurmd fork 出的任务） | 一个"定时器"事件 | `contribs/sim/sim_jobs.c` + 事件循环 |

而**调度器本身（priority、backfill、cons_tres 资源选择）全部是真实 Slurm 代码**——这正是它作为研究工具的价值：调度行为和真集群一致。

编译时通过 `-DSLURM_SIMULATOR` 宏区分模拟器代码路径。入口的包装手法（`contribs/sim/slurmctld_controller.c`）：

```c
#define main slurmctld_main
#include "../../src/slurmctld/controller.c"   /* 把真实 slurmctld 的 main 改名后整个包进来 */
#undef main

int main(int argc, char **argv)
{
    /* 模拟器自己的初始化：读 sim.conf、users.sim、sim.events，
       建立共享内存里的虚拟时钟，然后才调用真正的 slurmctld */
    ...
    slurmctld_main(slurmctld_argc, slurmctld_argv);
}
```

---

## 2. 虚拟时间系统（`sim_time.c`）

模拟器把 libc 的时间函数全部包了一层（链接期 `--wrap`），所有 Slurm 代码拿到的都是**虚拟时间**：

```c
int64_t get_sim_utime()
{
    int64_t cur_real_utime = get_real_utime();
    /* t_sim = t_real + shift + (scale - 1) * t_real  即 t_sim = scale*t_real + shift */
    int64_t cur_sim_time = cur_real_utime + *sim_timeval_shift
                         + (int64_t)((*sim_timeval_scale - 1.0) * cur_real_utime);
    return cur_sim_time;
}
```

- `scale` 与 `shift` 存放在**共享内存**（`sim.conf` 的 `SharedMemoryName`）中，保证多个组件看到同一个时钟。
- `sim.conf` 里 `ClockScaling=1000` 表示虚拟时间以 1000 倍速流逝：模拟 24 小时的集群运行，真实只需约 90 秒。
- 被包装的函数包括 `gettimeofday / time / sleep / usleep / nanosleep`，所以 Slurm 内部所有周期性线程（心跳、超时检查）都自动"跟着虚拟时间走"。

## 3. 事件系统（`sim_events.c/h`）——离散事件仿真的心脏

模拟器本质是一个**离散事件仿真器（DES）**：维护一条按虚拟时间排序的双向链表，事件类型定义在 `sim_events.h`：

```c
typedef enum {
    SIM_TIME_ZERO = 1001,        /* 哨兵：链表头 */
    SIM_TIME_INF,                /* 哨兵：链表尾 */
    SIM_NODE_REGISTRATION,       /* 节点上线注册 */
    SIM_SUBMIT_BATCH_JOB,        /* 提交一个批处理作业（对应 sim.events 的一行）*/
    SIM_COMPLETE_BATCH_SCRIPT,   /* 作业"跑完了"（定时器到点）*/
    SIM_EPILOG_COMPLETE,         /* epilog 完成，节点资源真正释放 */
    SIM_CANCEL_JOB,
    SIM_ACCOUNTING_UPDATE,       /* 周期性把状态刷进 slurmdbd/MySQL */
    SIM_PRIORITY_DECAY,          /* 周期性优先级衰减（fair-share 用）*/
    SIM_SET_DB_INDEX,
} sim_event_type_t;

typedef struct sim_event {
    int64_t when;                /* 事件发生的虚拟时间（微秒）*/
    struct sim_event *next, *previous;
    sim_event_type_t type;
    void *payload;               /* 例如 sim_event_submit_batch_job_t* */
} sim_event_t;
```

插入事件 = 按 `when` 在有序链表中找位置（`sim_insert_event2`）；`<=` 的比较保证**同一时刻的事件按到达顺序排队**，这是确定性（可复现）的关键之一。

### 3.1 sim.events 文件如何变成事件

启动时逐行解析 `sim.events`，例如：

```
-e submit_batch_job -dt 68 | --uid=alice -J jobid_2 -p gpu --gres=gpu:a100:2 -N 1 -n 12 -t 30 -sim-walltime 1841 pseudo.job -sleep 1841
```

- `-dt 68`：相对模拟起点 68 秒时触发 `SIM_SUBMIT_BATCH_JOB` 事件；
- `|` 之后是**近乎完整的 sbatch 命令行**，由 `sim_submit_batch_job_get_payload()` 用 `split_cmd_line` 切开存入 payload；
- `-t 30` 是**用户申请的 walltime**（30 分钟，调度器据此做规划）；
- `-sim-walltime 1841` 是**作业"实际"要跑多久**（1841 秒，模拟器据此安排完成事件）。
  两者分离正好复现了真实 HPC 中"用户申请时长 ≫ 实际用时"的现象；若 `-sim-walltime` 超过 `-t`，作业会被真实的超时逻辑杀掉（TIMEOUT）。
- 注意：解析器对以 `#` 开头的注释行有 bug，事件文件里**不能写注释**（我们之前踩过的坑）。

## 4. 主事件循环（`slurmctld_controller.c`）

模拟器砍掉了真实 slurmctld 的大部分后台线程（RPC 监听线程、agent 线程、backfill 线程……），把它们的工作全部收编进**单一主循环**，按虚拟时间顺序驱动：

```c
void sim_slurmctld_event_main_loop()
{
    _slurmctld_background(NULL);            /* 真实 Slurm 的后台维护入口 */
    while (1) {
        now = get_sim_utime();
        sim_main_thread_sleep_till = now + 1000000;   /* 推进 1 虚拟秒 */
        while (sim_main_thread_sleep_till > now)
            now = sim_events_loop();        /* 处理到期事件 + 手动调 scheduler/backfill */
        _slurmctld_background(NULL);        /* 周期性：作业超时检查、节点状态等 */
    }
}
```

`sim_events_loop()` 每轮做四件事：

1. **弹出所有到期事件**并分发（见第 5 节的作业生命周期）；
2. **手动调用调度器**：`schedule()`（主调度）与 `_attempt_backfill()`（回填调度）不再是独立线程，而是循环里按 `sched_interval / bf_interval` 直接函数调用——消除线程竞争，保证可复现；
3. 手动驱动 slurmdbd agent（记账消息队列）；
4. 检查退出条件：`sim.conf` 里 `TimeStop=1` 表示"最后一个作业完成后自动停机"。

## 5. 一个 Job 的完整生命周期

以 `jobid_2`（上面那行 GPU 作业）为例，串起全部机制：

### ① 创建（虚拟时刻 t=68s）

`SIM_SUBMIT_BATCH_JOB` 事件到期 → 事件循环调用 `submit_job()`（`contribs/sim/sim_sbatch.c`，一个内嵌的迷你 sbatch）：

- 解析 argv，填出真实的 `job_desc_msg_t`（分区、节点数、GRES、时限、uid……）；uid 从 `users.sim` 提供的假 passwd 里查（`alice → 1001`）；
- 调用**真实的** `slurm_submit_batch_job()`。消息不走网络：`src/common/slurm_protocol_api.c` 中 `#ifdef SLURM_SIMULATOR` 分支把 `REQUEST_SUBMIT_BATCH_JOB` 直接送进本进程的 RPC 处理函数（我们修的 `restrict_uid` bug 就在这条路径上）；
- 真实 Slurm 的 `_slurm_rpc_submit_batch_job()` 建立 `job_record`，进入优先级队列，日志出现 `JobId=2 InitPrio=... usec=...`；
- 同时模拟器在自己的"平行账本"（`sim_jobs.c` 的 `sim_job_t` 链表）里记一笔：`job_id=2, walltime=1841s`——这是 Slurm 本体不知道的"上帝视角"信息。

### ② 调度（真实 Slurm 代码，一行未改）

- `schedule()`：按优先级从队列取作业 → `select/cons_tres` 插件在满足分区/GRES/CPU 约束的节点里选资源；
- 若队头作业资源不够，`_attempt_backfill()` 用**用户申请的 walltime**（`-t`）构建时间轴，把小作业"回填"进空洞——这就是为什么申请时长的准确性会影响利用率，也是能耗研究里的经典变量；
- 成功后日志：`sched: Allocate JobId=2 NodeList=gpu-node001 ...`。

### ③ "执行"（模拟的核心 trick）

真实流程是 slurmctld 发 `REQUEST_BATCH_JOB_LAUNCH` 给 slurmd，slurmd fork 进程跑脚本。模拟器里：

- `slurmctld_agent.c` 的 `__wrap_agent_queue_request()` **拦截**这个 RPC，不发网络；
- 从 `sim_jobs` 账本查到该作业 `walltime=1841s`，直接向事件队列插入一个 `SIM_COMPLETE_BATCH_SCRIPT` 事件，`when = start_time + 1841s`；
- **没有任何进程被启动**。`pseudo.job` 脚本内容从未被读取，"执行"就是等一个定时器。
- 这期间虚拟时钟照常流逝，节点资源在 Slurm 账面上处于占用状态，其他作业照常排队/回填——**调度语义完全保真，只是省掉了真实计算**。

### ④ 完成与资源释放（t = start + 1841s）

- `SIM_COMPLETE_BATCH_SCRIPT` 到期 → 构造 `REQUEST_COMPLETE_BATCH_SCRIPT` 消息喂给真实的处理函数 → 日志 `_job_complete: JobId=2 done`；
- 紧接着插入 `SIM_EPILOG_COMPLETE` 事件 → 触发真实的 `job_epilog_complete()` → 节点资源释放，回填调度器立刻可以用它安排下一个作业；
- `SIM_ACCOUNTING_UPDATE` 周期事件把最终状态（start/end/state/tres）刷进 slurmdbd → MySQL，之后 `sacct` 可查。

### ⑤ 超时作业的路径

若 `-sim-walltime > -t`，真实的 `_slurmctld_background()` 超时检查会先发现作业超限 → 发 `REQUEST_KILL_TIMELIMIT`（被 agent 拦截）→ 模拟器把该作业标记 `requested_kill_timelimit` 并立即安排完成事件 → 作业以 TIMEOUT 状态结束。

## 6. 我们修过的三个 bug 在架构中的位置

| Bug | 所在环节 | 本质 |
|---|---|---|
| `restrict_uid is not set` fatal | ①创建：进程内 RPC 短路路径 | Slurm 23.11 新增安全校验，模拟器旁路没设置 `r_uid`，在 `slurm_protocol_api.c` 补上 `slurm_msg_set_r_uid(msg, SLURM_AUTH_UID_ANY)` |
| 只有 1 个节点上线 | 节点注册事件 | `SIM_NODE_REGISTRATION` 原来只注册本机节点；改为遍历全部配置节点逐个 `node_did_resp()`，多节点集群才能调度 |
| `Not implemented agent request` fatal | ③执行：agent 拦截层 | 周期性 `REQUEST_PING / HEALTH_CHECK / ACCT_GATHER_UPDATE` 没有对应模拟实现，直接丢弃即可 |

## 7. 对能耗/碳排放项目的意义：数据挂钩点

模拟器**本身不产生能耗数据**（没有真实硬件），但它精确产出能耗估算所需的全部"活动数据"：

- **每作业**：submit/start/end 时刻、节点列表、核数、GPU 数、分区、结束状态（来自 slurmctld.log 或 sacct/MySQL）；
- **每节点**：任意时刻的占用情况（可从作业分配反推 utilization 时间线）。

因此能耗估算的公式可以外挂在结果之上：

```
E_job = Σ_资源 (功率模型(资源类型, 利用率) × 占用时长) × PUE
碳排放 = E_job × 当时的电网碳强度 (gCO2/kWh)
```

- 功率模型可用 TDP 近似（如 A100≈400 W、Ice Lake 8358≈250 W/32 核）或接入 FastSim 论文里的回归模型；
- 这正是 FastSim 中 `ConsumedEnergyRaw` 字段扮演的角色——真实集群由 RAPL/IPMI 计量，模拟场景下由我们的功率模型合成。

## 8. 关键文件速查表

| 文件 | 作用 |
|---|---|
| `contribs/sim/slurmctld_controller.c` | main 包装、主事件循环、线程收编 |
| `contribs/sim/sim_events.c/h` | 事件队列、sim.events 解析 |
| `contribs/sim/sim_jobs.c/h` | 模拟器的作业"平行账本"（真实 walltime）|
| `contribs/sim/sim_time.c/h` | 虚拟时钟（scale/shift，共享内存）|
| `contribs/sim/sim_sbatch.c` | 内嵌 sbatch：事件 → 真实提交调用 |
| `contribs/sim/slurmctld_agent.c` | RPC 拦截层（launch/kill/ping…）|
| `src/common/slurm_protocol_api.c` | 进程内 RPC 短路（含 restrict_uid 修复）|
| `stanage-sim/slurm.conf, gres.conf` | 1:1 Stanage 拓扑（189 节点、6 分区）|
| `stanage-sim/sim.conf` | ClockScaling、共享内存名、事件文件路径 |
| `stanage-sim/sim.events` | 工作负载（每行一个作业提交事件）|
| `run-stanage.sh` | 一键构建 + 运行脚本 |
