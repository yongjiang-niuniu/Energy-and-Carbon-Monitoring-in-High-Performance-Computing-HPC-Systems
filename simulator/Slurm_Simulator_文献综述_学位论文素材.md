# Slurm Simulator 文献综述（学位论文素材）

> 项目：Energy and Carbon Monitoring in HPC Systems（Sheffield Stanage）  
> 本仓库模拟器代码：`slurm_simulator/`（基于 ubccr Slurm Simulator，Slurm 23.11）  
> 整理日期：2026-07-08  
> 论文 PDF 目录：`simulator/papers/`

---

## 1. 研究背景与动机

Slurm 是当前 HPC 领域最广泛使用的开源作业调度系统之一，其调度策略、回填（backfill）、公平共享（fair-share）、节点共享（node sharing）、分区与 QoS 等参数对集群利用率、作业等待时间和用户体验有显著影响。然而，在生产集群上直接调参往往不现实：一次配置变更可能需要数天甚至数周才能观察到效果，且可能对正在运行的作业产生不可预期的副作用。

**Slurm Simulator** 的核心思路是：保留真实 Slurm 调度器源码，仅替换与真实硬件、网络和时间相关的部分，从而在单机或小型环境中以加速虚拟时间重放历史或合成 workload，在不影响生产系统的前提下评估调度策略与配置变更。对于本课题——在 Stanage 上研究能耗与碳排放监测——模拟器提供了可重复的**作业到达、排队、开始、结束、资源占用**时间序列，可作为功率模型与碳强度（carbon intensity）外推的输入，而无需在真实集群上长时间占用资源。

Slurm Simulator 的发展跨越三个主要阶段：

| 阶段 | 代表工作 | 特点 |
|------|----------|------|
| 起源（2011） | Lucero (BSC) | 在 Slurm 2.x 源码上外挂 `sim_mgr`，事件驱动重放 trace |
| 工程化与验证（2015–2018） | Trofinoff & Benini；D'Amico et al. | 移植新版本、修复同步 bug、首次与真机对比验证 |
| 高性能参数分析（2017–2022） | Simakov et al. (UB/CCR) | 序列化 slurmctld、100× 加速、大规模参数实验；2022 版基于 Slurm 21.08+ 增量改造 |

---

## 2. 五篇核心文献逐篇分析

### 2.1 Lucero (2011) — Slurm Simulator 的最初实现

**文献**：Alejandro Lucero, *Slurm Simulator*, Slurm User Group Meeting (SLUG), September 2011.  
**文件**：`papers/Slurm Simulator - Phoenix Introduction (Lucero SLUG 2011).pdf`

#### 研究目的

BSC 运营多台 Slurm 集群（当时最大约 2500 节点），调度配置（公平共享树、QoS/用户/组限制、回填间隔等）对利用率影响显著。Moab 等系统已有 simulation mode，但 Slurm 缺乏等效工具。Lucero 的目标是：**在真实 Slurm 代码框架内重放作业 trace，比较不同调度配置，为管理员提供离线调参依据**。

#### 模拟器设计

- 基于 Slurm 2.1.9，后快速移植至 2.2.6；约 563 行新增、17 行删除，外加 ~2000 行 `sim_mgr.c` / `sim_lib.c`。
- 三进程架构：`sim_mgr`（控制虚拟时间、读取 trace、通过 Slurm API 提交作业）、`slurmctld`（标准控制器）、`slurmd`（简化版，不执行真实计算，仅按给定 duration 触发完成事件）。
- 输入：二进制 trace 文件（由历史 sacct 等转换）；输出：标准 Slurm 日志、job completion log、数据库文件。
- 所有 `time()` / `gettimeofday()` 等函数被重定向为返回模拟时间。

#### 方法

- 使用 MareNostrum 两个月真实 trace（~50,000 作业、489 用户）在单节点上重放。
- 对比「纯 fair-share」与「QoS 限制 + fair-share（Moab 当前配置）」两种策略。
- 硬件：Intel Xeon 2.5 GHz，8 核，12 GB 内存；启用 slurmdbd 与 backfill（硬编码限制 20 次循环）。

#### 实验结果

- 两种配置利用率接近（72% vs 71%），说明 QoS 限制在该 trace 下对利用率影响有限，但可能限制了对空闲节点的更好利用——模拟器使这类「如果当时换策略会怎样」的问题可量化。
- 证明了在 Slurm 源码内做 trace replay 的可行性，且移植成本较低。

#### 局限性

- 仅支持 Slurm 2.x，架构依赖多进程同步，大规模集群加速比有限。
- backfill 循环次数等参数硬编码，trace 格式为私有二进制格式，生态工具少。
- 未与真机 Slurm 做系统性对比验证；论文为 SLUG 演讲稿，非 peer-reviewed 全文。

---

### 2.2 Simakov et al. (2017) — 实现细节与参数分析（PMBS）

**文献**：N. A. Simakov et al., *A Slurm Simulator: Implementation and Parametric Analysis*, PMBS 2017, LNCS 10724, pp. 197–217. DOI: [10.1007/978-3-319-72971-8_10](https://doi.org/10.1007/978-3-319-72971-8_10)  
**文件**：`papers/A Slurm Simulator - Implementation and Parametric Analysis (Simakov PMBS 2017).pdf`  
**代码**：https://github.com/ubccr-slurm-simulator/（本仓库 `slurm_simulator/` 即基于此 lineage）

#### 研究目的

在 Lucero 与 Trofinoff 工作的基础上，解决**中等规模集群上模拟速度过慢、线程堆积导致 hang** 的问题，并系统评估 Slurm 参数（节点共享、backfill 参数、多控制器等）对吞吐与等待时间的影响，为 HPC 中心提供**可操作的离线调参工具**。

#### 模拟器设计

核心原则：**最小化 daemon/线程，slurmctld 内嵌离散事件循环，不再引入外部 sim_mgr**。

| 组件 | 真实 Slurm | 模拟器替换 |
|------|-----------|-----------|
| 虚拟时钟 | 系统时钟 | `--wrap` 劫持 `gettimeofday`/`time`/`sleep` 等 |
| slurmd / 作业进程 | fork 真实进程 | 控制器内 positive response + 定时完成事件 |
| RPC / agent 线程 | 多线程并发 | 序列化：主循环按到期时间调用调度、提交、释放 |
| mutex 锁 | pthread 锁 | dummy 占位（节省 ~40% 时间） |
| 编译 | debug + assert | release 模式进一步加速 |

仅保留 `slurmctld` + `slurmdbd` 两个 daemon。输入：`sim.conf` + 作业 trace（提交时刻、资源请求、实际运行时长）；配套 R 工具包 `RSlurmSimTools` 生成 trace。

#### 方法

1. **Micro-cluster 验证**：10 节点异构集群（CPU 类型、大内存、GPU），500 合成作业；7 次真 Slurm vs 120 次模拟，比较 job start time 分布。
2. **Rush 生产集群**：832 节点，65,000 历史作业（23.8 天），与单次历史 run 对比。
3. **参数实验**：节点 exclusive vs sharing；`bf_max_job_user` 10 vs 20。
4. **Stampede2 建模**：5,936 节点（KNL + SKX），合成 workload，测试单/双控制器 × 三种 node sharing 模式。

#### 实验结果

- **加速比**：Micro-cluster 上 12.9 h workload 约 17 s 完成（~112 simulated days/hour）；8,000 核异构集群约 **100×** 加速（20 天 workload ≈ 5 h 实机时间）。
- **验证**：模拟与真 Slurm 的 job start time 均值无显著差异；标准差略大（~15%）。参数变更（fair-share 权重 +20%）预测的 waiting time 变化区间覆盖真机观测。
- **节点共享**：Rush 集群上 exclusive 模式完成同一 workload 需多 **45%** 时间（等价于需大 45% 的集群容量）。
- **Stampede2**：SKX 分区上，双控制器 + node sharing 组合使平均等待时间降低约 **40%–51%**（见 Table 3）。

#### 局限性

- 简化项：忽略 epilog 延迟、节点故障、模拟开始前历史 fair-share 状态；Rush 对比仅有一次历史 run，start time 偏差 σ=12 h。
- backfill 在模拟中比真机快 10× 以上，需人工 scaling 补偿，否则 job 会过早启动。
- 调度具有**随机性**（周期性 routine 与作业到达的相对相位），需多次 run 或固定初始 delay。
- 基于较旧 Slurm 分支，后续版本需大量移植（2017 版后续已被 2022 版取代）。

---

### 2.3 Simakov et al. (2018) — 多控制器与节点共享（PEARC）

**文献**：N. A. Simakov et al., *Slurm Simulator: Improving Slurm Scheduler Performance on Large HPC Systems by Utilization of Multiple Controllers and Node Sharing*, PEARC '18. DOI: [10.1145/3219104.3219111](https://doi.org/10.1145/3219104.3219111)  
**文件**：`papers/Slurm Simulator - Multiple Controllers and Node Sharing (Simakov PEARC 2018).pdf`

#### 研究目的

针对 TACC Stampede2 等**天然分为多种节点类型**的超算，在部署前评估：（1）按节点类型拆分多个 Slurm controller 是否值得额外运维成本；（2）在大规模系统上启用 node sharing 是否会导致 backfill 过载；（3）两种策略组合的最优配置。

#### 模拟器设计

沿用 PMBS 2017 版 simulator；本节重点在**实验场景设计**而非新架构。Workload 由 Stampede1 历史作业抽样并映射到 Stampede2 的 KNL/SKX 节点比例。

#### 方法

- 系统：4,200 KNL + 1,736 SKX 节点。
- 变量：单/双 controller × SKX 上 no sharing / sharing-by-socket / sharing-by-core。
- 指标：平均等待时间（含 node-hours 加权版本）、backfill 循环耗时、time-limit 命中率、调度尝试次数。

#### 实验结果

- 独立 controller 使 SKX 作业等待时间降低约 **35%**；KNL 性能基本持平。
- 仅 SKX 启用 sharing 即可带来 ~25% 等待时间改善；与双 controller 组合最优，SKX 等待时间总计降低约 **40%**。
- 组合配置下 backfill 调度尝试次数接近翻倍，但 time-limit 触发率下降约 30%，说明 controller 负载分散后回填更高效。

#### 局限性

- node sharing 仅施加于 1,736 SKX 节点，结论不能外推到全 5,936 节点。
- Workload 为合成（Stampede1 迁移），非 Stampede2 生产 trace。
- 未考虑跨子集群作业、统一账户视图等运维复杂度。

---

### 2.4 D'Amico, Jokanović & Corbalán (2018) — 真机验证与确定性改进（PMBS）

**文献**：M. D'Amico, A. Jokanovic, J. Corbalán, *Evaluating SLURM Simulator with Real-Machine SLURM and Vice Versa*, PMBS 2018 (SC18 Workshops). DOI: [10.1109/PMBS.2018.8641556](https://doi.org/10.1109/PMBS.2018.8641556)  
**文件**：`papers/Evaluating SLURM Simulator with Real-Machine SLURM (DAmico PMBS 2018).pdf`

#### 研究目的

BSC 团队在 ScSF/Rodrigo 版 simulator 基础上，**首次将 simulator 输出与真实 Slurm 集群对跑验证**，量化旧版的非确定性与精度误差，并修复同步、RPC、调度触发等问题，使 simulator 达到可用于调度策略研究的可靠程度。

#### 模拟器设计

保留 Lucero 三进程架构（`sim_mgr` + `slurmctld` + `slurmd`），重点改进：

1. **多信号量同步**：消除 race condition 导致的「丢失模拟秒」。
2. **RPC / epilog 计数**：等待所有 epilog 完成后再推进模拟，避免作业 duration 被拉长。
3. **事件触发 FIFO 调度器**：在 job 到达与结束时刻触发调度，而非仅依赖定时 backfill。
4. 移植至 **Slurm 17.11**；支持 SWF ↔ trace ↔ jobcomp 转换工具链。

#### 方法

- **一致性**：Cirne 模型生成 4 组 5000 作业 / 3456 节点 workload，各 10 次 run，比较 job start time 方差。
- **精度**：10 节点真机，200 个 NAS benchmark 作业（CIRNE 到达模式映射），simulator vs 真机对比 wait/response/slowdown 等。
- **性能**：ANL Intrepid（68,936 作业 / 40,960 节点）、CEA Curie（198,509 作业 / 5,040 节点）trace。

#### 实验结果

- 旧版（SIM SCSF）：同输入 10 次 run 的 job start time 标准差可达 **22 min**；系统指标偏差最高 **12%**。
- 改进版（SIM V17）：**确定性**（10 次 run 完全一致）；相对真机偏差降至 **≤1.7%**；速度提升 **2.6×**。
- 用例：backfill interval 变更对系统 slowdown、调度器耗时的影响可在 simulator 中安全评估。

#### 局限性

- 真机精度实验仅 10 节点、~2 h窗口，规模远小于生产。
- 仍基于 BSC lineage 架构，与 UB 2017 版的序列化设计不同，两条代码线后续各自演进。
- 未建模异构 GPU 作业、动态节点、电源管理等扩展插件。

---

### 2.5 Simakov et al. (2022) — 新版准确模拟器（PEARC）

**文献**：N. A. Simakov et al., *Developing Accurate Slurm Simulator*, PEARC '22. DOI: [10.1145/3491418.3535178](https://doi.org/10.1145/3491418.3535178)  
**文件**：`papers/Developing Accurate Slurm Simulator (Simakov PEARC 2022).pdf`

#### 研究目的

2017 版 simulator 基于旧 Slurm，无法跟进 2018 年后多次 major release。作者发现：**调度本质上是随机过程**——priority/backfill  routine 执行时刻相对作业提交的 jitter 会导致不同 scheduling realization。新工作目标是：（1）基于 **Slurm 21.08** 构建可维护的新 simulator；（2）用 Virtual Cluster 获取同一 workload 的多次真机 baseline；（3） statistically 验证模拟 fidelity。

#### 模拟器设计

与 2017 版「重度序列化」相反，2022 版强调**最小侵入**：

- GCC `--wrap` 劫持时间函数与部分 pthread 函数，实现 opportunistic time skipping（无事件则快进到下一事件，上限 1 s）。
- constructor 初始化 simulator，避免改 `main()`；wrapper 暴露 static 变量供模拟层访问。
- 核心仅改 thread creation；作业规格采用与 Virtual Cluster 相同的**文本格式**（类 sbatch 参数 + `-dt` 提交偏移 + `-sim-walltime` 实际时长）。
- Docker Virtual Cluster：每物理节点一容器，自动 job feeder，随机 controller 启动 delay 模拟初始相位不确定性。

#### 方法

- Micro-cluster（10 节点，500 作业）：Virtual Cluster 20 次 reference run vs simulator。
- UB-HPC（217 节点，29,678 历史作业）：8 次 reference run。
- 统计检验：job wait time 向量间的欧氏距离 heatmap + **multivariate Wilcoxon test**。

#### 实验结果

- Micro-cluster：simulator 与 reference **统计上不可区分**。
- UB-HPC：fast CPU 上 simulator–reference 距离（906±180 h）接近 reference 内部距离（742±135 h）；slow CPU 偏差显著（2290±262 h）。按 job 平均仅 ~3 min，实践中可接受。
- 加速比 **20–40×**（低于 2017 版的 100×，因保留更多真实 Slurm 行为）。
- 作者指出：**相对变化（relative change）的预测往往比绝对值更可靠**——对参数调优场景仍足够。

#### 局限性

- 加速比下降；month-long workload 仍需 1–2 天实机时间。
-  fidelity 依赖运行 simulator 的硬件性能（线程同步耗时）。
- 论文为 4 页 extended abstract，技术细节少于 2017 全文。
- 本仓库使用的 ubccr 分支基于 **Slurm 23.11**，是 2022 版思路的进一步演进，但未必包含全部 PEARC '22 实验配置。

---

## 3. 横向对比与演进脉络

### 3.1 架构路线对比

| 维度 | BSC lineage (Lucero → D'Amico) | UB/CCR lineage (Simakov 2017 → 2022) |
|------|-------------------------------|--------------------------------------|
| 控制方式 | 外部 `sim_mgr` 驱动 | slurmctld 内嵌事件循环 或 wrap 劫持 |
| 进程模型 | sim_mgr + slurmctld + slurmd | 2017: slurmctld + slurmdbd；2022: 接近完整 Slurm + wrap |
| 确定性 | 2018 前非确定；D'Amico 修复 | 2017 即知随机性；2022 用统计方法应对 |
| 典型加速 | 中等 | 2017: ~100×；2022: ~20–40× |
| 验证深度 | 2018 首次真机 10 节点对比 | 2017 Micro + 生产 trace；2022 Virtual Cluster 统计验证 |

### 3.2 共同假设与简化

所有版本均**不执行真实用户程序**：作业时长由 trace 中的 `-sim-walltime`（或等价字段）指定。这意味着：

- 适合研究**调度与资源占用**（与本课题的能耗/碳排放建模直接相关）。
- 不适合研究应用级性能、I/O、网络 congestion、功率随负载动态变化等（需外接 power model，如本项目 `generate_workload.py` + `analyze_run.py` 的做法）。

### 3.3 主要局限（跨文献归纳）

1. **随机性**：调度结果对 controller 启动相位、硬件 jitter 敏感；单次 run 不足以支撑强结论，需多次 run 或统计检验。
2. **Backfill 时序**：模拟器中 backfill 执行远快于生产，必须 scaling 或 time skipping 校准，否则 job start time 系统性偏早。
3. **状态初始化**：历史 fair-share、节点 drain/fail、preemption 等常被人为忽略，模拟初期偏差较大。
4. **版本滞后**：simulator 分支往往落后 upstream Slurm 数个 major version，新插件（GPU cons_tres、power、cgroup v2 等）需手动移植。
5. **规模外推**：真机验证多在 10–500 节点；超大规模结论主要来自合成 trace 外推。

---

## 4. 对本 dissertation 的可用表述

### 4.1 方法学定位（Suggested dissertation text — English）

> To evaluate energy and carbon implications of scheduling decisions on the Stanage HPC system without interfering with production workloads, this project employs the Slurm Simulator developed by the UB Center for Computational Research (Simakov et al., 2017; 2022). Unlike discrete-event schedulers implemented from scratch, this simulator compiles the production Slurm source code in simulation mode, preserving the backfill scheduler, fair-share priority, and consumable-resource (cons_tres) selection logic that govern job placement on Stanage. Jobs are injected via a synthetic event trace specifying submission time, partition, node/core/GPU requests, requested walltime, and simulated runtime; the simulator advances virtual time at up to three orders of magnitude faster than real time, producing scheduler logs from which job start/end timestamps and resource occupancy can be extracted. Prior validation studies report mean job-start-time errors comparable to run-to-run variability of real Slurm (Simakov et al., 2017) and system-metric deviations below 2% against physical clusters when synchronization issues are addressed (D'Amico et al., 2018). We therefore treat simulator output as a faithful proxy for scheduling dynamics, and attach node-level power and carbon-intensity models downstream to estimate energy and CO₂e footprints.

### 4.2 方法学定位（中文）

> 本研究采用 UB/CCR 开发的 Slurm Simulator（Simakov 等，2017；2022），在不影响 Stanage 生产环境的前提下评估调度行为对能耗与碳排放的影响。该模拟器并非从零实现的离散事件模型，而是在编译期启用 `SLURM_SIMULATOR` 标志，保留 Stanage 实际使用的 backfill 调度器、公平共享优先级与 cons_tres 资源选择逻辑。作业通过合成事件 trace 注入，指定提交时刻、分区、节点/核/GPU 请求、申请 walltime 与模拟运行时长；虚拟时钟可相对实钟加速约 10²–10³ 倍，输出 slurmctld 日志供提取作业开始/结束时刻与资源占用曲线。已有研究表明，在修复同步问题后，模拟器相对真机的系统指标偏差可低于 2%（D'Amico 等，2018），job start time 的离散程度与真机多次运行间的差异相当（Simakov 等，2017）。因此，本课题将模拟器输出视为调度动力学的可靠代理，并在下游叠加节点级功率模型与电网碳强度以估算能耗与 CO₂e。

### 4.3 适用边界（Limitations paragraph — English）

> We acknowledge several limitations inherited from the Slurm simulation literature. First, simulated jobs do not execute application code; runtime and power are prescribed rather than measured, so results characterize scheduling-induced occupancy rather than workload-dependent energy dynamics. Second, scheduling is stochastic: a single simulation run may deviate from a historical production trace, particularly during the warm-up period when prior fair-share state and node failures are omitted. Third, our simulator fork (Slurm 23.11) may not perfectly reproduce every production-side delay (e.g., epilog latency, RPC contention) that affects backfill timing on the live Stanage controller. These limitations are acceptable for comparative analysis of scheduling policies and synthetic workload scenarios, which is the primary scope of this dissertation chapter, but absolute wait-time predictions should be interpreted with caution.

---

## 5. 参考文献

1. Lucero, A. (2011). *Slurm Simulator*. Slurm User Group Meeting (SLUG), Phoenix, AZ.  
2. Trofinoff, S., & Benini, M. (2015). *Using and Modifying the BSC Slurm Workload Simulator*. SLUG'15.  
3. Simakov, N. A., DeLeon, R. L., Innus, M. D., Jones, M. D., White, J. P., Gallo, S. M., Patra, A. K., & Furlani, T. R. (2018). A Slurm Simulator: Implementation and Parametric Analysis. In *PMBS 2017*, LNCS 10724, 197–217. https://doi.org/10.1007/978-3-319-72971-8_10  
4. Simakov, N. A., et al. (2018). Slurm Simulator: Improving Slurm Scheduler Performance on Large HPC Systems by Utilization of Multiple Controllers and Node Sharing. *PEARC '18*. https://doi.org/10.1145/3219104.3219111  
5. D'Amico, M., Jokanovic, A., & Corbalan, J. (2018). Evaluating SLURM Simulator with Real-Machine SLURM and Vice Versa. *PMBS 2018*, IEEE. https://doi.org/10.1109/PMBS.2018.8641556  
6. Simakov, N. A., DeLeon, R. L., Lin, Y., Hoffmann, P. S., & Mathias, W. R. (2022). Developing Accurate Slurm Simulator. *PEARC '22*. https://doi.org/10.1145/3491418.3535178  

---

## 6. 本地资源索引

| 文件 | 说明 |
|------|------|
| `simulator/papers/*.pdf` | 6 篇 PDF（含 Trofinoff 2015 补充材料） |
| `Slurm_Simulator_Internal_Architecture.md` | 本仓库 simulator 架构与配置说明 |
| `真实HPC_Workload模拟调研.md` | Stanage 合成 workload 与能耗外推方法 |
| `slurm_simulator/` | 可运行 simulator 源码与 Stanage 配置 |
