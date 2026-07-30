# 真实 HPC Workload 模拟调研

> 项目：Energy and Carbon Monitoring in HPC Systems
> 目标：用 Slurm Simulator 替代 placeholder 作业，模拟 AI / 物理模拟 / CPU·GPU 密集型混合负载

---

## 1. 核心结论：placeholder 能否替换？

**可以，且这是模拟器的正确用法。**

Slurm Simulator 中作业**从不真正执行**——`pseudo.job` 脚本不会被读取，没有真实进程。调度器看到的只有以下参数：

| 参数 | sim.events 字段 | 作用 |
|---|---|---|
| 提交时刻 | `-dt N` | 作业何时进入队列 |
| 分区 | `-p gpu / standard / ...` | 路由到哪个硬件池 |
| 节点数 | `-N` | MPI 规模 |
| 核数 | `-n` | CPU 占用 |
| GPU | `--gres=gpu:a100:2` | GPU 占用 |
| 申请 walltime | `-t` | 调度器做 backfill 规划 |
| 实际运行时长 | `-sim-walltime` | 模拟器安排完成事件的时刻 |

因此，只要这些参数服从真实 HPC 集群的统计规律，模拟出的**排队、回填、资源占用、完成时间**就是研究可用的活动数据；能耗/碳排放在此基础上用功率模型外推即可。

---

## 2. 真实 HPC Workload 的统计特征

来源：Feitelson Parallel Workloads Archive、Fugaku F-DATA、NERSC/ARCHER2 公开负载分析、Stanage 集群文档。

### 2.1 通用规律

- **运行时**：重尾分布（对数正态），大量短作业 + 少量超长作业
- **并行规模**：MPI 节点数偏好 2 的幂（1, 2, 4, 8, 16, 32, 64…）
- **到达过程**：非齐次泊松过程，白天强度约为夜间的 3 倍
- **申请 walltime**：实际用时的 1.5~10 倍，且取整到 15min/1h/4h/12h/24h 等档位
- **失败/超时**：约 5~8% 作业启动后迅速失败；约 5% 大 MPI 作业撞 walltime 超时

### 2.2 按 workload 类型

| 类型 | 典型分区 | 规模 | 时长 | 特征 |
|---|---|---|---|---|
| **HTC 参数扫描** | standard | 1 节点, 4~32 核 | 10min~3h | 数量最多，单作业能耗小 |
| **MPI 物理模拟** (CFD/MD/QCD) | standard | 2~32 节点 | 1~18h | node-hours 主力贡献者 |
| **Capability 大规模** | standard | 64~128 节点 | 4~16h | 数量少，占用大量节点 |
| **AI 训练 (A100)** | gpu | 1~2 节点, 4 GPU/节点 | 2~20h | GPU 能耗密度最高 |
| **AI 调试/推理** | gpu / gpu-h100 | 1 节点, 1~2 GPU | 2~90min | 短、频繁 |
| **LLM 训练 (H100-NVL)** | gpu-h100-nvl | 1 节点, 4 GPU | 4~20h | 最长、最高功率 |
| **大内存** (基因组/量子化学) | bigmem / hugemem | 1 节点 | 1~12h | 内存分区独占 |

---

## 3. 实现：workload 生成器

文件：`slurm_simulator/workload-gen/generate_workload.py`

```bash
cd slurm_simulator/workload-gen
python3 generate_workload.py --hours 12 --jobs 140 --seed 42
```

输出：
- `stanage-sim/sim.events.realistic` — 140 行事件（**不能含 `#` 注释**，解析器有 bug）
- `stanage-sim/workload_profile.csv` — 每作业的 class/user/资源/功率参数

默认 140 个作业、12 小时提交窗口，约 4100 node-hours，8 类作业按真实比例混合。

---

## 4. 运行与分析

```bash
cd slurm_simulator
./run-stanage.sh realistic          # 重放 sim.events.realistic
python3 workload-gen/analyze_run.py # 解析 slurmctld.log + workload_profile.csv
```

分析脚本输出 `stanage-sim/job_results.csv`，包含每作业的：
- 提交/开始/结束时刻、等待时间
- 能耗估算：`E = (cores × W_core + gpus × W_gpu) × runtime × PUE`
- 碳排放：`CO2 = E × CI`（默认 PUE=1.2, CI=150 gCO₂/kWh）

功率参数（TDP 近似）：

| 资源 | 功率 |
|---|---|
| Ice Lake 8358 CPU | 7.8 W/核 |
| A100 GPU | 400 W |
| H100 GPU | 350 W |
| H100-NVL GPU | 400 W |

---

## 5. 与 FastSim 的对比

| | Slurm Simulator (本方案) | FastSim |
|---|---|---|
| 数据需求 | 只需作业画像（可合成） | 需要真实 sacct 历史 dump |
| 调度保真度 | 真实 Slurm backfill/cons_tres | 简化调度模型 |
| 能耗数据 | 外推（功率模型 × 占用时长） | 直接读 ConsumedEnergyRaw |
| 适用场景 | 无真实数据时的探索性研究 | 有历史数据的精确回放 |

两者互补：本方案用于**没有真实测量数据**时的碳排放估算框架验证；FastSim 用于有 Stanage sacct 数据后的精确对比。

---

## 6. 已知限制

1. 模拟器不执行真实计算，CPU/GPU 利用率恒按 100% 估算（保守上界）
2. 功率模型用 TDP 近似，未考虑 DVFS、空闲功耗、网络/存储开销
3. 碳强度 CI 用固定值，未建模英国电网的小时波动
4. `sim.events` 解析器不支持 `#` 注释行
5. 长时间模拟需 patch `assoc_mgr.c`（无 slurmdbd 时 assoc 刷新失败导致 false `invalid account`）

---

## 7. 文件索引

| 文件 | 说明 |
|---|---|
| `workload-gen/generate_workload.py` | 真实 workload 生成器 |
| `workload-gen/analyze_run.py` | 结果分析 + 能耗/碳排放估算 |
| `stanage-sim/sim.events.realistic` | 140 作业事件文件 |
| `stanage-sim/workload_profile.csv` | 作业画像 |
| `stanage-sim/job_results.csv` | 模拟结果（运行后生成） |
| `run-stanage.sh realistic` | 一键运行 |
| `Slurm模拟器内部结构剖析.md` | 模拟器 job 生命周期详解 |
