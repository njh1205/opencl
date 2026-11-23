# HPOCL as HPC Infra — 定位、硬件映射与证据体系

> 用途：把 hpocl 从"混合预编码的 OpenCL kernel 集合"重新包装成**异构高性能计算 infra**。
> 本文只陈述仓库里已有证据（标注 file:line）与明确的投影模型（标注 PROJECTION），
> 不引入任何未经验证的加速数字。所有实测环境说明见 `docs/cost_model_real_shapes.md`。

**一句话定位（可直接对外用）**

> HPOCL: a cost-model-driven heterogeneous runtime for **hard-real-time
> physical-layer numerical linear algebra** on Zynq UltraScale+ RFSoC.

中文：面向硬实时物理层数值线性代数的、由成本模型驱动的异构计算运行时。

---

## 1. 为什么这算 HPC infra，而不是"一堆 kernel"

HPC infra 的价值不在峰值算力，而在四件事。这四件仓库里都有实体：

| HPC infra 要素 | hpocl 里的实体 | 证据 |
|---|---|---|
| **后端抽象 + 算子注册 + fallback chain** | `Runtime` 持有 device policy（`cpu\|gpu\|acc`）、buffer 生命周期、单一 kernel registry、event profiling | `runtime/hpocl_runtime.{h,cpp}`（55 KB）、`README.md:17-22` |
| **成本模型驱动的算子放置** | `jacobi_batch_flops` / `gram_batch_flops` / `min_flops` 决定逐算子留 PS 还是下 PL；`sched_bench` 用四角色（`costed` / `force` / `glue` / `glue!`）+ 逐 site 扫描验证放置决策 | `runtime/hpocl_schedule.cpp`、`README.md:77` |
| **数值可复现门禁 as CI** | 12 个 numpy oracle（`tools/*_ref.py`）+ 41 gate 的 FEC 层 + 静态门 `capacity_check` / `registry_check` / `glue_cost_check` / `allprog_check`；阈值分级为 bit-identical / 1 ulp / dB | `Makefile:277-345`、`README.md:71-115` |
| **最坏情况延迟确定性** | constant-latency bracket expansion：数据依赖的 `for (e<6 && psi_u(lo)>0)` → 6 次无条件 `psi_u` + priority mux | `README.md:923`（item 7） |

第五条是**可移植性**，而且它是可验证的而不是口号：同一份 `.cl` 源 + 同一份 host C++，跑在 x86 CPU / Apple GPU / Zynq PS / Zynq PL 四个后端；`allprog_check` 证明 10 个源能编成**一个** program、29/29 kernel 全部获取，且在 device 默认选项与强制 `-DHPOCL_FP32` 下各跑一次（`README.md:76`）。PL 侧 RTL 全部由工具从 OpenCL C 综合，**仓库里没有一行手写 Verilog**（`README.md:28-31`）。

### 三层 + 两横切（对外讲架构就画这张）

```
L2  调度层    hpocl_schedule.cpp   成本模型 → 逐算子 PS/PL 放置；C 通道批处理
                │
L1  算子层    29 kernels / 10 .cl   分解类 · 求解类 · datapath 类 · NN 类
                │
L0  后端层    hpocl_runtime.cpp     device policy · registry · profiling · fallback
                                    DOUBLE_FP_CONFIG 探测 → fp32/fp64 自动 marshal
────────────────────────────────────────────────────────────────────────────
X1  数值门禁  tools/*_ref.py (12)   bit-identical / ulp / dB 三级阈值 + 静态容量门
X2  构建可移植 build_xclbin.sh      同一份源 → x86 CPU · Apple GPU · A53 · PL .xclbin
```

算子层按角色分四类（这决定了它们吃哪种硬件，见 §3）：

| 类 | kernel | 主导瓶颈 |
|---|---|---|
| 分解类 | `jacobi_svd` `gmd_k` `ucd_k` `cbd_k` `power_iter` `deflate` | compute（DSP）+ 串行 sweep 依赖链 |
| 求解类 | `threshold_cf` `switch_threshold` | transcendental / 查表（BRAM·URAM）+ LUT |
| datapath 类 | `qam_map_k` `tx_combine_k` `channel_apply_k` `rx_combine_k` `modulo_fold_k` `softdemod_llr_k` `softdemod_folded_k` `gmi_k` `thp_precoder` `wbb_apply` `channel_cov` | memory（AXI/DDR 流量放大） |
| NN / 速率类 | `tf_*`(9) `mcs_select_k` | bandwidth（权重流）+ 查表 |

---

## 2. 负载画像：这不是 big-FLOPs HPC

包装成 HPC 的第一步是把 workload 说准。以 `paper_sweep`（K=8, Nt=128, Nr=4, NRF=32, d=4, T=32, null shape 8×28×32）一个**调度时刻**为例（`docs/cost_model_real_shapes.md` §2–§3）：

| 成分 | 算术量 | host 实测（throttled M4） | 瓶颈类型 |
|---|---|---|---|
| null-basis Jacobi 批（fp64, 9 sweeps 实测） | **22.86 Mflop**（cap-16 = 40.63 Mflop） | 4.19 ms = front_end 的 **94%** | compute + 串行 sweep |
| glue GEMM（eff / F_RF·N / gram(FN) / reduced / whiten / Fbb / power_norm） | 3.44 Mflop | 0.21 ms（4.7%；**A53 上约 15%**） | compute，但被 glue 开销主导 |
| Frf 常数模投影 | ~0 flop | 4 096 entries 投影 **5.86 µs**（本机实测） | **transcendental-bound**：4 096 `std::abs`(=hypot) + 4 096 mul。shipped 代码写成 `x/|x|` 而非生产的 `exp(i*angle(x))`，**实测 8.83× 更便宜**（51.75→5.86 µs，entry 的 2.2 ulp），`frontend_bench` 改前改后读数相同（`ref/frontend.cpp:149-178`）。`cost_model_real_shapes.md` §3 仍写 `16 384 (atan2+cos+sin+div)`，那是**生产形式**的定价，对本树已陈旧 |
| threshold solve（表路径，42 fills = 22 psi_u + 20 psi_du） | ~45 kflop | 0.022 ms（M=64） | transcendental：1 680 log + 1 930 exp |
| whitening + gbar SVD（4×4, C=8 ×2） | 0.034 Mflop | 0.008 ms | 留在 PS（`min_flops` 判定正确） |

对照物：**transformer student** 是 21.22 Mflop —— 比 null 批还少 —— 但每次 forward 要流 **5.05 MB fp32 权重 = ZU49DR 全部片上存储的 70%**，所以它在 PL 上是 **bandwidth-bound 0.5–2.1 ms**，而不是 flop 数暗示的 12.4 µs（`README.md:867-873`）。

**结论（这就是 infra 存在的理由）**：一个调度时刻内同时存在 compute-bound、memory-bound、transcendental-bound、latency-bound 四类算子，**不可能整体卸载到任何一个后端**。逐算子放置 + 成本模型 + 数值门禁，是这类负载唯一可行的工程形态。

---

## 3. 硬件映射表：ZU49DR 上每一块资源谁在用

器件：**XCZU49DR-2FFVF1760**（Zynq UltraScale+ RFSoC Gen3，-2 速度等级），板卡 QX_49DR_SV3。

| 硬件资源 | 规格 | 谁在用 / 该怎么用 | 现状 | 下一步 | 需板子 |
|---|---|---|---|---|---|
| **DSP48E2** | **4 272 个**，27×18 定点 MAC + 48-bit 累加器，**无浮点** | 所有复数 MAC：null Jacobi 批、glue GEMM、datapath | ⚠️ **未综合验证**。唯一真实综合是 4 个 kernel（`vadd`/`switch_threshold`/`thp_precoder`/`mcs_select_k`，合计 22 个乘法），吃 DSP 的 `decomp.cl`(221 乘) / `threshold_cf.cl`(228) / `datapath.cl`(100) / `tf_ops.cl`(77) **一个都没综合过** | 逐 kernel `v++ -c -t hw` 收 LUT/FF/DSP/Fmax，填 `vitis_flow_notes.md:97` 那张附录表 | 编译不需板子（需 Vitis） |
| **BRAM36 / URAM288** | `__constant` 池 = 500 Kibit = **14 BRAM36**（depth granularity 下最多 20）或 **2 URAM288** | deficit / Gauss-Hermite / MCS 查找表**常驻片上**，把 transcendental 换成查表 | ✅ **已落地且是最大的一笔收益**：实测 **6 187×@M=64 / 28 290×@M=256**；`cf_bench` 13 min → 3.2 s，`make check` ~25 min → 37 s（`README.md:918` item 1） | analytic `psi_du`（把 `d ln D/d ln t` 做成第 11 行表）→ Newton 整体 **2.5–2.8×**，代价 +6.25 KB，释放 6 个私有数组中的 4 个（item 6） | 精度可 DHV，容量需 BOARD |
| **PL LUT** | ~427 K | `exp/log → exp2/log2`（**-30~40% LUT/cycle**）；constant-latency 展开 | ⬜ item 9 未做 | 注意：**只对 PL 有效**，A53 上 ARMv8.0-A 无硬件 exp2/log2，实测 `log2` 比 `log` 更慢（`README.md:925`） | 否 |
| **DDR4 ×4 bank / AXI HP** | **~2.4 GB/s per AXI port** | H、F_RF、F_BB、per-symbol 缓冲 | ⚠️ **主瓶颈**：`channel_cov` 为 65.5 KB 有效数据读 **1.11 MB（17× 放大）**，~0.5 ms/次协方差更新（`README.md:928-931`）；transformer 5.05 MB/forward | local-memory tile（`WBB_LOCAL_MAX` 默认 128 = 1 KB/group，大 tile 留给 `-DWBB_LOCAL_MAX=<BRAM 预算>`）；`channel_cov` 外积平铺 | 是 |
| **PS OCM 256 KB** | 片上零延迟 scratch | whitening 4×4、gauge pin、nullity、trailing select 这些小对象 | ⬜ 未使用 | PS 侧固定大小栈数组（见 item 5：**164 `operator new` / 26 240 B per channel**，GCC/libstdc++ on aarch64 不做省略 → 结构上 ~2.07 M malloc/free 对 per channel，quadrature 路径最坏 ~90 ms/channel） | aarch64 g++ 才可测 |
| **PS 4× Cortex-A53 + NEON** | 128-bit SIMD：fp32 4 lane / fp64 2 lane；投影 **0.933 GFLOP/s**（1.4 GHz in-order, 1 fp64 flop/cycle, ×1.5 结构开销 —— PROJECTION） | fp64 控制面：BD/whitening、GMD/UCD、闭式解、safeguarded Newton、LMMSE shrink | ⚠️ 已跑通但**未向量化**：`Makefile:51` 是 `-O2`；`ref/`、`runtime/`、`include/` 里零个 `arm_neon.h` / `immintrin.h` / `__vector` / Eigen | **已实测，结论与直觉相反**：`-O3`（含 `-mcpu=native`）对热点是 **1.000× / 0.98–1.02×**，即**没有收益**，但**逐位相同**（102 291 个 dump 标量 0 个变）—— 因为 libc++ `complex` 的 `operator*` 带 Annex-G `isnan` 分支，`ref/decomp.cpp`/`frontend.cpp`/`pipeline.cpp` 里 `fmla.2d` 计数在 `-O2` 与 `-O3` 下**都是 0**。真正的杠杆是**手写保序 NEON**：见 §5.4 与 `docs/ps_neon_audit.md` | aarch64 才可测 |
| **PS 2× Cortex-R5F** | 裸机 / FreeRTOS + RPMsg | µs 级实时同步、触发、帧定时 | ⬜ **当前闲置，且这是明确决策**："hpocl 无硬实时控制任务，不为用而用"（`README.md:1655-1657`） | 只在 16T16R 射频数据通路上场：A53 Linux 非实时（ms 级抖动）不够（`README.md:1658-1661`） | 是 |
| **RFDC** | 8× RF-ADC @5 GSPS / 8× RF-DAC @9.85 GSPS | 空口收发；板上 Loopback 自测（无外部信号源，基带 IQ 由本地软件生成，与 `tools/` numpy oracle 同源） | ⬜ 未接（hpocl 是纯计算负载） | 需要板厂 RFDC + JESD204 参考设计 | 是 |
| **CMAC 100G** | — | 前传 / 上行 | ⬜ 未接 | 平台 `.xpfm` 里按需带（`README.md:1620-1621`） | 是 |
| **XRT + AXI DMA** | — | PS↔PL zero-copy、双缓冲、`Runtime::init("acc")` 枚举 PL 设备 | ⬜ runtime 已有 `acc` 策略与预综合二进制加载路径，未上板 | host binary 同一份源交叉编译 aarch64（`README.md:1624-1626`） | 是 |
| **AI Engine** | **无**（AIE 只在 Versal） | — | 所以 PL 向量算力只有 DSP48E2 + LUT，没有 VLIW 向量 tile 可退 | — | — |

⚠️ 两个必须写进对外材料的器件事实，否则会被内行一眼挑错：ZU49DR 的 slice 是 **DSP48E2 而非 DSP58**（DSP58 是 Versal），且 **DSP48E2 完全没有浮点**（`README.md:932-933`、`docs/x86_opencl_notes.md:43`）。

---

## 4. 精度架构：这是最"infra"的一块

通用框架（TVM / Halide / oneAPI）不解决的问题：**当目标 fabric 的物理上不支持某种精度时，运行时该怎么办。**

| 层 | 精度 | 依据 |
|---|---|---|
| PL | **强制 fp32**（`-DHPOCL_FP32`） | 双重硬阻塞：DSP48E2 无浮点；且 fp64 下 `__constant` 池 128 896 B = **1.97× 超预算**，`clCreateKernel` 直接 `CL_INVALID_BUFFER_SIZE (-61)` —— program 能 build，kernel 建不出来（`docs/x86_opencl_notes.md` Result 2、`build_xclbin.sh:19-31,54`） |
| PS | **fp64** | 控制面（GMD/UCD、闭式解、Newton、LMMSE）需要 fp64 的条件数余量 |
| 运行时 | **自动探测 + marshal** | 探 `CL_DEVICE_DOUBLE_FP_CONFIG`，按结果自动降级 fp32 并 marshal（`README.md:41-43`） |

而且 **fp64 路径不是删掉的死代码，是被门禁证明活着的**：`-DHPOCL_FP64` 下 7 个 case 全 PASS，`tau = 0.00e+00`，`cl = 7.61e-16 dB` vs 门禁 `1e-6 dB`（**1.3e9× margin**），且 **39/40 个 tau 值在 fp64 与 fp32 kernel 之间逐位相同**，第 40 个差最后一位（`docs/x86_opencl_notes.md` Result 1）。它只是不 ship。

这套东西的名字叫 **precision-aware operator placement + graceful degradation**，是可以直接写进 infra 叙事的能力，而不是"我们只支持 fp32"。

---

## 5. 性能证据体系（HPC 的语言）

### 5.1 必须分清的四个数量级

| 数 | 值 | 性质 |
|---|---|---|
| PL fp32 理论峰值 | 4 272 DSP48E2 × 2 flop × 200 MHz = **1.71 Tflop/s** | 理论 |
| ~~PL 唯一实测锚点~~ **已失效（2026-11-06）** | 原值 28×32 C=8 null 批 1.18–1.21 ms / 40.63 Mflop ≈ 34 Gflop/s。那 1.2 ms 是 **Apple OpenCL 的 `CL_PROFILING` event 时间，不是设备时间**：同一 launch 的 pipelined wall 是 **51.3 ms**（42×），执行 8 sweeps = 20.32 Mflop → **0.40–0.79 GFLOP/s**，比同机 fp64 host（4.19 ms）还慢 | **已被推翻** → `docs/hpc_roofline_measured.md` §2；PL 成本模型须用 wall-clock 重新基线（该文 §10 第①条） |
| **本机（M4 GPU）实测屋顶** | compute **547.7 GFLOP/s** · memory **88.7 GB/s** · ridge **6.18 flop/byte** · `exp` **22.2 Gexp/s** · dispatch floor **14.85 µs**（pipelined）/ **247 µs**（同步） | **实测**，`docs/hpc_roofline_measured.md` §3 |
| A53 | **0.933 GFLOP/s** | **PROJECTION，非实测** |
| host M4（无风扇，负载下 throttle 3–3.7×） | front_end 4.45 ms / schedule 4.67 ms @paper_sweep | 实测，但**只有比值与计数可跨机引用**，绝对 ms 不可 |
| 唯一真实综合 | `v++ -c -t hw` 4 个 kernel：**Fmax ≈ 205 MHz，107 KB** | 实测；但**不含任何吃 DSP 的 kernel**，且 DSP/LUT/FF 用量未记录 |

**诚实结论，必须主动写出来（2026-11-06 修订）**：

1. **仓库原先那个「唯一 PL 实测锚点」不成立。** 它是 Apple OpenCL 的 event 时间；该 event 列在本机对设备时间的低报幅度是 **42–490×（130 个 launch 的中位数 63.9×）**，隐含 3.9 TB/s（本机 DRAM 约 120 GB/s）与 22.1 Tflop/s（M4 GPU 全部 fp32 峰值的 4–5 倍）——都是物理不可能的数。pipelined wall 隐含 88.7 GB/s 与 517.8 GFLOP/s，正好落在物理该在的位置。**四重独立验证 + 仓库自己的 `sched_bench` 日志交叉验证**（含一次 launch 的 instant，wall 48–58 ms）。详见 `docs/hpc_roofline_measured.md` §2。
2. **因此「34 Gflop/s，距峰值 50×」这句话作废**，替换为：28×32 C=8 null 批在本机的真实设备时间是 **51.3 ms → 0.40–0.79 GFLOP/s**，比同机 fp64 host 路径（4.19 ms）**还慢**。也就是说：**把这一批 SVD 卸到本机 GPU 是负优化**，而 README 把 50.2 ms 归因于「driver launch latency」的解释是错的（那 50.2 ms 就是 kernel 本身）。
3. **什么活了下来**：所有 host wall-clock 数、所有正确性门禁、以及**同一 kernel 同一形状下两次 event 时间的比值**（偏差相消）—— cap-8/cap-16 = 0.98–1.006、`PSI_SCAN_N` 的 409/408 µs = 1.002 都成立，且已用 wall clock 复核。README 里 `wbb_apply` 的 tiling 表同理：**相对占用率效应成立，绝对 GFLOP/s 高约 40×**。
4. **一个反而更有价值的副产品**：kernel wall 时间随**静态 private footprint** 放大 11×（1.5 KB → 2.07 ms，22.8 KB → 23.3 ms）。这不是 launch 开销，是**寄存器/private 溢出**的执行代价 —— 在 PL 上直接对应 BRAM/寄存器压力，正是 `DECOMP_MAXM/MAXD` 必须按 kernel 分设上限的原因，也是 roofline 文档 §10 第③条的依据。
5. **PL 侧的真实缺口是 traffic 不是 DSP**：按仓库自己的 PL 常数（2.4 GB/s per AXI port、1.71 Tflop/s）投影，130 行中 **124 行在 PL 上是 traffic-bound** —— `channel_cov` **1428×**（复现了 README 的 1.11 MB / 17× / 0.465 ms）、glue GEMM 每 site **~1430×**、null 批 6.4×。加上结构性成因：串行迭代依赖链、内层 trip count 全是运行时变量（无法激进 unroll）、全库 8 个 `#pragma` 无一给综合器、零个 `fma`/`mad`（`cmac` 刻意保序）。

> 这个 gap 不是要藏起来的短板，恰恰是 infra 的卖点：**我们知道每个算子卡在 roofline 的哪个位置，并据此做放置决策**（`sched_bench` 的 `costed` vs `force` 对比就是"成本模型说的" vs "强行全下 PL"）。HPC 里这叫 performance modeling + operator placement，比堆 FLOPs 更接近 infra 的本质。

### 5.2 已经拿到、有据可查的加速比

| 优化 | 收益 | 性质 | 出处 |
|---|---|---|---|
| deficit 查找表（GH-48 quadrature → 表 + 4-pt Lagrange） | **6 187×@M=64 / 28 290×@M=256**；`cf_bench` 13 min→3.2 s；`make check` ~25 min→37 s；48.2 M exp / 6.1 M log 被消掉 | **实测** | `README.md:918` item 1、`cost_model_real_shapes.md` §3 |
| PS fallback 扫描网格统一（401→161 点，`psi_scan.clh`） | PS fallback **402 → 162 fills / 2.46×**，且 fp32 kernel **bit-identical** | 实测（DHV，已落地） | `README.md:923` item 7 |
| `DECOMP_MAXM` 16→32 / `MAXD` 8→40，8 个 null SVD 合成一次 `jacobi_svd`（C=8），sweeps 16→6 | front end **~19 ms → ~0.6 ms on PL**（~32×） | **投影**（算术相同，不涉数值） | `README.md:918` item 2 |
| 表路径三处 hoist（预算 `log λ_i`、`g_k`，grid index 对 `log u` 仿射，用 `t²=2uλ` 替代 `sqrt`） | threshold **1.21 → 0.35 ms** on A53；每通道去掉 ~5 900 log + 6 400 sqrt + 1 300 div | 投影；数值影响 ≤2.2e-5 dB（良态）/ ≤9.1e-3 dB（一个病态通道），与 `-ffp-contract=off` 同量级 | `README.md:918` item 3 |
| Newton 20 → 8 步 | 再 **2.5×**（0.35 → 0.15 ms） | 投影；需在表 `Psi` 上重测阶梯 | `README.md:918` item 4 |
| analytic `psi_du`（第 11 行表） | Newton 整体 **2.5–2.8×**（`psi_du` = 2.82–2.98× `psi_u`），释放 4/6 私有数组 | 投影；+6.25 KB 需 BOARD | `README.md:918` item 6 |
| `F_RF` 常数模投影改写（`exp(i*angle(x))` → `x/|x|`，并把 `1/sqrt(Nt)` 提出循环、除法改乘法） | **8.83×**（51.75 → 5.86 µs / 4 096 entries），entry 的 2.2 ulp，`frontend_bench` 改前改后读数相同 | **实测**（已 landed） | `ref/frontend.cpp:149-178` |
| device vs host event time 一致性 | `cf_bench` 5 configs × 5 交替轮：**409 µs vs 408 µs，ratio 1.002**（per config 1.000–1.011） | 实测 | `README.md:362` |

### 5.3 延迟预算（硬实时才是差异化）

| 算子 | 预算 | 约束类型 |
|---|---|---|
| `threshold_cf` | **~10–30 µs/channel** @200 MHz，4–8 lanes，40–80 DSP48E2 | **constant latency**（数据依赖分支已展平） |
| front end（null 批 + glue） | ~0.6 ms on PL（投影） vs 4.45 ms host 实测 | compute |
| `channel_cov` | ~0.5 ms/update | **memory**（17× 流量放大） |
| transformer student | 0.5–2.1 ms on PL / 7–15 ms on A53 | **bandwidth** → 明确判定"**不是快路径**" |
| 一个调度时刻（device event） | 409 µs（5 configs 合计，M4 GPU 实测） | — |

teacher/student 双轨 + runtime 可在线交叉校验（shadow mode）的设计（`README.md:23-27`），在 HPC 语境下的正确表述是：**模型替代不了精确解时，运行时支持两条路径共存并在线比对** —— 而且这里必须写清 student 的实测精度：**1.0722 dB held-out MAE / 2.3010 dB RMSE / 92.98% 决策准确率**，而控制律目标是 **0.05 dB**、精确解已达 **4.97e-14 dB**（39/40 通道）。所以 student 不上快路径，这是数据决定的，不是取舍模糊。

### 5.4 PS（A53）向量化：实测结论与优先级

全部数字来自 `docs/ps_neon_audit.md`（M4 上实测，A53 需板子复核；实验件在 `/tmp/psneon/`，14 个源文件 + 6 份 bit-dump + 91 份运行日志）。

| 手段 | 收益 | bit-exact？ | 判定 |
|---|---|---|---|
| `-O2` → `-O3`（含 `-mcpu=native`） | null Jacobi 批 **1.000×**、`front_end` **0.98–1.02×**（噪声内）、glue GEMM **0.994×**；只有 `threshold_closed_form` 动：**1.18×**（59.3→50.2 µs） | ✅ **0 / 102 291** 个标量变 | **不值得为提速改**（但因为免费且 bit-exact，改了也无害） |
| `-O3 -ffast-math` | `front_end` **1.19×** / glue **1.74×** | ❌ 84 083 个标量变；τ 最大 **3.97e-6 dB** vs `cf_bench` 的 **1e-9 dB** 行（**超 4000×**），`modem_bench` 的 bit-exact 行失败，且 `ref/threshold_cf.cpp:1033` 的 `INFINITY` 变成 UB | **禁用** |
| `-ffp-contract=off` | — | ❌ 75 487 个标量变 / τ 1.99e-6 dB | **不是安全方向**：现有 `-O2` baseline 已经是 FMA-contracted（`std::norm` → `fma(re,re,im*im)`，400 000/400 000），关掉反而动了数 |
| **手写 NEON，跨独立索引 `m` 向量化 `gemm_wx`** | **1.63–1.65×** | ✅ **0 / 1024** 元素变 | **最佳性价比**，且不需要板子即可验证 |
| 手写 NEON，column-major-per-sweep Jacobi | **1.45–1.49×** | ✅ | A53 上的数需要板子 |
| 手写 NEON，elementwise rotation apply | **1.71×** | ✅ **0 / 1920** | 占 `jacobi_batch_flops` 每对工作的 56%，**零 reduction 风险** |
| vDSP / Accelerate | **0.91–0.94×**（比标量还慢） | — | **不要用** |
| 允许重结合（reassociate） | 1.006–1.074× | ❌ | 收益太小，代价是 bit-exact，**不做** |

**优先级（按收益/风险）**：R1 不为提速改 flag（`-O3` 免费就顺手）→ **R2 bit-exact NEON `gemm_wx`（跨 `m`）** → R3 column-major NEON Jacobi（A53 数字需板子）→ R4 投影循环**不要动**（已经 8.83× 优化过），改为修 `cost_model_real_shapes.md` §3 那行陈旧定价 → R5 永不重结合 → R6 不放松门禁（R2+R3 已给 ~1.5–1.65×）。

**两条必须写进对外材料的判断**：

1. **A53 上控制面必须留在 fp64 → SIMD 天花板是 2 lane，不是 fp32 的 4 lane**；且 ARMv8.0-A NEON **没有向量 `exp`/`log`/`hypot`**，所以 threshold 路径（~13 000 个 transcendental ≈ 1.2 ms 中的 ~1 ms）和投影（4 096 `hypot`）**根本不可 SIMD 化**。可 SIMD 的只有 `gemm_wx`（跨 `m`）、elementwise apply、Gram 循环这三处。
2. **主导杠杆不是 SIMD**：同一批工作的 PL 卸载是 ~19 ms → ~0.6 ms（**~30×**，README 优化台账 item 2），而 PS 侧 bit-exact NEON 只有 **~1.5×**。另外一条**尚未评估但可能更值**的：把 C=8 的批拆到多个 A53 核上，**逐矩阵 bit-exact**，上限 ~4× —— 应该排在 NEON 之前评估。

---

## 6. 与通用 HPC infra 的差异化

被问"这跟 TVM / Halide / oneAPI 有什么区别"时，答这三条 —— 通用框架都不提供：

1. **数值等价门禁作为 CI 的一等公民。** 12 个独立 numpy oracle，阈值分三级（bit-identical / 1 ulp / dB），外加静态门（`capacity_check` 保证 host 镜像常量与 device 一致**并且**覆盖全部 21 个 config pool 的几何；`registry_check` 保证 `.cl` 里的 `__kernel void` 名、host registry、实际被调用的名字是同一组 29 个）。FEC 层 41 gate：`d_free = 10` 用三种独立方法验证（64 状态图上的 Dijkstra、两个分支预算下的分层 BFS、oracle 里 2^16 穷举），Viterbi 对 2^L 全部码字暴力比对到 5.4e-16 rel，CRC 校验值 0x31C3 / 0x29B1 / 0x21CF02 精确，无误码 BLER **0/1440** 帧，52 000 个单比特错误全部检出（`README.md:115`）。
2. **精度感知放置 + 自动降级。** fp64 在 fabric 上不是"慢"而是**硬阻塞**（`CL_INVALID_BUFFER_SIZE`），运行时靠探测 `CL_DEVICE_DOUBLE_FP_CONFIG` 决定 marshal 精度，同时保留被门禁证明正确的 fp64 路径。
3. **最坏情况延迟确定性。** constant-latency bracket expansion 把数据依赖循环换成固定 6 次计算 + priority mux。**并且有拒绝记录**：把 fallback 扫描换成固定 40 步 bisection 的方案被**实测否决**——`Psi` 在 dead-band 处非单调（`u→0` 时结构性趋 0），端点 bracket 检验不是 scale-covariant（把一个通道缩放 1e-4 会让 46 个退化 case 中的 14 个从 flag 1 翻到 flag 2），且 7 个数量级的带内平台上 bisection 的 `u0` 会落进不同 Newton basin，值 **30.6 dB 的 tau**（`README.md:923`）。**有决策记录的 infra 才是 infra**，这条比任何加速数字更能说明工程成熟度。

领域约束（说明为什么不能直接用通用框架）：一个调度周期内必须出结果；矩阵小（4×4 到 28×32）；批维度是 **C=8 通道**而不是 batch=256 图片；控制面必须 fp64；PL 必须 fp32；且延迟要 constant 而非 average。

---

## 7. 包装落地清单

### 7.1 需要新增/重排的文档

| 文件 | 内容 | 状态 |
|---|---|---|
| `docs/hpc_infra_positioning.md` | 本文 | ✅ |
| `docs/hpc_roofline_measured.md` | 逐算子 arithmetic intensity / achieved GFLOP·s⁻¹ / GB·s⁻¹ + roofline 图 + 三/四条上界 | 🔄 实测中（并行 agent） |
| `docs/ps_neon_audit.md` | A53 NEON 化审计：热点定位、`-O2` vs `-O3` vs `-mcpu=native` 实测、**是否打破 bit-identical 门禁** | 🔄 审计中（并行 agent） |
| ~~`docs/hardware_utilization_map.md`~~ | **不另建文件**（避免与本文 §3 重复）；每算子 LUT/FF/DSP/BRAM 实测一栏待 Vitis 可达后直接填进 §3 的表 | ⬜ 阻塞在工具链 |
| ~~`docs/latency_budget.md`~~ | **不另建文件**（避免与本文 §5.3 重复）；上板后把 worst-case / jitter 直接补进 §5.3 | ⬜ 需板子 |
| `docs/portability_matrix.md` | 6 后端 × 2 精度的通过矩阵 + 跨架构零漂移证据 + `__constant` 硬约束的两种失败形态 + **7 条诚实缺口** | ✅ 已建（从 `README.md:990-1000` 汇编） |

### 7.2 README 顶部改动（最小侵入）

把 `## Why this shape (the AI-infra story, honestly)` 改为 `## System architecture`，保留原有四条 bullet（它们本来就是 infra 语言），在其后插入：

- 一张 §1 的三层 ASCII 架构图；
- 一句负载画像（§2 的结论：四类瓶颈并存 → 必须逐算子放置）；
- 一张"数量级诚实表"（§5.1），把 1.71 Tflop/s / 34 Gflop/s / 0.933 GFLOP/s / 205 MHz 四个数一次讲清，避免读者自己算出 50× gap 时觉得被误导。

### 7.3 让它"真的像 infra"还缺的工程件

1. **稳定 API 层**：目前对外只有 bench 可执行文件。需要一个 C API（`hpocl_init` / `hpocl_design_instant` / `hpocl_threshold_cf`）+ 可选 Python binding，否则第三方无法把它当库用。
2. **逐 kernel 资源报告**：`v++ -c -t hw` 跑全 10 个源，收 LUT/FF/DSP/Fmax/编译时延，填 `vitis_flow_notes.md:97-98` 那张附录表。**这是当前最大的证据缺口**（现有 4 个已综合 kernel 合计 22 个乘法，未综合的 6 个合计 508 个）。
3. ~~**CI 上云**~~ → **已加 `.github/workflows/gates.yml`**：4 个 job —— (1) 静态自足门（无 device、无兄弟仓库，红灯必然是真红灯）；(2) 静态配置包络门（并列 checkout `ningjihang/6g_software_platform`，跑 `capacity_check` + `glue_cost_check`）；(3) **PoCL CPU device 上的全量 `make check`**（`DEV ?= cpu` 本来就是 Makefile 默认）+ 再用 `-DHPOCL_FP32` 跑一次 `allprog_check`；(4) 探索性 fp64（`continue-on-error`，因为树内没有 PoCL 跑 fp64 的记录，且 fp64 的 `__constant` 池超预算 1.97×）。不需要板子，也不需要 Vitis。**注意：文件已写并通过 YAML 校验，但尚未 push，所以还没被真实 runner 验证过。**
   - 途中发现一个 infra 级问题：`tools/` 下 **17 个文件**依赖仓库外的 `../mix_precode`，包括 2 个静态门与 8 个 bench 的 oracle。要么 vendor 一份 config pool 快照，要么把「必须与 `6g_software_platform` 并列 checkout」写成硬前提 —— CI 目前按后者实现（详见 `docs/portability_matrix.md` §6 缺口 8）。
   - `HPOCL_CLOPTS` 是 `getenv()` 读的**环境变量**（`runtime/allprog_main.cpp:53`、`bench_main.cpp:38`、`cf_main.cpp:94`），写成 `make target HPOCL_CLOPTS=...` 只会设一个不导出的 make 变量、**静默跑默认配置** —— workflow 里必须用 `env:` 传。
4. **扩展性实验（不需板子）**：
   - *强扩展*：固定总工作量，C（通道批）从 1→8→32，看 PS/PL 两侧的效率曲线；
   - *精度-性能权衡*：fp64 vs fp32 的 tau/sigma 偏差 vs 时间，画成一条曲线（数据已有一半，`x86_opencl_notes.md` Result 1）；
   - *放置策略 A/B*：`sched_bench` 的 `costed` vs `force` vs `glue` vs `glue!` 四角色已经是现成框架，只需把结果整理成"成本模型 vs 强行全卸载"的对比图。
5. **需要板子才有的**：jitter / worst-case latency、energy per scheduling instant、DDR 实测带宽（验证 2.4 GB/s per port 这个假设）、`WBB_LOCAL_MAX` 大 tile 是否真的在 PL 上回本 —— 已用 wall clock 复测：`WBB_LOCAL_MAX=4096` 在本机**慢 4.0–7.7×**（32 KB local 占满 → 占用率崩），而 `LLR_CACHE_MAX=4` **快 0.66–0.68×**（private 1 KB→16 B），两个符号相反的占用率效应，与 README 表同号（`docs/hpc_roofline_measured.md` §9）；PL 无 cache，结论应相反，仍**未验证**。

### 7.4 阻塞项（按关键路径排序）

| # | 阻塞 | 影响 | 解法 |
|---|---|---|---|
| 1 | **QX_49DR_SV3 的 `.xsa` / `.xpfm`**（AMD 官网没有，必须板厂给） | 目标板编译/链接/打包的唯一入口；现机器只有 zcu102/zcu104 | 向板厂申请，见 `docs/QX49DR_SV3_上板依赖清单.md` §1 |
| 2 | Vitis 工具链可达性（Windows 反向隧道当前 connection refused；WSL 安装未完成） | 逐 kernel 资源报告做不了 | 恢复隧道，或在 Linux 主机装 Vitis 2022.2 |
| 3 | 板上 Linux + XRT 2022.2 镜像 | 全部 BOARD 类验证 | PetaLinux 2022.2 BSP 或板厂镜像 |
| 4 | License 方案 | — | `上板依赖清单.md` §6 |

---

## 8. 诚实边界（对外材料里不要越过的线）

1. **树内没有综合报告**（`README.md:974-978` 已自我声明）。所有 PL 数字都是**结构性的 + 一个实测锚点**，不是 achieved 性能。
2. **M4 无风扇会 throttle 3–3.7×**，host 绝对 ms 不能跨机引用；可引用的是比值与计数。
3. **A53 与 PL 的绝对性能都是 PROJECTION**，文档里每次出现都必须带这个标注。
4. **transformer student 精度不达标**（1.0722 dB MAE vs 0.05 dB 目标），不能声称"学习法加速了控制律"，只能声称"双轨共存 + 在线交叉校验 + teacher 已达 4.97e-14 dB"。
5. **R5F / RFDC / CMAC / OCM 均未使用**，其中 R5F 闲置是明确决策而非遗漏。
6. **不要套用别处的加速数字**（例如无关项目的 kernel 优化倍数）。这里可辩护的最大实测收益是 deficit 表的 6 187× / 28 290×，它是"消掉 transcendental"的收益，不是"硬件加速"的收益，表述时必须说清。
