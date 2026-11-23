# 可移植性矩阵 — 后端 × 精度 × 门禁

> 纯整理文档：所有条目都指向仓库里已有的实测记录，未新增任何声明。
> 用途：HPC infra 叙事里"可移植性是特性而非口号"这一条的证据汇编
> （定位见 `docs/hpc_infra_positioning.md` §1、§4）。

---

## 1. 后端清单：哪些真跑过，哪些只是投影

| # | 后端 | 架构 / OS | OpenCL 实现 | `DOUBLE_FP_CONFIG` | `MAX_CONSTANT_BUFFER_SIZE` | `LOCAL_MEM_SIZE` | 实际跑过什么 | 证据 |
|---|---|---|---|---|---|---|---|---|
| 1 | **Apple M4 GPU** | arm64 macOS 26.6 | Apple `clang -x cl` | **`0`** → `Runtime::init` 置 `dev_fp64_=false`，全部 gate 以 `-DHPOCL_FP32` 编译；Apple clang **直接拒绝 `double`**（忽略 `cl_khr_fp64` pragma） | — | 32 768 B | 全套 bench + 全部静态门（开发/bring-up 后端） | `docs/x86_opencl_notes.md:1-12`、`README.md:994` |
| 2 | **NVIDIA Tesla P40** | x86_64 Ubuntu 20.04.6，48 核（`172.31.12.238`） | NVIDIA OpenCL **3.0**，driver 550.127.08 | **`0x3`** → fp64 kernel 可创建 | **65 536 B（恰好是 spec 最小值）** | 32 768 B（与 M4 相同） | **fp64 路径的首次真实执行**；`cf_bench -DHPOCL_FP64` 7/7 PASS；全套 fp32 门禁在 GCC 9.4 下重跑 | `docs/x86_opencl_notes.md:14-30`、`README.md:994` |
| 3 | **NVIDIA L4** | x86_64 Linux | NVIDIA（同 ICD 家族） | — | 65 536 B | — | `__constant` 超预算的**第二种失败形态**（编译器后端拒绝，见 §4） | `README.md:997-999` |
| 4 | **x86 CPU OpenCL device** | — | — | — | — | — | ⚠️ **`README.md:87` 的目标表里声称是 bring-up/regression 后端，但树内找不到任何执行记录**（P40 主机上只有 NVIDIA ICD，PoCL 明确"not needed and not installed"）。见 §6 缺口 1 | `README.md:87` vs `docs/x86_opencl_notes.md:9-10,29` |
| 5 | **Zynq PS（4× A53）** | aarch64 Linux | host C++（无需 OpenCL）或 CPU device | — | — | — | ⚠️ **仅投影**：0.933 GFLOP/s 模型（1.4 GHz in-order，1 fp64 flop/cycle，×1.5 结构开销）+ 逐 transcendental 周期数（exp 80 / log 70 / log2 55 / div 29 / sqrt 33） | `docs/cost_model_real_shapes.md:23-25` |
| 6 | **Zynq PL（XCZU49DR）** | Vitis `.xclbin` / XRT | `CL_DEVICE_TYPE_ACCELERATOR` | 无浮点 DSP（**DSP48E2，非 DSP58**） | — | BRAM/URAM | ⚠️ **仅编译过 4/10 个源**（`vadd`、`switch_threshold`、`thp_precoder`、`mcs_select_k`），Fmax ≈ 205 MHz、107 KB；**从未 link 出 `.xclbin`、从未加载执行**；LUT/FF/DSP 用量未记录 | `docs/vitis_flow_notes.md:73`、`README.md:974-978` |

---

## 2. 精度 × 后端

| 精度 | Apple M4 | Tesla P40 | L4 | A53（PS） | PL |
|---|---|---|---|---|---|
| **fp32**（ship 配置） | ✅ 全套门禁 | ✅ 全套门禁（GCC 9.4，零漂移） | ✅（作为 `__constant` 预算验证） | 投影 | ⚠️ 4/10 源编译过 |
| **fp64** | ❌ 设备不支持（`DOUBLE_FP_CONFIG=0`，编译器拒绝 `double`） | ✅ **已执行**：`tau=0.00e+00`，`cl=7.61e-16 dB` vs 门禁 `1e-6 dB`（**1.3e9× margin**），**39/40 个 tau 与 fp32 kernel 逐位相同**，第 40 个差最后一位 | ❌ 编译器阶段即失败（ptxas） | ✅ 控制面的 ship 精度 | ❌ **硬阻塞**，非性能问题（见 §4） |

fp64 **不是删掉的死代码**，是被门禁证明活着的代码路径 —— 只是不 ship。

---

## 3. 最硬的一条证据：跨架构数值一致性

整个 fp32 套件（`decomp` / `op` / `datapath` / `rate` / `chain` / `sched`，**156 个 gate 行**）在 **x86 Linux + 另一家 vendor ICD + GCC 9.4** 上重跑，相对 Apple M4 的结果：

- **零数值漂移**；
- fp32 Jacobi off-diagonal floor 在第二架构上复现为 **2.9e-9**（Apple 侧 1.4e-8..3.4e-8），即 `tol_exit = 1e-7` 在第二个架构上仍有 **~34× margin**；
- 只有 fp64 精度列按预期移动（1e-16 → 1e-7）。

出处：`README.md:1000`（"device-independent numerics" 行）。

**这一条对外怎么讲**：不是"代码能在两台机器上编译通过"，而是**同一套数值门禁在两种指令集、两种编译器、两家 OpenCL 实现上给出同一组数**，并且收敛裕度在第二个架构上被独立复现。这是可复现性（reproducibility）证据，比可移植性（portability）证据强一档。

---

## 4. 在第二个架构上撞出的硬约束 → 固化成静态门禁

这是"为什么需要 infra 而不是脚本"的最具体案例。

`__constant` 表池的实际占用由 `allprog_check` 按**当次构建真正选中的表**求和，再与 `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` 比较（measured, not argued）：

| 构建 | deficit 表 | Gauss-Hermite | CQI | 合计 | 相对 64 KB spec 最小值 |
|---|---|---|---|---|---|
| **fp32**（PL 默认） | 64 000 B | 384 B | 128 B | **64 512 B** | **98.4%，仅剩 1 024 B 余量** |
| **fp64** | — | — | — | **128 896 B** | **1.97× 超预算 → 硬失败** |

fp64 的失败在两台 NVIDIA 设备上呈现为**两种不同形态**：

1. **Tesla P40**：`clCreateKernel` 返回 `CL_INVALID_BUFFER_SIZE (-61)` —— program 能 build，kernel 建不出来；
2. **L4**：更早一级，在工具链里就死：`ptxas error : File uses too much global constant data (0x1f7a0 bytes, 0x10000 max)`（128 928 B = 128 896 B 表 + 32 B 编译器生成的字面量）。

**第二种形态对上板有直接含义**：限制是由**编译器后端**执行的，所以任何 host 侧检查都无法说服 vendor 工具链接受它，`.xclbin` 构建会在 `v++` 阶段而不是加载阶段失败。门禁因此把这个诊断打印在 "missing kernels" 报错**之前** —— 因为该失败模式的外部表现是"29 个 kernel 全都找不到"，极易误诊。

若 XRT 也报 64 KB，缓解手段各是一次生成器调用：`--N 800`（31.25 KB）、`--ms 64,256`（25 KB）、`--ms 64`（12.5 KB）。

出处：`README.md:997-999`、`docs/x86_opencl_notes.md` Result 2、`opencl/build/build_xclbin.sh:19-31,54`。

---

## 5. 不上板也能测到 PL 的代码路径

fabric 侧的两条配置分支被显式做成 dev-host 上可跑、且**bit-identical** 的门禁：

| 门禁 | flag | 覆盖的 PL 行为 |
|---|---|---|
| `make op_pl_bench` | `-DWBB_LOCAL_MAX=4096` | tiled `wbb_apply`（PL 的 local-memory 分支） |
| `make datapath_pl_bench` | `-DLLR_CACHE_MAX=4` | LLR **重算**回退分支（PL 上省 BRAM 的那条路） |
| `make allprog_check` | 设备默认选项 + 强制 `-DHPOCL_FP32` 各一次 | 10 个源编成**一个** program，29/29 kernel 全部获取 —— 等价于 `.xclbin` 跨 `.xo` link 做的事 |
| `make capacity_check` | — | host 镜像常量与 device 相等，**并且**覆盖全部 bench 几何 + `mix_precode/*/config_pools` 里全部 21 个 config |
| `make registry_check` | — | `.cl` 里的 `__kernel void` 名、host registry、实际被调用的名字是同一组 29 个 |

出处：`README.md:995-996`、`README.md:76`、`README.md:990`。

> 注意 `WBB_LOCAL_MAX` 大 tile 在 M4 上实测是 **tiled 338 < untiled 364 GFLOP/s**（有 cache 时重读本来就便宜，静态 local 数组反而亏）。PL 无 cache，结论应当相反，但**未验证** —— 这正是 `op_pl_bench` 存在的理由：保证 PL 真正会走的那条路径不是没人测过的路径（`README.md:1536-1543`）。

---

## 6. 缺口（诚实清单，按修复代价排序）

| # | 缺口 | 现状 | 修复代价 |
|---|---|---|---|
| 1 | **CPU OpenCL device 无执行记录**，但 `README.md:87` 的目标表把它列为 bring-up/regression 后端 | 声称与证据不一致 | 低：要么在任一台 x86 上装 PoCL 真跑一次并记录，要么把该行改为"host C++ 路径（不经 OpenCL）" |
| 2 | **PL 无综合报告**：4/10 源编译过，且这 4 个是全库最不吃 DSP 的（`thp_precoder` 9 个乘法 / `rate.cl` 8 / `switch_threshold` 5，合计 22），而 `decomp.cl` 221 / `threshold_cf.cl` 228 / `datapath.cl` 100 / `tf_ops.cl` 77 **一个都没综合过** | 所有 PL 数字都是结构性的；**而原先那个「唯一实测锚点」已于 2026-11-06 被推翻** —— 它是 Apple OpenCL 的 event 时间，本机 event 列低报设备时间 42–490×（中位 63.9×），28×32 C=8 批的真实 wall 是 51.3 ms 而非 1.20 ms（`docs/hpc_roofline_measured.md` §2） | 中：只需 Vitis（不需板子）跑 `v++ -c -t hw` 全 10 源，填 `docs/vitis_flow_notes.md:97` 的附录表；同时按 roofline §10 第①条用 wall-clock 重基线 PL 成本模型 |
| 3 | **`.xclbin` 从未 link/加载/执行过** | 上板链路全未验证 | 高：阻塞在板厂 `.xsa`/`.xpfm`（AMD 官网没有） |
| 4 | **A53 只有投影**，且 host 编译是 `-O2`，`ref/`、`runtime/`、`include/` 里零个 SIMD intrinsic/Eigen | NEON 完全未使用；但**已实测**：`-O3` 逐位相同却几乎零收益（`fmla.2d` 计数在 `-O2`/`-O3` 下都是 0，被 libc++ `complex::operator*` 的 Annex-G `isnan` 分支挡住），`-ffast-math` 有 1.19–1.74× 但**破门禁**（τ 3.97e-6 dB vs 1e-9 dB） | 中：走**手写保序 NEON**（`gemm_wx` 跨 `m` = 1.63–1.65× bit-exact），详见 `docs/ps_neon_audit.md` 与 `docs/hpc_infra_positioning.md` §5.4；**先评估把 C=8 批拆到多核**（逐矩阵 bit-exact，上限 ~4×） |
| 5 | **fp32 的 `__constant` 余量只有 1 024 B（1.6%）** | 任何表增长都会顶破 64 KB | 低但需决策：定一条"表预算策略"（何时切 `--N 800` / `--ms`），并让 `capacity_check` 在余量低于阈值时告警而非等到超 |
| 6 | ~~**没有 CI**~~ → **已加 `.github/workflows/gates.yml`**（4 job：静态自足门 / 静态配置包络门 / PoCL CPU 全量 `make check` / fp64 探索性非阻塞），用 PoCL CPU device，不需板子也不需 Vitis | 文件已写、YAML 已校验；**尚未 push，因此尚未被真实 runner 验证过** | 低：push 即生效 |
| 8 | **仓库不自足**：`tools/` 下 **17 个文件** import 兄弟仓库 `../mix_precode`（`github.com/ningjihang/6g_software_platform`，公开，main，63 MB）；其中 `check_capacity.py` / `glue_cost_table.py` 直接读 `../mix_precode/*/config_pools/*.json`（9 个 pool / 21 个 distinct config），**8 个 bench 的 oracle**（decomp / frontend / pipeline / modem / datapath / rx / lmmse / chain）经 `env_common.py` 依赖它；只有 `cf` / `rate` / `fec` 自足 | 单独 clone `njh1205/opencl` 无法跑全量门禁。好消息是两个静态门**响亮失败**而非空过（`check_capacity.py:196`、`glue_cost_table.py:64` 均 `raise SystemExit` 报 FAIL found no config pool），所以不会给出假绿灯 | 中：要么把 config pool 快照 vendor 进本仓库，要么把「必须与 6g_software_platform 并列 checkout」写成硬前提（CI 里已按后者实现） |
| 7 | **R5F / RFDC / CMAC / OCM 未使用** | R5F 闲置是**明确决策**（"hpocl 无硬实时控制任务，不为用而用"，`README.md:1655-1657`），只有 16T16R 射频通路才需要（A53 Linux ms 级抖动不够 µs 级帧定时） | 不在当前范围 |

---

## 一句话对外表述

> 同一份 `.cl` 源与同一份 host C++，在 **两种指令集（arm64 / x86_64）、两种编译器、两家 OpenCL 实现、三块 NVIDIA/Apple 设备**上通过同一套 **156 行数值门禁且零漂移**；在第二个架构上撞出的 `__constant` 硬限制被固化成静态门禁，并诊断出它在 vendor 工具链里的两种失败形态。PL 与 A53 目前是**编译验证 + 结构投影**，不是 achieved 性能 —— 这一条我们始终主动写明。
