# QX_49DR_SV3 (XCZU49DR) 上板跑通 Vitis OpenCL 依赖清单

> 目标：把 hpocl 的 29 个 kernel 编译成 xclbin，部署到 QX_49DR_SV3（ZCZU49DR-2FFVF1760），
> 从 PS 侧 Linux 用 XRT 调用 PL 加速器。
> 状态图例：✅ 已有 / ⬜ 缺（主阻塞） / 🔵 可选 / ⚠️ 需确认

---

## 0. 现状盘点（已具备）

| 项 | 状态 | 说明 |
|---|---|---|
| hpocl kernel 源码 | ✅ | kernels/ 10 源 29 kernel，纯 CL1.2 device code，无 host 混编、无 RFSoC 专用扩展 |
| kernel 正确性 | ✅ | Mac M4 GPU 全部 bench 通过（op/cf/decomp/datapath/chain/rx/pipeline/gh/tf） |
| Windows Vitis 2022.2 | ✅ | D:\xilinx\Vitis\2022.2（v++.bat），内置 base_platforms: zcu102/zcu104 |
| Windows Vivado 2022.2 | ✅ | D:\xilinx\Vivado\2022.2\bin\vivado.bat |
| Windows aarch64 交叉编译器 | ✅ | 已确认存在 |
| v++ -c 编译链路 | ✅ | zcu102 平台零下载验证：vadd/switch_threshold/thp_precoder/mcs_select_k 均出 .xo |
| 现成构建脚本 | ✅ | opencl/build/build_xclbin.sh + link.ini、scripts/run_board.sh、run_emu.sh |
| Windows 远程访问 | ✅ | ssh home-win（反向隧道），免密 |

---

## 1. 板厂资料依赖（最大缺口，AMD 官网没有）

| 依赖 | 状态 | 版本要求 | 用途 |
|---|---|---|---|
| **硬件平台 `.xsa`** | ⬜ | **Vivado 2022.2 导出**（必须与本链版本一致） | 平台构建的硬件输入（PS/PL 时钟、DDR、AXI、RFSoC 配置） |
| **Vitis 平台 `.xpfm`**（含 hw/、sw/） | ⬜ | 2022.2 | 若板厂直接提供，跳过平台构建步骤 |
| 板级 XDC 约束 + Vivado 工程 .xpr | 🔵 | 2022.2 | 需要改硬件/自己导出 xsa 时用 |
| 内存映射/地址表、PL 时钟清单 | 🔵 | — | link.ini 里 DDR 绑定（已预留 DDR[1..3] 注释） |
| PetaLinux 2022.2 BSP | 🔵 | 2022.2 | 生成 Linux 镜像 + 设备树（或板厂直接给镜像） |
| RF-ADC/RF-DAC 集成说明（RFDC/JESD204） | 🔵 | — | 若涉及 16T16R 射频通路（hpocl 本身是纯计算，不一定需要） |
| License 方案 | ⬜ | — | 见 §6 |

---

## 2. Linux 主机环境（上板全流程必需，Windows 到不了 -l/-p）

| 依赖 | 状态 | 建议 |
|---|---|---|
| Linux x86_64 主机 | ⬜ | 候选：172.31.12.238 (jihang-ning 个人容器, 1.4T 空闲, 需装 Vitis)；或新开虚拟机/真机 |
| Vitis 2022.2 **Linux 版**（含 Vivado 2022.2） | ⬜ | AMD 官网统一安装器，装完约 60GB，需 AMD 账号 |
| XRT 2022.2 | ⬜ | 板上/主机 runtime，版本必须 2022.2 |
| PetaLinux 2022.2（可选） | 🔵 | 生成 fsbl/u-boot/内核/设备树/rootfs |
| 磁盘 ≥100GB、内存 ≥16GB | ⚠️ | 满足才装 |

> Windows WSL2 跑 Vitis 2022.2：官方支持边界不确定，建议先用真 Linux/容器（jihang-ning 优先）。

---

## 3. 平台构建依赖（xsa → xpfm，只在 Linux）

```
板厂 .xsa
   + fsbl.elf · pmufw.elf · bl31.elf(ATF) · u-boot.elf · system.dtb · Linux Image · rootfs
   （来自 PetaLinux 构建 或 板厂 BSP 自带）
        ↓ Linux Vitis ("Xilinx → Platform" 或 vitis 命令行)
   platform 文件夹 = xxx.xpfm + hw/hw.xsa + sw/   ← v++ 消费的入口
```

---

## 4. kernel → xclbin 编译链路（hpocl 现成脚本）

| 阶段 | 命令 | 平台 | 状态 |
|---|---|---|---|
| kernel → .xo | `v++ -c -t hw $COMMON` | Windows ✅ / Linux | 已验证 3 个；COMMON=`-DHPOCL_FP32 -DNEWTON_ITERS=20 -I kernels` |
| .xo → .xclbin | `v++ -l -t hw --config link.ini` | **仅 Linux** | ⬜ 缺平台+license |
| 打包部署镜像 | `v++ -p`（BOOT.BIN 等） | **仅 Linux** | ⬜ |

> ⚠️ 坑：threshold_cf.cl 默认 fp64，**必须 -DHPOCL_FP32**，否则软浮点资源爆炸。
> ⚠️ 2D/3D + barrier kernel（channel_cov/datapath 5个/tf_ops 3个/deflate/power_iter/wbb_apply）未经实机验证，逐个确认。

---

## 5. PS 侧 host 运行依赖（板上）

| 依赖 | 状态 |
|---|---|
| hpocl_runtime.cpp + bench main | ✅ 已有 |
| XRT 库（libxrt_coreutil，2022.2） | ⬜ 随 Linux 镜像/RootFS 提供 |
| xclbin 部署到 /lib/firmware/hpocl.xclbin | ⬜ 由 §4 产出 |
| 测试向量 build/*.bin | ✅ 已有 |
| 板上启动介质（SD 卡: BOOT.BIN+boot.scr+Image+rootfs，或 QSPI） | ⬜ 随板/板厂 |
| aarch64 交叉编译 host | ✅ Windows 有；Linux 用 aarch64-linux-gnu-gcc 或板上直编 |

运行入口：`scripts/run_board.sh /lib/firmware/hpocl.xclbin`（内部 export HPOCL_XCLBIN + HPOCL_FP32=1）

---

## 6. 授权/账号

| 依赖 | 状态 | 说明 |
|---|---|---|
| AMD/Xilinx 账号 | ⬜ | 下载 Vitis Linux、PetaLinux、文档用 |
| Vivado Enterprise / Vitis license | ⬜ | `-c` HLS 综合免授权（已验证）；`-l`/`-p` 与真实实现**需要 license**；ZU49DR 为 RFSoC 大器件，通常需更高授权档 |
| 板厂 license 方案 | ⬜ | 问供应商是否有现成 license 或节点 |

---

## 7. 一句话依赖链（可视化）

```
[板厂] xsa ──→ [Linux Vitis + PetaLinux 软件栈] ──→ xpfm (平台)
                                                       │
[hpocl .cl] ──v++ -c──→ .xo ──v++ -l──→ hpocl.xclbin ──v++ -p──→ BOOT.BIN/SD
                                                       │
                              [板上 Linux + XRT] + [aarch64 host] ──→ 跑 bench
```

---

## 8. 当前最短缺的三件事（按优先级）

1. **QX_49DR_SV3 的 .xsa（Vivado 2022.2 导出）** —— 找板厂，无它一切白搭
2. **一台 Linux 主机装 Vitis 2022.2**（含 license）—— Windows 只能到 -c
3. **License 方案确认** —— 问板厂 + AMD 账号
