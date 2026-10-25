# Vitis 2022.2 流程支持与上板准备（hpocl）——结论速览、证据链、缺失清单

> 范围：hpocl（OpenCL→Zynq MPSoC 的 AI-infra 链路：SVD/GMD/UCD 闭式解 +
> Transformer 阈值决策 + 调制解调全链路）在 Vitis 2022.2 上的工具链支持现状与上板准备。
> 本文所有技术结论均在远程 Windows 机（Vitis 2022.2，`D:\xilinx\Vitis\2022.2`）上实测；
> **数值/报错原文不得改动**。只列缺失清单，不下载。
> 关联：README「Vitis/Zynq 状态」小节。

## 1. 结论速览

- **emulation 仿真流程（sw_emu / hw_emu）在 Windows 上被工具硬拒**：`v++ -c -k ... -t sw_emu`
  与 `-t hw_emu` 立即报
  `ERROR: [v++ 82-987] Sorry, emulation flows are not yet supported on this operating system.`
  ——这是 Vitis 对 OS 的限制（emulation 为 Linux 独占），**不是 license 或安装问题**。
- **Windows 可用的核心路径是 `v++ -c -t hw`（kernel HLS 综合 → .xo）**：OpenCL kernel
  可以直接映射为 RTL IP，这是"同一份 kernel 既能跑 CPU/GPU 验证（本地 `make check`），
  又能过 Vitis HLS 映射到 Zynq PL"这一可移植性卖点的直接证据，也是 AI-infra 包装（
  算子 + 运行时 + CPU/GPU/PL 三目标）对 PL 一侧的最小闭环。
- **Windows 上跑 `-t hw` 必须给完整平台 `.xpfm` 路径**：只写平台名（如
  `--platform xilinx_zcu102_base_202220_1`）找不到平台——默认平台仓库
  `D:/xilinx/Vitis/2022.2/platforms` 是空的，实际平台文件在
  `D:\xilinx\Vitis\2022.2\base_platforms\` 子目录下。
- **最小 kernel 实测：Fmax ≈ 205 MHz，约 1 分钟产出 .xo（107 KB）**。这是
  "OpenCL kernel → RTL IP"的核心验证：kernel 源无需改写即可进入 PL 综合。
- **机器上只有 zcu102 / zcu104 平台，缺 ZCU49DR 平台**——目标板
  (QX_49DR_SV3, XCZU49DR-2FFVF1760) 的 `.xsa/.xpfm` 需另行获取（见缺失清单）。
- 结论：**当前可用** = Windows 上逐 kernel `-t hw` 综合验证；**当前不可用** =
  hw_emu 仿真、`v++ -l` 链接、xclbin 打包、上板镜像——这些都需要 Linux Vitis。

## 2. 证据链（远程 Windows 机实测）

### 2.1 emulation 被硬拒（Windows → Linux 独占）

```
D:\xilinx\Vitis\2022.2\bin> v++ -c -k <kernel> -t sw_emu --platform <xpfm> <src.cl>
ERROR: [v++ 82-987] Sorry, emulation flows are not yet supported on this operating system.
```

`-t hw_emu` 行为相同、同一报错。工具参数正确、平台路径正确时依旧触发，定位为
**OS 支持矩阵硬限制**（Vitis 2022.2 emulation 只支持 Linux 宿主），与 license、
安装完整性无关。Windows 上 sw_emu/hw_emu 两条路径在本项目上**不可用**。

### 2.2 `v++ -c -t hw`（kernel HLS 综合 → .xo）Windows 可用

- 必须显式给出完整平台文件路径；只写平台名会在空仓库
  `D:/xilinx/Vitis/2022.2/platforms` 里找不到（平台实际在
  `base_platforms\` 子目录）：
  ```
  D:\xilinx\Vitis\2022.2\base_platforms\xilinx_zcu102_base_202220_1\hw\hw.xsa   (板上壳)
  D:\xilinx\Vitis\2022.2\base_platforms\xilinx_zcu102_base_202220_1\hw_emu\hw_emu.xsa (仿真壳)
  D:\xilinx\Vitis\2022.2\base_platforms\xilinx_zcu102_base_202220_1\sw\   boot/ + qemu/ + xrt/
  ```
- 最小 kernel 实测：`v++ -c -k <k> -t hw --platform <完整 .xpfm>` 约 **1 分钟**出
  `.xo`（**107 KB**），综合报告 **Fmax ≈ 205 MHz**。
- 结论：hpocl 的 `kernels/*.cl`（threshold_cf / wbb_apply / thp_precoder /
  channel_cov / tf_ops 等）可以直接在这条路径上逐个过 `-t hw` 综合，得到 RTL IP，
  验证 kernel 可移植性；但 Windows 上止步于 `.xo`。

### 2.3 Windows 安装里可用的周边工具

| 组件 | 路径/状态 |
|---|---|
| 平台文件（zcu102） | `D:\xilinx\Vitis\2022.2\base_platforms\xilinx_zcu102_base_202220_1\`（hw / hw_emu / sw 齐全） |
| 平台文件（zcu104） | `base_platforms\` 下另有 zcu104；**无 ZCU49DR** |
| aarch64 交叉编译器 | `D:\xilinx\Vitis\2022.2\gnu\aarch64\nt\aarch64-linux\bin\aarch64-linux-gnu-gcc.exe`（可用，PS 侧 host 交叉编译） |
| vitis_hls | 可用（HLS 综合/分析，对应 `v++ -c -t hw` 的独立入口） |
| 许可证 | 现机器只有 MATLAB 的；见缺失清单 (4) |

## 3. Windows vs Linux 流程支持矩阵

| 流程环节 | Windows Vitis 2022.2 | Linux Vitis 2022.2 | 状态/缺口 |
|---|---|---|---|
| kernel 综合 `v++ -c -t hw` → `.xo`（OpenCL→RTL IP） | ✅ 可用（需完整 .xpfm；1 min, Fmax≈205 MHz, 107KB） | ✅ | Windows 已验证，可移植性卖点成立 |
| 仿真 `v++ -c -t sw_emu` / `-t hw_emu` | ❌ `[v++ 82-987]`（OS 硬拒） | ✅ | **Linux 独占**，本项目路上最大缺口 |
| 链接 `v++ -l` → 生成平台镜像 | ❌（依赖 emulation/打包链路） | ✅ | 需 Linux |
| 打包 xclbin / 上板镜像（boot 等） | ❌ | ✅ | 需 Linux（含 XRT） |
| aarch64 host 交叉编译 | ✅（gnu\aarch64\nt 工具链可用） | ✅ | 已有，PS 侧 host 可先行 |
| vitis_hls（独立 HLS） | ✅ | ✅ | 可用 |
| 目标板平台（ZCU49DR .xsa/.xpfm） | ❌ 缺 | ❌ 缺 | 见缺失清单 (3) |

一句话：**Windows 目前 = "kernel→RTL IP 综合验证"平台；Linux 才是完整流程
（仿真→链接→打包→上板）的宿主**。

## 4. 缺失清单（只列，不下载）

| # | 缺失项 | 标注 | 用途 |
|---|--------|------|------|
| 1 | **Linux Vitis 2022.2（约 40–60 GB）**，装到 WSL Ubuntu 22.04 或 Linux 机 | 需下载 | hw_emu 仿真、`v++ -l` 链接、打包、上板全部需要 |
| 2 | **XRT 运行时**（随 Linux Vitis 提供，`/opt/xilinx/xrt`；目标板 rootfs 里也要装） | 需下载（随 Vitis） | PL 设备枚举、`Runtime::init("acc")`、xclbin 加载 |
| 3 | **ZCU49DR 的 `.xsa` / `.xpfm`**（目标板平台；现机器只有 zcu102/zcu104） | 需申请（板卡厂商/QX_49DR_SV3 平台工程） | 目标板编译/链接/打包的唯一平台 |
| 4 | **Xilinx 许可证**（现机器只有 MATLAB 的） | 需申请；zcu102 hw_emu 通常 WebPACK/免费够用；**ZCU49DR 上板是否需额外授权：待确认** | 综合/仿真/上板授权 |
| 5 | **上板硬件/线缆/SD 卡镜像流程**（板子、JTAG/网口、启动镜像、格式化与烧写流程） | 需准备（硬件+流程文档） | 真机上板验证 |

## 5. 下一步建议（按顺序）

1. **Windows 侧可先行**：用 zcu102 平台把 hpocl 的 `kernels/*.cl` 逐个
   `v++ -c -t hw --platform <zcu102 完整 .xpfm>` 综合，收集每 kernel 的
   Fmax/资源（LUT-FF-DSP）/编译时延，作为论文附录的 PL 综合证据表；同时用
   `aarch64-linux-gnu-gcc` 把 host runtime 交叉编译成静态/动态二进制，验证 PS 侧可执行。
2. **补齐 Linux 宿主**：WSL Ubuntu 22.04 或 Linux 机安装 Linux Vitis 2022.2（40–60 GB），
   获得 hw_emu + `v++ -l` + 打包能力；优先在 hw_emu 上跑通
   `PLATFORM=<zcu102 .xpfm> opencl/build/build_xclbin.sh hw_emu` 的完整链路。
3. **申请 ZCU49DR 平台**：向板卡厂商/平台工程要 QX_49DR_SV3 的 `.xsa` 与由它构建的
   `.xpfm`（PS + PL DDR4 ×4 + RFDC + CMAC 100G）；拿到后即可把
   `PLATFORM=` 换成目标板平台。
4. **XRT 与 rootfs**：Linux Vitis 装好后确认 `/opt/xilinx/xrt`，并规划目标板 rootfs
   的 XRT 部署（zcu102 平台 `sw/xrt/` 可先作为参考模板）。
5. **上板（硬件齐备后）**：整理 SD 卡镜像流程（boot 分区 + rootfs + xclbin），
   复跑 README「Zynq bring-up checklist」的 1–5 步；ZCU49DR 授权问题在申请平台时
   一并确认。

---

*本文档只描述工具链状态与缺失项；不包含任何下载动作。所有据以成文的实测结论见 §2。*
