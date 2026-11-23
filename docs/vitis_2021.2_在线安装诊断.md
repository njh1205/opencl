# Vitis 2021.2 在线安装诊断（2026-10-26）

## 环境状态（已就绪 ✅）
- Windows 11 x64 + WSL2 (v2.7.8) + **Ubuntu 20.04.6**（D 盘 vhdx，955G 空闲）
- WSL 已配清华 apt 源（675KB/s）、gcc 9.4/make/python3.8、OpenCL 头、libtinfo5 软链
- 安装目录 `/opt/Xilinx`、组件接收目录 `/data/xilinx_pkgs`
- 安装器：`C:\Users\27147\Downloads\Xilinx_Unified_2021.2_1021_0703_Lin64.bin`（MD5 通过，Makeself 包，仅含 xsetup 框架，**不含组件**）

## 在线下载实测结论
- **通道是通的**：实际下载服务器 `amd-ax-dl.entitlenow.com`（**不走 account.amd.com**，无需账号登录）✅
- 第一次会话成功下载 18 个 payload 分片（~135MB）后失败
- **失败根因**（xsetup 日志）：
  1. 下载 URL 带 `expires=` 签名，**有效期仅约 3 分钟**（st 会话 10:47:43 过期）
  2. entitlenow.com 国内下载慢，赶不上签名时效 → 大面积 ERROR
  3. 重启后卡在 `https://www.xilinx.com/bin/public/webinstall/oauth/validate`（授权验证），进程退出
- 结论：webinstaller 免费临时授权通道在慢网下不可用；需要快网络或正式凭据/离线包

## 三条出路（按推荐）
1. **换能快速连接 AMD/Xilinx 的网络**（手机 5G 热点走代理/VPN，或办公室外网出口）→ 重跑 `set s id nohup printf '1\n' | xsetup -b Install -a XilinxEULA,3rdPartyEULA -l /opt/Xilinx -e "Vitis Unified Software Platform"`
2. **问板厂要 2021.2 完整离线包**（bin + 全套 tar.gz，40-60GB）→ 放 `/data/xilinx_pkgs`，用 `./xsetup -b Install -c config` 离线装（最稳，零网络依赖）
3. **AMD 账号在能上网处登录，跑 `xsetup -b AuthTokenGen` 生成 token 文件** → token 文件拷回，用 token 授权在线下载

## 就绪的自动化（组件包一到即跑）
- WSL 依赖 ✅ / 安装目录 ✅ / 清华源 ✅
- 待执行：解包 bin → batch 安装到 /opt/Xilinx → 构建 2021.2 平台（用 `lt49dr_ps_wrapper.xsa` 2021.2）→ v++ 编译 hpocl kernel → sw_emu/hw_emu
