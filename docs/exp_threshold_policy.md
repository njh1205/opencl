# exp_threshold_policy —— 闭式 SVD/THP 切换阈值（hpocl 部署网格）数值实验

> 实验脚本：`tools/exp_threshold_policy.py`（250 行，numpy/scipy/matplotlib，不动任何既有文件）
> 输出目录：`build/experiments/`（5 个逐配置 `cfg<i>_*.csv`、`summary_all_configs.csv`、`threshold_policy_main.png`）
> 正式运行：`cd hpocl && time python3 tools/exp_threshold_policy.py --n-ch 400 --seed 20261024 --workers 10`
> 实测耗时：**5 分 07 秒**（wall, 10 worker）——见文末“运行时间与折中”。

## 1. 方法（公式与口径）

闭式切换阈值直接复用已被 cf_bench 逐数值对拍过的部署实现
`cf_ref.tau_policy(lam, M, d, K, variant)`（闭合形式 + 安全 Newton，`iters=30` 收敛口径；
`flag=2` ⇒ `τ=±∞`，分支序恒一）。`lam` 为每用户功率白化 BD 等效信道 `G_k`
（`AnalogPrecoder → build_effective_channels → build_bd_digital_basis →
build_power_whitened_bd_channel → svd(G_k)`，白色奇异值平方 `s[:d]²`），生成逻辑与
`pipeline_ref.py` 完全一致（直接 import 复用 `dummy_env`，仅 import 不修改）。

信道：每配置独立 `default_rng(seed)` 采 400 个 i.i.d. 复高斯 Rayleigh 信道 `H∈C^{K×Nr×Nt}`
（每元素 `(x+j y)/√2`），NRF>Nt 是 hpocl 部署配置的既有口径（生产代码同款，白化矩阵
`R` 求逆溢出警告为已知良性，脚本内屏蔽并注释说明）。

指标（脚本内均有公式注释；本文件列口径）：

1. **THP 占比** `thp_frac(γ) = P_ch[γ ≥ τ(lam)]`，`γ∈{0,1,…,40} dB`（总 SNR，
   `snr_per_stream = 10^(γ/10)/(K·d)`）。`τ=+∞` ⇒ 恒 SVD、`τ=−∞` ⇒ 恒 THP。
2. **决策边际** `margin = |γ − τ|`，仅有限 τ 信道统计（median/p10/p90）；“翻转点附近”
   的量化 = 交叉 γ（thp_frac=0.5 经过处）上的 margin 中位。
3. **分支速率差** `rate_delta(γ) = R_THP − R_SVD`，每信道每分支
   `R = Σ_streams log2(1 + ρ_realized)`，`ρ_realized` 来自
   `DigitalStructuredPrecoder.design_precoder(G, snr_ps, strategy)`（variant0→gmd，
   variant1→ucd）。在 γ∈{10,20,30} dB 以及 γ=τ（截断到 [0,40]）四点统计分布。
4. **transformer vs 闭式**（找到真实 checkpoint 才跑）：
   `MAE/RMSE = mean/|τ_tf − τ|`、`agree = P_{(ch,γ)}[(γ≥τ_tf)==(γ≥τ)]`、
   `|τ_tf − τ|` 的 p50/p90 与 CDF `P(|Δ|≤k), k∈{1,2,3,5} dB`。

## 2. 配置网格与结果表

`(K, Nt, Nr, NRF, d, M, variant)`，每配置 400 信道，seed=20261024。所有 400/400 信道
都是有限交叉（`finite_frac=1.0, flag2_frac=0.0`）。

| cfg | 配置 (K,Nt,Nr,NRF,d,M,var) | τ 中位 dB (p10–p90) | 交叉 γ (thp=0.5) | thp@20dB / @30dB | margin@交叉 (dB) | ΔR@τ 中位 (bps/Hz) | TF MAE (dB) | TF 决策一致 agree | TF \|Δτ\| p90 (dB) | P(\|Δτ\|≤3 dB) |
|----:|---------------------------|--------------------|------------------|------------------|------------------|--------------------|------------|-------------------|---------------------|-----------------|
| 0 | (2,4,2,8,2,64,0)  GMD  | 21.5 (19.0–24.2) | 22 | 0.24 / 1.00 | 1.45 | −0.10 | 1.36 | 0.966 | 2.43 | 0.98 |
| 1 | (2,4,2,8,2,64,1)  UCD  | 21.0 (18.5–23.3) | 21 | 0.31 / 1.00 | 1.33 | −0.12 | 1.71 | 0.959 | 3.20 | 0.87 |
| 2 | (3,6,2,12,2,16,0) GMD  | 19.1 (16.6–22.1) | 19 | 0.71 / 1.00 | 1.54 | −0.49 | 3.48 | 0.915 | 4.77 | 0.28 |
| 3 | (4,8,2,16,2,64,0) GMD  | 24.5 (22.5–27.0) | 25 | 0.00 / 0.99 | 1.25 | −0.24 | 2.41 | 0.941 | 3.82 | 0.75 |
| 4 | (2,8,2,12,2,64,1) UCD  | 17.0 (15.4–18.6) | 17 | 0.98 / 1.00 | 0.85 | −0.01 | 2.11 | 0.948 | 3.29 | 0.81 |

（τ 为有限信道分布；ΔR@τ = γ=τ 处 R_THP−R_SVD 中位；“TF 决策一致” = 同一 (信道,γ) 网格上
transformer 与闭式得到的模式相同比例。）

## 3. 结果说明

- 闭式阈值在 5 个部署配置上都给出有限交叉点：THP 分支占比从 17–25 dB 区间平滑上升到
  高 SNR 全选；交叉 γ 与 τ 分布中位高度一致（≤0.5 dB），决策边际中位 0.85–1.5 dB。
- 在解析切换点 γ=τ 处，THP 分支相对 SVD 的速率差中位为 −0.01 至 −0.49 bps/Hz（负值 =
  闭式阈值略微保守，SVD 分支在切换点仍保有少量速率余量），高 SNR 时两分支速率收敛（ΔR→0）。
- 预训练 transformer（真实 checkpoint，见下）与闭式解决策一致 0.915–0.966；OOD 配置
  （M=16 的 cfg2、K4/Nt8 的 cfg3）偏差最大（MAE 3.5/2.4 dB，正偏/负偏），提示现有
  checkpoint 的训练配置与部署网格不同，若要在 hpocl 上部署 AI 阈值需用本网格重训。

## 4. transformer 对比的来源（重要，避免误导）

- **使用真实预训练权重，不是随机权重**：
  `mix_precode/_archive_20261018/bulk/results/svd_ucd_threshold_poolv2_3x1000_fast/poolv2_3x1000_fast_model.pt`
  —— `SpectralSetTransformerThresholdRegressor`，d_model=256, layers=4, heads=8,
  token_dim=5（[log σ, 去均值 log σ, 功率占比, 用户位置, 流位置]），config_dim=10；
  在 generalization_pool_v2 的 3 个配置（(2,16,4,8,4)/(3,24,6,18,3)/(4,32,4,16,2)）各
  1000 条 CDL-A 信道上训练，其自带 metrics（`poolv2_3x1000_fast_metrics.json`）：
  **test MAE=1.072 dB, test RMSE=2.30 dB, test 决策准确率 0.930**。随机权重只存在于
  `hpocl/build/tf_weights.bin`（`tools/make_random_tf_weights.py` 的 bench 用向量），与本文
  结果无关。
- **加载与校验**：本机无 torch，脚本内置纯 numpy 的 `.pt` 读取（FloatStorage→float32，
  不改任何文件），权重经 `tools/transformer_ref.py`（hpocl tf_bench 同一 numpy 前向）推理；
  并用该 checkpoint 自带数据集自检——numpy 前向在 120 样本子集上的 MAE≈0.80 dB，
  与公开 test MAE 1.07 dB 一致，证明权重映射/前向正确。
- **OOD 口径与退化特征处理**：训练配置在 Nt16–32、NRF8–18，本文部署网格更小（Nt4–8）。
  训练中恒定特征（bits=6 与 NRF/Nt=0.5）的 config 标准差≈0，评估时对该两维用 std=1 的
  恒等尺度（z=(x−mean)/1），其余维用真实训练统计量；逐配置结果已注明 OOD 偏差。
- **导出**：`tools/export_transformer.py`（→ HPOCL 扁平 fp32 权重）需要 torch，本机无法
  运行；在训练集群上执行 `python3 tools/export_transformer.py <上述 .pt> build/experiments/tf_weights.bin`
  即可落盘部署（本实验评估用的就是同一批权重，仅经 numpy 镜像前向）。

## 5. 运行时间与折中

- 正式运行 **306 s wall**（10 worker 并行）：用户 CPU 2311 s ≈ 2000 信道，平均约
  **1.2 s CPU/信道**；单信道耗时主要被 `tau_policy` 的 30 步安全 Newton 主导：
  M=16/K=3 约 0.15 s，M=64/K=4 约 2.5 s，信道生成 <10 ms，速率设计 ~1–3 ms。
- 并行口径：每配置按 `JOB_CH=50` 切块（400 信道 ⇒ 8 块/配置，共 40 个任务，10 worker
  共享进程池），总墙钟 = 最重块 ≈ 125 s × 波次系数，实测 5:07。串行运行同一负载 ≈ 39 min。
- 若需更快/更小的复跑：`--n-ch 100` 约 1.5–2 min，`--n-ch 50` 约 1 min；
  脚本无 MPI/GPU 依赖，`--workers` 可按机器核数调。

## 6. 如何复跑

```bash
cd /Users/ning/Desktop/源文件/biren/hpocl
python3 -m py_compile tools/exp_threshold_policy.py          # 语法自检
python3 tools/exp_threshold_policy.py --n-ch 400 --seed 20261024 --workers 10
```

产物（均在 `build/experiments/` 下，可复现）：
- `cfg{i}_k{Nt}_..._var{v}.csv`（i=0..4，每文件 41 行 = SNR 0..40 dB）：
  `snr_db, thp_frac, svd_frac, flag2_frac, tau_mean, tau_p10, tau_p90, margin_median, margin_p10, margin_p90`
- `summary_all_configs.csv`：每配置一行，含 τ 分布、`rate_delta_med_{10,20,30}db`、
  `rate_delta_{med,p10,p90}_at_tau`、以及 `tf_mae_db/tf_rmse_db/tf_bias_db/tf_agree_frac/
  tf_gap_med_db/tf_gap_p90_db/cdf_le{1,2,3,5}`。
- `threshold_policy_main.png`：主图 2 子图 —— (a) `thp_frac(γ)` 各配置曲线；
  (b) 各配置有限 τ 的箱线分布。

seed 固定 ⇒ 每次复跑数字一致；抽查可用 `--n-ch 100`（约 2 min）。
