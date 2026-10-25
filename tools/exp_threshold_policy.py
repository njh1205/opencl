#!/usr/bin/env python3
"""Paper-level numerics for the hpocl SVD-vs-THP(GMD/UCD) switching threshold.
Closed form = cf_ref.tau_policy(lam,M,d,K,variant); flag 2 -> tau = +-inf.
Metrics: 1) thp_frac(g)=P_ch[g>=tau]; 2) margin=|g-tau|; 3) rate_delta(g)=R_THP-R_SVD
(R=sum log2(1+rho_realized), g in {10,20,30} dB, g=tau); 4) transformer tau_tf vs
tau_policy: MAE/bias, decision agreement, |tau_tf-tau| gap stats.
Outputs (created-only): build/experiments/cfg<i>_*.csv, summary_all_configs.csv,
threshold_policy_main.png; run: python3 tools/exp_threshold_policy.py [--n-ch 400]"""
import argparse, io, pathlib, pickle, sys, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.linalg import svd
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MIX = ROOT.parent / "mix_precode"
for p in (str(HERE), str(MIX / "classical"), str(MIX)):
    if p not in sys.path: sys.path.insert(0, p)
from env_common import dummy_env          # shared dimension shell (was: pipeline_ref)
from analog_precoder import AnalogPrecoder
from digital_precoder import DigitalStructuredPrecoder
import cf_ref                         # deployed closed-form threshold
import ckpt_io                        # shared checkpoint reader / weight mapping
import transformer_ref as tfref
GRID = [(2,4,2,8,2,64,0),(2,4,2,8,2,64,1),(3,6,2,12,2,16,0),(4,8,2,16,2,64,0),(2,8,2,12,2,64,1)]
SNR_DB = np.arange(0.0, 41.0, 1.0); FIXED_GAMMA_DB = (10.0, 20.0, 30.0)
CKPT = MIX / "_archive_20261018/bulk/results/svd_ucd_threshold_poolv2_3x1000_fast/poolv2_3x1000_fast_model.pt"
CKPT_DS = CKPT.with_name("poolv2_3x1000_fast_dataset.npz")
# Checkpoint reading and the state_dict -> weight mapping live in tools/ckpt_io.py
# (shared with tools/export_transformer.py, which produces the deployed binary).
# Keeping one implementation is what makes tf_export_check meaningful: it proves
# the *deployed* path, and this experiment uses the same mapping.
def build_tf(path):
    """checkpoint -> (transformer_ref weight dict, eval-mode metadata)."""
    ck = ckpt_io.load_ckpt_numpy(path)
    W, dims = ckpt_io.state_dict_to_weights(ck)
    nrm = ckpt_io.norm_from_artifacts(ck["artifacts"], dims)
    # T=9 is this pool's token budget (K*d = 8 real tokens + 1 pad), not a model
    # constant; it comes from the dataset, see transformer_stats().
    return W, dict(D=dims["d_model"], L=dims["layers"], H=dims["heads"], T=9,
                   fmean=nrm["feature_mean"], fstd=nrm["feature_std"],
                   cmean=nrm["config_mean"], cstd=nrm["config_std"],
                   tmean=nrm["target_mean"], tstd=nrm["target_std"])
def gen_lam_rates(cfg, seed, n):
    """n channels -> lam, tau_policy, flag, rate_delta(:, 10/20/30 dB, at tau)."""
    K, Nt, Nr, NRF, d, M, var = cfg
    env = dummy_env(K, Nt, Nr, NRF, d)
    prec = DigitalStructuredPrecoder(NRF, d)
    thp = ("gmd", "ucd")[var]
    rng = np.random.default_rng(seed)
    lam_a = np.zeros((n, K * d)); tau_a = np.full(n, np.nan)
    flag_a = np.zeros(n, np.int8); delta = np.zeros((n, len(FIXED_GAMMA_DB) + 1))
    with warnings.catch_warnings():    # benign NRF>Nt whitening overflow (prod too)
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n):
            H = (rng.normal(0, 1, (K, Nr, Nt)) + 1j * rng.normal(0, 1, (K, Nr, Nt))) / np.sqrt(2)
            Frf = AnalogPrecoder(Nt, NRF).build_precoder(H, d)
            eff = env.build_effective_channels(H, Frf)
            lam, Gs = [], []
            for k in range(K):
                N = env.build_bd_digital_basis(eff, k)
                _, _, G = env.build_power_whitened_bd_channel(eff, Frf, k, bd_basis=N)
                s = svd(G, compute_uv=False)
                Gs.append(G); lam += list(s[:d] ** 2)
            lam = np.array(lam)
            tau, flag = cf_ref.tau_policy(lam, M, d, K, var)
            lam_a[i], tau_a[i], flag_a[i] = lam, tau, flag
            gammas = list(FIXED_GAMMA_DB) + ([float(np.clip(tau, 0, 40))] if np.isfinite(tau) else [])
            for j, g in enumerate(gammas):
                snr_ps = 10 ** (g / 10) / (K * d)
                svd_r = sum(float(np.log2(1 + prec.design_precoder(G, snr_ps, strategy="svd")[1]).sum()) for G in Gs)
                thp_r = sum(float(np.log2(1 + prec.design_precoder(G, snr_ps, strategy=thp)[1]).sum()) for G in Gs)
                delta[i, j] = thp_r - svd_r
    return dict(lam=lam_a, tau=tau_a, flag=flag_a, delta=delta)
JOB_CH = 50   # channels per parallel chunk (finer chunks -> better load balance)
TF_SELFCHECK_MAE_DB = 2.0   # trained student's own MAE on this pool is ~0.6 dB
TF_AGREE_BAND_DB = 5.0      # decision-relevant SNR band around each channel's tau
def run_config(cfg, res):
    lam = np.concatenate([r["lam"] for r in res])
    tau = np.concatenate([r["tau"] for r in res])
    flag = np.concatenate([r["flag"] for r in res])
    delta = np.concatenate([r["delta"] for r in res])
    fin = np.isfinite(tau)
    rows = []
    for g in SNR_DB:
        marg = np.abs(g - tau[fin])
        # NaN tau (saturated / ill-conditioned channels) made thp_frac and
        # svd_frac both False, so the two did not sum to 1. The C++ decision is
        # `mode = (gamma_tot_dB >= tau_dB)`, which is false for NaN, i.e. SVD --
        # mirror that explicitly so the fractions partition the channels.
        thp = np.where(np.isnan(tau), False, g >= tau)
        rows.append(dict(snr_db=g, thp_frac=float(thp.mean()),
                         svd_frac=float((~thp).mean()), flag2_frac=float((flag == 2).mean()),
                         tau_mean=float(tau[fin].mean()), tau_p10=float(np.percentile(tau[fin], 10)),
                         tau_p90=float(np.percentile(tau[fin], 90)),
                         margin_median=float(np.median(marg)),
                         margin_p10=float(np.percentile(marg, 10)),
                         margin_p90=float(np.percentile(marg, 90))))
    return dict(cfg=cfg, lam=lam, tau=tau, flag=flag, delta=delta, fin=fin, rows=rows)
def transformer_stats(cfg, W, meta, lam, tau, fin):
    """tokens (extract_bd_spectral_tokens schema) -> tau_tf; stats vs tau."""
    K, Nt, Nr, NRF, d = cfg[:5]
    cfg_feat = np.array([K, Nt, Nr, NRF, d, K * d, 6.0, NRF / Nt, K * d / NRF, d / Nr], np.float32)
    tau_tf = np.zeros(len(lam))
    dims = dict(T=meta["T"], heads=meta["H"], layers=meta["L"], d_model=meta["D"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, l in enumerate(lam):
            s = np.sqrt(np.maximum(np.asarray(l), 1e-12)).reshape(K, d)
            log_s = np.log(s)
            tok = np.stack([log_s, log_s - log_s.mean(),
                            s ** 2 / np.maximum((s ** 2).sum(), 1e-12),
                            np.linspace(0.5/K, 1-0.5/K, K)[:, None].repeat(d, 1),
                            np.linspace(0.5/d, 1-0.5/d, d)[None, :].repeat(K, 0)], -1)
            pad = np.zeros((meta["T"], 5), np.float32); pad[:K * d] = tok.reshape(-1, 5)
            mask = np.zeros(meta["T"], np.float32); mask[:K * d] = 1.0
            tn = np.where(mask[:, None].astype(bool), (pad - meta["fmean"]) / meta["fstd"], 0.0)
            cn = ((cfg_feat - meta["cmean"]) / meta["cstd"]).astype(np.float32)
            tau_n, _ = tfref.forward(W, dims, tn, mask, cn, np.array([20.0], np.float32))
            tau_tf[i] = tau_n * meta["tstd"] + meta["tmean"]
    # Agreement restricted to the decision-relevant band |g - tau| <= band.
    # Pooling over the whole 0..40 dB grid made this metric ~1 by construction:
    # at the grid extremes both laws pick the same branch for nearly every
    # channel, so the number said nothing about the interesting region.
    band = np.abs(SNR_DB[:, None] - tau[None, :]) <= TF_AGREE_BAND_DB
    same = (SNR_DB[:, None] >= tau[None, :]) == (SNR_DB[:, None] >= tau_tf[None, :])
    agree = float(same[band].mean()) if band.any() else float("nan")
    gap = tau_tf[fin] - tau[fin]
    return dict(agree=agree, mae=float(np.mean(np.abs(gap))),
                rmse=float(np.sqrt(np.mean(gap ** 2))), bias=float(np.median(gap)),
                gap_med=float(np.median(np.abs(gap))), gap_p90=float(np.percentile(np.abs(gap), 90)),
                cdf=[float((np.abs(gap) <= k).mean()) for k in (1, 2, 3, 5)])

def write_plots(summaries, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib unavailable -> CSV only]"); return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in summaries:
        ax[0].plot(SNR_DB, [r["thp_frac"] for r in s["rows"]],
                   label="k%d nt%d nr%d nrf%d d%d M%d v%d" % s["cfg"], lw=1.5)
        ax[1].boxplot(s["tau"][s["fin"]], positions=[s["idx"] + 1], widths=0.6, showfliers=False)
    ax[0].set_xlabel(r"total SNR $\gamma$ [dB]"); ax[0].set_ylabel("THP fraction")
    ax[0].set_title("(a) closed-form THP branch share vs SNR"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    ax[1].set_xticks(range(1, len(summaries) + 1))
    ax[1].set_xticklabels(["%d\n%d/%d/%d/%d/M%d/v%d" % s["cfg"] for s in summaries], fontsize=7)
    ax[1].set_ylabel(r"finite $\tau$ [dB]"); ax[1].set_title("(b) threshold distribution")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "threshold_policy_main.png", dpi=150)
    print("  wrote", out_dir / "threshold_policy_main.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ch", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out_dir = ROOT / "build" / "experiments"; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    W = meta = None
    if CKPT.exists():
        W, meta = build_tf(CKPT)
        print(f"[tf] checkpoint: {CKPT.name} D={meta['D']} L={meta['L']} H={meta['H']}")
        if CKPT_DS.exists():            # numpy-forward selfcheck on its own dataset
            ds = np.load(CKPT_DS); errs = []
            dims = dict(T=meta["T"], heads=meta["H"], layers=meta["L"], d_model=meta["D"])
            for i in range(0, 120, 30):
                ti, mi, ci = ds["tokens"][i], ds["token_masks"][i], ds["config_features"][i]
                tn = np.where(mi[:, None].astype(bool), (ti - meta["fmean"]) / meta["fstd"], 0.0)
                cn = ((ci - meta["cmean"]) / meta["cstd"]).astype(np.float32)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    tau_n, _ = tfref.forward(W, dims, tn, mi, cn, np.array([20.0]))
                errs.append(abs(tau_n * meta["tstd"] + meta["tmean"] - ds["thresholds_db"][i]))
            mae = float(np.mean(errs))
            # was an unconditional ": OK" -- a self-check that cannot fail is
            # worse than none, because it reads as evidence.
            if not (mae < TF_SELFCHECK_MAE_DB):
                raise SystemExit(f"[tf] selfcheck FAILED: numpy forward disagrees with "
                                 f"the checkpoint's own labels, MAE={mae:.3f}dB "
                                 f">= {TF_SELFCHECK_MAE_DB}dB")
            print(f"  [tf] selfcheck MAE={mae:.3f}dB on checkpoint dataset "
                  f"(< {TF_SELFCHECK_MAE_DB}dB): OK")
    else:
        print("[tf] NO checkpoint under mix_precode -> transformer part skipped")
    chunks = {i: [] for i in range(len(GRID))}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = []
        for idx, cfg in enumerate(GRID):
            for j in range(-(-a.n_ch // JOB_CH)):    # ceil(n_ch/JOB_CH) chunks
                n = min(JOB_CH, a.n_ch - j * JOB_CH)
                futs.append((idx, cfg, ex.submit(gen_lam_rates, cfg, a.seed + 10000 * idx + j, n)))
        for idx, cfg, f in futs:
            chunks[idx].append(f.result())
    summaries = []
    for idx, cfg in enumerate(GRID):
        t1 = time.time()
        s = run_config(cfg, chunks[idx])
        s["idx"] = idx
        print(f"  [cfg {idx}] {cfg} n={a.n_ch} tau_mean={s['tau'][s['fin']].mean():.2f}dB "
              f"finite={s['fin'].mean():.2f} flag2={np.mean(s['flag']==2):.2f} ({time.time()-t1:.0f}s)", flush=True)
        fname = f"cfg{idx}_k{cfg[0]}_nt{cfg[1]}_nr{cfg[2]}_nrf{cfg[3]}_d{cfg[4]}_m{cfg[5]}_var{cfg[6]}.csv"
        with open(out_dir / fname, "w") as f:
            f.write(",".join(s["rows"][0]) + "\n")
            f.writelines(",".join(f"{r[c]:.6g}" for c in s["rows"][0]) + "\n" for r in s["rows"])
        if W is not None:
            s["tf"] = transformer_stats(cfg, W, meta, s["lam"], s["tau"], s["fin"])
            print(f"  [tf] cfg {idx}: MAE={s['tf']['mae']:.3f}dB agree={s['tf']['agree']:.4f} gap_p90={s['tf']['gap_p90']:.2f}dB", flush=True)
        summaries.append(s)
    write_plots(summaries, out_dir)
    cols = ["cfg_id","K","Nt","Nr","NRF","d","M","variant","n_ch","finite_frac","flag2_frac",
            "tau_mean","tau_p10","tau_p90","rate_delta_med_10db","rate_delta_med_20db",
            "rate_delta_med_30db","rate_delta_med_at_tau","rate_delta_p10_at_tau","rate_delta_p90_at_tau"]
    if W is not None:
        # the tf_* columns only exist when a checkpoint was found; emitting the
        # header unconditionally produced a 30-column header over 20-column rows.
        cols += ["tf_mae_db","tf_rmse_db","tf_bias_db",
                 f"tf_agree_frac_pm{int(TF_AGREE_BAND_DB)}db","tf_gap_med_db",
                 "tf_gap_p90_db","cdf_le1","cdf_le2","cdf_le3","cdf_le5"]
    else:
        print("[tf] WARNING: no checkpoint -> summary_all_configs.csv omits all tf_* columns")
    with open(out_dir / "summary_all_configs.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for s in summaries:
            c, tau, fin, dl = s["cfg"], s["tau"], s["fin"], s["delta"]
            row = [str(x) for x in (c[0], c[1], c[2], c[3], c[4], c[5], c[6], a.n_ch, fin.mean(), np.mean(s["flag"] == 2))]
            row += [f"{x:.4f}" for x in (tau[fin].mean(), np.percentile(tau[fin], 10), np.percentile(tau[fin], 90))]
            row += [f"{np.median(dl[:, j]):.4f}" for j in range(len(FIXED_GAMMA_DB))]
            row += [f"{x:.4f}" for x in (np.median(dl[:,-1]), np.percentile(dl[:,-1], 10), np.percentile(dl[:,-1], 90))]
            if W is not None:
                t = s["tf"]
                row += [f"{t['mae']:.4f}", f"{t['rmse']:.4f}", f"{t['bias']:.4f}", f"{t['agree']:.4f}", f"{t['gap_med']:.4f}", f"{t['gap_p90']:.4f}"] + [f"{v:.4f}" for v in t["cdf"]]
            line = [str(s["idx"])] + row
            assert len(line) == len(cols), f"row has {len(line)} fields, header {len(cols)}"
            f.write(",".join(line) + "\n")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_dir}")
if __name__ == "__main__":
    main()
