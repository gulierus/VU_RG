"""
Seed-agregační pipeline pro bias²/variance Varianty C.

**Účel (přesně):** rozlišit STRUKTURÁLNÍ bias (vlastnost architektury) od OPTIMALIZAČNÍHO
rezidua konkrétního běhu. Načte N checkpointů natrénovaných s různým seedem, pro každý spočítá
bias² plateau a sklon variance, a reportuje **per-seed hodnoty + jejich rozptyl napříč seedy +
diagnostiku shody**. Teprve *shoda* plateau přes seedy opravňuje napsat „strukturální bias";
pokud se plateau rozcházejí, závěr je opačný (optimalizační artefakt).

Klíč k férovosti: všechny modely se hodnotí na **STEJNÉ** fixní sadě úloh, query a tahů support
setu (pevné data-seedy nezávislé na model-seedu). Tím se izoluje variace *mezi modely* od variace
dat.

Použití:
    # více seedů (cíl):
    python background/variant_c_seed_aggregation.py \
        models/pfn_seg_best_seed0.pth models/pfn_seg_best_seed1.pth models/pfn_seg_best_seed2.pth
    # bez argumentů: vezme models/pfn_seg_best.pth + všechny models/pfn_seg_best_seed*.pth
Výstup: background/variant_c_seed_results.json + figures/VariantC_pfn_seg/fig_C_07_seed_bias.png
"""
import os, sys, glob, json, math
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

GRID = 64
PAIR_BUDGET = 128
N_SUPP = [1, 2, 4, 8, 16, 32, 64]
PLATEAU_N = [16, 32, 64]            # „velké n" pro odečet bias² plateau
REGIMES = {"Easy": 0.40, "Hard": 0.05}
SIGMA = {"Easy": 0.10, "Hard": 0.25}
DATA_SEED = 777                     # pevné pro VŠECHNY modely (fér srovnání)
N_QUERIES = 8                       # fixních úloh/režim
N_DRAWS = 24                        # tahů support setu na (úloha, n_supp)


# ---- generátor + oracle (přesná kopie z notebooku Varianty C) ----
def _icdf(p):
    return math.sqrt(2.0) * torch.erfinv(2.0 * torch.as_tensor(p, dtype=torch.float32) - 1.0)

def _cov(H, W, ls, os_=1.0):
    iy = torch.arange(H); ix = torch.arange(W)
    dy = torch.minimum(iy, H - iy).float(); dx = torch.minimum(ix, W - ix).float()
    return (os_ ** 2) * torch.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) / (2.0 * ls ** 2))

def _fields(n, H, W, ls, os_, gen):
    c = _cov(H, W, ls, os_); Lam = torch.fft.fft2(c).real.clamp_min(0.0); sL = Lam.sqrt()
    xi = torch.randn(n, H, W, generator=gen) + 1j * torch.randn(n, H, W, generator=gen)
    return (torch.fft.fft2(sL[None] * xi) / math.sqrt(H * W)).real

def _phi(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

def oracle(x, ls, sig, tau, H=GRID, W=GRID, os_=1.0):
    c = _cov(H, W, ls, os_); Lam = torch.fft.fft2(c).real.clamp_min(0.0)
    mu = torch.fft.ifft2(Lam / (Lam + sig ** 2) * torch.fft.fft2(x)).real
    s = float((Lam * sig ** 2 / (Lam + sig ** 2)).mean().clamp_min(1e-12).sqrt())
    return _phi((mu - tau) / s)

def _gc():
    if DEVICE == "mps":
        torch.mps.empty_cache(); torch.mps.synchronize()

def load_model(path):
    from universeg import universeg
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = universeg(pretrained=False)
    m.load_state_dict(ck["model_state_dict"])
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck

@torch.no_grad()
def _predict(m, q, si, sl):
    lo = m(q.to(DEVICE), si.to(DEVICE), sl.to(DEVICE)).cpu()
    _gc()
    return torch.sigmoid(lo)


def bias_var_curve(m, lsf, sig):
    """Per-model bias²(n) a var(n) na FIXNÍ sadě N_QUERIES úloh (stejné pro všechny modely).
    bias_ij = E_S[p̂] − p_oracle (fixní query); var_ij = Var_S[p̂]. Vrací (b2[n], var[n])."""
    ell = lsf * GRID; tau = float(_icdf(1 - 0.5))
    gq = torch.Generator().manual_seed(DATA_SEED)         # fixní query napříč modely
    queries = []
    for _ in range(N_QUERIES):
        f = _fields(1, GRID, GRID, ell, 1.0, gq)
        x = (f + sig * torch.randn(1, GRID, GRID, generator=gq))[0]
        queries.append((x, oracle(x, ell, sig, tau)))
    b2_by_n = []; var_by_n = []
    for S in N_SUPP:
        eff = max(1, min(16, PAIR_BUDGET // S))
        b2s = []; vrs = []
        gs = torch.Generator().manual_seed(DATA_SEED + 1000 + S)   # fixní support tahy napříč modely
        for x, por in queries:
            preds = []; done = 0
            while done < N_DRAWS:
                cur = min(eff, N_DRAWS - done)
                si = []; sl = []
                for _ in range(cur):
                    fs = _fields(S, GRID, GRID, ell, 1.0, gs)
                    ims = fs + sig * torch.randn(S, GRID, GRID, generator=gs)
                    si.append(ims[None, :, None]); sl.append((fs > tau).float()[None, :, None])
                si = torch.cat(si); sl = torch.cat(sl)
                qb = x[None, None].expand(cur, -1, -1, -1)
                preds.append(_predict(m, qb, si, sl)[:, 0]); done += cur
            preds = torch.cat(preds)
            b2s.append(float(((preds.mean(0) - por) ** 2).mean()))
            vrs.append(float(preds.var(0).mean()))
        b2_by_n.append(float(np.mean(b2s))); var_by_n.append(float(np.mean(vrs)))
    return np.array(b2_by_n), np.array(var_by_n)


def summarize_seed(m):
    """Pro jeden model: bias² plateau (mean přes PLATEAU_N) a sklon variance (log-log) per režim."""
    out = {}
    for reg, lsf in REGIMES.items():
        b2, var = bias_var_curve(m, lsf, SIGMA[reg])
        plateau = float(np.mean([b2[N_SUPP.index(n)] for n in PLATEAU_N]))
        ok = var > 0
        slope = float(np.polyfit(np.log(np.array(N_SUPP)[ok]), np.log(var[ok]), 1)[0])
        out[reg] = {"bias2_curve": b2.tolist(), "var_curve": var.tolist(),
                    "bias2_plateau": plateau, "var_slope": slope}
    return out


def agreement_verdict(plateaus):
    """plateaus: list per-seed bias² plateau. Vrací (mean, std, cv, verdikt)."""
    a = np.asarray(plateaus, float)
    if len(a) < 2:
        return float(a.mean()), float("nan"), float("nan"), "POTŘEBA ≥2 SEEDŮ"
    mean, std = float(a.mean()), float(a.std(ddof=1))
    cv = std / mean if mean > 0 else float("nan")
    verdikt = ("SHODA → strukturální" if cv < 0.25 else
               "ROZPTYL → optimalizační reziduum" if cv > 0.5 else "NEJEDNOZNAČNÉ (mezi 0,25–0,5)")
    return mean, std, cv, verdikt


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = ([p for p in ["models/pfn_seg_best.pth"] if os.path.exists(p)]
                 + sorted(glob.glob("models/pfn_seg_best_seed*.pth")))
    paths = [p for p in dict.fromkeys(paths) if os.path.exists(p)]
    if not paths:
        print("Žádné checkpointy. Předej cesty nebo dej models/pfn_seg_best*.pth"); sys.exit(1)
    print(f"Device: {DEVICE} | seedů: {len(paths)}")
    for p in paths:
        print("  -", p)

    per_seed = []
    for p in paths:
        m, ck = load_model(p)
        s = summarize_seed(m)
        per_seed.append({"path": p, "config_seed": ck.get("config", {}).get("seed"), **s})
        for reg in REGIMES:
            print(f"[{os.path.basename(p)}] {reg}: bias² plateau={s[reg]['bias2_plateau']:.5f} "
                  f"var_slope={s[reg]['var_slope']:.2f}", flush=True)
        del m
        _gc()

    # --- agregace přes seedy ---
    print("\n=== AGREGACE PŘES SEEDY (rozliš strukturální vs optimalizační bias) ===")
    agg = {}
    for reg in REGIMES:
        plateaus = [d[reg]["bias2_plateau"] for d in per_seed]
        slopes = [d[reg]["var_slope"] for d in per_seed]
        mean, std, cv, verdikt = agreement_verdict(plateaus)
        agg[reg] = {"plateaus": plateaus, "plateau_mean": mean, "plateau_std": std,
                    "plateau_cv": cv, "verdict": verdikt,
                    "slope_mean": float(np.mean(slopes))}
        cvs = f"{cv:.2f}" if cv == cv else "—"
        print(f"{reg}: plateau per-seed={['%.5f'%x for x in plateaus]}  "
              f"mean={mean:.5f} std={std if std==std else float('nan'):.5f} CV={cvs}  → {verdikt}")

    # --- uložení + figura ---
    os.makedirs("figures/VariantC_pfn_seg", exist_ok=True)
    with open("background/variant_c_seed_results.json", "w") as f:
        json.dump({"paths": paths, "per_seed": per_seed, "aggregate": agg,
                   "config": {"data_seed": DATA_SEED, "n_queries": N_QUERIES, "n_draws": N_DRAWS,
                              "plateau_n": PLATEAU_N}}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {"Easy": "#2ca02c", "Hard": "#d62728"}
    for j, reg in enumerate(REGIMES):
        plateaus = agg[reg]["plateaus"]; x = np.full(len(plateaus), j) + np.linspace(-0.1, 0.1, len(plateaus))
        ax.scatter(x, plateaus, color=colors[reg], zorder=3, label=f"{reg} (per-seed)")
        m_, s_ = agg[reg]["plateau_mean"], agg[reg]["plateau_std"]
        if s_ == s_:
            ax.errorbar(j, m_, yerr=s_, fmt="_", color="k", capsize=6, zorder=2)
    ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(list(REGIMES))
    ax.set_ylabel("bias² plateau (per pixel)")
    ax.set_title("bias² plateau per seed — shoda ⇒ strukturální, rozptyl ⇒ optimalizační")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig("figures/VariantC_pfn_seg/fig_C_07_seed_bias.png", dpi=150, bbox_inches="tight")
    print("\nVýsledky: background/variant_c_seed_results.json")
    print("Figura:    figures/VariantC_pfn_seg/fig_C_07_seed_bias.png")
    if len(paths) < 2:
        print("\n⚠ Jen 1 seed — pro verdikt strukturální/optimalizační natrénuj 2–3 modely s jiným "
              "SEED a přidej jejich checkpointy (pfn_seg_best_seed{1,2}.pth).")


if __name__ == "__main__":
    main()
