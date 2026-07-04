"""
Q4b: OOD outputscale experiment for Experiment 6.
Analogous to Q4 (OOD lengthscale) in the notebook.
Runs with a fixed seed (55, same as Q4) and 200 instances per outputscale value.

Usage:
    cd /Users/ruslanguliev/VU_RG
    python3 background/run_q4b.py
"""
import os, sys, math, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "PFNs"))

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

from pfns.train import train, MainConfig, OptimizerConfig, TransformerConfig, BatchShapeSamplerConfig
from pfns.model.encoders import EncoderConfig
from pfns.model.bar_distribution import BarDistributionConfig
from pfns.priors.prior import AdhocPriorConfig
from pfns.priors.fast_gp import get_batch as get_batch_for_gp

# must be defined before torch.load
def get_batch_for_gp_random_hps(batch_size, seq_len, num_features,
                                  device="cpu", hyperparameters=None, **kwargs):
    hps = {"lengthscale": 0.3, "outputscale": 0.1, "noise": 1.0}
    return get_batch_for_gp(batch_size, seq_len, num_features,
                             device=device, hyperparameters=hps, **kwargs)


def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if "num_features" in cfg:
        nf = cfg["num_features"]; mds = cfg["max_dataset_size"]
        borders = ckpt["criterion"].borders.tolist(); crit = ckpt["criterion"]
        nlayers = cfg.get("nlayers", 6)
        get_fn = get_batch_for_gp; pkw = {"num_features": nf, "hyperparameters": cfg.get("hps", {})}
    else:
        nf = cfg["priors"][0]["prior_kwargs"]["num_features"]
        mds = cfg["batch_shape_sampler"]["max_seq_len"]
        borders = cfg["model"]["criterion"]["borders"]
        nlayers = cfg["model"].get("nlayers", 6)
        get_fn = get_batch_for_gp_random_hps; pkw = {"num_features": nf, "hyperparameters": {}}; crit = None

    mc = MainConfig(
        priors=[AdhocPriorConfig(get_batch_methods=[get_fn], prior_kwargs=pkw)],
        optimizer=OptimizerConfig("adamw", lr=0.0003),
        model=TransformerConfig(
            criterion=BarDistributionConfig(full_support=True, borders=borders),
            emsize=512, nhead=8, nhid=1024, nlayers=nlayers,
            features_per_group=1, attention_between_features=False,
            encoder=EncoderConfig(constant_normalization_mean=0.5,
                                   constant_normalization_std=math.sqrt(1/12))
        ),
        batch_shape_sampler=BatchShapeSamplerConfig(
            batch_size=2, max_seq_len=mds, min_num_features=nf, max_num_features=nf
        ),
        epochs=1, steps_per_epoch=1, num_workers=0,
    )
    res = train(mc, device=device, reusable_config=False)
    model = res["model"]
    model.load_state_dict(ckpt["model_state_dict"])
    if crit is not None:
        model.criterion = crit
    model.to(device).eval()
    nlayers_actual = model.transformer_layers.num_layers
    print(f"  Loaded {os.path.basename(path)}: nlayers={nlayers_actual}, epoch={ckpt.get('epoch','?')}")
    return model


def rbf_kernel(x1, x2, ls, osc=1.0):
    dist_sq = (x1[:, None] - x2[None, :]) ** 2
    return osc * np.exp(-dist_sq / (2.0 * ls**2))


def gp_posterior(tx, ty, te, ls, noise, osc=1.0):
    K   = rbf_kernel(tx, tx, ls, osc) + noise * np.eye(len(tx))
    Ks  = rbf_kernel(te, tx, ls, osc)
    try:
        L     = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, ty))
        mean  = Ks @ alpha
    except np.linalg.LinAlgError:
        mean = np.zeros(len(te))
    return mean


def pfn_predict(model, train_x, train_y, test_x):
    with torch.no_grad():
        logits = model(train_x[None].to(device),
                       train_y[None].to(device),
                       test_x[None].to(device))
    return model.criterion.mean(logits)[0].detach().cpu().numpy()


def generate_datasets(n_context, n_test, n_inst, hps, seed=55):
    torch.manual_seed(seed)
    np.random.seed(seed)
    datasets = []
    for _ in range(n_inst):
        batch = get_batch_for_gp(
            batch_size=1, seq_len=n_context + n_test, num_features=1,
            device="cpu", hyperparameters=hps)
        train_x = batch.x[0, :n_context, :]
        train_y = batch.y[0, :n_context]
        test_x  = batch.x[0, n_context:, :]
        test_y  = batch.y[0, n_context:]
        datasets.append((train_x, train_y, test_x, test_y))
    return datasets


def compute_ood_mse_osc(model, osc_values, ls, n_context, n_test, n_inst, noise):
    results = {}
    for osc in osc_values:
        hps = {"lengthscale": ls, "noise": noise, "outputscale": osc}
        datasets = generate_datasets(n_context, n_test, n_inst, hps, seed=55)
        pfn_mses, ref_mses = [], []

        for train_x, train_y, test_x, test_y in datasets:
            tx    = train_x.numpy().reshape(-1)
            ty    = train_y.numpy().reshape(-1)
            te    = test_x.numpy().reshape(-1)
            ty_te = test_y.numpy().reshape(-1)
            try:
                gp_c  = gp_posterior(tx, ty, te, ls, noise, osc)
                pfn_m = pfn_predict(model, train_x, train_y, test_x)
                mse_pfn = float(np.mean((pfn_m - gp_c)**2))
                mse_ref = float(np.mean((gp_c  - ty_te)**2))
                if np.isfinite(mse_pfn) and mse_pfn < 1e6:
                    pfn_mses.append(mse_pfn)
                if np.isfinite(mse_ref):
                    ref_mses.append(mse_ref)
            except Exception:
                pass

        pfn_mean = float(np.nanmean(pfn_mses)) if pfn_mses else float("nan")
        pfn_se   = float(np.std(pfn_mses) / np.sqrt(len(pfn_mses))) if pfn_mses else float("nan")
        ref_mean = float(np.nanmean(ref_mses)) if ref_mses else float("nan")
        rel_mse  = pfn_mean / max(ref_mean, 1e-8) if np.isfinite(pfn_mean) else float("nan")

        results[osc] = {"pfn_mean": pfn_mean, "pfn_se": pfn_se,
                        "ref_mean": ref_mean, "rel_mse": rel_mse}
        print(f"  osc={osc:6.3f}:  MSE_PFN={pfn_mean:.5f},  ref={ref_mean:.5f},  rel={rel_mse:.3f}")

    return results


def print_table(results, osc_values, train_lo=0.439, train_hi=2.276):
    print(f"\n{'osc':>8}  {'MSE_PFN':>12}  {'MSE_ref':>12}  {'rel_MSE':>10}  Stav")
    print("-" * 62)
    for osc in osc_values:
        r   = results[osc]
        ood = osc < train_lo or osc > train_hi
        tag = " <- OOD" if ood else ""
        print(f"{osc:>8.3f}  {r['pfn_mean']:>12.5f}  {r['ref_mean']:>12.5f}  "
              f"{r['rel_mse']:>10.3f}{tag}")


if __name__ == "__main__":
    MODEL_PATH = os.path.join("models", "pfn_rand_hps_6layer.pth")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join("models", "pfn_rand_hps_6L_1000epoch.pth")

    print(f"\nLoading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    OSC_VALUES = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    LS         = 0.3
    NOISE      = 0.01
    N_CONTEXT  = 40
    N_TEST     = 10
    N_INST     = 200

    print(f"\nQ4b: OOD outputscale, l={LS}, noise={NOISE}, n={N_CONTEXT}, inst={N_INST}")
    print(f"     osc_values={OSC_VALUES}\n")

    results = compute_ood_mse_osc(model, OSC_VALUES, LS, N_CONTEXT, N_TEST, N_INST, NOISE)
    print_table(results, OSC_VALUES)
