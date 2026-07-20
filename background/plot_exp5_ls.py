"""
Přegeneruje obrázky Experimentu 5 (krabicový graf + čára mediánu) do
figures/GP1_random_model/ z IDENTICKÉ pipeline jako tabulka 3.6, tedy
collect_stats.run_exp5 (50 instancí = 5 seedů x 10 realizací, fixní kernel
s pravým ℓ). Nemusí se kvůli tomu spouštět celý collect_stats (Exp 1-8).

Použití:
    cd /Users/ruslanguliev/VU_RG
    python3 background/plot_exp5_ls.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_stats as cs

# Checkpoint se pickle-uje s referencí na get_batch_for_gp_random_hps v __main__
# namespace, proto ho definujeme ZDE PŘED torch.load (stejná past jako v noteboocích).
get_batch_for_gp = cs.get_batch_for_gp
def get_batch_for_gp_random_hps(batch_size, seq_len, num_features,
                                 device="cpu", hyperparameters=None, **kwargs):
    hps = {"lengthscale": 0.3, "outputscale": 0.1, "noise": 1.0}
    return get_batch_for_gp(batch_size, seq_len, num_features,
                             device=device, hyperparameters=hps, **kwargs)

LS_LIST = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
N_SEEDS = 5

rand_model = cs.load_model(os.path.join("models", "pfn_rand_hps_latest_epoch_500.pth"))

# 5 seedů x 10 realizací = 50 instancí na ℓ (přesně jako main() v collect_stats)
raw5 = {ls: [] for ls in LS_LIST}
for seed in range(N_SEEDS):
    r = cs.run_exp5(rand_model, seed, ls_list=LS_LIST)
    for ls in LS_LIST:
        raw5[ls].extend(r[ls])
    print(f"  seed={seed}: " + " ".join(f"ls{ls}={np.median(r[ls]):.5f}" for ls in LS_LIST),
          flush=True)

# Kontrola: mediány musí sedět na stats_averaged.json (tj. na tabulku 3.6).
auth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_averaged.json")
if os.path.exists(auth_path):
    auth = json.load(open(auth_path)).get("exp5_rand", {})
    print("\n=== kontrola medián vs stats_averaged.json (tabulka 3.6) ===")
    worst = 0.0
    for ls in LS_LIST:
        new = float(np.median(raw5[ls])); old = auth.get(str(ls), {}).get("median")
        if old is None:
            continue
        rel = abs(new - old) / max(old, 1e-12)
        worst = max(worst, rel)
        print(f"  ℓ={ls:<5} nový={new:.6g}  json={old:.6g}  odchylka={rel:.2%}")
    print("VERDIKT:", "OK — sedí na tabulku" if worst < 0.01 else f"POZOR: max odchylka {worst:.1%}")

cs.save_exp5_figures(raw5)
print("Hotovo.")
