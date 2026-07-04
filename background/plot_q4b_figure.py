"""
Generate Q4b OOD outputscale figure from pre-computed results
and save to figures/ (project root) for inclusion in the article.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Results from run_q4b.py (seed=55, n=200, n_context=40, ls=0.3, noise=0.01)
OSC_VALUES = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
RESULTS = {
    0.01:  {"pfn_mean": 0.00058, "ref_mean": 0.01059, "rel_mse": 0.055},
    0.05:  {"pfn_mean": 0.00037, "ref_mean": 0.01089, "rel_mse": 0.034},
    0.1:   {"pfn_mean": 0.00034, "ref_mean": 0.01103, "rel_mse": 0.031},
    0.3:   {"pfn_mean": 0.00038, "ref_mean": 0.01126, "rel_mse": 0.034},
    0.5:   {"pfn_mean": 0.00045, "ref_mean": 0.01137, "rel_mse": 0.040},
    1.0:   {"pfn_mean": 0.01091, "ref_mean": 0.01154, "rel_mse": 0.946},
    2.0:   {"pfn_mean": 0.04763, "ref_mean": 0.01173, "rel_mse": 4.060},
    3.0:   {"pfn_mean": 0.09982, "ref_mean": 0.01185, "rel_mse": 8.422},
    5.0:   {"pfn_mean": 0.38458, "ref_mean": 0.01202, "rel_mse": 31.992},
    10.0:  {"pfn_mean": 1.80979, "ref_mean": 0.01227, "rel_mse": 147.443},
    20.0:  {"pfn_mean": 6.75831, "ref_mean": 0.01256, "rel_mse": 538.169},
    50.0:  {"pfn_mean": 30.95136, "ref_mean": 0.01299, "rel_mse": 2383.324},
}

TRAIN_LO = 0.439
TRAIN_HI = 2.276
LS = 0.3
NOISE = 0.01

osc_arr = np.array(OSC_VALUES)
rel_mse = np.array([RESULTS[o]["rel_mse"] for o in OSC_VALUES])

fig, ax = plt.subplots(figsize=(11, 5))

ax.axvspan(TRAIN_LO, TRAIN_HI, alpha=0.12, color='green',
           label=f'Trenovaci rozsah osc ~ LogN(0,0.5): [{TRAIN_LO:.2f}, {TRAIN_HI:.2f}]')
ax.axhline(1.0, color='green', ls=':', lw=1.8,
           label='Referencni hodnota (rel_MSE = 1)')
ax.plot(osc_arr, rel_mse, 'o-', color='steelblue', lw=2.2, ms=8,
        label='rel_MSE = MSE_PFN / MSE_oracle')

ax.set_xscale('log')
ax.set_xlabel('Outputscale (log skala)', fontsize=12)
ax.set_ylabel('Relativni MSE  (MSE_PFN / MSE_oracle)', fontsize=12)
ax.set_title(
    f'Q4b: OOD outputscale (l={LS}, noise={NOISE}, n=40)\n'
    f'Zelena oblast = 5.-95. percentil trenovaci distribuce LogN(0, 0.5)',
    fontsize=12
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = os.path.join(ROOT, "figures", "GP2_exp6_hp_identification", "fig_exp6_q4b_ood_outputscale.png")
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"Saved: {out_path}  ({os.path.getsize(out_path)//1024} KB)")
