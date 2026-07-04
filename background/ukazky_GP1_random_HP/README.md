# Ukázky — GP1 experimenty s náhodnými hyperparametry

Sada obrázků z notebooku [`experiments/Experiments_from_GP1_random_model.ipynb`](../../experiments/Experiments_from_GP1_random_model.ipynb).

**Model:** jeden 6-vrstvý PFN trénovaný na **distribuci hyperparametrů** (random-HP), nikoliv na fixních HP. Musí tedy identifikovat správné HP (lengthscale, noise, outputscale) přímo z kontextových dat. Architektura: `emsize=512`, `nhead=8`, `nhid=1024`, `nlayers=6`.

Popisy níže jsou převzaté z interpretačních buněk notebooku.

---

## Experiment 1 — Predikce a uncertainty (PFN vs GP oracle)

| Obrázek | Popis |
|---|---|
| ![prior](fig_rand_01_prior_samples.png) | **`fig_rand_01_prior_samples.png`** — vzorky z GP prioru s náhodnými HP; ukázka, jak vypadají trénovací funkce. |
| ![uncertainty](fig_rand_02_uncertainty.png) | **`fig_rand_02_uncertainty.png`** — Predikce a uncertainty: **modrá (PFN)** = model, který musel sám odhadnout HP z dat; **zelená (GP)** = oracle se znalostí správných HP. Porovnáváme tvar a škálu uncertainty. |

## Experiment 2 — Struktura attention přes vrstvy

| Obrázek | Popis |
|---|---|
| ![attn-all](fig_rand_03_attention_all_layers.png) | **`fig_rand_03_attention_all_layers.png`** — attention matice napříč všemi vrstvami. Hledáme asymetrii: Test→Train aktivní, Train→Test nulový. |

## Experiment 3 — Attention vs RBF kernel

| Obrázek | Popis |
|---|---|
| ![attn-detail](fig_rand_04_attention_detail.png) | **`fig_rand_04_attention_detail.png`** — detail: attention vrstva 0 vs poslední vrstva, Test→Train, a přímé porovnání attention vs RBF kernel. Attention má podobný trend jako RBF, ale je **ostřejší**; entropie přes vrstvy ukazuje měnící se šířku. |

## Experiment 4 — Vliv velikosti kontextu

| Obrázek | Popis |
|---|---|
| ![ctx-mse](fig_rand_05_context_size_mse.png) | **`fig_rand_05_context_size_mse.png`** — MSE(PFN, reálný model) vs MSE(GP) podle velikosti kontextu (stejná GP realizace). Pro **malý kontext PFN vyhrává**, ale jakmile kontext naroste, **PFN prohrává**. |
| ![ctx-grid](fig_rand_05b_context_size_grid.png) | **`fig_rand_05b_context_size_grid.png`** — grid predikcí a kontext→kontext attention vs RBF pro různá `n_ctx`. Skrytím bodů nutíme attention sledovat i vzdálenější body. PFN trénovaný na distribuci HP dělá pravděpodobně **implicitní marginalizaci** přes možné HP. |

## Experiment 5 — Přesnost odhadu lengthscale

| Obrázek | Popis |
|---|---|
| ![ls-acc](fig_rand_06_ls_accuracy.png) | **`fig_rand_06_ls_accuracy.png`** — přesnost PFN pro různé lengthscale. Čím **divočejší funkce** (menší LS), tím hůř PFN odhaduje — nepomáhá ani přidání kontextových bodů. |
| ![ls-pred](fig_rand_06b_ls_predictions.png) | **`fig_rand_06b_ls_predictions.png`** — PFN vs GP predikce pro sadu různých lengthscale. |

## Nové experimenty — Identifikace hyperparametrů

Postup: data jsou z GP(LS=0.1). PFN dostane 3, 5, 10, 20, 40, 60 bodů a predikuje. Predikci porovnáme se dvěma GP — **zelená (correct):** GP(LS=0.1), **červená (wrong):** GP(LS=0.3, 3× delší). Blíž zelené = PFN identifikoval LS z dat.

| Obrázek | Popis |
|---|---|
| ![hp-ls01](fig_rand_07a_hp_ident_ls01.png) | **`fig_rand_07a_hp_ident_ls01.png`** — identifikace HP, data z GP(LS=0.1), varianta 1. |
| ![hp-ls02](fig_rand_07b_hp_ident_ls02.png) | **`fig_rand_07b_hp_ident_ls02.png`** — identifikace HP, data z GP(LS=0.1), varianta 2. |
| ![pfn-ml2](fig_rand_08_pfn_vs_ml2.png) | **`fig_rand_08_pfn_vs_ml2.png`** — přesnost a rychlost: PFN vs Type-II ML (marginální věrohodnost). |
| ![marg](fig_rand_09_marginalization.png) | **`fig_rand_09_marginalization.png`** — je PFN blíž k ML-II bodovému odhadu, nebo k plné marginalizaci přes HP? |

---

*Obrázky byly extrahované skriptem [`background/extract_figures.py`](../extract_figures.py) (prefix `fig_rand_*`). Zdroj: `experiments/Experiments_from_GP1_random_model.ipynb`.*
