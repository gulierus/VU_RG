# Project Transformer — PFN as GP Approximator

Výzkumný projekt: Jak Prior-Fitted Networks (PFN) aproximují inferenci Gaussovských procesů.

## TODO

### Experimenty s modelem trénovaným na distribuci hyperparametrů

- Ověřit, zda PFN rozpozná správný lengthscale — porovnat MSE s GP se správnými vs špatnými HP
- Zjistit, kolik context pointů stačí pro identifikaci hyperparametrů
- Porovnat PFN vs Type-II ML (přesnost + rychlost inference)
- Otestovat chování mimo trénovaný rozsah HP (OOD lengthscale)
- Vizualizovat attention matice pro různé lengthscale — ověřit zda krátký LS → lokálnější attention
- Vizuální porovnání predikcí PFN vs GP pro různé lengthscale v jednom řádku grafů
- Zopakovat experimenty konvergence a vlivu kontextu pro tři různé lengthscale (0.1, 0.3, 0.7)

### Experiment: PFN inference na reálné GP realizaci (pro všechny modely)

Vygenerovat kompletní GP realizaci s mnoha body (ground truth = skutečné samplované hodnoty `y`), PFN dát pouze malou podmnožinu těchto bodů a měřit jak dobře PFN odhadne zbývající skutečné hodnoty. Na rozdíl od stávajícího Experimentu 5 (kde ground truth = GP posteriorní střední hodnota) jde zde o predikci skutečných nasamplovaných `y`, což je těžší úkol — zahrnuje i šum a náhodnost realizace. Spustit pro všechny tři modely (20 epoch, 100 epoch, random HP).

## Struktura repozitáře

```
VU_RG/
├── experiments/          # Hlavní experimenty
│   ├── HOW_PFN_APPROX_GP_KERNEL_big_model.ipynb    # Analýza velkého modelu (100 epoch)
│   └── HOW_PFN_APPROX_GP_KERNEL_small_model.ipynb  # Analýza malého modelu (20 epoch)
├── train/                # Trénování modelu
│   └── PFN_TRAIN_SETUP.ipynb   # Setup a spuštění tréninku
├── examples/             # Pomocné příklady (GP metody, TabPFN)
├── background/           # Matematické podklady
│   └── PFN-GP.md         # Vztah mezi PFN a GP
├── models/               # Uložené modely (ignorovány gitem)
├── requirements.txt
└── setup.sh
```

## Quickstart

```bash
# 1. Naklonuj repozitář
git clone <repo-url> && cd VU_RG

# 2. Nainstaluj závislosti a PFNs
bash setup.sh

# 3. Spusť Jupyter z rootu projektu (důležité pro správné cesty k modelům)
jupyter notebook
```

## Workflow

1. **Trénování:** `train/PFN_TRAIN_SETUP.ipynb` — natrénuje PFN na GP prioru, uloží model do `models/`
2. **Experimenty:** `experiments/HOW_PFN_APPROX_GP_KERNEL_*.ipynb` — načte model a spustí analýzu

## Experimenty (shrnutí)

| Experiment | Popis |
|---|---|
| 1 | Konvergence mean funkce k prioru + kalibrace variance |
| 2 | Analýza attention vah přes vrstvy |
| 3 | Porovnání attention vs RBF kernel |
| 4 | PFN vs Nadaraya-Watson vs true GP |
| 5 | Vliv velikosti kontextu na MSE |
| 6 | Analýza jedné attention hlavy — NW hypotéza |

## TODO

- Finish the experiments with GP with fixed hyperparameter
  - Does prediction at points far from measurement converge to the prior?
  - Is the variance underestimated?
- Analyze relations in [background/PFN-GP.md](background/PFN-GP.md)
  - Visualize weights of y, attention vs kernel
- Train more advanced PFN for GP with distribution of hyperparameters
  - Check if PFN generalizes across different GPs
  - Analyze if attention learned the correct kernel
