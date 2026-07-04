# WORKFLOW — od experimentu k obrázku v práci

Tento dokument popisuje **jedinou správnou cestu** obrázku od notebooku do textu práce.
Cílem je, aby se obrázky nemíchaly a aby vše šlo znovu vygenerovat.

```
 notebook (experiments/*.ipynb)          text práce (text/Kapitola*.tex)
        │  spustit buňky, vzniknou            ▲  \includegraphics{figures/<exp>/...}
        │  inline PNG výstupy                 │  (přes \graphicspath -> ../figures/)
        ▼                                     │
 background/extract_figures.py  ──────►  figures/<exp>/*.png
        (extrahuje PNG z outputů         (JEDINÝ zdroj obrázků do práce,
         a pojmenuje je)                  v kořeni, tříděné po experimentech)
```

**Rozdělení:** kořen = generování (notebooky, `figures/`, `background/`, `models/`);
`text/` = psaní (LaTeX). Práce se kompiluje z `text/` a obrázky si tahá z `../figures/`
díky `\graphicspath{{../}{./}}` v preambuli — v `.tex` se proto píše prostě `figures/<exp>/...`.

## Zásady

1. **`figures/` v kořeni je jediné místo s obrázky do práce.** Nezakládej druhou `figures` složku, needituj obrázky ručně na dvou místech.
2. **Obrázky se negenerují ručně** — vznikají spuštěním notebooku a poté `extract_figures.py`. Ruční PNG (např. schéma `figures/BarDistribution.png`) jsou výjimka a patří přímo do `figures/`.
3. **Cesty v `.tex` vždy `figures/<experiment>/<jméno>.png`** (relativně z kořene, odkud se kompiluje `VU_RG.tex`).

## Pipeline krok za krokem

```bash
# 1. Spusť experiment (z KOŘENE projektu), buňky vyprodukují inline obrázky
jupyter notebook experiments/Experiment_6_from_GP2.ipynb

# 2. Ulož notebook (obrázky jsou uložené v .ipynb jako base64 PNG outputy)

# 3. Vyextrahuj všechny obrázky do figures/<exp>/
python3 background/extract_figures.py

# 4. Zkompiluj práci Z ADRESÁŘE text/ — cesty už na obrázky ukazují
cd text && latexmk -pdf VU_RG.tex
```

`extract_figures.py` čte outputy buněk z notebooků, dekóduje `image/png` a ukládá do
`figures/<PREFIX_DIR>/`. Přátelská jména dává mapa `KNOWN_NAMES` v tom skriptu
(klíč `(prefix, cell_index, fig_index_v_buňce)`); neznámé obrázky dostanou
`fig_<prefix>_cell<NN>_fig<N>.png`. **Když do notebooku přidáš graf**, přidej řádek
do `KNOWN_NAMES`, ať má stabilní název, na který se dá odkázat z `.tex`.

## Mapa: experiment → notebook → složka → kapitola

| Experiment | Notebook (`experiments/`) | Složka (`figures/`) | Používá kapitola |
|---|---|---|---|
| GP1 big model (6L fixed-HP) | `Experiments_from_GP1_big_model.ipynb` | `GP1_big_model/` (`fig_big_*`) | **Kapitola3a** |
| GP1 random-HP model | `Experiments_from_GP1_random_model.ipynb` | `GP1_random_model/` (`fig_rand_*`) | **Kapitola3b** |
| GP2 Exp. 1 — label vs kernel | `Experiment_1_from_GP2.ipynb` | `GP2_exp1_label_kernel/` (`fig_exp1_*`) + kurátorské `ch2_label_kernel/` (`fig_ch2_*`) | **Kapitola3c** |
| GP2 Exp. 2 — Neumannova řada | `Experiment_2_from_GP2.ipynb` | `GP2_exp2_neumann/` (`fig_exp2_*`) | *(zatím necitováno)* |
| GP2 Exp. 5 — lokalizace | `Experiment_5_from_GP2.ipynb` | `GP2_exp5_localization/` (`fig_exp5_*`) | *(zatím necitováno)* |
| GP2 Exp. 6 — identifikace HP | `Experiment_6_from_GP2.ipynb` | `GP2_exp6_hp_identification/` (`fig_exp6_*`) | *(zatím necitováno)* |
| GP2 Exp. 7 — misspecifikace | `Experiment_7_from_GP2.ipynb` | `GP2_exp7_misspecification/` (`fig_exp7_*`) | *(zatím necitováno)* |

Poznámky:
- `ch2_label_kernel/` je **kurátorská** (ručně vybraná) sada obrázků pro kapitolu o label vs. kernel; její čísla musí sedět s articelem `background/GP2_experiments_article.md`. Syrové výstupy Exp. 1 jsou v `GP2_exp1_label_kernel/`.
- `figures/_archiv_kuratorske/` = staré/záložní varianty, **nepoužívané v aktuálním textu**.
- Obrázek `fig_exp6_q4b_ood_outputscale.png` se generuje zvlášť skriptem
  `background/plot_q4b_figure.py` (z předpočítaných výsledků `run_q4b.py`) do
  `figures/GP2_exp6_hp_identification/`.

## Autoritativní čísla

Číselné výsledky do textu ber z `background/GP1_experiments_article.md` a
`background/GP2_experiments_article.md` (průměry přes seedy/instance), ne z paměti.
Agregované statistiky GP1 počítá `background/collect_stats.py` →
`background/stats_averaged.json`.

## Přidání nového obrázku do práce — checklist

1. Přidej/uprav graf v příslušném notebooku, spusť buňku, **ulož notebook**.
2. Doplň řádek do `KNOWN_NAMES` v `background/extract_figures.py` (stabilní jméno).
3. `python3 background/extract_figures.py` → obrázek je v `figures/<exp>/`.
4. V kapitole (`text/Kapitola*.tex`): `\includegraphics[...]{figures/<exp>/<jméno>.png}` + `\caption` + `\label{fig:...}`.
5. `cd text && latexmk -pdf VU_RG.tex` a zkontroluj, že se obrázek vysázel.
