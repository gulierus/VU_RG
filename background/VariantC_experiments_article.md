# Varianta C — výsledky (skutečný PFN pro segmentaci měřený proti oraclu)

*Autoritativní čísla experimentu Varianta C (viz [workflow_A_C_for_PFN.md](workflow_A_C_for_PFN.md),
část II). Generováno notebookem
[experiments/Experiment_C_PFN_segmentation.ipynb](../experiments/Experiment_C_PFN_segmentation.ipynb);
obrázky v `figures/VariantC_pfn_seg/`. Neber čísla z paměti — ber je odsud.*

## Setup

- **Model:** UniverSeg (random init, `pretrained=False`), ~1,18 M parametrů, per-pixel Bernoulli
  hlava (per-pixel BCE). Natrénovaný na HELIOSu (`pfn_seg_train.py`); checkpoint
  `models/pfn_seg_best.pth`.
- **Generativní model úlohy (prior Π):** $f\sim\mathcal{GP}(0,k_\ell)$ na **toru** (periodické
  RBF), $\text{image}=f+\mathcal N(0,\sigma^2)$, $\text{mask}=\mathbf 1[f>\tau]$, kde
  $\tau=\Phi^{-1}(1-\text{fg})$ (outputscale = 1). Trén. rozsahy: $\ell\in[0{,}05,0{,}40]\cdot\text{GRID}$
  (log-uniform), $\sigma\in[0{,}01,0{,}30]$, fg $\in[0{,}15,0{,}85]$, GRID = 64, $S\in\{1,2,4,8,16,32,64\}$
  (randomizované per batch).
- **Oracle (pravý posterior):** pole je na toru → $K$ je cirkulantní → vše se diagonalizuje 2D
  FFT. $\mu_{\text{post}}=\mathcal F^{-1}[\tfrac{\Lambda}{\Lambda+\sigma^2}\mathcal F(\text{image})]$,
  $p_{\text{oracle}}=\Phi(\tfrac{\mu_{\text{post}}-\tau}{s_{\text{post}}})$, $\Lambda=\mathcal F(k_\ell)$.
  Exaktní a levný (ne $O(\text{GRID}^3)$). Prior maska = $\Phi(-\tau)=\text{fg}$ (uniformní).
- **Ověření oraclu:** sampler má Var$(f)\approx1{,}003$, empirická autokovariance sedí na jádro;
  Wiener filtr denoisuje (MSE$(\mu,f)\ll$ MSE(image,$f$)).

**Stav modelu (poctivě):** trénink se ke konci **rozešel** (loss $\to10^8$ v epoše 249–250);
pojistkový `best` checkpoint drží dřívější dobré váhy (loss ~0,062, epocha ~8, pix-acc 0,97;
`epoch` pole v checkpointu je omylem 250). Model je výborný segmenter (Dice ~0,99) a — jak
ukazují měření — i **věrný aproximátor posterioru**. Logity na jistých pixelech saturují
($|\text{logit}|$ až ~300), ale jen tam, kde s oraclem souhlasí.

## Hlavní tvrzení

**Skutečný PFN natrénovaný na explicitním prioru věrně aproximuje pravý Bayesovský posterior —
v průměru i v nejistotě.** Amortizační chyba se koncentruje do tvrdých / OOD úloh (ne do počtu
kontextu); variance s kontextem mizí, bias je malý, ale strukturálně přetrvává; kalibrace je
dobrá s mírnou over-sharpness na hranicích. To je pozitivní 2D validace jádra PFN myšlenky proti
oraclu — pokračování GP2 měřené proti pravdě.

### (1) Fidelita k oraclu a její rozpad

Průměrné $|\hat p_{\text{PFN}}-p_{\text{oracle}}|$ (L1 per pixel), 40 úloh/buňka, náhodné $\sigma$.

| režim (ls_frac) | n=1 | n=4 | n=16 | n=64 |
|---|---|---|---|---|
| **OOD-short** (0,03, pod rozsahem) | 0,0341 | — | — | 0,0316 |
| **Hard** (0,05) | 0,0083 | — | — | 0,0052 |
| **Medium** (0,15) | 0,0068 | — | — | 0,0054 |
| **Easy** (0,40) | 0,0220 | — | — | 0,0082 |
| **OOD-long** (0,60, nad rozsahem) | 0,0351 | — | — | 0,0120 |

**Čtení:** (a) **silná osa = tvrdost/OOD** — největší gap má OOD-short (drsné aliasované pole
pod trén. rozsahem, ~0,034 a s kontextem neklesá); OOD-long startuje vysoko (0,035), ale
s kontextem klesne k trénovaným hodnotám (0,012). Asymetrie extrapolace jako v GP2: delší $\ell$
model dožene, kratší ne. (b) **slabá osa = $n_{\text{supp}}$** — protože query je *hustý*
pozorovaný obrázek pole, i $n_{\text{supp}}=1$ skoro stačí; support jen dodává práh a šum.

### (2) Kolaps k prioru — *negativní* výsledek

Hard režim, fg = 0,5, $\sigma$ = 0,25, 48 úloh/buňka.

| $n_{\text{supp}}$ | 1 | 2 | 8 | 32 | 64 |
|---|---|---|---|---|---|
| $d_{\text{oracle}}=|\hat p-p_{\text{oracle}}|$ | 0,0133 | 0,0111 | 0,0101 | 0,0098 | 0,0096 |
| $d_{\text{prior}}=|\hat p-\text{fg}|$ | 0,473 | 0,474 | 0,474 | 0,474 | 0,474 |

Kolaps k uniformní prior masce **nenastává**: $d_{\text{oracle}}$ je malé a ploché i při
$n_{\text{supp}}=1$ (PFN je pořád mnohem blíž pravdě než uniformní fg). Důvod: hustý query
identifikuje úlohu i z jednoho support páru. Kolaps z Varianty A tedy vyžaduje *řídký*
neinformativní kontext — není to univerzálie amortizace, ale důsledek řídkosti dat.

### (3) bias² + variance rozklad (per pixel, fixní úloha)

Fixní query, $n_{\text{draws}}=24$ tahů support setu.

| úloha | bias² (n=1 → 64) | variance (n=1 → 64) |
|---|---|---|
| Easy | 0,0036 → 0,0035 | 0,0039 → ~0,0000 |
| Hard | 0,0016 → 0,0017 | 0,0009 → ~0,0000 |

**Variance s kontextem mizí** (kvalitativně $O(n^{-1/2})$, empiricky až strměji), zatímco
**bias² se drží na malé kladné hodnotě** (plateau) — přesně Naglerova předpověď (globální
attention porušuje locality → neredukovatelný bias), 2D analog GP2 rozkladu. Bias je malý,
což je konzistentní s dobrou fidelitou. (Nefitujeme degenerovaný joint `bias²+c/n` — past z Ch.3.)

### (4) Kalibrace vs oracle — PFN reprezentuje nejistotu věrně

Sada Hard+Medium+Easy úloh, per-pixel $\hat p_{\text{PFN}}$ vs $p_{\text{oracle}}$.

- Podíl **nejistých** pixelů (oracle $\in[0{,}1,0{,}9]$): **0,051** (tenké hranice).
- Střední entropie predikce: PFN **0,042** vs oracle **0,036** (vše); PFN **0,522** vs oracle
  **0,552** (nejistá množina) — PFN nejistotu na hranicích **zachycuje**.
- ECE vůči oraclu: **0,0049** (vše), **0,0369** (jen nejistá množina).

Rozdělení $\hat p_{\text{PFN}}$ a $p_{\text{oracle}}$ se **téměř překrývají** (histogram: oba
U-tvar s hranou uprostřed). Zbývá jen **mírná over-sharpness** na hranicích (reliability binovaná
podle oraclu je lehce S-tvaru). Saturace logitů je jen na jistých pixelech, kde s oraclem souhlasí,
takže neškodí — model rekalibraci (temperature scaling na logitech, past C.8) prakticky nepotřebuje.

## Obrázky (`figures/VariantC_pfn_seg/`)

- `fig_C_01_qualitative.png` — image / pravá maska / oracle / PFN / |rozdíl| (Easy & Hard).
- `fig_C_02_fidelity.png` — fidelita vs $n_{\text{supp}}$ přes 5 režimů tvrdosti/OOD.
- `fig_C_03_collapse.png` — $d_{\text{oracle}}$ vs $d_{\text{prior}}$ (negativní výsledek).
- `fig_C_04_bias_variance.png` — bias² vs variance a sklon $n^{-1/2}$.
- `fig_C_05_calibration.png` — histogram predikcí + reliability vs oracle.

## Limity a most A ↔ C

- **Nestabilní trénink:** `best` je fakticky early-stopped (epocha ~8). Přesto model věrně
  aproximuje oracle; čistěji natrénovaný (menší LR / silnější clip / stop kolem epochy 10) by byl
  ještě hladší.
- **Torus:** oracle i generátor sdílí periodické (cirkulantní) jádro → oracle exaktní a levný (FFT).
- **A vs C:** A ukázala kolaps na reálném modelu proti *operační* prior masce (heuristika, řídký
  support). C měří proti **explicitní** prior masce a **pravému** posterioru — a ukazuje, že
  *skutečný* PFN na matched prioru je k posterioru mnohem věrnější (a nekolabuje, protože query je
  hustý). Rozdíl A↔C při stejné rodině architektury (UniverSeg) izoluje **efekt PFN trénovací
  procedury** (explicitní prior + PPD/BCE loss).

**Závěr C:** *Skutečný PFN na explicitním prioru věrně aproximuje pravý posterior (fidelita
vysoká, nejistota zachycená); amortizační chyba roste s tvrdostí a OOD hyperparametrů (asymetricky
v $\ell$), ne s počtem hustého kontextu; variance mizí $O(n^{-1/2})$, bias malý ale strukturálně
přetrvává; kalibrace dobrá s mírnou over-sharpness na hranicích. Bias–variance účet amortizované
segmentace měřený proti pravdě.*
