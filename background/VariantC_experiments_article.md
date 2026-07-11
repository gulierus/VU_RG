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

**Stav modelu:** trénink na HELIOSu **konvergoval** (loss z 0,15 na ~0,02, pix-acc 0,99, cosine LR
do nuly, `best` = finální epocha 250). Grad-clip cestou zachytil dva přechodné výkyvy gradientu
(epochy 76–77, `gnorm`~10⁴) a běh se během jedné epochy srovnal — proto nekolaboval. Model je
výborný segmenter (Dice ~0,99) a — jak ukazují měření — i **věrný aproximátor posterioru**. Logity
na jistých pixelech saturují ($|\text{logit}|$ až ~300), což je u sebejistého BCE segmenteru
normální a projevuje se jen tam, kde s oraclem souhlasí. (Pozn.: lokálně uložený log
`pfn_seg.o235929` je *jiný, starší* běh s LR=1e-3 bez grad-clipu, který se rozešel natrvalo — ne
tento model.)

## Hlavní tvrzení

**Skutečný PFN natrénovaný na explicitním prioru je in-distribution skoro Bayes-optimální a věrně
aproximuje pravý Bayesovský posterior — v průměru i v nejistotě.** Excess risk nad Bayes floor je
na trénovaných režimech jen ~0,004–0,006 BCE. Amortizační chyba se koncentruje do tvrdých / OOD
úloh (ne do počtu kontextu); variance s kontextem mizí, bias je malý, ale strukturálně přetrvává;
kalibrace je dobrá s mírnou over-sharpness na hranicích. To je pozitivní 2D validace jádra PFN
myšlenky proti oraclu — pokračování GP2 měřené proti pravdě.

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

### (5) PFN vs Bayes floor — jak blízko je k optimu

Oracle má **vlastní** neredukovatelnou BCE (Bayes floor = podmíněná entropie $H(y\mid\text{image})$;
masku prahujeme z čistého $f$, ale pozorujeme jen zašuměný obraz). Žádný prediktor pod ni nejde.
BCE PFN i oraclu vůči **pravé** tvrdé masce (S=16, $\sigma$ mix, 40 úloh/režim):

| režim | PFN BCE | Bayes floor (oracle) | excess risk |
|---|---|---|---|
| OOD-short | 0,1164 | 0,0586 | **+0,0577** |
| **Hard** | 0,0463 | 0,0420 | **+0,0044** |
| **Medium** | 0,0315 | 0,0250 | **+0,0065** |
| Easy | 0,0480 | 0,0327 | +0,0153 |
| OOD-long | 0,0403 | 0,0272 | +0,0131 |

**Čtení:** in-distribution (Hard, Medium) je PFN od Bayes floor vzdálený jen ~0,004–0,006 → naučil
se **skoro Bayes-optimální** prediktor. Trénovací plató na loss ~0,019 tedy **není zaseknutí — je
to sezení těsně nad Bayes floor**. Excess risk (= amortizační cena) vyskočí ~9× až **OOD-short**,
přesně kde amortizace selhává. Easy má mírně vyšší excess než Hard, protože hladká pole mají široké
nejisté hranice, kde ta mírná over-sharpness (sekce 4) stojí nejvíc BCE.

## Obrázky (`figures/VariantC_pfn_seg/`)

- `fig_C_01_qualitative.png` — image / pravá maska / oracle / PFN / |rozdíl| (Easy & Hard).
- `fig_C_02_fidelity.png` — fidelita vs $n_{\text{supp}}$ přes 5 režimů tvrdosti/OOD.
- `fig_C_03_collapse.png` — $d_{\text{oracle}}$ vs $d_{\text{prior}}$ (negativní výsledek).
- `fig_C_04_bias_variance.png` — bias² vs variance a sklon $n^{-1/2}$.
- `fig_C_05_calibration.png` — histogram predikcí + reliability vs oracle.
- `fig_C_06_bayes_floor.png` — PFN BCE vs Bayes floor + excess risk per režim.

## Limity a most A ↔ C

- **Trénink konvergoval čistě** (loss ~0,019, pix-acc 0,991, epocha 250), takže výsledky
  neomezuje trénovací nestabilita; saturace logitů je normální projev sebejistého segmenteru.
- **Torus:** oracle i generátor sdílí periodické (cirkulantní) jádro → oracle exaktní a levný (FFT).
- **A vs C:** A ukázala kolaps na reálném modelu proti *operační* prior masce (heuristika, řídký
  support). C měří proti **explicitní** prior masce a **pravému** posterioru — a ukazuje, že
  *skutečný* PFN na matched prioru je k posterioru mnohem věrnější (a nekolabuje, protože query je
  hustý). Rozdíl A↔C při stejné rodině architektury (UniverSeg) izoluje **efekt PFN trénovací
  procedury** (explicitní prior + PPD/BCE loss).

**Závěr C:** *Skutečný PFN na explicitním prioru je in-distribution skoro Bayes-optimální (excess
risk nad Bayes floor ~0,004–0,006) a věrně aproximuje pravý posterior (fidelita vysoká, nejistota
zachycená); amortizační chyba roste s tvrdostí a OOD hyperparametrů (asymetricky
v $\ell$), ne s počtem hustého kontextu; variance mizí $O(n^{-1/2})$, bias malý ale strukturálně
přetrvává; kalibrace dobrá s mírnou over-sharpness na hranicích. Bias–variance účet amortizované
segmentace měřený proti pravdě.*
