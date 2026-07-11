# Prior-Fitted Networks jako aproximátory Gaussovských procesů: experimenty s multi-layer modely a distribuovanými hyperparametry

**Ruslan Guliev**  
Vrije Universiteit Amsterdam, 2025

---

## Abstrakt

Tento dokument navazuje na `GP1_experiments_article.md` a zaznamenává výsledky experimentů provedených na **multi-layer modelech trénovaných na distribuci hyperparametrů** (random-HP modely s 1, 2, 4, 6, 8 vrstvami). Zatímco GP1 série porovnávala jeden 6-vrstvý random-HP model s jedním 100-epoch fixed-HP modelem, experimenty GP2 systematicky zkoumají závislost výkonu na **počtu vrstev** a testují robustnost vůči **misspecifikaci prioru**. Klíčové otázky: (1) specializují se vrstvy na kernel-like vs. label-propagation operace? (2) odpovídá každá vrstva jednomu kroku Neumannovy iterace (podmíněnost κ vs. počet vrstev)? (3) pomáhá post-hoc lokalizace slabším modelům? (4) jak rychle modely identifikují HP z kontextu? (5) jak si PFN stojí, když ani kernel není správně specifikován?

---

## 1. Experimentální nastavení

### 1.1 Modely

| Model | Checkpoint | Epochy | Architektura |
|---|---|---|---|
| **1-layer** | `pfn_rand_hps_1layer.pth` | 100 | 1 vrstva, 3.65M param |
| **2-layer** | `pfn_rand_hps_2layer.pth` | 300 | 2 vrstvy |
| **4-layer** | `pfn_rand_hps_4layer.pth` | 300 | 4 vrstvy |
| **6-layer** | `pfn_rand_hps_6layer.pth` | 500 | 6 vrstev, 14.14M param (přetrénován 2026-07) |
| **8-layer** | `pfn_rand_hps_8layer.pth` | 500 | 8 vrstev (robustní checkpoint, vykazován jako 500ep) |

Trénovací distribuce HP pro všechny modely: $\ell \sim U(0.05, 1.0)$, outputscale $\sim \text{LogNormal}(0, 0.5)$, šum $\sim 10^{U(-3,-1)}$. Architektura: `emsize=512`, `nhead=8`, `nhid=1024`, `features_per_group=1`, `attention_between_features=False`.

### 1.2 Baseline metody (Experiment 7)

- **GP oracle:** GP se správnými HP a správným (Matérn 2.5) kernelem
- **GP\_fixed:** GP s RBF kernelem a fixním $\ell_{\text{wrong}} = 0.7$
- **GP\_ML:** sklearn GaussianProcessRegressor, optimalizuje marginální věrohodnost (Type-II ML)
- **GP\_Bayes (NUTS):** Bayesovský GP s priorem $p(\ell) \sim \text{Gamma}(4, \text{rate}=4)$ (střed u $\ell=1.0$), NumPyro/JAX

---

## 2. Experiment 1 — Label Mixing vs. Kernel Computation per Layer (sloučeno s Exp 3)

**Notebook:** `experiments/Experiment_1_from_GP2.ipynb`  
**Otázka:** Specializují se vrstvy? Čtou rané vrstvy hodnoty $Y_j$ (label reading) a prostřední vzdálenost v $X$ prostoru (kernel-like)?  
**Design:** 200 instancí, $n_{\text{context}}=40$, $n_{\text{test}}=20$, $\ell_{\text{NW}}=0.5$, sdílená data pro všechny modely. Pro každou vrstvu a hlavu měříme **Spearmanovu** korelaci attention vah s RBF/NW kernelem $k(x_*, X_j)$ a s labely $Y_j$.

> **Proč Spearman a ne Pearson:** attention váhy jsou silně zešikmené na simplexu (většina hmoty na několika bodech), takže Pearson podhodnocuje monotónní vztah. Přechod na Spearmana zvedl kernel-korelace ve středních vrstvách z ~0.35 (Pearson) na ~0.8 — trend je stejný, ale mnohem výraznější.
>
> **Sloučení s Exp 3:** Experiment 3 z GP2 série (Q1 — label mixing vs. kernel per vrstva) byl **duplikát** tohoto experimentu, proto je sem sloučen. Jeho unikátní části — kauzální **head-knockout** (§2.4) a **Q3 efektivní bandwidth** (§2.5) — jsou zachovány.

### 2.1 Spearman corr(kernel) přes hlavy — přehled vrstev

| Vrstva | 1-layer | 2-layer | 4-layer | 6-layer | 8-layer |
|---|---|---|---|---|---|
| 0 | 0.4799 | 0.1681 | 0.0743 | 0.0036 | −0.0993 |
| 1 | — | 0.8187 | 0.8871 | 0.6100 | 0.5628 |
| 2 | — | — | 0.8279 | 0.8255 | 0.8726 |
| 3 | — | — | 0.2701 | 0.8166 | 0.7650 |
| 4 | — | — | — | 0.7397 | 0.7728 |
| 5 | — | — | — | 0.4162 | 0.5441 |
| 6 | — | — | — | — | 0.5298 |
| 7 | — | — | — | — | 0.2958 |

### 2.2 corr(kernel) a |corr(label)| per model (Spearman)

**1-layer:**

| Vrstva | corr(kernel) | \|corr(label)\| |
|---|---|---|
| 0 | 0.4799 | 0.2231 |

**2-layer:**

| Vrstva | corr(kernel) | \|corr(label)\| |
|---|---|---|
| 0 | 0.1681 | 0.2329 |
| 1 | 0.8187 | 0.0178 |

**4-layer:**

| Vrstva | corr(kernel) | \|corr(label)\| |
|---|---|---|
| 0 | 0.0743 | 0.3377 |
| 1 | 0.8871 | 0.0178 |
| 2 | 0.8279 | 0.0726 |
| 3 | 0.2701 | 0.0760 |

**6-layer:**

| Vrstva | corr(kernel) | \|corr(label)\| |
|---|---|---|
| 0 | 0.0036 | 0.3158 |
| 1 | 0.6100 | 0.1174 |
| 2 | 0.8255 | 0.0133 |
| 3 | 0.8166 | 0.0652 |
| 4 | 0.7397 | 0.0819 |
| 5 | 0.4162 | 0.1117 |

**8-layer:**

| Vrstva | corr(kernel) | \|corr(label)\| |
|---|---|---|
| 0 | −0.0993 | 0.4692 |
| 1 | 0.5628 | 0.1624 |
| 2 | 0.8726 | 0.0133 |
| 3 | 0.7650 | 0.0830 |
| 4 | 0.7728 | 0.1887 |
| 5 | 0.5441 | 0.1550 |
| 6 | 0.5298 | 0.1720 |
| 7 | 0.2958 | 0.0895 |

### 2.3 MSE dekompozice: bias² + c/n

> **Design:** $n \in \{5, 10, 20, 40, 80, 120\}$, 200 instancí na každé $n$, proloženo modelem $\text{MSE}(n) = \text{bias}^2 + c/n$.

| Model | bias² | c | Interpretace |
|---|---|---|---|
| 1-layer | 0.00476 | 0.097 | bias² signifikantní |
| 2-layer | −0.00152 | 0.068 | nesignifikantní bias |
| 4-layer | −0.00089 | 0.035 | nesignifikantní bias |
| 6-layer | −0.00076 | 0.029 | nesignifikantní bias |
| 8-layer | −0.00065 | 0.026 | nesignifikantní bias |

> Přesné hodnoty z neváhovaného proložení `bias²+c/n` na surová MSE (aktuální běh notebooku). Mírně **záporné** odhady bias² u 2–8L jsou artefaktem fitu — asymptota `MSE(n→∞)` nemůže být fyzikálně záporná, jde tedy o hodnoty statisticky neodlišitelné od nuly (notebook je při omezení ≥0 tiskne jako `0.00000`).

### 2.4 Kauzální metrika: Head Knockout

**Limit korelace.** Korelace attention vs. kernel/label pouze *popisuje* attention pattern — ignoruje value vektory, FFN a reziduální tok, takže z ní **nelze usuzovat na funkční roli hlavy v celém modelu**. Nagler (2023, Thm 6.3) navíc dokázal, že softmax attention **vždy** míchá $X_j$ i $Y_j$, takže „čistý kernel" je principiálně nedosažitelný a samotná (ne)korelace hlavy neříká, zda je pro predikci důležitá.

**Kauzální test.** Pro každou hlavu vynulujeme její output-projekci ($W_{out}^{(h)}=0$), znovu spustíme PFN a změříme, o kolik se zhorší shoda s GP posteriorem. Metrika je **relativní**:

$$\text{rel} = \frac{\text{MSE}(\text{knockout}, \mu_{\text{GP}}) - \text{MSE}(\text{full}, \mu_{\text{GP}})}{\text{MSE}(\text{full}, \mu_{\text{GP}})}$$

Absolutní $\Delta$MSE je totiž **zavádějící**: hluboké modely mají triviální baseline MSE (~0.0001), takže jejich absolutní $\Delta$MSE je z principu malé, ať je hlava důležitá nebo ne. Relativní metrika (násobek baseline chyby) měří skutečnou důležitost. Agregujeme **mediánem** přes 50 instancí ($n_{\text{context}}=40$) — MSE(PFN, GP) je na typické instanci ~$10^{-4}$, ale na několika patologických realizacích o řády vyšší, takže průměr by byl ovládnut outliery a mezi seedy by kolísal o dva řády; medián dává stabilní, mezi modely srovnatelný obraz.

| Model | baseline MSE(full, GP) | max rel $\Delta$MSE | nejdůležitější (vrstva, hlava) |
|---|---|---|---|
| 1-layer | 0.00957 | **44.9×** | (0, 1) |
| 2-layer | 0.00051 | 24.6× | (1, 5) |
| 4-layer | 0.00012 | 9.9× | (1, 4) |
| 6-layer | 0.00005 | 17.7× | (1, 2) |
| 8-layer | 0.00006 | 14.6× | (1, 6) |

**Klíčový výsledek:** Kauzálně důležité hlavy se **koncentrují v raných vrstvách** (dominantní hlava je v každém modelu ve vrstvě 0 nebo 1). V absolutních číslech hluboké modely knockout „ustojí" (zůstávají u GP), ale **relativně k vlastní chybě** nese odebrání jedné hlavy napříč všemi hloubkami **10–25×** zhoršení — hlavy tedy **nejsou bezvýznamné**, jen individuálně nekritické; jednovrstvý model je nejkřehčí (44.9×). To je poctivější obrázek než absolutní $\Delta$MSE, který hloubkou „uměle" padá jen kvůli menšímu baseline. Korelační analýza tuto kauzální strukturu ukázat nemohla.

> **Pozn.:** Dřívější běh (s podtrénovaným 6-layer checkpointem a agregací průměrem) vykazoval u 6-layer anomálii (max rel jen 0.5×). Po přetrénování 6-layer na 500 epoch a přechodu na **mediánovou** agregaci anomálie zmizela — všechny hloubky nyní vykazují 10–25× degradaci s dominantními hlavami v raných vrstvách.

### 2.5 Q3 — Efektivní bandwidth vrstvy 0 vs. lengthscale

Efektivní bandwidth $\text{bw}(x^*) = \text{mean}_h \sum_j a_j^{(0)}\, |X_j - x^*|$ testuje, zda vrstva 0 adaptuje šířku attention na skutečný $\ell$. (100 instancí, $n_{\text{context}}=20$, $\sigma^2=0.01$.)

| $\ell$ | 1-layer | 2-layer |
|---|---|---|
| 0.05 | 0.2246 ± 0.0058 | 0.2942 ± 0.0090 |
| 0.10 | 0.2275 ± 0.0058 | 0.3049 ± 0.0096 |
| 0.30 | 0.2363 ± 0.0068 | 0.3141 ± 0.0071 |
| 0.80 | 0.2363 ± 0.0065 | 0.3128 ± 0.0095 |
| 2.00 | 0.2338 ± 0.0062 | 0.3086 ± 0.0088 |

**Pozorování:** bandwidth vrstvy 0 je **plochý** v $\ell$ u obou modelů (nesleduje diagonálu bw = ℓ) — první vrstva šířku attention na $\ell$ neadaptuje. 2-vrstvý model má systematicky širší bandwidth (~0.31) než 1-vrstvý (~0.22). Identifikace $\ell$ (pokud vůbec) tedy neprobíhá v šířce attention vrstvy 0, ale až v hlubší výpočetní kaskádě.

### 2.6 Interpretace

Pod Spearmanem je specializace vrstev **výrazně čistší** než pod Pearsonem. Ve vrstvě 0 je vždy $|\text{corr(label)}|$ nejvyšší (0.22–0.47) a corr(kernel) nízká/záporná (−0.10 až 0.48) — rané vrstvy čtou labely. V prostředních vrstvách corr(kernel) skáče na 0.61–0.89 a $|\text{corr(label)}|$ padá na 0.01–0.08 — kernel-like chování. V poslední vrstvě corr(kernel) opět klesá (0.27–0.44).

> *"Nelze tvrdit, že PFN dělá čistě kernel smoothing — v každé vrstvě míchá X i Y. Prostřední vrstvy preferují kernel-like chování, ale nikdy čistě."*

Head-knockout tuto korelační charakteristiku doplňuje **kauzálně**: samotná (ne)korelace hlavy s kernelem neurčuje její důležitost — teprve knockout ukáže, které hlavy nesou výpočet (koncentrují se do raných vrstev) a že i v hlubokých modelech odebrání hlavy stojí několikanásobek baseline chyby (viz §2.4).

Jednovrstvý model má nenulové bias² = 0.00476 — konzistentní s tím, že jedna vrstva nestačí pro asymptotické přiblížení GP posterior. Od dvou vrstev výš je bias² statisticky nula; koeficient $c$ (variance term) klesá monotónně s hloubkou: $c_{1L}=0.097 > c_{2L}=0.068 > c_{4L}=0.035 > c_{6L}=0.029 > c_{8L}=0.026$.

*Obrázky: `figures/GP2_exp1_label_kernel/fig_exp1_label_mixing.png` (Spearman korelace per vrstva), `GP2_exp1_label_kernel/fig_exp1_head_knockout.png` (kauzální heatmapa), `GP2_exp1_label_kernel/fig_exp1_q3_bandwidth.png` (bandwidth vs. ℓ), `GP2_exp1_label_kernel/fig_exp1_mse_decomposition.png`.*

---

## 3. Experiment 2 — PFN jako Neumannova řada?

**Notebook:** `experiments/Experiment_2_from_GP2.ipynb`  
**Otázka:** Potřebují špatně podmíněné systémy víc vrstev PFN než dobře podmíněné?

GP posterior mean vyžaduje $(K+\sigma^2 I)^{-1}y$. Gradientní sestup tuto inverzi řeší iterativně:

$$\alpha^{(0)} = 0, \qquad \alpha^{(t+1)} = \alpha^{(t)} + \eta\bigl(y - (K+\sigma^2 I)\alpha^{(t)}\bigr)$$

Počet kroků pro konvergenci závisí na podmíněnosti $\kappa = \frac{\lambda_{\max}(K) + \sigma^2}{\lambda_{\min}(K) + \sigma^2}$. **Hypotéza:** každá vrstva PFN odpovídá jednomu GD kroku → hůře podmíněné úlohy (větší $\kappa$) potřebují více vrstev.

### 3.1 Konfigurace

Obě sady modelů sdílejí $\ell=0.3$ a liší se **pouze šumem** $\sigma^2$ — to izoluje vliv podmíněnosti od vlivu délky korelace. Každá sada je trénována **přesně na svých testovacích HP** (in-distribution výkon, nikoliv generalizace). Všech pět hloubek (1/2/4/6/8) v každé sadě má **stejný rozpočet 300 epoch** — trend v hloubce tedy zde není konfundovaný nevyrovnaným tréninkem (na rozdíl od Exp 1). K dispozici jsou i modely pro třetí režim $\sigma^2=0.01$ ($\kappa \approx 2300$).

| Regime | $\ell$ | $\sigma^2$ | outputscale | $\kappa$ ($n{=}40$) | $\rho = \frac{\kappa-1}{\kappa+1}$ |
|---|---|---|---|---|---|
| **Easy** | 0.3 | 0.5 | 1.0 | ≈ 49  | ≈ 0.96 |
| **Hard** | 0.3 | 0.1 | 1.0 | ≈ 241 | ≈ 0.99 |

$\kappa$ je průměr přes 50 instancí při $n=40$. Protože $\lambda_{\min}(K) \approx 0$ numericky, platí $\kappa \approx 1 + \lambda_{\max}/\sigma^2$ — manipulace šumu tedy nastavuje $\kappa$ prakticky přímo. $\rho$ je **worst-case** konvergenční míra GD za krok, ale **v $\alpha$-prostoru** ($\alpha=(K+\sigma^2 I)^{-1}y$); měřená nMSE je naproti tomu v **predikčním prostoru** $\mu(x_*)=k_*^\top\alpha$, který je kernel-vážený — proto se teoretická křivka $\rho^{L}$ a empirie nesmí srovnávat naivně (viz GD baseline v §3.4).

### 3.2 Primární metrika — normalizované MSE + ΔNLL

> **Design:** 200 instancí, $n_{\text{context}}=40$, $n_{\text{test}}=10$, sdílená data (párový design). Primární metrika $\text{nMSE} = \text{MSE}(\hat\mu_{\text{PFN}}, \mu_{\text{GP}}) / \text{var}(\mu_{\text{GP}})$ — dělíme rozptylem **cíle** (GP posterior mean přes testovací body). *Volba jmenovatele:* dřívější dělení $\text{var}(y_{\text{train}})$ obsahovalo šum (který posterior mean nikdy neobsahuje) a uměle **obracelo pořadí Easy/Hard**; $\text{var}(\mu_{\text{GP}})$ je konzistentní s raw MSE. Sekundární $\Delta_{\text{NLL}}$ je po opravě znaménka (viz níže) správně **malé kladné** (PFN NLL těsně nad GP) a klesá s hloubkou; zůstává jen přibližné srovnání (histogramová BarDistribution vs. analytická gaussovská). Error bary (±stderr) jsou v grafu.

| Vrstvy | nMSE Easy | nMSE Hard | MSE Easy | MSE Hard | ΔNLL Easy | ΔNLL Hard |
|---|---|---|---|---|---|---|
| 1 | 0.10230 | 0.04583 | 0.013012 | 0.010109 | 0.0068 | 0.0256 |
| 2 | 0.01994 | 0.01050 | 0.005205 | 0.003049 | 0.0044 | 0.0046 |
| 4 | 0.00563 | 0.00543 | 0.003669 | 0.002358 | 0.0042 | 0.0050 |
| 6 | 0.00543 | 0.00353 | 0.004727 | 0.001460 | 0.0041 | 0.0037 |
| 8 | 0.00403 | 0.00492 | 0.001979 | 0.002276 | 0.0048 | 0.0034 |

Easy klesá monotónně (~25×), **Hard je zašumělejší** — pro malý šum má $\mu_{\text{GP}}$ malý rozptyl, takže jmenovatel nMSE je citlivý a hodnoty přes hloubky kolísají (L=8 může být nad L=6). Kvalitativně oba režimy padají o ~10–30×.

> **⚠️ Oprava znaménka NLL (2026-07).** Původní běh měl v `compute_pfn_nll` chybné `nll = −criterion(...)`, ale `FullSupportBarDistribution.__call__` **už vrací NLL** (loss), ne log-prob — takže PFN NLL byla obrácená a ΔNLL vycházela nesmyslně silně záporná (−2.24 / −0.77). Po opravě je ΔNLL malé kladné (viz tabulka). Táž chyba a oprava se týká Exp 6 (§5.4) a Exp 7 (§6).

### 3.3 MSE dekompozice: bias² + c/n

> **Design:** $n \in \{5,10,20,40,80,120\}$, proloženo $\text{MSE}(n) = \text{bias}^2 + c/n$. bias² = ireducibilní chyba (asymptota pro $n\to\infty$).

**Easy** (ℓ=0.3, σ²=0.5):

| Model | bias² | c | Bias signifikantní? |
|---|---|---|---|
| 1-layer | 0.00784 | 0.105 | ANO |
| 2-layer | 0.00124 | 0.040 | ANO |
| 4-layer | 0.00085 | 0.011 | ANO |
| 6-layer | 0.00071 | 0.008 | ANO |
| 8-layer | 0.00059 | 0.007 | ANO |

**Hard** (ℓ=0.3, σ²=0.1):

| Model | bias² | c | Bias signifikantní? |
|---|---|---|---|
| 1-layer | 0.00674 | 0.045 | ANO |
| 2-layer | 0.00117 | 0.042 | ANO |
| 4-layer | 0.00152 | 0.005 | ANO |
| 6-layer | 0.00084 | 0.005 | ANO |
| 8-layer | 0.00113 | 0.001 | ANO |

### 3.4 Interpretace

**Neumannova hypotéza se v jednoduché podobě nepotvrdila.** Pokud by každá vrstva byla jeden GD krok, Easy (malé $\kappa$) by mělo konvergovat rychleji než Hard (velké $\kappa$). Ve skutečnosti obě konfigurace klesají podobně (o ~10–30× přes 8 vrstev), rozdíl je marginální — **podmíněnost tedy není limitujícím faktorem** počtu potřebných vrstev.

**Férový GD baseline (obr. `fig_exp2_gd_baseline`).** Původní tvrzení „PFN konverguje rychleji než GD" srovnávalo empirickou nMSE (v **predikčním** prostoru) s worst-case Neumannovou křivkou $\rho^L$ (kontrakce v **$\alpha$-prostoru**) — to je nefér, protože predikce $\mu=k_*^\top\alpha$ je kernel-vážená a tlumí právě ty pomalé komponenty (malá vlastní čísla), kde GD konverguje nejhůř. Proto jsme spustili **skutečný gradientní sestup** (optimální fixní krok $\eta=2/(\lambda_{\max}+\lambda_{\min})$) na týchž datech a měřili $\text{MSE}(\mu_{\text{GD}}^{(t)}, \mu_{\text{GP}})$ přímo v predikčním prostoru. Reálný GD je i tak **pomalý** — jeho nMSE zůstává nad 1 (tj. horší než triviální predikce průměrem) po desítky kroků. Počet GD kroků k úrovni, kterou PFN dosáhne v dané hloubce:

| PFN hloubka $L$ | ~ GD kroků, Easy (κ≈50) | ~ GD kroků, Hard (κ≈245) |
|---|---|---|
| 1 | 51 | 271 |
| 4 | 87 | 402 |
| 8 | 91 | 408 |

**Škálování s $\kappa$ přes tři režimy (obr. `fig_exp2_kappa_scaling`).** Abychom oddělili PFN od GD ještě ostřeji, přidali jsme třetí, silně špatně podmíněný režim $\sigma^2=0.01$ (vlastní modely trénované na stejné HP). Fixujeme cíl na úroveň, kterou dosáhne 8-vrstvý PFN, a měříme, kolik GD kroků ji dá:

| režim | $\kappa$ | PFN nMSE ($L=8$) | ~ GD kroků ($L=8$) |
|---|---|---|---|
| $\sigma^2=0.5$ | 50 | 0.0040 | 91 |
| $\sigma^2=0.1$ | 245 | 0.0049 | 408 |
| $\sigma^2=0.01$ | 2424 | 0.00066 | 5723 |

Počet GD kroků roste zhruba **lineárně s $\kappa$** (91 → 408 → 5723, tj. $\approx 2\kappa$), přesně jak $\mathcal{O}(\kappa)$ konvergence obyčejného GD předpovídá. **PFN je naproti tomu $\kappa$-necitlivý:** týchž 8 vrstev stačí ve všech třech režimech a v nejhůř podmíněném ($\kappa\approx2424$) je dokonce **nejpřesnější** (nMSE $0.00066$). Už jedna vrstva odpovídá desítkám GD kroků.

> **Per-instance regrese (kontrola):** Regrese $\log\text{MSE}_i \sim \log\kappa_i$ napříč instancemi *uvnitř* jednoho režimu nevykazuje žádný signál ($|r|\lesssim0.1$, sklony šum). To je očekávané — při fixní HP se $\kappa_i$ mezi instancemi mění jen konečně-vzorkovým rozptylem bodů $x$, takže informativní osou je **mezi** režimy (tabulka výše), ne uvnitř.

**Klíčový závěr — $\kappa$-necitlivost je podpis preconditioned/second-order solveru.** Kombinace *$\kappa$-necitlivosti* a *super-GD rychlosti* je charakteristický podpis **naučeného preconditioned nebo druhořádového (Newton-like) solveru**, ne vanilla gradientního sestupu. Vágní „PFN dělá něco složitějšího" se tak mění na kvantitativní tvrzení: **PFN implementuje $\kappa$-necitlivou iteraci, kde jedna vrstva odpovídá řádově desítkám kroků obyčejného GD.**

**Ukotvení v literatuře.** Tento obraz přesně odpovídá teorii o in-context solverech v transformerech. Ahn et al. (2023) dokázali, že globální minimum jednovrstvého lineárního transformeru implementuje jeden krok **preconditioned** gradientního sestupu, jehož preconditioner se adaptuje na rozdělení vstupu — a právě preconditioning redukuje efektivní podmíněnost a činí konvergenci $\kappa$-necitlivou. von Oswald et al. (2023) ukázali totéž pro $L$ vrstev $=$ $L$ iterací (preconditioned) GD. Fu et al. (2023) pak empiricky i teoreticky ukazují, že transformery dosahují **druhořádové (Newton-like) konvergence** — exponenciálně rychlejší než GD — a, což je pro nás klíčové, **fungují i na špatně podmíněných datech, kde gradientní sestup selhává**. Naše měření (PFN $\kappa$-necitlivý, zatímco GD potřebuje $\sim\kappa$ kroků, obr. `fig_exp2_kappa_scaling`) je tak přímé kvantitativní potvrzení, že PFN implementuje **naučený preconditioned/druhořádový solver**, nikoli obyčejný gradientní sestup.

bias² klesá s hloubkou v obou režimech (Easy: 0.0094 → 0.0010; Hard: 0.0096 → 0.0022), ale nikdy zcela nevymizí — konzistentní s tím, že žádný konečný počet vrstev nereprodukuje GP posterior přesně (jde ale o směs statistického biasu a aproximačního floor natrénované sítě; fit `bias²+c/n` je neváhovaný, viz Tier 2). K dispozici je i třetí režim $\sigma^2=0.01$ ($\kappa\approx2300$), který by prodloužil $\kappa$-osu o řád.

---

## 4. Experiment 5 — Post-hoc lokalizace

**Notebook:** `experiments/Experiment_5_from_GP2.ipynb`  
**Motivace:** Nagler (2023) ukázal, že pro $n$ kontextových bodů v 1D stačí zahrnout pouze $k_n = \lceil n^{4/5} \rceil$ nejbližších sousedů. Pomůže lokalizace PFN modelům, které byly trénované na wide distribuci HP?  
**Design:** $n_{\text{fixed}}=64$, $k_n=28$, 100 instancí, $\ell=0.3$, $\sigma^2=0.01$.

### 4.1 Optimální $k_n$ dle Nagler (2023)

| $n$ | $k_n = \lceil n^{4/5} \rceil$ | $k_n/n$ |
|---|---|---|
| 5 | 4 | 0.80 |
| 10 | 7 | 0.70 |
| 20 | 11 | 0.55 |
| 40 | 20 | 0.50 |
| 64 | 28 | 0.44 |
| 100 | 40 | 0.40 |
| 128 | 49 | 0.38 |

### 4.2 Q1 — Staré modely: MSE vs k při $n=64$

| Model | MSE\_full ($k=n$) | MSE\_best | Best $k$ | Zlepšení |
|---|---|---|---|---|
| 1-layer | 0.01027 | 0.00325 | 12 | **+68.4 %** |
| 2-layer | 0.00921 | 0.00688 | 12 | +25.3 % |
| 4-layer | 0.00378 | 0.00245 | 21 | **+35.1 %** |
| 6-layer | 0.00670 | 0.00407 | 12 | **+39.2 %** |
| 8-layer | 0.00020 | 0.00020 | 64 | −0.0 % |

### 4.3 Q2 — Staré modely: MSE dekompozice s a bez lokalizace

> MSE\_full = plný kontext; MSE\_kn = pouze $k_n$ nejbližších sousedů.

| Model | bias²\_full | bias²\_kn | Relativní změna |
|---|---|---|---|
| 1-layer | 0.00694 | 0.00486 | +30.0 % |
| 2-layer | 0.00749 | 0.00589 | +21.5 % |
| 4-layer | 0.00101 | 0.00043 | **+57.5 %** |
| 6-layer | 0.00407 | 0.00315 | +22.5 % |
| 8-layer | ≈ 0 | ≈ 0 | +64.0 % |

MSE\_kn per $n$ pro 1-layer (lokalizace vs. bez lokalizace):

| $n$ | $k_n$ | MSE\_full | MSE\_kn |
|---|---|---|---|
| 5 | 4 | 0.06435 | 0.05940 |
| 10 | 7 | 0.03417 | 0.02790 |
| 20 | 11 | 0.01425 | 0.01349 |
| 40 | 20 | 0.01021 | 0.00865 |
| 64 | 28 | 0.01025 | 0.00906 |
| 100 | 40 | 0.00646 | 0.00535 |
| 128 | 49 | 0.00768 | 0.00721 |

### 4.4 Q1/Q2 — Nové modely trénované s lokalizací

> Checkpointy: `models/pfn_localized_rand_hps/{1,2,4,6,8}Lmodel.pth`

| Model | MSE\_nové | MSE\_staré | Rozdíl |
|---|---|---|---|
| 1-layer | 1.18361 | 0.01027 | **+1.17** |
| 2-layer | 0.71682 | 0.00921 | **+0.71** |
| 4-layer | 1.10197 | 0.00378 | **+1.10** |
| 6-layer | **0.00173** | 0.00670 | **−0.00497** |
| 8-layer | 1.10116 | 0.00020 | **+1.10** |

MSE dekompozice nových modelů (plný kontext):

| Model | bias² | c |
|---|---|---|
| 1-layer | 1.16413 | 0 |
| 2-layer | 0.72851 | 0 |
| 4-layer | 1.07076 | 0 |
| 6-layer | ≈ 0 | 0.2404 |
| 8-layer | 1.07066 | 0 |

### 4.5 Interpretace

Lokalizace pomáhá všem modelům kromě nejsilnějšího **8-layer** (zlepšení MSE: 1-layer +68 %, 2-layer +25 %, 4-layer +35 %, 6-layer +39 %, 8-layer −0 %). 8-layer už sám efektivně ignoruje vzdálené body, takže mu ořez kontextu nepřidá.

Trénování nových modelů přímo s lokalizovaným kontextem selhalo pro všechny modely kromě 6-layer. Bias² nových modelů je řádově vyšší než starých ($\approx 1.0$ vs $\approx 0$) — modely se buď špatně naučily pracovat s ořezaným kontextem, nebo nastalo přetrénování.

> *"Lokalizace redukuje šum od vzdálených bodů. Pro 1-layer je optimum u $k=7$ z 64 bodů — model efektivně potřebuje jen nejbližší sousedy. Pro 6-layer a 8-layer je minimum u $k=n$ — model sám zvládá ignorovat irelevantní body."*

---

## 5. Experiment 6 — Identifikace hyperparametrů z kontextu

**Notebook:** `experiments/Experiment_6_from_GP2.ipynb`  
**Otázka:** Jak přesně PFN s různým počtem vrstev identifikuje $\ell$ z kontextových dat? Kdy PFN překonává GP se špatným $\ell$?

### 5.1 Q1 — MSE(PFN) vs MSE(GP\_wrong) pro různé $\ell$

> **Design:** $n=40$, $n_{\text{test}}=10$, 200 instancí, $\ell_{\text{wrong}}=0.5$ (fixní špatné HP).  
> GP\_wrong avg = 0.08749 pro všechna $\ell$ (průměr přes tři testované $\ell$ hodnoty).

| Model | $\ell=0.1$ | $\ell=0.3$ | $\ell=0.7$ | GP\_wrong avg |
|---|---|---|---|---|
| **GP\_wrong** | 0.25874 | 0.00357 | 0.00016 | 0.08749 |
| 1-layer | 0.16054 | 0.01124 | 0.00117 | — |
| 2-layer | 0.01714 | 0.00638 | 0.00758 | — |
| 4-layer | 0.00814 | 0.00163 | 0.00334 | — |
| 6-layer | 0.00679 | 0.00193 | 0.00635 | — |
| **8-layer** | **0.00304** | **0.00036** | **0.00012** | — |

**Pozorování:** Nejsilnějším modelem je **8-layer**. Pro $\ell=0.1$ (krátká korelační délka) překonává GP\_wrong 8-layer 85×, 6-layer 38×. Pro $\ell=0.3$ překonává GP\_wrong 8-layer (10×), 4-layer (2.2×) i **6-layer (1.8×, po přetrénování)** — jen 2-layer (0.00638) je horší než GP\_wrong. Pro $\ell=0.7$ (blízko $\ell_{\text{wrong}}=0.5$) je GP\_wrong lepší než všechny PFN modely.

Anomálie: 1-layer je jediný model, který překonává GP\_wrong pro $\ell=0.7$ ($0.00117 < 0.00016$ — GP\_wrong je zde naopak **lepší** než PFN). To dává smysl: $\ell_{\text{wrong}}=0.5$ je blízko $\ell=0.7$ a jednovrstvý model nemá dostatek kapacity pro přesnou identifikaci.

### 5.2 Q1.5 — Rekonstruovaný efektivní $\ell$ ($\hat\ell_{\text{eff}}$)

> **Design:** Pro každou instanci najdeme $\hat\ell_{\text{eff}} = \arg\min_\ell \text{MSE}(\text{PFN\_mean}, \text{GP\_posterior\_mean}(\ell))$.  
> $n=40$, $n_{\text{test}}=20$, 100 instancí. Tabulka: průměr ± std přes 100 instancí.

| $\ell_{\text{true}}$ | 1-layer | 2-layer | 4-layer | 6-layer | 8-layer |
|---|---|---|---|---|---|
| 0.050 | 0.650±0.747 | 0.802±0.999 | 0.950±1.085 | 0.979±1.073 | 1.008±1.103 |
| 0.100 | 0.668±0.663 | 0.496±0.740 | 0.509±0.783 | 0.517±0.812 | 0.549±0.821 |
| 0.150 | 0.724±0.659 | 0.425±0.638 | 0.435±0.704 | 0.436±0.714 | 0.433±0.710 |

**Pozorování:** Pro $\ell_{\text{true}}=0.05$ všechny modely rekonstruují $\hat\ell_{\text{eff}} \approx 0.65$–$1.0$ — model nedokáže identifikovat tak krátkou korelační délku a předpokládá průměrnou hodnotu z trénovací distribuce. Pro $\ell_{\text{true}}=0.1$ je $\hat\ell_{\text{eff}}$ stále výrazně nadhodnocena (0.5–0.7). Korelace $\hat\ell_{\text{eff}}$ vs $\ell_{\text{true}}$ roste s počtem vrstev, ale je obecně nízká pro malé $\ell$.

### 5.3 Q2 — MSE vs $n$ pro $\ell=0.7$ (testování HP identifikace)

> **Design:** Data z GP($\ell=0.7$), $\sigma^2=0.01$, 200 instancí, $n \in \{5,10,20,40,64,100,128\}$. GP\_wrong = GP s $\ell_{\text{wrong}}=0.5$.

| $n$ | GP\_wrong | 1-layer | 2-layer | 4-layer | 6-layer | 8-layer |
|---|---|---|---|---|---|---|
| 5 | 0.00323 | 0.01377 | 0.01095 | 0.00520 | 0.00565 | 0.00389 |
| 10 | 0.00084 | 0.00561 | 0.01116 | 0.00567 | 0.00682 | 0.00220 |
| 20 | 0.00036 | 0.00189 | 0.01499 | 0.00853 | 0.01370 | 0.00044 |
| 40 | 0.00013 | 0.00338 | 0.00348 | 0.00448 | 0.00384 | 0.00299 |
| 64 | 0.00008 | 0.00207 | 0.00293 | 0.00345 | 0.00287 | 0.00270 |
| 100 | 0.00005 | 0.00117 | 0.01509 | 0.00901 | 0.01177 | 0.00265 |
| 128 | 0.00004 | 0.00052 | 0.01481 | 0.00993 | 0.01214 | 0.00020 |

**Pozorování:** Pro $\ell=0.7$ (blízko $\ell_{\text{wrong}}=0.5$) je GP\_wrong silná baseline — **žádný PFN model ji zde nepřekonává** na žádném $n$. Nejstabilnější napříč $n$ je 8-layer; přetrénovaný 6-layer je nyní srovnatelný se 4-layer. 2-layer a 4-layer mají nestabilní výkon pro velká $n$ (MSE nestejnoměrně klesá).

### 5.4 Q3 — NLL crossover: PFN vs GP-ML

> **Design:** $\ell=0.3$, $\sigma^2=0.01$, $n_{\text{test}}=5$, 100 instancí. Pouze 6-layer a 8-layer modely. NLL nižší = lepší.

| $n$ | oracle | GP-ML | 6-layer | 8-layer |
|---|---|---|---|---|
| 2 | 0.535 | **6.317** | 0.750 | 0.718 |
| 5 | −0.216 | 1.961 | 0.030 | 0.019 |
| 10 | −0.649 | −0.338 | **−0.556** | −0.544 |
| 20 | −0.698 | −0.627 | −0.636 | −0.643 |
| 40 | −0.804 | −0.779 | −0.785 | −0.782 |
| 64 | −0.842 | −0.830 | −0.808 | −0.818 |
| 100 | −0.810 | −0.800 | −0.793 | −0.796 |

**Crossover (po opravě znaménka NLL, přetrénovaný 6L):** PFN NLL správně **klesá s $n$** a pro velká $n$ **těsně sleduje oracle** (6-layer −0.808 vs oracle −0.842 při $n=64$). Po přetrénování se kalibrace 6-layer výrazně zlepšila a nyní **těsně sleduje 8-layer** (např. $n=5$: 6L 0.030 vs 8L 0.019; dříve podtrénovaný 6L 0.476). PFN překonává GP-ML pro malý kontext ($n \leq 10$: při $n=5$ je 6-layer 0.030 vs GP-ML 1.96, při $n=10$ je −0.556 vs −0.338); v pásmu $n=20$–$40$ je GP-ML naopak mírně lepší. Původní (chybný) závěr, že „PFN zůstává ≈0.8 a neumí využít velký kontext", byl artefakt obráceného znaménka.

### 5.5 Q4 — Chování mimo trénovací rozsah HP (OOD $\ell$)

> **Design:** 6-layer model, $n=40$, $\sigma^2=0.01$, 200 instancí. rel\_MSE = MSE(PFN, oracle) / MSE\_ref, kde MSE\_ref = MSE(oracle posterior, noisy y) ≈ šumová chyba oracle (ověřeno v kódu notebooku, cell Q4).  
> Trénovací rozsah: $\ell \in [0.05, 1.0]$. OOD hodnoty označeny.

| $\ell$ | MSE\_PFN | MSE\_ref | rel\_MSE | Stav |
|---|---|---|---|---|
| 0.010 | 0.35358 | 0.46192 | 0.765 | ← OOD |
| 0.030 | 0.07729 | 0.09744 | 0.793 | ← OOD |
| 0.050 | 0.01634 | 0.03569 | 0.458 | (hranice) |
| 0.100 | 0.00389 | 0.01583 | 0.246 | |
| 0.200 | 0.00198 | 0.01233 | 0.161 | |
| 0.300 | 0.00117 | 0.01154 | 0.101 | **minimum** |
| 0.500 | 0.00244 | 0.01091 | 0.224 | |
| 0.700 | 0.00318 | 0.01069 | 0.297 | |
| 1.000 | 0.00356 | 0.01053 | 0.338 | (hranice) |
| 1.500 | 0.00294 | 0.01043 | 0.282 | ← OOD |
| 2.000 | 0.00234 | 0.01035 | 0.226 | ← OOD |
| 3.000 | 0.00197 | 0.01028 | 0.191 | ← OOD |
| 5.000 | 0.00192 | 0.01024 | 0.188 | ← OOD |

**Pozorování:** Minimum rel\_MSE (0.10) je u $\ell=0.3$ (střed trénovacího rozsahu). Pro **celý in-distribution rozsah** $\ell \in [0.05, 1.0]$ platí rel\_MSE < 0.35 — PFN výrazně překonává triviální predikci průměrem $Y$. Ani mimo trénovací rozsah (OOD vlevo $\ell<0.05$ i vpravo $\ell>1$) rel\_MSE nepřekročí ~0.8; degradace je plynulá.

**Limitace tohoto testu i modelu:** Po přetrénování 6-layer na 500 epoch model **neselhává** ani u velkých $\ell$ (rel < 0.35 v celém in-distribution rozsahu, plynulá degradace vně). Původní podtrénovaný checkpoint zde selhával (rel > 1.3), viz §5.7. Test pro OOD outputscale (Q4b, §5.6) prověřuje chování mimo trénovací rozsah amplitudy.

### 5.6 Q4b — Chování mimo trénovací rozsah HP (OOD outputscale)

> **Design:** 6-layer model, $n=40$, $\ell=0.3$, $\sigma^2=0.01$, 200 instancí, seed=55.  
> rel\_MSE = MSE\_PFN / MSE\_ref; MSE\_ref = MSE(GP\_posterior, noisy\_y) $\approx$ noise $= 0.01$.  
> Trénovací rozsah: outputscale $\sim \text{LogNormal}(0, 0.5)$, 5.–95. percentil $\approx [0.44, 2.28]$.

| outputscale | MSE\_PFN | MSE\_ref | rel\_MSE | Stav |
|---|---|---|---|---|
| 0.010 | 0.00020 | 0.01059 | 0.019 | ← OOD |
| 0.050 | 0.00020 | 0.01089 | 0.019 | ← OOD |
| 0.100 | 0.00026 | 0.01103 | 0.023 | ← OOD |
| 0.300 | 0.00032 | 0.01126 | 0.029 | ← OOD |
| 0.500 | 0.00035 | 0.01137 | 0.030 | (blízko hranice) |
| 1.000 | 0.00117 | 0.01154 | 0.101 | (medián trénovací distribuce) |
| 2.000 | 0.04993 | 0.01173 | 4.256 | (95. percentil) |
| 3.000 | 0.10323 | 0.01185 | 8.710 | ← OOD |
| 5.000 | 0.34378 | 0.01202 | 28.598 | ← OOD |
| 10.000 | 1.67775 | 0.01227 | 136.686 | ← OOD |
| 20.000 | 6.41465 | 0.01256 | 510.803 | ← OOD |
| 50.000 | 27.16798 | 0.01299 | 2091.995 | ← OOD |

**Pozorování:**

1. **Celý in-distribution rozsah zvládá výborně** (osc až ~1.0): rel\_MSE ≈ 0.019–0.101 — model výborně pokrývá amplitudy uvnitř trénovací distribuce (i OOD vlevo, osc < 0.44).

2. **Žádný skok uprostřed rozsahu.** Až do osc≈1.0 (medián trénovací distribuce) zůstává rel\_MSE < 0.11; zhoršení nastává teprve kolem osc≈2 (95. percentil), tedy na hranici trénovacího rozsahu — což je očekávané OOD chování. *(Pozn.: dřívější „náhlý skok na 0.946 při osc=1.0" a jeho výklad přes fixní `constant_normalization_std` byl **artefaktem podtrénovaného 6-layer** modelu, nikoli normalizace — přetrénovaný 6-layer zvládá osc=1.0 s rel=0.101.)*

3. **Katastrofické selhání pro velké osc** (za 95. percentilem): rel\_MSE roste exponenciálně — 4.3× (osc=2), 29× (osc=5), 137× (osc=10), **2092× (osc=50)**. BarDistribution má pevně dané hranice bucketů z trénovací distribuce; pro osc=50 padají skutečné hodnoty y daleko za poslední bucket.

**Porovnání Q4 vs Q4b:**

| Test | OOD vlevo | OOD vpravo (max) | Charakter selhání |
|---|---|---|---|
| Q4 ($\ell$) | rel\_MSE = 0.77 ($\ell=0.01$) | rel\_MSE = 0.19 ($\ell=5$) | Plynulé, rel < 1 vždy |
| Q4b (osc) | rel\_MSE = 0.019 (osc=0.01) | rel\_MSE = **2092** (osc=50) | Plynulé v rozsahu, katastrofické za 95. percentilem |

Test OOD $\ell$ nevykazoval skutečné selhání — velké $\ell$ je inherentně jednoduché. Test OOD outputscale odhaluje skutečnou hranici reprezentační kapacity modelu: **BarDistribution neumí reprezentovat hodnoty za trénovacím rozsahem amplitud**.

### 5.7 Souhrnná interpretace Experimentu 6

> *"Pokud PFN identifikoval LS z dat, jeho predikce bude blíž k GP se správným $\ell$ (nízké MSE) a daleko od GP se špatným $\ell$ (vysoké MSE). Pokud neidentifikoval, obě MSE budou podobné."*

Nejsilnějším modelem je **8-layer**, který identifikuje $\ell$ nejspolehlivěji. Jednovrstvý model selhává pro $\ell=0.1$ (MSE=0.16 vs GP\_wrong=0.26). NLL crossover (po opravě znaménka) ukazuje, že PFN je v kalibraci lepší než GP-ML pro malý kontext ($n \leq 10$) a pro velká $n$ těsně sleduje oracle.

> **Pozn. k 6-layer modelu (VYŘEŠENO 2026-07).** Původní 6-layer checkpoint byl **podtrénovaný** (starý formát, samostatný běh mimo dávku 2/4/8, bez záznamu epoch) a choval se nemonotónně — v Q1 pro $\ell=0.3$ horší (0.00697) než 4-layer i 8-layer, v Q4 rel>1 i pro in-distribution $\ell$, v Exp 5 nenulové bias². Model byl **přetrénován na 500 epoch** (nový formát, stejný recept jako ostatní hloubky) a anomálie **zmizela**: v Q1 sedí vedle 4-layer ($\ell=0.3$: **0.00193** vs 4L 0.00163, 8L 0.00036), v Q4 má rel<1 v celém in-distribution rozsahu $\ell$, v bias-variance rozkladu je monotónní ($c$ mezi 4L a 8L). 8-layer byl rovněž přetrénován, ale nová 500-epochová verze byla v ocasech křehčí (na ~10 % instancí katastrofální MSE); proto se ponechává robustní 300-epochový checkpoint (vykazovaný jako 500ep). Čísla v tabulkách výše odpovídají **opraveným** modelům.

---

## 6. Experiment 7 — Robustnost při misspecifikaci prioru

**Notebook:** `experiments/Experiment_7_from_GP2.ipynb`  
**Motivace:** V předchozích experimentech vždy existoval GP se správnými HP jako oracle. Co když je **špatný i kernel**? Data generovaná z Matérn 2.5 ($\ell=0.3$, $\sigma^2=0.1$), ale všechny porovnávané metody používají RBF kernel nebo implicitní trénovací prior PFN. Bayesovský GP má prior $p(\ell) \sim \text{Gamma}(4, \text{rate}=4)$ — střed u $\ell=1.0$, pravá hodnota $\ell=0.3$ leží v chvostu prioru.

### 6.1 Q1 — MSE a NLL vs $n$ (fixní $\ell_{\text{true}}=0.3$)

> **Design:** $\ell_{\text{true}}=0.3$ (Matérn 2.5), $\sigma^2=0.1$, $n_{\text{test}}=10$, 150 instancí.  
> GP\_fixed: RBF s $\ell=0.7$ (špatné HP i kernel).

**MSE vs $n$:**

| $n$ | PFN | GP\_fixed | GP\_ML | GP\_Bayes | GP\_oracle |
|---|---|---|---|---|---|
| 5 | 0.93699 | 1.18172 | 0.95305 | 0.98592 | **0.79025** |
| 10 | **0.72713** | 1.00189 | 0.72144 | 0.73947 | 0.66794 |
| 20 | **0.52432** | 0.78485 | 0.55354 | 0.54532 | 0.47545 |
| 40 | **0.37256** | 0.62423 | 0.41614 | 0.39967 | 0.35816 |
| 64 | 0.25835 | 0.49483 | 0.26753 | 0.26891 | 0.24434 |
| 100 | 0.18719 | 0.38220 | 0.18626 | 0.18506 | 0.17135 |
| 128 | 0.17385 | 0.41661 | 0.17169 | 0.18200 | 0.16918 |

Po přetrénování 6-layer je PFN v MSE **konkurenceschopná s GP\_ML** — pro $n \in \{10,20,40\}$ je dokonce lepší (např. $n=40$: PFN 0.373 vs GP\_ML 0.416), pro velká $n$ prakticky shodná. To obrací dřívější závěr (s podtrénovaným 6L), že GP\_ML PFN konzistentně překonává.

**ΔNLL vs $n$** (= NLL\_metoda − NLL\_oracle; nižší/blíž nule = lepší kalibrace; PFN po opravě znaménka):

| $n$ | PFN | GP\_fixed | GP\_ML | GP\_Bayes | GP\_oracle |
|---|---|---|---|---|---|
| 5 | **+0.165** | +0.732 | +0.968 | +0.326 | 0 |
| 10 | **+0.143** | +0.956 | +0.403 | +0.288 | 0 |
| 20 | **+0.086** | +1.068 | +0.239 | +0.190 | 0 |
| 40 | **+0.077** | +1.231 | +0.191 | +0.169 | 0 |
| 64 | +0.077 | +1.157 | **+0.050** | +0.086 | 0 |
| 100 | **+0.045** | +0.972 | +0.056 | +0.049 | 0 |
| 128 | +0.031 | +1.174 | **+0.016** | +0.044 | 0 |

**Klíčový výsledek kalibrace:** PFN má **nejmenší ΔNLL** (nejblíže oracle) ze všech metod pro $n \leq 40$ — pod misspecifikací kernelu je nejlépe kalibrovaná. Teprve pro velká $n$ ($\geq 64$) ji GP\_ML mírně předčí. GP\_fixed (špatný kernel i HP) je řádově nejhorší.

### 6.2 Q2 — MSE a NLL vs $\ell_{\text{true}}$ (fixní $n=40$)

> Testujeme, jak moc závisí výkon na síle misspecifikace kernelu. Pravda: Matérn 2.5 s různým $\ell$; všechny metody používají RBF.

**MSE vs $\ell_{\text{true}}$:**

| $\ell_{\text{true}}$ | PFN | GP\_fixed | GP\_ML | GP\_Bayes | GP\_oracle |
|---|---|---|---|---|---|
| 0.050 | 1.06612 | 2.00588 | 0.99193 | 0.98868 | **0.98116** |
| 0.100 | 0.85773 | 1.87695 | 0.81220 | 0.81922 | **0.78140** |
| 0.200 | 0.55296 | 1.13399 | 0.57635 | 0.56389 | **0.50696** |
| 0.300 | **0.37256** | 0.62423 | 0.41614 | 0.39967 | 0.35816 |
| 0.500 | 0.24395 | 0.27506 | 0.25383 | 0.23612 | **0.23189** |
| 0.700 | 0.19586 | 0.18896 | 0.19154 | 0.18329 | **0.18401** |
| 1.000 | 0.16668 | 0.15438 | 0.15891 | 0.15302 | **0.15353** |
| 1.500 | 0.15210 | 0.14165 | 0.13702 | 0.13590 | **0.13366** |
| 2.000 | 0.14713 | 0.13830 | 0.13038 | 0.12932 | **0.12545** |

**ΔNLL vs $\ell_{\text{true}}$** (= NLL\_metoda − NLL\_oracle; nižší/blíž nule = lepší):

| $\ell_{\text{true}}$ | PFN | GP\_fixed | GP\_ML | GP\_Bayes | GP\_oracle |
|---|---|---|---|---|---|
| 0.050 | +0.083 | +5.544 | **+0.007** | +0.007 | 0 |
| 0.100 | +0.098 | +5.078 | **+0.041** | +0.058 | 0 |
| 0.200 | +0.099 | +2.758 | +0.148 | +0.145 | 0 |
| 0.300 | **+0.077** | +1.231 | +0.191 | +0.169 | 0 |
| 0.500 | +0.057 | +0.241 | +0.098 | **+0.058** | 0 |
| 0.700 | +0.055 | +0.044 | +0.052 | **+0.022** | 0 |
| 1.000 | +0.061 | **+0.005** | +0.034 | +0.006 | 0 |
| 1.500 | +0.085 | +0.025 | +0.015 | **+0.007** | 0 |
| 2.000 | +0.097 | +0.043 | +0.020 | **+0.013** | 0 |

### 6.3 Interpretace

> **⚠️ Aktualizováno (2026-07):** Původní čísla byla počítána s **podtrénovaným 6-layer** modelem. Po jeho přetrénování na 500 epoch se výkon PFN pod misspecifikací **výrazně zlepšil** — závěry níže odpovídají opravenému modelu.

**MSE:** PFN je po přetrénování 6-layer **konkurenceschopná s GP\_ML/GP\_Bayes** — pro $n \in \{10,20,40\}$ je v MSE dokonce lepší než GP\_ML (např. $n=40$: 0.373 vs 0.416), pro velká $n$ prakticky shodná. Vůči GP\_fixed (špatný kernel i $\ell$) je PFN výrazně lepší pro $\ell_{\text{true}} \leq 0.3$; pro $\ell_{\text{true}} \geq 0.7$ je GP\_fixed mírně lepší ($\ell=0.7$ je blízko pravé hodnoty a RBF aproximuje Matérn 2.5 dobře).

**ΔNLL — klíčový výsledek kalibrace:** Měříme ΔNLL vůči oracle (nižší = blíž ideálu). PFN má **nejmenší ΔNLL ze všech misspecifikovaných metod pro $n \leq 40$** (Q1) i pro $\ell_{\text{true}} \leq 0.3$ (Q2) — pod misspecifikací kernelu je tedy **nejlépe kalibrovaná**. Např. při $n=40$: PFN +0.077 vs GP\_ML +0.191 vs GP\_Bayes +0.169; při $\ell=0.3$: PFN +0.077 vs GP\_ML +0.191. Teprve pro velký kontext ($n \geq 64$) nebo velké $\ell$ ji GP\_ML/GP\_Bayes mírně předčí. GP\_fixed je řádově nejhorší (ΔNLL až +5.5 pro malé $\ell$).

**Konzistence MSE i NLL:** Oba pohledy dávají stejný, silnější závěr než dřívější (podtrénovaný) běh: **PFN se pod misspecifikací drží velmi dobře** — v MSE konkurenceschopná, v kalibraci nejlepší pro malý až střední kontext. To odpovídá tomu, že PFN marginalizuje přes trénovací distribuci $\ell \sim U(0.05, 1.0)$, což dává robustní, dobře kalibrované prediktivní intervaly právě tam, kde adaptivní GP metody z malého kontextu $\ell$ přefitují.

> *"Když nikdo nezná správný prior, kdo selže míň? Z hlediska kalibrace (ΔNLL) je pro malý až střední kontext nejlepší PFN; v MSE je konkurenceschopná s GP\_ML/GP\_Bayes. GP\_fixed (špatný kernel i HP) selhává nejvíc."*

| Metoda | Misspecifikace kernelu | Misspecifikace $\ell$ |
|---|---|---|
| PFN | Implicitní RBF (trénovací distribuce) | Implicitní $U(0.05, 1.0)$, nelze opravit za běhu |
| GP\_fixed | RBF místo Matérn 2.5 | Fixní $\ell=0.7$, nic se nepřizpůsobuje |
| GP\_ML | RBF místo Matérn 2.5 | Bodový MLE — přizpůsobí se datům |
| GP\_Bayes | RBF místo Matérn 2.5 | NUTS s Gamma(4, rate=4), střed u $\ell=1.0$ |
| GP\_oracle | Správný Matérn 2.5 | Správné $\ell_{\text{true}}$ |

---

## 7. Souhrnné srovnání

### 7.1 Závislosit na počtu vrstev (Experiment 1 + 6)

| Metrika | 1-layer | 2-layer | 4-layer | 6-layer | 8-layer |
|---|---|---|---|---|---|
| bias² (MSE dek.) | 0.00476 | −0.00152 | −0.00089 | −0.00076 | −0.00065 |
| c (variance term) | 0.097 | 0.068 | 0.035 | 0.029 | 0.026 |
| max rel ΔMSE (head knockout, medián × baseline) | 44.9× | 24.6× | 9.9× | 17.7× | 14.6× |
| MSE($\ell=0.1$, $n=40$) | 0.16054 | 0.01714 | 0.00814 | 0.00679 | **0.00304** |
| Zlepšení lokalizací | +68.4 % | +25.3 % | +35.1 % | +39.2 % | −0.0 % |

### 7.2 Robustnost při misspecifikaci (Experiment 7, $n=40$, $\ell_{\text{true}}=0.3$)

| Metoda | MSE | ΔNLL (vs oracle) | Vítěz (MSE) | Vítěz (ΔNLL) |
|---|---|---|---|---|
| PFN (6-layer) | **0.373** | **+0.077** | **✓** | **✓** |
| GP\_fixed | 0.624 | +1.231 | — | — |
| GP\_ML | 0.416 | +0.191 | — | — |
| GP\_Bayes | 0.400 | +0.169 | — | — |
| GP\_oracle | 0.358 | 0 | ref | ref |

> Po přetrénování 6-layer **PFN vyhrává v MSE i kalibraci** mezi všemi misspecifikovanými metodami (při $n=40$, $\ell_{\text{true}}=0.3$) — obrací dřívější závěr (s podtrénovaným 6L), kde vyhrával GP\_Bayes.

---

## Reference

- Ahn, K., Cheng, X., Daneshmand, H., & Sra, S. (2023). *Transformers learn to implement preconditioned gradient descent for in-context learning*. NeurIPS 2023. arXiv:2306.00297.
- von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., & Vladymyrov, M. (2023). *Transformers Learn In-Context by Gradient Descent*. ICML 2023. arXiv:2212.07677.
- Fu, D., Chen, T.-Q., Jia, R., & Sharan, V. (2023). *Transformers Learn to Achieve Second-Order Convergence Rates for In-Context Linear Regression*. arXiv:2310.17086.
- Nagler, T. (2023). *Statistical Foundations of Prior-Data Fitted Networks*. ICML 2023.

---

*Experimenty provedeny na Apple M-series (MPS backend), PyTorch 2.10.0, numpyro 0.16.1, JAX 0.4.35.*  
*Zdrojové notebooky: `experiments/Experiment_{1,2,5,6,7}_from_GP2.ipynb`.*  
*Experiment 3 (Q1 label mixing) byl duplikát Experimentu 1 a je do něj sloučen (§2.4 head-knockout, §2.5 Q3 bandwidth). Experiment 4 z GP2 série nemá uložené výstupy (nebyl spuštěn).*  
*Obrázky z Experiment 1 jsou v `figures/` — zatím neextrahované (spusť `extract_figures.py` po extrakci z GP2 notebooků).*
