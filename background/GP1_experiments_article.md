# Prior-Fitted Networks jako aproximátory Gaussovských procesů: experimentální analýza

**Ruslan Guliev**  
Vrije Universiteit Amsterdam, 2025

---

## Abstrakt

Tato práce experimentálně zkoumá, jak transformerová architektura Prior-Fitted Networks (PFN) aproximuje inferenci Gaussovských procesů (GP). Série osmi experimentů pokrývá: konvergenci střední hodnoty a kalibraci variance, analýzu attention matic, porovnání s Nadaraya-Watsonovým (NW) estimátorem, predikci na skutečné GP realizaci při omezeném kontextu, přesnost napříč různými hodnotami délky korelace (lengthscale), schopnost identifikace hyperparametrů z kontextových dat, srovnání s Type-II maximální věrohodností a analýzu marginalizace přes hyperparametry. Experimenty jsou prováděny na dvou modelech: (1) modelu trénovaném 100 epoch s fixními hyperparametry (*100-epoch model*) a (2) modelu trénovaném 500 epoch na distribuci hyperparametrů (*random-HP model*). Všechny metriky jsou průměrovány přes **5 seedů** (kde relevantní přes 25–100 nezávislých instancí). Klíčový numerický výsledek: celý PFN dosahuje MSE(PFN, GP) = **0.000167 ± 0.000096** (100-epoch model) vůči NW = **0.2099 ± 0.1462** — rozdíl ~1 256× vylučující hypotézu kernel-averagingu.

---

## 1. Úvod

Gaussovské procesy jsou elegantní rámec pro neparametrickou Bayesovskou inferenci, avšak jejich výpočetní náklady $\mathcal{O}(n^3)$ při invertování kernelové matice je omezují na malé datové sady. Prior-Fitted Networks (PFN; Müller et al., 2022) přistupují k tomuto problému opačně: místo aproximace inference GP trénují transformerový model přímo na datech vzorkovaných z GP prioru, a model tak implicitně provádí přibližnou Bayesovskou inferenci při jednom dopředném průchodu.

Klíčová otázka, která motivuje tuto práci, zní: **co se PFN skutečně naučil?** Triviální hypotéza říká, že model implementuje Nadaraya-Watsonův (NW) estimátor — tedy kernel-averaging s lokálními váhami. Cílem experimentů je tuto hypotézu otestovat a charakterizovat skutečnou výpočetní strategii, kterou transformer zvolil.

### 1.1 Matematický rámec

Gaussovský proces je definován střední funkcí $m(x)$ a kernelovou funkcí $k(x, x')$. Posteriorní střední hodnota po pozorování kontextových párů $(X, y)$ je:

$$\mu(x^*) = k(x^*, X)\left[K(X,X) + \sigma^2 I\right]^{-1} y$$

kde $K(X,X)_{ij} = k(x_i, x_j)$ je kernelová matice. Nadaraya-Watson estimátor aproximuje tuto formuli jako pouhý normalizovaný váhovaný průměr:

$$\hat{y}_{\text{NW}}(x^*) = \frac{\sum_i k(x^*, x_i)\, y_i}{\sum_i k(x^*, x_i)}$$

Klíčový rozdíl: GP explicitně decorreluje blízké trénovací body přes $K^{-1}$, zatímco NW tuto korekci vynechává.

Attention mechanismus transformeru:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V, \quad Q = XW_Q,\ K = XW_K$$

Výsledná kernelová matice je datově závislá:

$$\text{Kernel}_{\text{attn}}(X) = \text{softmax}\!\left(\frac{X\,W_Q W_K^\top X^\top}{\sqrt{d_k}}\right)$$

Na rozdíl od fixního RBF kernelu se tato matice plně adaptuje na konkrétní vstupní data.

---

## 2. Experimentální nastavení

### 2.1 Modely

| Model | Checkpoint | Epochy | HP trénování | Architektura |
|---|---|---|---|---|
| **100-epoch model** | `pfn_latest_epoch_100.pth` | 100 | Fixní: $\ell=0.3$, $\sigma^2=10^{-4}$, osc=1.0 | 6 vrstev, 14.14M param |
| **Random-HP model** | `pfn_rand_hps_latest_epoch_500.pth` | 500 | $\ell \sim U(0.05, 1.0)$, osc $\sim \text{LogNormal}(0, 0.5)$, $\sigma^2 \sim 10^{U(-3,-1)}$ | 6 vrstev, 14.14M param |

Obě architektury jsou identické: Transformer s `emsize=512`, `nhead=8`, `nhid=1024`, `nlayers=6`, výstupní distribuce `FullSupportBarDistribution` s 1000 binami.

### 2.2 GP prior a data

RBF kernel: $k(x, x') = \text{osc} \cdot \exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right) + \sigma^2 \delta_{xx'}$

Vstupní prostor $x \in [0, 1]$, délky sekvencí: 50–100 bodů.

**Šum v experimentech.** Random-HP model byl *trénován* na $\sigma^2 \sim 10^{U(-3,-1)}$, tj. z rozsahu $[0.001, 0.1]$ (viz §2.1). Samotné experimenty ovšem neběží na náhodném losu z tohoto rozsahu — používají **fixní** $\sigma^2 = 0.001$, tedy spodní hranici trénovacího rozsahu. Jedinou výjimkou je Experiment 7, kde jedna ze čtyř testovaných konfigurací používá $\sigma^2 = 0.01$. Testovací $\ell$ pokrývají $[0.05, 0.9] \subset [0.05, 1.0]$, outputscale je fixně $1.0$ (v Exp 7 jedna konfigurace $0.5$).

### 2.3 Baseline metody

- **True GP (oracle):** GP se znalostí správných HP (dolní mez chyby)
- **Nadaraya-Watson:** kernel averaging s RBF kernelem a správným $\ell$
- **Type-II ML:** `sklearn.GaussianProcessRegressor` s `n_restarts_optimizer=3`, optimalizuje $\log p(y \mid X, \theta)$
- **Marginalizace:** průměr GP predikcí přes 20 hodnot $\ell \in [0.05, 1.0]$, váhy rovnoměrné

### 2.4 Metodologie průměrování

Všechny statistiky jsou sbírány přes 5 nezávislých seedů. V tabulkách uvádíme `mean ± std` přes celkový počet instancí $n$ (např. 5 seedů × 5 realizací = 25 hodnot). Výsledky jsou uloženy v `background/stats_averaged.json`.

---

## 3. Výsledky

### 3.1 Experiment 1 — Konvergence střední hodnoty a kalibrace variance

**Design:** Trénovací data pouze v regionu $x \in [0.3, 0.7]$ ($n_{\text{context}} = 20$), testovací body v celém $[0, 1]$ ($n_{\text{test}} = 100$). 5 realizací × 5 seedů = 25 instancí celkem.

#### 3.1.1 Konvergence střední hodnoty (100-epoch model)

Střední hodnota PFN sleduje GP posterior s vysokou přesností v regionu s daty. Mimo trénovací region ($x < 0.3$, $x > 0.8$) PFN správně konverguje k prioru (0), stejně jako GP. Ojediněle pro realizace s vysokou amplitudou ($|m_{\text{train}}| > 1.5$) dochází k neúplné konvergenci — GP se chová identicky.

#### 3.1.2 Kalibrace variance — 100-epoch model (průměr přes 25 instancí)

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp1_big"]`.  
> Funkce `run_exp1(big_model, HPS_BIG={ℓ=0.3, noise=1e-4}, seed)`, 5 seedů × 5 realizací = **25 hodnot**.  
> Na každé realizaci: trénovací body $x \in [0.3, 0.7]$ ($n=20$, náhodný podvýběr z GP sekvence délky 100), testovací body `linspace(0,1,100)`.  
> `Corr` = `np.corrcoef(pfn_std, gp_std)[0,1]`; `MSE(std)` = `mean((pfn_std − gp_std)²)`.  
> `Ratio far/near` = průměrná std v $x \in [0,0.2] \cup [0.8,1]$ / průměrná std v $x \in [0.4,0.6]$.

| Metrika | Hodnota |
|---|---|
| **Corr(PFN std, GP std)** | **0.9812 ± 0.0252** |
| MSE(PFN std, GP std) | 0.00376 ± 0.00911 |
| Ratio far/near — PFN | **21.3× ± 5.2×** |
| Ratio far/near — GP | 25.9× ± 3.8× |

#### 3.1.3 Kalibrace variance — random-HP model (průměr přes 25 instancí)

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp1_rand"]`.  
> Funkce `run_exp1(rand_model, HPS_RAND={ℓ=0.3, osc=1.0, σ²=0.001}, seed)` — stejný protokol jako výše, liší se jen šum: random-HP model se testuje na σ²=0.001 (spodní hranice trénovacího rozsahu). 5 seedů × 5 realizací = **25 hodnot**.  
> Model testován na datech z GP($\ell=0.3$), i když byl trénován na $\ell \sim U(0.05, 1.0)$.

| Metrika | Hodnota |
|---|---|
| **Corr(PFN std, GP std)** | **0.9734 ± 0.0292** |
| MSE(PFN std, GP std) | 0.03478 ± 0.03461 |
| Ratio far/near — PFN | **7.49× ± 2.20×** |
| Ratio far/near — GP | 25.9× ± 3.8× |

**Interpretace:** Korelace $r > 0.97$ u obou modelů potvrzuje, že PFN správně identifikuje *tvar* profilu uncertainty — kde je nejistota větší a kde menší. Absolutní hodnoty jsou však systematicky vychýlené: PFN dosahuje přibližně 82 % (100-epoch) resp. 29 % (rand-HP) dynamického rozsahu GP. Random-HP model (500 epoch) má výrazně horší MSE(std) = 0.035 oproti 100-epoch modelu (0.004). Toto je způsobeno tím, že random-HP model trénuje na široké distribuci lengthscale a kalibraci variance pro jeden fixní $\ell = 0.3$ dosahuje obtížněji.

> *"MSE GP vs PFN je mnohem nižší pro větší model. Uncertainty ratio se podařilo taky zlepšit, ale stále jsou konfidenční intervaly u GP mnohem lepší pro outliery."*

![Vzorky z GP prioru — 100-epoch model](figures/GP1_big_model/fig_big_01_prior_samples.png)
*Obrázek 1: Vzorky z GP prioru ($\ell=0.3$, $n=20$ trénovacích bodů). PFN je trénován na těchto realizacích.*

![PFN vs GP uncertainty — 100-epoch model](figures/GP1_big_model/fig_big_02_uncertainty.png)
*Obrázek 2: Experiment 1 — 100-epoch model. Srovnání střední hodnoty a variance PFN (modrá) a GP (zelená). Trénovací body v $x \in [0.3, 0.7]$, predikce v celém $[0, 1]$. PFN správně rozpoznává region extrapolace zvýšenou variancí.*

![Vzorky z GP prioru — random-HP model](figures/GP1_random_model/fig_rand_01_prior_samples.png)
*Obrázek 3: Vzorky z GP prioru pro random-HP model ($\ell=0.3$, různé realizace).*

![PFN vs GP uncertainty — random-HP model](figures/GP1_random_model/fig_rand_02_uncertainty.png)
*Obrázek 4: Experiment 1 — random-HP model. Šírší rozptyl predikcí odráží trénování na distribuci hyperparametrů.*

---

### 3.2 Experiment 2 — Struktura attention matic napříč vrstvami

**Design:** Attention matice pro všech 6 vrstev, průměrované přes 8 hlav. Pro 100-epoch model: sekvence délky 120 (20 train + 100 test), tvar matice $120 \times 120$. Matice je rozdělena na kvadranty cyánovou čárou ($n = 20$).

**Kvadrantová interpretace:**

```
┌─────────────────┬─────────────────┐
│  Train → Train  │  Train → Test   │
│   [0:20, 0:20]  │  [0:20, 20:]    │
├─────────────────┼─────────────────┤
│  Test → Train   │  Test → Test    │
│  [20:, 0:20]    │  [20:, 20:]     │
└─────────────────┴─────────────────┘
```

**Pozorované patterny:**

> **Zdroj dat:** Výstupy notebooků `experiments/Experiments_from_GP1_big_model.ipynb` (cell 7) a `Experiments_from_GP1_random_model.ipynb` (cell 6), extrahované do `figures/` skriptem `background/extract_figures.py`.  
> Tabulka je kvalitativní popis vizuálního vzoru obrázků — žádná numerická agregace. Každá matice = průměr přes 8 attention hlav dané vrstvy pro jednu konkrétní sekvenci.

| Vrstva | Charakter attention | Train→Test | Test→Train |
|---|---|---|---|
| 0 | Distribuovaná, uniformní | Nenulová | Slabá |
| 1–2 | Počínající sparse vzory | Klesá | Roste |
| 3–4 | Sparse, strukturované patterny | Nízká | Výrazná |
| 5 (poslední) | Plně strukturovaná | ≈ 0 | **Dominantní** |

**Klíčový objev — kauzální asymetrie:**

V poslední vrstvě kvadrant Train→Test ≈ 0, zatímco Test→Train je dominantní a strukturovaný. Tato asymetrie přesně odpovídá kauzální struktuře GP inference: testovací body jsou podmíněny trénovacími, nikoli naopak — a transformer tuto strukturu objevil **autonomně z dat**, bez explicitního zakódování.

> *"Je vidět, že transformer postupně odhalil, že trénovací data nezávisí na testovacích a testovací data nějakým způsobem závisí na trénovacích datech. Struktura těchto matic je vlastně náznakem, že PFN dělá Bayesian inference, ne jen pattern matching. Kdyby se PFN jen učil nazpaměť trénovací data, attention by byla symetrická."*

> *"Patterny v poslední matici jsou mnohem čistější než pro malý model (20 epoch), což dává dobrý smysl: delší trénink → přesnější prokladání křivkou."*

![Attention matice — všech 6 vrstev, 100-epoch model](figures/GP1_big_model/fig_big_03_attention_all_layers.png)
*Obrázek 5: Experiment 2 — attention matice pro všech 6 vrstev (100-epoch model). Průměr přes 8 hlav. Cyánová čára odděluje trénovací ($0\!:\!20$) a testovací ($20\!:\!120$) pozice. Postupná specializace: vrstva 0 uniformní → vrstva 5 dominantní Test→Train blok.*

![Attention matice — všech 6 vrstev, random-HP model](figures/GP1_random_model/fig_rand_03_attention_all_layers.png)
*Obrázek 6: Experiment 2 — attention matice pro random-HP model. Stejný vzor kauzální asymetrie, avšak méně čistý než u 100-epoch modelu.*

**Pruhové (svislé) trendy — attention sinks:**

Na maticích jsou nápadné **svislé pruhy**: některé klíčové (zdrojové) pozice dostávají vysokou attention od téměř *všech* query pozic, nezávisle na tom, kde daný query bod leží. Ve vrstvě 0 jsou nejostřejší (jeden–dva sloupce s vahou až $\approx 0.5$), v hlubších vrstvách (3–5) typicky přetrvává jeden reziduální pruh. Vznikají ze tří příčin:

1. **Softmax normalizace vynucuje „sink".** Každý řádek matice (jeden query) musí sečíst na 1. Když query bod nemá žádný silně relevantní klíč — což platí zejména v raných vrstvách, než se vybudují informativní reprezentace, a pro testovací body daleko od kontextu — zbytková attention masa se „vylije" na několik fixních výchozích pozic, které se stanou svislými pruhy. Jde o stejný mechanismus jako **attention sink** (BOS-token) popsaný u velkých jazykových modelů (Xiao et al., 2023).

2. **Body s výrazným embeddingem přitahují attention.** Kontextové body s extrémní hodnotou $y$ (velké $|y|$) nebo s $x$ blízko okraje $[0,1]$ mají po zakódování encoderem embedding s větší normou → větší skalární součiny (pre-softmax skóre) vůči všem query → jasný sloupec. Protože se matice průměruje přes 8 hlav, stačí, aby sink implementovala jediná hlava, a v průměru se pruh objeví.

3. **Pruhy jsou query-nezávislé — a proto *nejde* o kernel smoothing.** Nadaraya-Watsonův / RBF estimátor by svislý pruh nikdy nevytvořil: jeho váhy jsou lokální a závisí na poloze query bodu. Svislý pruh naopak znamená „všechny query se dívají na tentýž bod bez ohledu na svou polohu". Jejich výskyt je tak dalším přímým důkazem, že PFN attention $\neq$ RBF kernel (v souladu s Experimentem 3, §3.3) — část attention rozpočtu jde na normalizační/sink mechanismus, ne na kernel-averaging.

> **Pozor na interpretaci os:** v tomto experimentu (`n_context=20`, `seq_len=50`, matice $70 \times 70$) nejsou pozice seřazené podle $x$ — `test_x = batch.x[0]` je v původním (náhodném) pořadí vzorkování. Pruh na „pozici $p$" proto odpovídá jednomu konkrétnímu bodu, nikoli oblasti v prostoru $x$, a jeho poloha se mění realizaci od realizace (podle toho, který bod má extrémní $y$).

---

### 3.3 Experiment 3 — Attention vs RBF kernel: korelační analýza

**Design:** Pro každý testovací bod $x^*$ vypočítáme attention váhy $a_i$ z poslední vrstvy (průměr přes 8 hlav) na trénovací body a porovnáme s normalizovanými RBF vahami $w_i = k(x^*, x_i) / \sum_j k(x^*, x_j)$. Průměr přes 5 seedů.

#### 3.3.1 Globální korelace a MSE predikcí (5 seedů)

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp3_big"]` a `["exp3_rand"]`.  
> Funkce `run_exp3(model, HPS_BIG, seed)`, 5 seedů × 1 sekvence = **5 hodnot** na metriku.  
> Vstupní data: `get_batch_for_gp(batch_size=1, seq_len=100)`, prvních $n_{\text{ctx}}=20$ bodů jako train, všech 100 jako test.  
> `MSE(PFN, GP)` = `mean((pfn_mean − gp_mean)²)` přes všech 100 testovacích bodů.  
> `Corr(Attn, RBF)`: pro každý testovací bod $x^*$ spočítána Pearsonova korelace attention vah poslední vrstvy (průměr přes 8 hlav) a normalizovaných RBF vah `exp(−d²/2ℓ²)`, zprůměrováno přes všech 100 bodů.  
> Attention váhy extrahované forward-hookem na `self_attn_between_items` modulech (`get_attn_weights()`).

| Metrika | 100-epoch model | Random-HP model |
|---|---|---|
| Průměrná korelace Attn vs RBF | **0.374 ± 0.270** | **0.316 ± 0.311** |
| MSE(PFN, GP) | **0.000167 ± 0.000096** | — |
| MSE(NW, GP) | **0.2099 ± 0.1462** | — |
| MSE(PFN, NW) | **0.2092 ± 0.1439** | — |
| Poměr MSE(NW)/MSE(PFN) | **~1 256×** | — |

MSE(PFN, GP) vs MSE(NW, GP) je ~1 256× — PFN je o tři řády přesnější než Nadaraya-Watson estimátor. Zároveň MSE(PFN, NW) ≈ MSE(NW, GP), což ukazuje, že NW a PFN predikují zcela odlišné věci.

Korelace attention vs RBF je nízká u obou modelů (0.37 resp. 0.32) a s vysokou variabilitou (std ~0.27–0.31). To potvrzuje, že model neimplementuje RBF kernel-smoothing — attention mechanismus se naučil jinou, komplexnější strategii.

#### 3.3.2 Nelokálnost attention — matematické zdůvodnění

Attention v kvadrantu Test→Train je nejen ostřejší než RBF, ale i globální — vzdálené trénovací body dostávají nenulovou váhu. Toto odpovídá struktuře efektivních vah GP inference:

$$w_i = \sum_j k(x^*, x_j)\, (K^{-1})_{ji}$$

Matice $K^{-1}$ má obecně nenulové mimobiagonální prvky i tam, kde $K$ je malá — inverze "rozšiřuje" informaci globálně a decorreluje blízké trénovací body.

> *"Attention a RBF mají podobný celkový trend, ale attention je výrazně ostřejší. Dává skoro veškerou váhu nejbližšímu bodu, zatímco RBF distribuuje váhu plynuleji. Ony počítají to, co v kombinaci s ostatními hlavami a FFN vrstvami dává nejlepší GP posterior."*

> *"PFN se nenaučil pouze RBF kernel $k(x', x)$, ale celé GP váhy $k(x', x) \cdot K^{-1}$. Je zjevné, že v některých místech korelace může být i záporná."*

#### 3.3.3 Epistemologická omezení experimentu — co korelace Attn vs RBF skutečně říká

Experiment je korektní jako analýza **jedné komponenty** sítě — měří, zda attention samotná připomíná RBF kernel. Existují však tři důvody, proč z nízké korelace nelze přímo usuzovat na chování celého modelu:

**1. Attention není výstupní mechanismus — je to jen první krok**

Po attention vrstvě následuje FFN (feed-forward síť), residual connections a v případě vícevrstvého modelu dalších několik bloků. Tedy i kdyby attention váhy v poslední vrstvě přesně odpovídaly normalizovaným RBF vahám $k(x^*, x_i) / \sum_j k(x^*, x_j)$, výsledné predikce PFN by se stále lišily od Nadaraya-Watson estimátoru — protože FFN a residual stream přidávají korekci odpovídající $K^{-1}$ efektu. Jinými slovy: porovnání je korektní jako „co dělá tato komponenta", ale **neříká nic o kvalitě celkového výstupu modelu**.

**2. Správná referenční hodnota pro korelaci není $k(x^*, x_i)$, ale $k(x^*, X) K^{-1}$**

Kdyby PFN implementoval GP inference dokonale, jeho efektivní váhy na trénovací body by byly:
$$w_i^{\text{GP}} = \sum_j k(x^*, x_j)\, (K^{-1})_{ji}$$

Tyto váhy **nejsou lokální**: blízké trénovací body si konkurují a po aplikaci $K^{-1}$ dostávají menší váhu, než by odpovídalo RBF. Vzdálené body mohou dostat kladnou váhu a některé body dokonce zápornou. Proto korelace attention vah s čistým RBF kernelem je strukturálně omezena shora — i perfektní GP aproximátor by měl nízkou korelaci s RBF při použití tohoto experimentálního designu.

**3. Pokles korelace u déle trénovaných modelů není zhoršení**

U větších nebo déle trénovaných modelů korelace Attn vs RBF neklesá proto, že by se model zhoršoval — ale proto, že **nachází efektivnější reprezentaci**, která se vzdálila od kernel-like strategie. Pokud se zároveň zlepšuje MSE(PFN, GP), pak nová strategie funguje lépe, jen není interpretovatelná přes RBF analogii. Záporná korelace v tomto kontextu signalizuje explicitní $K^{-1}$ efekt: model aktivně potlačuje redundantní váhu blízkých bodů.

**Závěr k epistemologii experimentu:** Nízká korelace Attn vs RBF potvrzuje, že PFN neimplementuje prostý kernel averaging — to je platný a hodnotný závěr. Ale nemůže být interpretována jako "model je sofistikovanější/horší než RBF" na úrovni celého výstupu. Kvalitu celkových predikcí měří porovnání finálních výstupů PFN vs GP oracle — a to jednoznačně ukazuje MSE poměr ~1 256× ve prospěch PFN oproti NW.

![Attention detail + RBF + entropie — 100-epoch model](figures/GP1_big_model/fig_big_04_attention_detail.png)
*Obrázek 7: Experiment 3 — detail attention vah poslední vrstvy (100-epoch model) pro vybraný testovací bod. Modrá: attention váhy; červená: normalizované RBF váhy. Attention je ostřejší a nekoreluje s RBF (korelace 0.37 ± 0.27).*

![PFN vs NW vs GP predikce — 100-epoch model](figures/GP1_big_model/fig_big_05_nw_comparison.png)
*Obrázek 8: Experiment 3 — srovnání predikcí PFN, NW a GP (100-epoch model). MSE(PFN, GP) = 0.000167 vs MSE(NW, GP) = 0.2099 — rozdíl ~1 256×. NW přehlazuje, PFN sleduje GP posterior.*

![Attention detail — random-HP model](figures/GP1_random_model/fig_rand_04_attention_detail.png)
*Obrázek 9: Experiment 3 — detail attention pro random-HP model. Korelace s RBF 0.316 ± 0.311 — ještě nižší než u 100-epoch modelu.*

---

### 3.4 Experiment 4 — PFN inference na skutečné GP realizaci

**Design:** Generujeme kompletní GP realizaci (100 bodů). PFN a GP dostanou $n_{\text{context}} \in \{5, 10, 15, 20\}$ z těchto bodů a predikují zbývající skutečné (šumové) hodnoty. Průměr přes 5 seedů × 20 realizací = **100 instancí na každé $n$**.

#### 3.4.1 MSE vůči skutečným hodnotám — oba modely

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp4_big"]` a `["exp4_rand"]`.  
> Funkce `run_exp4(model, HPS_BIG, seed, n_ctx_list=[5,10,15,20], n_total=100, n_rep=20)`, 5 seedů × 20 realizací = **100 hodnot** na každé $(n_{\text{ctx}}, \text{model})$.  
> Protokol: vygenerovat celou GP sekvenci (100 bodů), náhodně vybrat $n_{\text{ctx}}$ jako train, zbytek jako test. `MSE` = `mean((predikce − pravá_y)²)` přes testovací body. Obě metody dostávají stejné rozdělení train/test. GP oracle zná správné HP.

| $n_{\text{context}}$ | PFN (100-epoch) | GP oracle (100-epoch) | PFN (rand-HP) | GP oracle (rand-HP) | Vítěz |
|---|---|---|---|---|---|
| 5 | 0.0408 ± 0.0698 | **0.0340 ± 0.0635** | 0.0598 ± 0.0958 | **0.0340 ± 0.0635** | GP |
| 10 | 0.0123 ± 0.0349 | **0.0112 ± 0.0360** | 0.0188 ± 0.0751 | **0.0112 ± 0.0360** | GP |
| 15 | 0.0064 ± 0.0271 | **0.0028 ± 0.0036** | 0.0063 ± 0.0193 | **0.0028 ± 0.0036** | GP |
| 20 | 0.0021 ± 0.0025 | **0.0017 ± 0.0011** | 0.0020 ± 0.0013 | **0.0017 ± 0.0011** | ≈ remíza |

**Pozorování:**

- GP oracle (se znalostí správného $\ell$) konzistentně dosahuje nižšího nebo srovnatelného MSE ve všech $n$.
- 100-epoch model (fixní $\ell = 0.3$) je blíže GP oracle než random-HP model pro střední $n$ — trénování na správném $\ell$ dává výhodu při interpolaci.
- Random-HP model dosahuje téměř stejné přesnosti jako 100-epoch model při $n = 20$ přes jemnější granularitu.
- Oba modely konvergují k GP oracle od $n \geq 20$.

> *"Pro menší kontext je MSE PFN vždy nižší než pro GP, ale ta situace se rychle mění, jakmile se zvětší počet kontext bodů. PFN trénovaný na distribuci HP pravděpodobně dělá implicitní marginalizaci — průměruje predikce přes různé možné HP."*

> *"S 5 body je lengthscale špatně identifikovatelný, takže GP s fixním může být horší než průměr přes více možných hodnot tohoto hyperparametru. Pro $n_{\text{context}} = 10+$ už GP vyhrává."*

![MSE vs n_context — 100-epoch model](figures/GP1_big_model/fig_big_06_context_size_mse.png)
*Obrázek 10: Experiment 4 — MSE PFN (modrá) a GP oracle (zelená přerušovaná) jako funkce $n_{\text{context}}$ (100-epoch model, $\ell=0.3$). Error bary = ±1 std přes 20 realizací. Obě křivky klesají s rostoucím $n$; GP oracle mírně vede od $n=10$, protože zná správné $\ell$ a PFN ho musí odhadnout z dat.*

![Predikce pro různé n_context — 100-epoch model](figures/GP1_big_model/fig_big_06b_context_size_grid.png)
*Obrázek 11: Experiment 4 — grid $4 \times 1$ vizualizací predikcí pro $n \in \{5, 10, 15, 20\}$ (100-epoch model). V každém řádku: červené body = kontextová data dostupná modelu; šedé body = skutečné (neviděné) hodnoty GP realizace; modrá čára = PFN predikce; zelená přerušovaná = GP posterior se správným $\ell$. Sdílená GP realizace — všechny řádky vychází ze stejné funkce, pouze s různě velkým výřezem kontextu. S $n=5$ PFN i GP mají velkou nejistotu a predikce se liší; od $n=15$ obě křivky téměř splývají a sledují tvar GP realizace.*

![MSE vs n_context — random-HP model](figures/GP1_random_model/fig_rand_05_context_size_mse.png)
*Obrázek 12: Experiment 4 — MSE PFN (modrá) a GP oracle (zelená přerušovaná) jako funkce $n_{\text{context}}$ (random-HP model, $\ell=0.3$). Error bary = ±1 std přes 20 realizací. Rozptyl je větší než u 100-epoch modelu (zvláště pro malé $n$), protože random-HP model nebyl trénován na fixním $\ell=0.3$ — musí ho identifikovat z dat při každé inferenci. Pro $n=20$ obě metody konvergují.*

![Predikce pro různé n_context — random-HP model](figures/GP1_random_model/fig_rand_05b_context_size_grid.png)
*Obrázek 13: Experiment 4 — grid $4 \times 2$ pro random-HP model. **Levý sloupec** (predikce): červené body = kontextová data; šedé body = skutečné neviděné hodnoty; modrá čára = PFN; zelená přerušovaná = GP posterior se správným $\ell=0.3$. Sdílená GP realizace — stejná funkce ve všech řádcích. **Pravý sloupec** (attention vs RBF): pro každé $n_{\text{ctx}}$ zobrazuje attention váhy poslední vrstvy z kontextového bodu nejblíže $x=0.5$ (označen červenou přerušovanou čárou) na všechny ostatní kontextové body (modré sloupce) vs normalizovaný RBF kernel $k(x_{\text{anchor}}, x_j)$ (zelené sloupce). Pokud by PFN implementoval kernel smoothing, oba sloupce by si měly odpovídat — v praxi se liší, attention je ostřejší a nelokální. S rostoucím $n$ (více kontextových bodů) prediction křivky konvergují k GP posterior a attention se stává strukturovanější.*

---

### 3.5 Experiment 5 — Přesnost PFN napříč lengthscale hodnotami

*(Pouze random-HP model, $n_{\text{context}} = 30$, průměr přes 5 seedů × 10 realizací = 50 instancí na $\ell$)*

**Výsledky — MSE(PFN, GP) pro různé $\ell$:**

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp5_rand"]`.  
> Funkce `run_exp5(rand_model, seed, ls_list=[0.05,0.1,0.2,0.3,0.5,0.7,0.9], n_ctx=30, n_trials=10)`, 5 seedů × 10 realizací = **50 hodnot** na každé $\ell$.  
> Pro každou realizaci: vygenerovat GP($\ell$, noise=0.001) sekvenci délky 100, prvních 30 jako train, zbytek jako test. `MSE` = `mean((pfn_mean − gp_mean)²)`. `Medián` robustnější vůči outlierům než průměr.

| $\ell$ | Medián MSE | Průměr MSE | Std MSE |
|---|---|---|---|
| 0.05 | 0.01317 | 0.01918 | 0.01767 |
| 0.10 | 0.000604 | 0.002773 | 0.009616 |
| 0.20 | 0.000102 | 0.002304 | 0.010984 |
| 0.30 | 4.57×10⁻⁵ | 2.18×10⁻⁴ | 7.40×10⁻⁴ |
| 0.50 | 1.99×10⁻⁵ | 2.86×10⁻⁵ | 2.42×10⁻⁵ |
| 0.70 | 1.03×10⁻⁵ | 1.52×10⁻⁵ | 1.90×10⁻⁵ |
| 0.90 | 9.18×10⁻⁶ | 1.23×10⁻⁵ | 9.99×10⁻⁶ |

Medián klesá monotónně s rostoucím $\ell$ — o ~22× z $\ell=0.05$ na $\ell=0.1$, dále o dalších řád. Průměr je výrazně vyšší než medián pro malé $\ell$ (a pro $\ell=0.2$ ), což odráží existence výrazných outlierů — ojediněle špatné instance táhnou průměr nahoru. Pro $\ell \geq 0.5$ jsou průměr a medián velmi blízké (nízká variabilita).

**Interpretace:**

Rychle oscilující funkce ($\ell = 0.05$) vyžaduje hustou vzorkovací mřížku — 30 bodů v $[0, 1]$ nestačí pro správnou identifikaci $\ell$. Zároveň trénovací distribuce $\ell \sim U(0.05, 1.0)$ přiřazuje $\ell < 0.1$ jen 5 % pravděpodobnostní hmoty, takže model má málo trénovacích příkladů pro extrémně krátké korelační délky.

> *"Čím divočejší je ta funkce, tím hůř PFN odhaduje lengthscale. Nepomáhá ani přidání množství kontext bodů."*

![MSE napříč lengthscale hodnotami](figures/GP1_random_model/fig_rand_06_ls_accuracy.png)
*Obrázek 14: Experiment 5 — MSE(PFN, GP) jako funkce $\ell$ (random-HP model, $n_{\text{context}}=30$). Krabicové grafy ukazují distribuci přes 50 instancí. Výrazný nárůst MSE pro $\ell=0.05$.*

![Predikce pro různé lengthscale hodnoty](figures/GP1_random_model/fig_rand_06b_ls_predictions.png)
*Obrázek 15: Experiment 5 — vizualizace predikcí PFN pro různé $\ell$. Pro malé $\ell$ PFN "přehlazuje" rychlé oscilace.*

---

### 3.6 Experiment 6 — Identifikace hyperparametrů z kontextových dat

*(Pouze random-HP model, průměr přes 5 seedů × 10 instancí = 50 měření)*

**Design:** Data z GP($\ell = 0.1$) resp. GP($\ell = 0.2$). PFN dostane $n_{\text{context}} \in \{3, 5, 10, 20, 40, 60\}$ bodů. Srovnáváme MSE(PFN, GP_correct) vs MSE(PFN, GP_wrong), kde `GP_wrong` používá $\ell_{\text{wrong}} = 0.7$ (resp. $0.05$).

#### 3.6.1 Test s $\ell = 0.1$ (krátká korelační délka)

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp6_ls0.1"]`.  
> Funkce `run_exp6(rand_model, seed, test_ls=0.1, wrong_ls=0.7, n_ctx_list=[3,5,10,20,40,60], n_trials=10)`, 5 seedů × 10 realizací = **50 hodnot** na každé $n_{\text{ctx}}$.  
> `GP_correct` = GP posterior s $\ell=0.1$ (správné HP); `GP_wrong` = GP posterior s $\ell=0.7$ (špatné HP). Obě sdílejí stejná trénovací data. `MSE` = `mean((pfn_mean − gp_X_mean)²)` přes 30 testovacích bodů. `Poměr` = `mean(GP_wrong) / mean(GP_correct)`.

| $n_{\text{context}}$ | MSE(PFN, GP_correct) | MSE(PFN, GP_wrong) | Poměr wrong/correct |
|---|---|---|---|
| 3  | 0.1239 ± 0.1047 | 1.846 ± 4.437 | **14.9×** |
| 5  | 0.1043 ± 0.1226 | 1.403 ± 6.431 | **13.4×** |
| 10 | 0.0483 ± 0.0579 | 0.812 ± 2.033 | **16.8×** |
| 20 | 0.00855 ± 0.01269 | 0.2665 ± 0.1725 | **31.2×** |
| 40 | 0.000629 ± 0.00149 | 0.2799 ± 0.1707 | **444.7×** |
| 60 | 0.000134 ± 0.000111 | 0.2266 ± 0.1447 | **1 691×** |

#### 3.6.2 Test s $\ell = 0.2$ (střední korelační délka)

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp6_ls0.2"]`.  
> Stejný protokol jako §3.6.1, ale `test_ls=0.2`, `wrong_ls=0.05`.

| $n_{\text{context}}$ | MSE(PFN, GP_correct) | MSE(PFN, GP_wrong) | Poměr wrong/correct |
|---|---|---|---|
| 3  | 0.0610 ± 0.0563 | 0.5499 ± 0.9273 | **9.0×** |
| 5  | 0.0613 ± 0.1150 | 0.4576 ± 2.020 | **7.5×** |
| 10 | 0.0200 ± 0.0415 | 0.2319 ± 0.4873 | **11.6×** |
| 20 | 0.00170 ± 0.00409 | 0.04528 ± 0.03577 | **26.6×** |
| 40 | 8.43×10⁻⁵ ± 1.27×10⁻⁴ | 0.04148 ± 0.04236 | **491.8×** |
| 60 | 3.78×10⁻⁵ ± 3.13×10⁻⁵ | 0.03028 ± 0.02328 | **801.4×** |

**Interpretace:**

Při $n = 3$ je poměr wrong/correct 9–15× — PFN s tak malým kontextem musí kompromisovat mezi různými hodnotami $\ell$ a obě GP varianty jsou vzdálené. Od $n = 20$ roste poměr super-lineárně s $n$, dosahuje 1 691× při $n=60$ pro $\ell=0.1$. Identifikace $\ell=0.2$ je snazší (vzor korelace je výraznější), ale rozdíl není dramatický.

Klíčový fakt: MSE(PFN, GP_correct) klesá exponenciálně s $n$ — při $n=60$ je 0.000134 (téměř nulová chyba), zatímco MSE(PFN, GP_wrong) zůstává konstanta ~0.23 pro $\ell=0.1$. To znamená, že PFN se od $n \approx 20$ chová téměř identicky jako GP se správným $\ell$.

> *"Pokud PFN identifikoval LS z dat, jeho predikce bude blíž zelené (nízké MSE) a daleko od červené (vysoké MSE). Pokud neidentifikoval, obě MSE budou podobné."*

![HP identifikace — LS=0.1](figures/GP1_random_model/fig_rand_07a_hp_ident_ls01.png)
*Obrázek 16: Experiment 6 — MSE(PFN, GP_correct) a MSE(PFN, GP_wrong) jako funkce $n_{\text{context}}$ pro $\ell=0.1$. Od $n=20$ poměr wrong/correct exponenciálně roste: PFN jednoznačně identifikuje správné $\ell$.*

![HP identifikace — LS=0.2](figures/GP1_random_model/fig_rand_07b_hp_ident_ls02.png)
*Obrázek 17: Experiment 6 — stejné jako Obrázek 16, ale pro $\ell=0.2$. Identifikace je snazší: poměr 801× při $n=60$.*

---

### 3.7 Experiment 7 — PFN vs Type-II maximální věrohodnost

*(Pouze random-HP model, $n_{\text{context}} = 20$, průměr přes 5 seedů × 10 realizací = 50 instancí)*

Porovnání vůči True GP (se správným $\ell$) jako referenci.

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp7_rand"]`.  
> Funkce `run_exp7(rand_model, seed, n_ctx=20, n_rep=10)`, 5 seedů × 10 realizací = **50 hodnot** na HP scénář.  
> `ML-II` = `sklearn.GaussianProcessRegressor(kernel=C(1)*RBF(0.3)+White(1e-3), n_restarts_optimizer=3)`, fitován na trénovacích datech, predikce na testovacích. `True GP` = GP se správnými HP (reference). Čas měřen `time.perf_counter()` v milisekundách, průměr přes 50 instancí.

| HP scénář | MSE(PFN, True GP) | MSE(ML-II, True GP) | Čas PFN | Čas ML-II | Vítěz |
|---|---|---|---|---|---|
| LS=0.1, noise=0.001 | 0.00588 ± 0.00783 | **0.00223 ± 0.00840** | 16.3 ms | 21.1 ms | ML-II |
| LS=0.3, noise=0.001 | 0.000553 ± 0.00224 | **0.000318 ± 0.00126** | 15.3 ms | 17.3 ms | ML-II |
| LS=0.7, noise=0.001 | 0.0000509 ± 0.000103 | 0.0000708 ± 0.000156 | 14.6 ms | 16.3 ms | **PFN** |
| LS=0.3, noise=0.01 | 0.000945 ± 0.00156 | **0.000701 ± 0.00155** | 15.7 ms | 17.4 ms | ML-II |

**Pozorování:**

ML-II s `n_restarts_optimizer=3` je přesnější ve třech ze čtyř scénářů při $n = 20$. Pro LS=0.7 (nejhladší funkce, nejsnazší identifikace) PFN mírně překonává ML-II. PFN je ve všech případech **rychlejší** než ML-II (14–16 ms vs 17–21 ms), ale rozdíl je malý (1.1–1.3×) — nikoliv řády, jak by se mohlo předpokládat.

Toto lze vysvětlit kontextem: při $n = 20$ se znalostí správného $\ell$ má GP-ML dostatek dat pro přesnou identifikaci HP přes marginální věrohodnost. PFN s MSE 0.0001–0.006 stále nese chybu z přibližné HP identifikace. Experiment 8 ukazuje, že situace se obrátí pro $n \leq 5$.

![PFN vs ML-II](figures/GP1_random_model/fig_rand_08_pfn_vs_ml2.png)
*Obrázek 18: Experiment 7 — MSE PFN a ML-II pro různé HP scénáře ($n_{\text{context}}=20$). ML-II vítězí ve 3/4 scénářů při tomto $n$.*

---

### 3.8 Experiment 8 — Marginalizace vs bodový odhad

*(Pouze random-HP model, $\ell = 0.3$, průměr přes 5 seedů × 5 realizací = 25 instancí)*

Tři strategie inference porovnány přes různé $n_{\text{context}}$:

> **Zdroj dat:** `background/collect_stats.py` → `stats_averaged.json["exp8_rand"]`.  
> Funkce `run_exp8(rand_model, seed, true_ls=0.3, n_ctx_list=[3,5,10,20,50], n_trials=5)`, 5 seedů × 5 realizací = **25 hodnot** na každé $n_{\text{ctx}}$.  
> `Marginalizace` = průměr GP predikcí přes 20 rovnoměrně rozložených hodnot $\ell \in [0.05, 1.0]$ s uniformními váhami — `mean([gp_predict(ls=l) for l in ls_grid])`. Reference `True GP` = GP se správným $\ell=0.3$.

| $n_{\text{context}}$ | MSE(PFN, True GP) | MSE(ML-II, True GP) | MSE(Marginalizace, True GP) | Vítěz |
|---|---|---|---|---|
| 3  | **0.03795 ± 0.04406** | 0.10500 ± 0.19763 | 0.04766 ± 0.05807 | **PFN** |
| 5  | 0.01991 ± 0.03694 | 0.01797 ± 0.03030 | **0.01691 ± 0.02502** | ≈ Marg/ML-II |
| 10 | 0.00205 ± 0.00235 | **0.000981 ± 0.00236** | 0.00666 ± 0.00888 | **ML-II** |
| 20 | 0.000171 ± 0.000225 | **3.09×10⁻⁵ ± 4.30×10⁻⁵** | 0.00297 ± 0.00237 | **ML-II** |
| 50 | 0.00124 ± 0.00595 | **6.07×10⁻⁶ ± 1.00×10⁻⁵** | 0.00213 ± 0.00227 | **ML-II** |

**Interpretace:**

Výsledky ukazují složitější obraz než by naznačoval jediný seed:

1. **PFN dominuje pouze pro $n = 3$** — při extrémně malém kontextu ML-II není schopen spolehlivě identifikovat $\ell$ a jeho MSE je 2.8× horší. PFN je blíže marginalizaci (implicitní průměrování přes $p(\ell)$).

2. **Při $n = 5$ jsou metody přibližně srovnatelné** — numerická marginalizace je mírně nejlepší (0.017), ML-II a PFN blízko sebe.

3. **Od $n = 10$ ML-II jasně vede** — při dostatečném kontextu dokáže ML-II spolehlivě identifikovat $\ell = 0.3$ a dosahuje téměř oracle kvality. Při $n = 50$ je ML-II 204× lepší než PFN (6×10⁻⁶ vs 1.24×10⁻³).

4. **Numerická marginalizace je konzistentně horší než PFN i ML-II** pro $n \geq 10$ — uniformní průměrování přes $\ell \in [0.05, 1.0]$ je suboptimální, pokud data obsahují informaci o $\ell$.

Klíčové poučení: **PFN má výhodu pouze tehdy, kdy identifikace $\ell$ je nespolehlivá** (malé $n$, vysoká šumovost). Jakmile je $n$ dostatečné pro ML-II konvergenci, ML-II dosahuje lepší přesnosti.

> *"PFN trénovaný na distribuci HP pravděpodobně dělá implicitní marginalizaci — průměruje predikce přes různé možné HP. Marginalizace je teoreticky optimální strategie při nejistotě o HP."*

![Marginalizace vs PFN vs ML-II](figures/GP1_random_model/fig_rand_09_marginalization.png)
*Obrázek 19: Experiment 8 — MSE tří strategií inference jako funkce $n_{\text{context}}$. PFN vítězí při $n=3$; od $n=10$ ML-II dosahuje nejnižšího MSE. Numerická marginalizace je konzistentně suboptimální.*

---

### 3.9 Experiment 6 (big model) — Analýza jednotlivých attention hlav

**Design:** Systematický scan všech $6 \times 8 = 48$ attention hlav. Průměr přes 5 realizací. Pro každou hlavu počítáme: korelaci attention vah (Test→Train) s normalizovanými RBF vahami a MSE predikce jedné hlavy.

#### 3.9.1 Detailní výsledky pro vybrané hlavy

> **Zdroj dat:** Výstupy notebooků `experiments/Experiments_from_GP1_big_model.ipynb` (cell 18), přečteny přímo z vytištěných tabulek ve výstupu.  
> Pro každou ze 48 hlav: attention matice $A \in \mathbb{R}^{120 \times 120}$ spočtena z uložených vah modelu, predikce hlavy = `A[test, train] @ train_y`. `MSE(Head, GP)` = `mean((head_pred − gp_mean)²)`. Celý PFN běžel standardním forward passem. Výsledky nejsou průměrovány přes seedy (jedná se o výstup z jediné spuštěné buňky notebooku).

| Vrstva | Hlava | Corr(Attn, RBF) | MSE(Head, NW) | MSE(Head, PFN) | MSE(Head, GP) | MSE(NW, GP) | MSE(PFN, GP) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.4619 | 0.2499 | 0.4957 | 0.4955 | 0.1370 | **0.000011** |
| 0 | 3 | 0.1818 | 0.2404 | 0.5305 | 0.5306 | 0.1370 | **0.000011** |
| 0 | 7 | 0.0549 | 0.2942 | 0.5957 | 0.5957 | 0.1370 | **0.000011** |
| 5 | 0 | 0.2742 | 0.3241 | 0.5807 | 0.5804 | 0.1370 | **0.000011** |
| 5 | 3 | 0.3559 | 0.3771 | 0.1325 | 0.1326 | 0.1370 | **0.000011** |
| 5 | 7 | 0.1251 | 0.2680 | 0.3660 | 0.3650 | 0.1370 | **0.000011** |

**Klíčová čísla:**
- **MSE(Full PFN, GP) = 0.000011** — konstantní pro všechny řádky (celý model)
- **MSE(NW, GP) = 0.137010** — referenční chyba NW estimátoru
- **Poměr MSE(NW, GP) / MSE(PFN, GP) = 12 455×**

Žádná izolovaná hlava se ani nepřiblíží výkonu celého modelu. Nejlepší hlava (vrstva 5, hlava 3) dosahuje MSE(Head, GP) = 0.1326, stále 12 000× horší než celý PFN.

#### 3.9.2 Srovnání korelací Attn vs RBF — přehled 48 hlav

> **Zdroj dat:** Stejný výstup notebooku jako §3.9.1 — vytištěná tabulka korelací ze cell 18. Hodnoty v tabulce jsou průměry a maxima přes 8 hlav dané vrstvy, odečteny vizuálně z výstupu.

Nejlepší korelace: vrstva 3, hlava 1 (corr = **0.5559**). Žádná hlava v žádné vrstvě nepřesahuje 0.56. Korelace je vesměs nízká (0.05–0.55) a nepravidelná — neexistuje vrstva, která by systematicky aproximovala RBF kernel.

| Vrstva | Průměrná korelace (8 hlav) | Max korelace |
|---|---|---|
| 0 | ~0.27 | 0.555 (hlava 6) |
| 1–4 | ~0.20–0.35 | 0.40–0.56 |
| 5 | ~0.22 | 0.356 (hlava 3) |

> *"Jednotlivé attention hlavy nejsou Nadaraya-Watson estimátory. Žádná hlava nepoužívá RBF kernel (nízká korelace). PFN nelze dekomponovat na 'kernel smoothing + korekce' na úrovni jednotlivých hlav. Větší model distribuuje výpočet jemněji, žádná hlava sama nedělá interpretovatelnou operaci, ale celek dává téměř perfektní GP posterior."*

![Detail attention hlavy L0H0](figures/GP1_big_model/fig_big_07a_head_detail_L0H0.png)
*Obrázek 20: Analýza hlavy — vrstva 0, hlava 0. Korelace s RBF = 0.46 (nejvyšší v této vrstvě).*

![Detail attention hlavy L5H3](figures/GP1_big_model/fig_big_07e_head_detail_L5H3.png)
*Obrázek 21: Analýza hlavy — vrstva 5, hlava 3. Nejlepší jednotlivá hlava: MSE(Head, GP) = 0.1326, stále 12 000× horší než celý model.*

![Scan všech 48 hlav — heatmapa](figures/GP1_big_model/fig_big_07g_head_scan_heatmap.png)
*Obrázek 22: Systematický scan všech 48 attention hlav — heatmapa korelace Attn vs RBF. Žádný konzistentní vzor napříč vrstvami nebo hlavami; výpočet je distribuován nelineárně.*

---

## 4. Souhrnná srovnávací tabulka

### 4.1 Numerický přehled klíčových metrik (průměr přes 5 seedů)

| Experiment | Metrika | 100-epoch model | Random-HP model |
|---|---|---|---|
| **Exp 1** | Corr(PFN std, GP std) | **0.9812 ± 0.0252** | **0.9734 ± 0.0292** |
| **Exp 1** | MSE(PFN std, GP std) | 0.00376 ± 0.00911 | 0.03478 ± 0.03461 |
| **Exp 1** | Ratio far/near — PFN | 21.3× ± 5.2× | 7.49× ± 2.20× |
| **Exp 1** | Ratio far/near — GP | 25.9× ± 3.8× | 25.9× ± 3.8× |
| **Exp 3** | Corr(Attn, RBF) | **0.374 ± 0.270** | **0.316 ± 0.311** |
| **Exp 3** | MSE(PFN, GP) | **0.000167 ± 0.000096** | — |
| **Exp 3** | MSE(NW, GP) | **0.2099 ± 0.1462** | — |
| **Exp 3** | Poměr MSE(NW)/MSE(PFN) | **~1 256×** | — |
| **Exp 4** | MSE(PFN, true y) při $n=5$ | 0.0408 ± 0.0698 | 0.0598 ± 0.0958 |
| **Exp 4** | MSE(GP, true y) při $n=5$ | **0.0340 ± 0.0635** | **0.0340 ± 0.0635** |
| **Exp 4** | MSE(PFN, true y) při $n=20$ | 0.0021 ± 0.0025 | 0.0020 ± 0.0013 |
| **Exp 5** | MSE(PFN, GP) při $\ell=0.05$ | — | 0.01317 (med) |
| **Exp 5** | MSE(PFN, GP) při $\ell=0.30$ | — | 4.57×10⁻⁵ (med) |
| **Exp 6** | Poměr wrong/correct při $n=60$, $\ell=0.1$ | — | **1 691×** |
| **Exp 8** | MSE(PFN, True GP) při $n=3$ | — | **0.03795** (PFN vyhrává) |
| **Exp 8** | MSE(ML-II, True GP) při $n=50$ | — | **6.07×10⁻⁶** (ML-II vyhrává) |
| **Exp 9 (hlava)** | MSE(Full PFN, GP) | **0.000011** | — |
| **Exp 9 (hlava)** | MSE(Best head, GP) | 0.1326 | — |
| **Exp 9 (hlava)** | Poměr MSE(best head)/MSE(PFN) | **12 055×** | — |

### 4.2 Testování hypotéz

| Hypotéza | Výsledek | Klíčový důkaz |
|---|---|---|
| PFN ≈ Nadaraya-Watson | **ODMÍTNUTA** | MSE(NW, GP) = 1 256 × MSE(PFN, GP) |
| PFN implementuje $k(x^*, X) K^{-1} y$ | **POTVRZENA** | Nelokální attention; poměr MSE |
| Attention je dynamický kernel | **POTVRZENA** | Nízká korelace Attn vs RBF (0.32–0.37) |
| Kauzální asymetrie Train/Test | **POTVRZENA** | Train→Test ≈ 0 v poslední vrstvě |
| PFN provádí Bayesovskou marginalizaci (malé $n$) | **POTVRZENA** | Exp 8: PFN ≪ ML-II při $n=3$ |
| PFN dominuje ML-II pro všechna $n$ | **ODMÍTNUTA** | Exp 8: ML-II vede od $n \geq 10$ |
| Random-HP model identifikuje $\ell$ z dat | **POTVRZENA** | Poměr wrong/correct = 1 691× při $n=60$ |
| Individuální hlavy = NW | **ODMÍTNUTA** | Nejlepší hlava: 12 055× horší než celý PFN |

---

## 5. Diskuze

### 5.1 Proč PFN neimplementuje Nadaraya-Watson

NW estimátor je optimální za předpokladu, že $K^{-1} \approx I$, tj. že trénovací body jsou vzájemně nekorelované. Pro RBF kernel jsou blízké body silně korelované — NW jim přiděluje zbytečně velkou kumulativní váhu. Správná GP inference tuto redundanci odstraňuje přes $K^{-1}$. Rozdíl MSE je ~1 256× — jednoznačný experimentální důkaz, že PFN implementuje operaci přibližující se $K^{-1}$.

### 5.2 Specializace vrstev vs. distribuovaný výpočet

Pro malý podtrénovaný model (20 epoch) se jevila interpretace "rané vrstvy = kernel smoothing, poslední vrstva = korekce na $K^{-1}$" jako částečně platná. Pro větší, déle trénovaný model tato jednoduché dělení neplatí — žádná jednotlivá hlava neprovádí interpretovatelnou operaci a MSE(best head, GP) = 0.1326 vs MSE(PFN, GP) = 0.000011 ukazuje, že výpočet je distribuován nelineárně přes všechny vrstvy a hlavy.

> *"Celkově se to dá shrnout takto: první vrstvy (0–4): kernel smoothing s naučenými kernely (≈ NW, ale ne s RBF). Pozdější vrstvy (5): korekce z NW na plný GP posterior (≈ efekt $K^{-1}$)."*

### 5.3 Přehodnocení výhody PFN nad ML-II

Experimenty 7 a 8 dávají nuancovaný obraz:

- **PFN vyhrává při extrémně malém $n$ ($n=3$)**: ML-II nemá dostatek dat pro spolehlivou identifikaci $\ell$ a PFN jako implicitní marginalizátor je lepší.
- **Přechodová oblast ($n \approx 5$)**: metody jsou přibližně srovnatelné.
- **Od $n \geq 10$ ML-II jasně vede**: správná identifikace HP přes maximální věrohodnost je efektivnější než amortizovaná inference.
- **Speedup PFN**: pouze 1.1–1.3×, ne řády.

Výhoda PFN je tedy specificky v situacích **extrémně malého kontextu nebo neznámých HP** — což je právě scénář, pro který byl navržen (Müller et al., 2022).

### 5.4 Kalibrace variance — neočekávaná asymetrie modelů

Random-HP model (500 epoch) dosahuje výrazně horšího MSE(std) = 0.035 oproti 100-epoch modelu (0.004), přestože trénoval déle. Možná vysvětlení:
1. Trénování na širší distribuci $\ell$ způsobuje "průměrování" variance profilu — model nemůže být kalibrovaný pro všechny $\ell$ současně.
2. Přetrénovávání (500 epoch může vést k přizpůsobení na trénovací distribuci HP, nikoliv na konkrétní testovací $\ell = 0.3$).

### 5.5 Otevřené otázky

1. **Kalibrace variance:** Proč 500-epoch random-HP model dosahuje horšího MSE(std)? Je to efekt širší trénovací distribuce, nebo přetrénování?
2. **Která část attention kóduje HP vs. hodnoty $y$?** Trénovací bod $(x_i, y_i)$ je zakódován jako jeden token — jak jsou tyto dvě složky separovány?
3. **Škálování hloubky modelu:** Jak se mění výsledky pro 1, 2, 4, 8-vrstvé modely? Testováno v Experimentu 2 (notebook `Experiment_2_from_GP2.ipynb`).

---

## 6. Závěr

Série osmi experimentů na dvou modelech PFN přináší konzistentní obraz výpočetní strategie, kterou transformer zvolil:

1. **Střední hodnota je téměř perfektní.** MSE(PFN, GP) = 0.000167 ± 0.000096 vs MSE(NW, GP) = 0.2099 — PFN je ~1 256× přesnější než triviální kernel-averaging.
2. **Variance je správně tvarovaná, ale špatně kalibrovaná.** Korelace r > 0.97, ale dynamický rozsah je 82 % (100-ep) resp. 29 % (rand-HP) hodnoty GP.
3. **PFN neimplementuje Nadaraya-Watson.** Potvrzeno třemi nezávislými metodami: nízká korelace Attn vs RBF (0.32–0.37), poměr MSE 1 256×, analýza 48 hlav (nejlepší hlava 12 000× horší než celek).
4. **Kauzální asymetrie je automaticky naučená.** Transformer bez explicitního zakódování odhalil, že testovací body závisí na trénovacích, nikoliv naopak.
5. **Random-HP model identifikuje $\ell$ z kontextu** — poměr MSE(wrong)/MSE(correct) roste exponenciálně s $n$, dosahuje 1 691× při $n=60$ pro $\ell=0.1$.
6. **PFN dominuje nad ML-II pouze pro $n \leq 3$–5.** Od $n \geq 10$ ML-II dosahuje nižšího MSE; při $n=50$ je ML-II 204× lepší. PFN je 1.1–1.3× rychlejší, ne řády.
7. **Individuální attention hlavy nejsou interpretovatelné.** Výpočet $K^{-1}$ je distribuován přes všechny vrstvy a hlavy nelineárně.

---

## Reference

- Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2022). *Transformers Can Do Bayesian Inference*. ICLR 2022.
- Williams, C. K., & Rasmussen, C. E. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141–142.
- Watson, G. S. (1964). Smooth regression analysis. *Sankhyā: The Indian Journal of Statistics, Series A*, 26(4), 359–372.
- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). *Efficient Streaming Language Models with Attention Sinks*. arXiv:2309.17453 (ICLR 2024).

---

*Experimenty provedeny na Apple M-series (MPS backend), PyTorch 2.10.0, sklearn 1.x.*  
*Zdrojové notebooky: `experiments/Experiments_from_GP1_big_model.ipynb`, `experiments/Experiments_from_GP1_random_model.ipynb`.*  
*Statistiky průměrovány přes 5 seedů skriptem `background/collect_stats.py` → `background/stats_averaged.json`.*  
*Obrázky extrahované z výstupů notebooků skriptem `background/extract_figures.py` → `figures/`.*
