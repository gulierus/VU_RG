# Workflow: PFN pro segmentaci — Varianty A a C

Podrobný návod ke dvěma navazujícím experimentům. **Varianta A** diagnostikuje patologie
amortizované inference na *existujícím* in-context segmenteru (UniverSeg); **Varianta C**
staví *skutečný* PFN na syntetickém prioru s dostupným oraclem a měří tytéž patologie
rigorózně proti pravdě. A je levný falsifikační test premisy; C je vědecké jádro a přímé
pokračování GP2 do 2D binárního targetu.

---

## 0. Slovník mapování GP2 → segmentace

Aby byl zbytek čitelný, fixuji překlad pojmů, se kterými pracuješ v GP2, do segmentačního
světa:

| GP2 (regrese) | Segmentace (Varianty A/C) |
|---|---|
| target `y ∈ ℝ` | maska `y ∈ {0,1}^{H×W}` (per-pixel label) |
| PPD `p(y* \| x*, D)` = Gaussovská hustota | per-pixel `p(y*_ij = 1 \| image, support)` = Bernoulli |
| support/context `D = {(x_i, y_i)}` | support set `S = {(image_k, mask_k)}` |
| BarDistribution NLL | per-pixel binary cross-entropy (BCE) |
| Easy/Hard přes lengthscale ℓ | Easy/Hard přes ℓ latentního GP (viz C) |
| bias k prioru daleko od dat | posun predikované masky k marginální (prior) masce |
| var → 0 při O(n^{−1/2}) | totéž per pixel s rostoucím `n_supp` |

Klíčové pro celý dokument: **maska** je matice stejného rozměru jako obrázek, per pixel
0/1. Model nevydává tvrdou masku, ale per-pixel pravděpodobnost `p̂_ij ∈ [0,1]`; prahování
(typicky 0,5) z ní dělá tvrdou masku. Objekt kalibrace je právě `p̂_ij`.

---

# ČÁST I — VARIANTA A

## A.1 Myšlenka

UniverSeg je amortizovaný in-context segmenter: podmíní se na support set dvojic
(obrázek, maska) a jedním forward passem segmentuje query obrázek, bez retrénování. To je
**strukturálně tatáž myšlenka jako PFN** (amortizuj přes úlohy, inferuj z kontextu), i když
UniverSeg *není* PFN — netrénuje se KL kritériem na syntetických tazích z explicitního
priora, nýbrž supervizovaně na MegaMedical (53 datasetů, 26 domén, 16 modalit).

Naglerova teorie (Thm 5.4, locality condition) tvrdí, že jakýkoli amortizovaný prediktor
s **globální** (softmax) attention má neredukovatelný bias: daleko od dat se predikce vrací
k prioru, protože nenulová váha zůstává i na vzdálených bodech. UniverSeg váhuje support
přes CrossBlock (feature-averaging mechanismus s globálním dosahem), takže **predikce A je:
i tento ne-PFN model musí dědit bias k prioru, protože sdílí architektonickou příčinu.**

Pokud se to potvrdí, jev **není** artefakt PFN trénovací procedury, ale vlastnost
amortizace/globální attention obecně. To je silnější tvrzení, než jaké bys kdy prokázal
na vlastním PFN samotném (jeden model nerozliší „vlastnost architektury" od „vlastnost mé
konkrétní tréninkové procedury").

## A.2 Cíl

Falsifikovatelně změřit dvě věci, **striktně odděleně**, jako funkci degradace vstupní
informace:

1. **Bias / kolaps k prioru:** posouvá se predikovaná maska k marginální (prior) masce,
   když se support set zmenšuje nebo posouvá do OOD?
2. **Kalibrace:** jak se mění over-/under-confidence per-pixel pravděpodobností — **směr
   NEPŘEDPOKLÁDEJ, změř ho.** (Původní survey chyboval, když tvrdil „variance is
   under-estimated" jako univerzálii; u tebe je kalibrace přes trénink nestabilní.)

Výstupní tvrzení, které chceš mít podložené: *„In-context segmentery dědí prior-collapse
předpovězený PFN teorií; kalibrace se chová takto [směr změřen empiricky]."*

## A.3 Setup

**Model.** Předtrénovaný UniverSeg, veřejné váhy (github.com/JJGO/UniverSeg), zmražený.
Netrénuje se nic. Pozor: nativní inference UniverSeg průměruje přes `K` náhodných support
setů, `ŷ = 1/K Σ f_θ(x, S_k)`. Pro čistotu experimentu **toto průměrování vypni** nebo ho
drž fixní (K=1), jinak zaměníš efekt velikosti support setu s efektem ensemblingu přes
support sety.

**Data — rozhodnutí.** Kotevní dataset je **WBC (bílé krvinky)**, jako nejjednodušší
možnost; **mozková MRI** (např. OASIS) je záložní varianta, pokud by WBC nešel použít
(chybějící/nekompatibilní veřejné anotace, problém se zarovnáním apod.).

Zdůvodnění volby WBC jako primárního:
- **Nejčistší prior maska.** Buňka je centrovaná, kompaktní blob → per-pixel marginální
  foreground `E[y]` (viz A.5) je smysluplná bez netriviální registrace. To je tvrdé
  omezení metriky kolapsu (i): funguje jen na **prostorově zarovnaných** datech.
- **Minimum confoundů.** Jednoduchá geometrie, jedna dominantní struktura, žádné tenké
  rozvětvené útvary (na rozdíl od cév sítnice, kde by prior maska byla nesmysl).
- **Held-out eval set UniverSeg** → veřejné váhy si na něm vedou dobře a existuje reference.

Záložní **mozková MRI** (komory / bílá hmota, roughly registrované): prior maska je stále
rozumná, struktur je víc, takže mírně bohatší, ale i mírně zašuměnější signál. Použij ji,
jen pokud WBC selže na úrovni dat.

Tři vrstvy vůči tréninku UniverSeg (definují **OSU 2 — OOD-ness**, viz A.4):
- *In-distribution / mírné OOD (kotva):* WBC (příp. mozková MRI) — held-out, ale
  modalita/anatomie je v MegaMedical zastoupena; model si vede dobře.
- *Silné OOD:* modalita/anatomie podreprezentovaná či chybějící v MegaMedical (prakticky
  ověřený případ je **prostate MR**, použit v navazující analýze UniverSeg právě jako test
  generalizace mimo trénink).

**Caveat k ID/OOD zařazení:** přesný train/held-out split MegaMedical si **ověř** v repu
(github.com/JJGO/UniverSeg, resp. MegaMedical) předtím, než dataset označíš za ID vs OOD —
jinak si podkopeš druhou osu.

Pro definici „prior masky" (viz A.5) obecně potřebuješ **prostorově zarovnaná** data — u WBC
a roughly-registered mozkové MRI je průměrná maska smysluplná; u nezarovnaných dat prior
maska ztrácí význam a metriku (i) musíš buď zarovnat, nebo nahradit (viz past A.7).

**Prostředí.** MacOS + PyTorch lokálně stačí (inference ~142 ms / 128×128, žádný trénink).
Pro velké sweepy případně HELIOS, ale A je záměrně levná.

## A.4 Nezávislé proměnné — dvě osy degradace

Dvě **ortogonální** osy, každou měň při fixaci druhé:

- **Osa 1 — velikost support setu `n_supp`:** sweep {1, 2, 4, 8, 16, 32, 64}. Malé
  `n_supp` = málo lokální informace o úloze → tlak k prioru.
- **Osa 2 — OOD-ness support setu:** ID → held-out → silné OOD (viz A.3). Support set
  je vzdálený v feature prostoru od toho, co model zná.

Kříž těchto os (7 × 3 = 21 buněk) je jádro designu. Míchat je do jedné proměnné je chyba
— rozdělil bys efekt řídkosti dat od efektu distribučního posunu.

## A.5 Závislé proměnné — co přesně měřit

### (i) Kolaps k prioru (bias term)

UniverSeg nemá explicitní prior, takže „prior masku" definuješ **operačně** a tu definici
obhájíš:

```
prior_mask_ij = E[y_ij]  ≈  (1/M) Σ_{m=1}^{M} mask_m,ij
```

tj. per-pixel marginální foreground pravděpodobnost přes referenční množinu `M` masek
z dané domény (po zarovnání). Je to „defaultní" maska bez informace o konkrétním query.

Pak měř **dvě vzdálenosti současně**, obě jako funkci (n_supp, OOD-ness):

- `d_truth = distance(prediction, ground_truth)` — roste s degradací.
- `d_prior = distance(prediction, prior_mask)` — klesá, pokud nastává kolaps.

Jako `distance` použij soft metriku na pravděpodobnostních mapách (per-pixel MSE nebo
symetrickou KL mezi `p̂` a prior/GT mapou), **ne** jen Dice na tvrdých maskách — Dice
zahodí informaci o důvěře, kterou právě chceš. Reportuj i Dice zvlášť jako čitelný proxy.

**Podpis kolapsu** = crossover: jak data degradují, `d_truth ↑` a `d_prior ↓`. To je přímo
bias člen `E[q(y|x,D)] − p₀(y|x)` v režimu neinformativních dat.

### (ii) Kalibrace

Nad per-pixel `p̂_ij`:
- **Reliability diagram** (binned): pro pixely s `p̂ ≈ b` změř empirickou frekvenci
  foreground; kalibrovaný model má frekvenci ≈ b.
- **ECE** (expected calibration error) jako skalární shrnutí.
- **Odděleně uvnitř objektu vs v hraničním pásmu** (dilatace−eroze GT masky) — ambiguita
  žije na hranici, průměr přes celou masku ji rozmělní.

**Směr (over/under-confidence) reportuj jako výsledek, ne jako předpoklad.**

## A.6 Statistický protokol

- **≥ 50–100 query obrázků** na buňku (n_supp × OOD), ne jeden.
- **Víc náhodných tahů support setu** na buňku (aby ses zprůměroval přes to, *které*
  konkrétní příklady support obsahuje) — jinak měříš šum konkrétního tahu.
- Reportuj **distribuce a error bars**, ne bodové hodnoty. To odpovídá kritice nízké
  statistické síly, kterou máš z GP2 Ch.3.
- Fixuj a loguj seedy tahů support setu.

## A.7 Pasti (čti před spuštěním)

- **Prior maska vyžaduje zarovnání.** Bez registrace je průměrná maska nesmysl. Pokud
  data nezarovnáš, nahraď metriku (i) posunem k *per-dataset majority masce* nebo to
  omez na roughly-registered domény.
- **Kalibrační škálování ≠ σ-scaling.** Pro binární per-pixel Bernoulli je „σ" špatně
  definovaná. Rekalibraci dělej **temperature scalingem na logitech** (jeden parametr T)
  nebo Platt scalingem, ne κ na reziduích jako v regresi.
- **NLL unit mismatch.** Nesrovnávej per-pixel BCE agregovanou přes masku jako by to byla
  jedna nezávislá veličina — sousední pixely jsou korelované, agregát není součet
  nezávislých. Používej NLL konzistentně (stejná agregace všude) a nesrovnávej s žádnou
  spojitou hustotou (tvůj známý problém z BarDistribution vs Gaussian NLL).
- **Nezaměň osy.** n_supp efekt vs OOD efekt drž oddělené.
- **UniverSeg nemá pravý prior** — celá metrika (i) je *operační* aproximace. To je
  principiální limit A a přesně důvod, proč následuje C.

## A.8 Deliverable A

Graf(y) `d_truth` a `d_prior` vs (n_supp, OOD-ness) s vyznačeným crossoverem + reliability
diagramy a ECE per buňku. Tvrzení: *„In-context segmentery dědí prior-collapse; kalibrace
se chová [směr]. Jev existuje na reálném modelu → má smysl postavit kalibrovaný PFN (C)."*

---

# ČÁST II — VARIANTA C

## C.1 Myšlenka

A ukázala, *že* jev existuje, ale měřila proti *operační* prior masce — nemá ground truth
posterior. C tuto slabinu odstraní: postaví hračku, kde je **prior explicitní** a **pravý
posterior spočitatelný oraclem**. Pak natrénuje **skutečný PFN** a měří amortizační bias
*proti pravdě*, ne proti heuristice. Je to tvůj GP2 setup zvednutý ze skalárů na 2D binární
masky.

**Generativní model úlohy (prior):**
```
ℓ, τ, σ²  ~  Π                       # hyperparametry úlohy (task) z priora
f         ~  GP(0, k_ℓ)  on H×W grid # latentní pole, k_ℓ = RBF s lengthscale ℓ
image     =  f + ε,  ε ~ N(0, σ²)    # pozorovaný "obrázek" (zašuměné f)
mask_ij   =  1[f_ij > τ]             # binární segmentace prahováním
```
Jedna úloha = fixní `(ℓ, τ, σ²)`. Prior přes úlohy = `Π` nad hyperparametry + náhodnost GP.
Každý „obrázek" v support setu i query je nezávislá GP realizace se **stejnými**
hyperparametry. Support set tedy modelu komunikuje úlohu (ℓ = hladkost, τ = kde prahovat /
kolik foreground, σ² = šum) — **amortizovaná inference hyperparametrů**, přesně jako PFN.

## C.2 Cíl

Změřit **fidelitu PFN vůči pravému posterioru** a její **rozpad** v přesně těch osách, které
znáš z GP2:
- s tvrdostí úlohy (krátké ℓ = Hard, dlouhé ℓ = Easy — přímý analog Easy/Hard),
- s OOD hyperparametrů (mimo tréninkový rozsah `Π`),
- s velikostí support setu `n_supp` (→ 0 = kolaps k prioru, teď s **explicitně definovanou**
  prior maskou).

A rozložit chybu na **bias² + variance** per pixel — analog tvého bias²+c/n rozkladu.

## C.3 Oracle — jak spočítat pravý posterior

Pro **známé** `(ℓ, τ, σ²)` a pozorovaný query image (husté zašuměné `f` na celé mřížce):

1. GP posterior nad latentním `f` na každém pixelu daný hustým zašuměným pozorováním je
   Gaussovský:
   ```
   f | image  ~  N(μ_post, Σ_post),   μ_post = K(K + σ²I)^{-1} · image
   ```
   kde `K` je H·W × H·W kernelová matice (RBF, lengthscale ℓ) na mřížce.
2. Pravá posterior-predictive maska per pixel:
   ```
   p_oracle_ij = P(f_ij > τ | image) = Φ( (μ_post,ij − τ) / s_post,ij )
   ```
   kde `s_post,ij = sqrt(Σ_post,ii)`.

Tohle je **ground truth**, proti kterému měříš PFN. `p_oracle` s **známými** hyperparametry
je „best possible given the task"; PFN musí toto dohnat, ačkoli hyperparametry **nezná** a
inferuje je ze support setu. Gap = amortizační chyba.

**Prior maska (explicitní!):** marginálně, bez pozorování, `P(f_ij > τ) = Φ(−τ / σ_f)`, kde
`σ_f` je output-scale GP. To je pravá „defaultní" maska, ke které PFN kolabuje při `n_supp
→ 0`. Na rozdíl od A je tu definována přesně.

**Náklady oraclu:** hustý GP posterior na H·W pixelech = inverze `(K + σ²I)`. Drž mřížku
malou (32×32 = 1024, nebo 48×48). Precompute Choleskyho faktorizaci per `(ℓ, σ²)`.
Krátké ℓ → špatně podmíněná `K` (velké κ) → tvoje známé κ/nugget/jitter téma; přidej
nugget pro stabilitu.

## C.4 Architektura PFN

**Doporučení: UniverSeg-lite** (zmenšený CrossBlock encoder-decoder) s **per-pixel Bernoulli
hlavou** (1 logit per pixel, sigmoid). Vstup: query image + support set (image, mask) dvojic.

Klíčový důvod pro tuto volbu: **A i C pak používají srovnatelnou architekturu**, takže
rozdíl ve výsledcích je přičitatelný **trénovací proceduře** (syntetický explicitní prior +
PPD/BCE loss u C vs MegaMedical supervize u A), ne architektuře. To je čistý kontrolovaný
kontrast.

Zásadní bod: **PFN definuje trénovací objektiv + prior, ne konkrétní architektura.** Nagler
zachází i s window smootherem a stromem jako s PFN. UniverSeg-lite trénovaný na tazích
z explicitního priora minimalizací per-pixel cross-entropy **je** validní PFN. Nemusíš
stavět čistý pixel-token transformer (drahý); stačí, aby loss a data odpovídaly PFN
receptu.

Konstanty na začátek (škáluj dle GPU): malý encoder ~2–4 úrovně, support set do 64, mřížka
32×32.

## C.5 Trénink

Jeden krok:
1. Sampluj úlohu `(ℓ, τ, σ²) ~ Π`.
2. Sampluj `n_supp + 1` obrázků: pro každý nezávislá GP realizace `f`, `image = f + ε`,
   `mask = 1[f > τ]`.
3. `n_supp` dvojic (image, mask) → support; 1 (image → mask) → query target.
4. Loss = **per-pixel BCE** mezi PFN logity a query maskou, průměr přes pixely. To je
   diskrétní analog GP2 BarDistribution NLL (binární target → BCE, žádná BarDistribution
   není potřeba).

**Prior `Π` (návrh rozsahů):**
- `ℓ` log-uniform přes rozsah pokrývající Easy i Hard (např. ℓ ∈ [0.05, 0.5] normalizované
  k velikosti mřížky).
- `τ` omez tak, aby foreground fraction `Φ(−τ/σ_f)` byla v rozumném pásmu (např. [0.1, 0.9])
  — jinak degenerativní masky (samé 0 / samé 1).
- `σ²` malé kladné pásmo; případně fixuj pro první běh a variuj až pro třetí datový bod
  (analog tvého σ²=0.01 nápadu na třetí κ bod).

**Reprodukovatelnost / RNG past (tvoje známá):** generuj GP data **na CPU a pak přesuň na
device**. Na MPS používá batch generátor separátní RNG neovlivněné standardním seedováním
— definitivní fix je CPU wrapper (`get_batch_cpu()`), který data vyrobí na CPU a teprve pak
`.to(device)`. Jinak nebudeš mít deterministické běhy.

## C.6 Závislé proměnné — co měřit

### Fidelita k oraclu (přímé měření amortizačního biasu)

Per pixel `|p̂_PFN_ij − p_oracle_ij|`, průměr přes pixely a query; nebo symetrická KL /
`KL(oracle ‖ PFN)`. Toto je jediné **čisté** měření v celém plánu — všude jinde chybí
referenční posterior.

### Rozpad fidelity (Easy/Hard, OOD, n_supp)

Klíčový graf: jak fidelita **klesá**, když
- ℓ → krátké (Hard),
- hyperparametry jdou mimo `Π` (OOD extrapolace — z GP2 víš, že model extrapoluje dobře na
  *delší* ℓ, ale katastrofálně selhává na *kratší*; ověř, že totéž platí ve 2D),
- `n_supp → 0` (a ověř kolaps: `p̂_PFN → Φ(−τ/σ_f)`, tj. k **explicitní** prior masce).

### Bias² + variance rozklad (per pixel)

Pro fixní úlohu a mnoho query realizací:
```
bias_ij     = E_query[p̂_PFN_ij] − p_oracle_ij
variance_ij = Var_query[p̂_PFN_ij]
```
Očekávání z Naglera: variance mizí ~O(n_supp^{−1/2}) (softmax → diminishing sensitivity),
bias **strukturálně nemizí** (globální attention porušuje locality). Fituj `bias² + c/n`
křivku jako v GP2 — a pozor na to, aby fit nenarazil na hranice parametrů (tvoje Ch.3 past
v Hard/8L případě): zkontroluj bounds a reportuj, když je fit degenerovaný.

### Kalibrace vs oracle (ne vs empirie)

Protože máš oracle, testuj over/under-confidence **vůči pravému posterioru**, ne jen vůči
empirické frekvenci. To je čistší: reliability diagram, kde „pravda" je `p_oracle`, ne
tvrdý label.

## C.7 Most A → C (proč obě, a jak spolu mluví)

- **A** ukázala kolaps na reálném modelu, ale proti *operační* prior masce (heuristika).
- **C** měří tentýž kolaps proti **explicitní** prior masce `Φ(−τ/σ_f)` a proti **pravému**
  posterioru → potvrdí (nebo vyvrátí) A rigorózně.
- Protože architektura C je UniverSeg-lite (srovnatelná s A), rozdíl v kalibraci mezi
  A a C izoluje **efekt PFN trénovací procedury** (explicitní prior + PPD loss).

Souhrn tvrzení práce: *„Amortizovaná in-context segmentace dědí PFN patologie (A, reálný
model). Skutečný PFN na explicitním prioru (C) tyto patologie kvantifikuje proti oraclu:
variance mizí O(n^{−1/2}), bias k prioru strukturálně přetrvává kvůli globální attention,
kalibrace se chová [změřeno]. To je bias–variance účet amortizované segmentace."*

## C.8 Pasti (čti před spuštěním)

- **Podmíněnost kernelu.** Krátké ℓ → velké κ → nestabilní oracle. Nugget/jitter, dvojitá
  přesnost pro oracle inverzi.
- **Degenerativní masky.** Omez `τ` na rozumnou foreground fraction, jinak trénink kolabuje.
- **RNG na MPS.** CPU wrapper (viz C.5), jinak nereprodukovatelné.
- **NLL srovnatelnost.** Drž agregaci BCE konzistentní; per-pixel korelace přes GP znamená,
  že agregovaná NLL není součet nezávislých — pro srovnání vs oracle to je OK, pokud je
  stejná metrika všude.
- **Rekalibrace.** Temperature scaling na logitech, ne σ-scaling.
- **Bias/variance fit bounds.** Kontroluj, zda `bias² + c/n` fit nenarazil na hranice
  (Ch.3 zkušenost).
- **Škála mřížky vs náklady oraclu.** 32×32 na start; zvětšuj až po ověření, že oracle
  pipeline běží.

## C.9 Deliverable C

(1) Fidelita PFN↔oracle jako funkce (ℓ, OOD, n_supp) s vyznačeným rozpadem; (2) per-pixel
bias²+variance rozklad s fitem a ověřením O(n^{−1/2}) variance a nemizejícího biasu; (3)
kalibrace vs oracle; (4) explicitní demonstrace kolapsu `p̂_PFN → Φ(−τ/σ_f)` při `n_supp→0`.

---

# Pořadí a proč

**A → C.** A je nejlevnější (nulový trénink, veřejné váhy, výsledek za týdny) a je to
falsifikační test premisy — pokud in-context segmentery kolaps *nemají*, celá teze padá a
nemá smysl stavět C. C je pak rigorózní měření proti oraclu v kontrolovaném prostředí,
přímé pokračování GP2, které de-riskuje jakýkoli budoucí skok na reálná medicínská data.

A ptá se: *„dědí hotové amortizované segmentery patologie, které PFN teorie předpovídá?"*
C ptá se: *„když postavím opravdový PFN, jak daleko je od pravdy a jak se ta vzdálenost
rozpadá?"* První motivuje a de-riskuje, druhá je vlastní vědecké jádro.
