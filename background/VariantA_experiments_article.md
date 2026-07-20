# Varianta A — výsledky (diagnostika kolapsu k prioru na zmraženém UniverSeg)

*Autoritativní čísla experimentu Varianta A (viz [workflow_A_C_for_PFN.md](workflow_A_C_for_PFN.md),
část I). Generováno notebookem
[experiments/Experiment_A_UniverSeg_segmentation.ipynb](../experiments/Experiment_A_UniverSeg_segmentation.ipynb);
obrázky v `figures/VariantA_universeg/`. Neber čísla z paměti — ber je odsud.*

## Setup

- **Model:** předtrénovaný UniverSeg (veřejné váhy, 1,18 M parametrů), zmražený, `eval`.
  Voláme jej přímo na **jednom** support setu → K=1 (nativní ensembling přes K tahů vypnut),
  aby se efekt velikosti support setu nezaměnil s efektem ensemblingu.
- **Cíl (target):** binární maska **celé buňky** (cytoplazma ∪ jádro) u WBC; bílá hmota
  (seg4, label 3) u OASIS.
- **Dvě osy degradace** (drženy odděleně, A.4):
  1. **Velikost kontextu `n_supp ∈ {1,2,4,8,16,32,64}` — PRIMÁRNÍ, čistě měřená osa kolapsu.**
     Nezávislá na doméně; nese hlavní tvrzení A.
  2. **Doménová osa — robustnost napříč doménami, NE „OOD-ness".** WBC-JTSC, WBC-CV (jiná
     akvizice téže úlohy), OASIS (mozková MRI). *Oprava oproti dřívější verzi:* původní nálepky
     „ID → mírné OOD → silné OOD" byly **chyba** — přidělili jsme je apriorně, ne změřili. Podle
     tréninkového splitu UniverSeg jsou WBC i OASIS *held-out demo datasety* s modalitou
     zastoupenou v MegaMedical (mikroskopie i MRI patří mezi 16 trénovacích modalit), takže
     **žádný z nich není ověřené silné OOD** (WBC i OASIS jsou na témže tieru „held-out, viděná
     modalita"; přesnou příslušnost ke splitu jsme neověřovali). Osa proto měří **posun úlohy /
     akvizice**, ne vzdálenost od tréninku. Skutečný OOD bod (neviděná modalita, např. prostate
     MR) v datech **není** → možné rozšíření. **Proč ne feature-distance osa:** vzdálenost v
     enkodéru *od kotvy WBC* by měřila špatnou veličinu — mozková MRI je v featurách daleko od WBC,
     ale model MRI zná, takže by dostala vysoké skóre bez kolapsu; „distance-from-training" bez
     featur MegaMedical rigorózně nepostavíme.
- **Statistický protokol:** 60 query obrázků × 3 tahy support setu na buňku (180 pozorování;
  WBC-CV má menší test pool → 90), reportujeme průměr ± SEM přes surové per-query hodnoty.
  Seedy tahů jsou logované v `variant_a_results.json`.
- **Prior maska (operační):** `E[y] ≈ (1/M) Σ mask_m` přes support pool domény (A.5). WBC je
  kompaktní centrovaný blob → prior maska smysluplná (foreground 0,315 JTSC / 0,226 CV);
  OASIS bílá hmota je roughly-registered (foreground 0,187), strukturovaná, ale zašuměnější.

## Hlavní tvrzení

**In-context segmenter (ne-PFN) dědí prior-collapse předpovězený PFN teorií (Nagler, locality
condition): při ubývání kontextu se predikce vrací k prioru, kalibrace je nestabilní ve směru
over-confidence.** Jev tedy není artefakt PFN trénovací procedury, ale vlastnost
amortizace / globální attention.

### (i) Kolaps k prioru — crossover `d_truth` × `d_prior`

Podpis kolapsu je crossover: s klesajícím `n_supp` roste `d_truth` (predikce dál od pravdy)
a predikce se přibližuje prior masce (`d_prior` je při `n_supp=1` nejnižší **vůči** `d_truth`).
`d_*` = per-pixel soft MSE (Brier) mezi `p̂` a GT / prior maskou.

| doména | metrika | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---|---|---|---|---|---|---|---|---|
| WBC-JTSC | `d_truth` | **0,1295** | 0,0579 | 0,0378 | 0,0220 | 0,0203 | 0,0152 | **0,0140** |
| WBC-JTSC | `d_prior` | 0,1045 | 0,0593 | 0,0502 | 0,0510 | 0,0502 | 0,0489 | 0,0495 |
| WBC-CV | `d_truth` | 0,1075 | 0,0812 | 0,0568 | 0,0466 | 0,0387 | 0,0319 | 0,0301 |
| WBC-CV | `d_prior` | 0,0728 | 0,0585 | 0,0428 | 0,0445 | 0,0372 | 0,0378 | 0,0364 |
| OASIS | `d_truth` | 0,0735 | 0,0471 | 0,0379 | 0,0324 | 0,0291 | 0,0278 | 0,0271 |
| OASIS | `d_prior` | 0,0437 | 0,0392 | 0,0405 | 0,0417 | 0,0409 | 0,0409 | 0,0408 |

**Čtení:** u všech tří domén je při `n_supp=1` predikce **blíž prior masce než pravdě**
(`d_prior < d_truth`), tj. kolaps. Jak `n_supp` roste, `d_truth` klesá pod `d_prior`
(crossover kolem `n=2` u JTSC, `n≈8–16` u CV, `n≈4` u OASIS) a `d_prior` se stabilizuje na
plató — predikce se od defaultní masky vzdálí a přilne k pravdě. `d_prior` **nekonverguje
k nule**, protože silný model má predikci datově specifickou, ne marginální.

### (ii) Výkon (Dice) — saturace kontextem

| doména | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---|---|---|---|---|---|---|---|
| WBC-JTSC | 0,611 | 0,857 | 0,917 | 0,952 | 0,957 | 0,967 | 0,969 |
| WBC-CV | 0,591 | 0,747 | 0,842 | 0,877 | 0,900 | 0,916 | 0,921 |
| OASIS | 0,731 | 0,839 | 0,872 | 0,889 | 0,901 | 0,904 | 0,906 |

Dice roste a **saturuje** (plató od `n_supp≈8–16`). To je přímý 2D analog Experimentu 5 z GP2
(context size → error saturace) a plató popsané v UniverSeg. Plató se liší napříč doménami (JTSC
0,97 > CV 0,92) — to je posun úlohy/akvizice, **ne OOD-ness**; OASIS má vyšší Dice při `n=1` díky
velké, konzistentně umístěné struktuře.

### (iii) Kalibrace — směr MĚŘENÝ, ne předpokládaný (a je DOMÉNOVĚ ZÁVISLÝ)

Přesně jak varuje návod (A.5 ii: „směr NEPŘEDPOKLÁDEJ, změř ho"), směr kalibrace **není
univerzální** — liší se podle domény. Dvě různá měření to potvrzují.

**Reliability diagram (standardní smysl, `fig_A_03`):** empirická frekvence foreground vs
predikované `p̂`. Křivky **nad** úhlopříčkou = under-confidence, **pod** = over-confidence.
- **WBC (JTSC i CV): under-confident** — křivky nad úhlopříčkou (model podstřeluje `p̂`
  foreground). S rostoucím `n_supp` se JTSC blíží úhlopříčce (ECE 0,128 → 0,008).
- **OASIS: over-confident** — křivky pod úhlopříčkou, prakticky nezávisle na `n_supp`
  (ECE ≈ 0,032 konstantní).

**Znaménkový calib_gap = průměr (confidence − accuracy)** s confidence = `max(p̂, 1−p̂)`
(smysl „mám-li pravdu"): `>0` = over-confident. Je kladný všude, nejsilněji při `n_supp=1`,
s rostoucím kontextem klesá k nule.

| doména | calib_gap n=1 | n=8 | n=64 |
|---|---|---|---|
| WBC-JTSC | +0,0850 | +0,0103 | +0,0042 |
| WBC-CV | +0,0582 | +0,0296 | +0,0166 |
| OASIS | +0,0114 | +0,0057 | −0,0005 |

Obě metriky nejsou v rozporu — měří jiný pojem (frekvence foreground vs. správnost tvrdého
rozhodnutí). Klíčové sdělení: **kalibrace je nestabilní a doménově závislá** (WBC under-,
OASIS over-confident v reliability smyslu), takže tvrzení „variance je univerzálně
podstřelená" by bylo chybné — což je přesně varování z původního survey.

**ECE odděleně uvnitř objektu vs v hraničním pásmu** (dilatace−eroze GT, A.5): ambiguita žije
na hranici. ECE v hraničním pásmu je řádově vyšší než uvnitř a klesá s `n_supp` pomaleji.

| doména | ECE hranice (n=1 → n=64) | ECE vnitřek (n=1 → n=64) |
|---|---|---|
| WBC-JTSC | 0,2298 → 0,1303 | 0,1337 → 0,0066 |
| WBC-CV | 0,2328 → 0,1229 | 0,0830 → 0,0275 |
| OASIS | 0,0874 → 0,0939 | 0,0219 → 0,0047 |

U OASIS je hraniční ECE prakticky konstantní přes celý sweep — nejistota na hranici bílé hmoty
je neredukovatelná ani velkým kontextem, kdežto vnitřek se plně vyřeší (0,0047).

## Obrázky (`figures/VariantA_universeg/`)

- `fig_A_01_collapse_crossover.png` — `d_truth`↑ a `d_prior`↓ per doména (podpis kolapsu).
- `fig_A_02_dice.png` — Dice vs `n_supp` (saturace + napříč doménami). *(Titulek figury zní
  „OOD-ness" — legacy název; správně jde o robustnost napříč doménami, viz osa 2. Opraví se při
  příštím přeběhu notebooku.)*
- `fig_A_03_reliability.png` — reliability diagramy (doména × vybrané `n_supp`).
- `fig_A_04_calibration.png` — ECE celek/hranice + znaménkový calibration gap.
- `fig_A_05_prior_masks.png` — operační prior masky (obhajoba metriky (i)).

## Limity (A.7) a most k C

- **Doménová osa NENÍ OOD osa (oprava).** Nálepky ID/mírné/silné OOD byly apriorní a věcně
  chybné (WBC i OASIS jsou held-out demo datasety s viděnou modalitou). Primární a čistý důkaz
  kolapsu proto nese **osa `n_supp`** (měřená, doménově nezávislá), ne doménová osa. Skutečný OOD
  bod (neviděná modalita) v datech chybí; feature-distance osu jsme záměrně nestavěli, protože
  „distance-from-anchor" měří jinou vzdálenost než tu, co kolaps pohání (viz osa 2 v Setupu).
- **Jednotné vyprávění A↔C.** Kolaps v A táhne **řídkost kontextu** (`n_supp`), ne novost domény
  — což je *tatáž* story jako v C: identifikovatelná (hustá) úloha nekolabuje bez ohledu na to,
  jak „cizí" je. Kolaps je řízen nedostatkem informace, ne vzdáleností domény. To je silnější
  a napříč variantami konzistentní tvrzení než vynucená OOD osa.
- UniverSeg **nemá pravý prior** → metrika (i) je *operační* aproximace (průměrná maska).
  Právě tuto slabinu odstraní Varianta C explicitním priorem + oraclem posterioru.
- Prior maska vyžaduje **prostorové zarovnání** — platí pro WBC (centrovaná buňka) i
  roughly-registered OASIS; u nezarovnaných dat by ztratila význam.
- Kalibrace nerekalibrována; případná rekalibrace = temperature scaling na logitech, ne
  σ-scaling (binární per-pixel Bernoulli).
- **Reprodukovatelnost:** MPS po mnoha forward passech / velkém support setu tiše korumpuje
  výstup (Dice→0) a při `batch×n_supp` velkém padá na OOM. Fix v notebooku: `torch.mps.empty_cache()`
  + `synchronize()` po každém batchi a adaptivní `eff_batch = min(batch, pair_budget//n_supp)`.

**Závěr A:** *In-context segmentery dědí prior-collapse předpovězený PFN teorií (crossover
`d_truth`×`d_prior` při ubývání kontextu); kalibrace je nestabilní a doménově závislá (WBC
under-, OASIS over-confident) a nejhorší na hranici objektu. Jev existuje na reálném modelu →
má smysl postavit kalibrovaný PFN (Varianta C) a měřit tytéž patologie proti oraclu.*
