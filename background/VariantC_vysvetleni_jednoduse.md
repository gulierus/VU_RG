# Varianta C úplně jednoduše

*Vysvětlení bez matematiky a bez odborných zkratek. Každý pojem vysvětlíme, jakmile se poprvé
objeví (tučně). Úplně dole je slovníček na rychlé dohledání.*

---

## Co vlastně řešíme (jedna věta)

Chceme zjistit, jestli se **rychlá umělá inteligence** umí rozhodovat skoro stejně dobře jako
**ideální (dokonalý) statistický výpočet**, a hlavně kde se od něj začne lišit.

Aby to nebylo abstraktní, budeme celou dobu používat jednu analogii: **učíme asistenta obtahovat
skvrny na mlhavých fotkách.**

---

## 1. Úloha: obtáhnout skvrnu na mlhavé fotce

Máme černobílé, trochu **mlhavé** (zašuměné) fotky. Na každé je nějaká oblast (skvrna), kterou
chceme **obtáhnout**. Tomuhle obtahování se odborně říká **segmentace**. Výsledek obtažení je
**maska**: obrázek stejné velikosti, kde je každý bod (pixel) buď *uvnitř* skvrny (bílá), nebo
*venku* (černá).

Každá série fotek se řídí třemi „skrytými knoflíky", které dopředu neznáme:

- **Jak velké a hladké jsou skvrny.** Odborně **lengthscale** (značka `ℓ`). Malé `ℓ` = drobné,
  rozčepýřené skvrny (těžké). Velké `ℓ` = velké, hladké plochy (snadné).
- **Kolik z fotky je uvnitř skvrny.** To řídí **práh** (značka `τ`) a s ním **foreground** =
  podíl bílé v masce. (Nižší práh → víc bílé.)
- **Jak moc je fotka zašuměná.** Odborně **šum** (značka `σ²`). Čím větší šum, tím větší mlha a
  tím hůř se pozná, kde skvrna přesně končí.

Jedna „úloha" = jedno nastavení těch tří knoflíků.

---

## 2. Jak asistentovi řekneme, jaká jsou pravidla: ukázky

Asistent ta tři čísla (`ℓ`, `τ`, `σ²`) nezná. Řekneme mu je nepřímo: **ukážeme mu pár hotových
příkladů** — fotek, u kterých už *je* maska správně nakreslená. Téhle sadě příkladů se říká
**support set** (nebo **kontext**). Z nich asistent pozná, jaká pravidla platí pro tuhle sérii.

- Počtu příkladů budeme říkat **velikost kontextu** (značka `n_supp`). Někdy dáme jeden příklad,
  někdy 64.
- Že asistent řeší úlohu **z příkladů, bez přeučování**, se odborně říká **in-context** učení.

To, že se asistent jednou dlouho učí (na milionu cvičných úloh) a *pak už* odpovídá okamžitě, se
jmenuje **amortizace**. Představ si studenta, který se jednou naučí postup, a pak ho na každý nový
příklad aplikuje hned, místo aby ho pokaždé znovu složitě odvozoval. Tohle je celý trik: draho
zaplatíš jednou (trénink), pak jsou odpovědi levné a rychlé.

**Náš model se jmenuje PFN.** Je to přesně takový vycvičený asistent: viděl milion cvičných úloh,
a teď z pár příkladů okamžitě nakreslí masku.

---

## 3. Model nekreslí jen černobíle: říká i „jak jsem si jistý"

Dobrý asistent nedá jen tvrdou černobílou masku. U každého pixelu řekne **číslo od 0 do 100 %** =
„jak jsem si jistý, že sem patří skvrna". Téhle mapě jistoty se odborně říká **posterior** (nebo
**pravděpodobnostní mapa**). „Posterior" prostě znamená **nejlepší informovaný odhad poté, co jsem
viděl data**. Uvnitř skvrny bude blízko 100 %, venku blízko 0 %, a přesně na hranici třeba 50 %
(protože tam si nejsme jistí).

Pro srovnání: **prior** je opak — to je „defaultní" očekávání *ještě než* něco uvidíme. Kdyby
asistent neměl žádnou informaci, hádal by prostě průměr (třeba „půlka fotky bývá skvrna").

---

## 4. Tajná zbraň Varianty C: máme „klíč se správnými odpověďmi"

Tady je jádro celého experimentu. V našem umělém světě si ta tři pravidla (`ℓ`, `τ`, `σ²`)
**určujeme sami**. Proto umíme spočítat **dokonalou odpověď** — nejlepší možnou mapu jistoty,
jakou vůbec jde z dané fotky udělat, kdyby člověk ta pravidla znal.

Tomuhle dokonalému výpočtu říkáme **oracle**. Ber to jako **matematika, který pravidla zná a
spočítá bezchybný výsledek.** Je to náš klíč se správnými odpověďmi, naše **pravda**.

A teď to důležité:

- **Oracle pravidla ZNÁ.** (Je to teoretické maximum.)
- **PFN pravidla NEZNÁ** — musí je uhodnout z těch pár příkladů.

Rozdíl mezi nimi = **cena za to, že PFN hádá.** Odborně se tomu říká **amortizační chyba**. Celá
Varianta C není nic jiného než pečlivé měření tohohle rozdílu, v různých situacích.

*(Poznámka na okraj: aby se ten dokonalý výpočet dal udělat rychle, necháváme obrázek „omotat
dokola" jako v Pac-Manovi. Odborně **torus**. Je to jen matematická vychytávka, na pochopení
výsledků nezáleží.)*

---

## 5. Co jsme zjistili (polopatě)

### a) PFN kreslí skoro to samé co matematik

Když položíme vedle sebe mapu od oraclu a od PFN, jsou **skoro k nerozeznání**. To je hlavní
pozitivní zpráva: rychlý natrénovaný model opravdu umí to, co slibuje — dá skoro dokonalý
Bayesovský odhad. A poprvé to umíme dokázat proti *skutečné pravdě*, ne proti nějakému odhadu.

### b) PFN je skoro tak dobrý, jak to vůbec jde (Bayes floor)

Klíčová myšlenka: **ani matematik (oracle) nemůže být dokonalý.** Fotka je mlhavá, takže se prostě
nedá u každého pixelu poznat jistě, jestli je uvnitř. Existuje tedy **nejmenší možná chyba, pod
kterou se nedostane nikdo** — ani ten nejlepší na světě. Téhle nepřekročitelné hranici říkáme
**Bayes floor** (dá se to číst jako „nejlepší dosažitelné skóre v testu").

A výsledek: **PFN je od téhle nejlepší možné hranice jen kousíček.** Konkrétně na běžných
(tréninku podobných) úlohách ztrácí jen `~0,004–0,006` bodu. To znamená, že se naučil **skoro
optimálně** — líp už to skoro nejde.

Tomu „o kolik je PFN horší než teoretické maximum" se říká **excess risk** (přebytečná chyba).
Malý excess risk = skoro dokonalé.

*(Bonus: díky tomu teď víme, že trénink neskončil „zaseknutý" na nějaké chybě — skončil přesně
tam, kde je ta nepřekročitelná hranice. Nešlo být lepší.)*

### c) Kde se to pokazí: na úlohách nepodobných tréninku

Model je skvělý na tom, co zná. Ale když mu dáme úlohu **hodně jinou, než na čem se učil**
(odborně **OOD** = *out-of-distribution*, „mimo trénink"), spadne. Nejhůř dopadá na skvrnách,
které jsou **mnohem drobnější** než všechno, co kdy viděl — tam je excess risk skoro `10×` větší.
Zajímavé: opačný extrém (mnohem *hladší* skvrny než v tréninku) zvládá docela dobře. Selhává
tedy **nesymetricky** — drsnější úlohy jsou pro něj mnohem horší než hladší.

### d) Počet ukázek skoro nevadí

Čekali bychom, že s jednou ukázkou bude model ztracený a bude hádat průměr. **Nestalo se.** Už
z jedné ukázky pozná úlohu skoro stejně dobře jako z 64. Důvod: sama ta fotka, kterou má
obtáhnout, prozradí skoro všechno; ukázky jen doplní, kde přesně je práh. (V dřívější Variantě A,
kde měl model informace *řídce*, se ke „průměru" vracel — tady ne.)

### e) Dva druhy chyby: „roztřesenost" a „systematická chybička"

Statistici chybu rozkládají na dvě části:

- **Variance (roztřesenost):** jak moc se odpověď mění podle toho, *které* konkrétní ukázky zrovna
  dostal. S víc ukázkami se odpověď **ustálí** — variance mizí. To se u nás potvrdilo.
- **Bias (systematická chybička):** malá chyba, která **nezmizí ani s nekonečně ukázkami**. Drží
  se hlavně na hranicích skvrn. U nás je malá, ale **nezmizí** — a protože model je jinak skoro
  dokonalý, víme, že tahle chybička je „vestavěná" do způsobu, jak model funguje, ne že by se
  málo naučil.

### f) Jistota (kalibrace) je taky skoro správně

Ještě jsme se ptali: když model řekne „jsem si jistý na 70 %", stane se to opravdu v 70 %
případů? Tomuhle „sedí procenta?" se říká **kalibrace**. Výsledek: model je **skoro kalibrovaný** —
jeho jistota odpovídá matematikovi, i na mlhavých hranicích. Jediná drobná vada: na úzkém pruhu
kolem hranice je **trošku přehnaně sebejistý** (řekne „skoro 100 %" tam, kde by měl přiznat „tak
napůl"). Odborně **over-sharp** / **over-confident**. Ale je to malé.

---

## 6. Proč je to důležité (celkově)

- **Potvrdili jsme slib PFN.** Rychlý model, který se učí z příkladů, opravdu umí skoro dokonalé
  (Bayesovsky optimální) rozhodování — a poprvé to ukazujeme proti *skutečné pravdě*, ne proti
  odhadu.
- **Víme, kde se pokazí.** Ne kvůli málu příkladů, ale když je úloha **nepodobná tréninku**
  (hlavně drsnější). To je praktické varování: pozor na data mimo trénovací rozsah.
- **Navazuje to na zbytek práce.** Je to ta samá analýza jako u našich 1D pokusů (série GP),
  jen povýšená na 2D obrázky, a měřená proti dokonalému klíči.

---

## Slovníček (rychlé dohledání)

| Pojem | Polopaticky |
|---|---|
| **Segmentace** | Obtáhnout na obrázku oblast (co je „uvnitř" a co „venku"). |
| **Maska** | Výsledek obtažení: obrázek, kde je každý pixel bílý (uvnitř) nebo černý (venku). |
| **PFN** | Náš rychlý model. Vycvičený asistent, co z pár ukázek hned kreslí masku. |
| **Support set / kontext** | Pár ukázkových fotek s už hotovou maskou, ze kterých se pozná úloha. |
| **`n_supp` (velikost kontextu)** | Kolik ukázek jsme dali (1 až 64). |
| **In-context** | Řešit z ukázek, bez přeučování modelu. |
| **Amortizace** | Zaplať draho jednou (trénink), pak odpovídej rychle a zadarmo. |
| **`ℓ` (lengthscale)** | Jak velké/hladké jsou skvrny. Malé = drobné a drsné, velké = hladké. |
| **`τ` (práh) / foreground** | Kde se „ořízne" bílá; foreground = kolik % fotky je uvnitř. |
| **`σ²` (šum)** | Jak moc je fotka mlhavá/zrnitá. |
| **Prior** | Defaultní očekávání *než* něco uvidíme (třeba „prostě hádej průměr"). |
| **Posterior / mapa jistoty** | Nejlepší odhad *poté*, co jsem viděl data: u každého pixelu 0–100 %. |
| **Oracle** | Matematik, co pravidla zná a spočítá dokonalou odpověď. Náš „klíč". |
| **Amortizační chyba** | O kolik je hádající PFN horší než oracle, který pravidla zná. |
| **Bayes floor** | Nejmenší možná chyba, pod kterou se nedostane nikdo (kvůli mlze/šumu). |
| **Excess risk** | O kolik je PFN horší než teoretické maximum (malý = skoro dokonalé). |
| **OOD (out-of-distribution)** | Úloha „mimo trénink", nepodobná tomu, na čem se model učil. |
| **Bias (systematická chybička)** | Chyba, co nezmizí ani s nekonečně ukázkami. |
| **Variance (roztřesenost)** | Kolísání odpovědi podle toho, které ukázky dostal; mizí s víc ukázkami. |
| **Kalibrace** | „Sedí procenta?" Když řekne 70 %, stane se to v 70 % případů. |
| **Over-sharp / over-confident** | Přehnaně sebejistý: říká 0/100 % tam, kde by měl hedgovat 50 %. |
| **Torus** | Obrázek „omotaný dokola" (Pac-Man). Jen matematická vychytávka pro rychlý oracle. |
