# Jak PFN aproximuje GP posterior: Experimentální analýza

## Přehled projektu

Cílem tohoto výzkumu bylo experimentálně prozkoumat, jakým způsobem Prior-Fitted Networks (PFN) — transformerové modely trénované na syntetických datech z Gaussian Process prioru — aproximují Bayesovskou inferenci. Konkrétně nás zajímalo, zda se uvnitř transformeru dá identifikovat známá statistická operace (kernel smoothing, Nadaraya-Watson estimátor), nebo zda model řeší GP inferenci kvalitativně odlišným způsobem.

Experimenty byly provedeny na dvou modelech se stejnou architekturou (6 vrstev, 8 attention hlav, embedding dimenze 512, `features_per_group=1`, `attention_between_features=False`), ale s různou délkou tréninku. Oba modely byly trénovány na fixních GP hyperparametrech (lengthscale=0.3, outputscale=1.0, noise=1e-4).

| Parametr | Malý model | Velký model |
|----------|-----------|-------------|
| Epochy | 20 | 100 |
| Kroky/epocha | 200 | 150 |
| Celkem viděných datasetů | ~256 000 | ~960 000 |
| GP hyperparametry | Fixní (LS=0.3, OS=1.0, noise=1e-4) | Fixní (LS=0.3, OS=1.0, noise=1e-4) |

---

## Experiment 1: Konvergence k prioru a uncertainty kalibrace

### Myšlenka

Prvním krokem bylo ověřit základní funkčnost modelu. Pokud PFN správně aproximuje GP posterior, měl by splňovat dvě vlastnosti: za prvé, jeho mean predikce by měla odpovídat GP mean, a za druhé, jeho uncertainty (konfidenční intervaly) by měla odpovídat GP uncertainty. Zvlášť nás zajímalo chování v extrapolaci — co se stane, když trénovací data pokrývají jen část prostoru a model musí predikovat i daleko od nich.

### Postup

PFN dostal trénovací data pouze v regionu x=0.3–0.7 (20 bodů) a predikoval v celém rozsahu x=0.0–1.0. Výstup jsme porovnali s pravým GP posteriorem, který zná správné hyperparametry. Měřili jsme korelaci mezi PFN a GP směrodatnou odchylkou (zda model ví *kde* být nejistý), MSE mezi nimi (zda ví *jak moc*), a poměr uncertainty daleko vs blízko od dat (dynamický rozsah).

### Výsledky a interpretace

U malého modelu jsme pozorovali, že mean predikce je téměř perfektní — modrá (PFN) a zelená (GP) křivka se prakticky překrývají i v extrapolačních oblastech. To bylo povzbudivé. Problém se ale ukázal v uncertainty: korelace mezi PFN a GP std byla vysoká (0.94–0.96), což znamená, že model správně rozpoznává *tvar* uncertainty — ví, že daleko od dat má být nejistější než blízko. Absolutní hodnoty ale nesedí. PFN má std=0.95 daleko od dat tam, kde GP má 0.37, a std=0.05 blízko dat tam, kde GP má 0.01. Příliš vysoká baseline a příliš malý dynamický rozsah. Poměr far/near byl u PFN 5–15×, zatímco u GP 12–29×.

U velkého modelu se situace výrazně zlepšila. Korelace tvaru vzrostla na 0.99+, MSE mezi PFN a GP std kleslo 5–50×. Uncertainty poměr se zvýšil na 17–24× u PFN, ale GP poměr se také zvýšil na 36–80×. PFN tedy stále dosahuje zhruba polovičního ratia oproti GP. Na grafu je vidět, že modrý pás (PFN ±2σ) kopíruje zelený (GP ±2σ) mnohem věrněji než u malého modelu, ale v extrapolaci je stále širší.

### Závěr

Delší trénink výrazně zlepšuje uncertainty kalibraci. Tvar uncertainty je u většího modelu téměř perfektní. Přetrvávající gap v extrapolačním ratiu (~50% GP) může být buď otázka ještě delšího tréninku, nebo fundamentální limitace architektury — BarDistribution má omezený dynamický rozsah kvůli diskretizaci výstupu, a model nikdy během tréninku neviděl scénáře s extrémně sparse daty v malém regionu.

---

## Experiment 2: Struktura attention matic

### Myšlenka

Attention matice je klíčové okno do toho, jak transformer interně zpracovává informaci. Pokud PFN dělá GP inferenci, měli bychom v attention vidět nějakou strukturu, která odpovídá matematice GP posterioru — konkrétně, test body by se měly „dívat" na trénovací data, protože predikce závisí na pozorovaných hodnotách.

### Postup

Vizualizovali jsme attention matice pro všech 6 vrstev, průměrované přes 8 hlav. Matice je přirozeně rozdělená na čtyři kvadranty podle toho, zda jde o train nebo test bod v roli query (řádky) nebo key (sloupce): Train→Train (vlevo nahoře), Train→Test (vpravo nahoře), Test→Train (vlevo dole), Test→Test (vpravo dole).

### Výsledky a interpretace

Oba modely vykazují konzistentní strukturu, která se vyvíjí přes vrstvy. Vrstva 0 má relativně distribuovanou attention — váhy jsou rozložené přes mnoho pozic, žádný kvadrant výrazně nedominuje. To interpretujeme jako explorativní fázi, kde model sbírá globální informaci o rozložení bodů.

Vrstvy 1–4 se postupně zostřují. Attention se stává sparse — jen několik pozic má vysokou váhu, zbytek je blízko nule. Prostřední vrstvy zřejmě slouží k internímu zpracování a redistribuci informace.

Vrstva 5 (poslední) vykazuje nejzajímavější pattern: Test→Train kvadrant je jasně aktivní (test body silně attendují na train body), zatímco Train→Test je téměř černý (trénovací body ignorují testovací). Tato asymetrie je přesně to, co bychom očekávali od GP inference — predikce na test bodech závisí na pozorovaných datech, ale pozorovaná data nezávisí na tom, kde predikujeme. Model tento kauzální vztah objevil autonomně z dat, bez jakéhokoliv explicitního zakódování.

Rozdíl mezi modely je v čistotě tohoto patternu. Velký model má v poslední vrstvě výrazně ostřejší kontrast mezi aktivním Test→Train kvadrantem a zbytkem matice. To odpovídá jeho lepší přesnosti — čistější attention znamená přesnější přenos informace z train dat do predikce.

### Závěr

Celková struktura attention (distribuovaná → sparse → Test→Train) je stabilní vlastnost architektury, ne artefakt konkrétního tréninku. Je to důkaz, že PFN provádí Bayesovskou inferenci — kdyby model jen memoroval trénovací data nebo dělal pattern matching, attention by byla symetrická.

---

## Experiment 3: Porovnání attention s RBF kernelem

### Myšlenka

GP prior našeho modelu používá RBF kernel. Pokud transformer aproximuje GP inferenci, nabízí se otázka, zda se jeho attention váhy podobají RBF kernelu — tj. zda body blíže k test bodu dostávají vyšší attention, s gaussovským profilem závislým na vzdálenosti.

### Postup

Pro konkrétní test body jsme porovnali attention váhy z poslední vrstvy s RBF kernel vahami. Měřili jsme korelaci mezi oběma vektory vah pro každý test bod zvlášť i celkově. Vizualizovali jsme oba profily vedle sebe a sledovali entropii attention přes vrstvy.

### Výsledky a interpretace

Na vizualizaci je patrné, že oba profily mají podobný celkový trend — bližší body dostávají více váhy. Ale attention je výrazně ostřejší než RBF. Zatímco RBF distribuuje váhu plynule podle gaussovské křivky, attention koncentruje téměř veškerou váhu na nejbližší bod s prudkým poklesem pro vzdálenější body. Průměrná korelace je relativně nízká — 0.32 u velkého modelu, 0.3–0.6 u malého.

To dává smysl, když si uvědomíme, co model optimalizuje. Attention hlavy nepotřebují počítat čistou kernel similarity k(x*, xᵢ). Potřebují počítat to, co v kombinaci s ostatními hlavami a FFN vrstvami dá nejlepší GP posterior. A pro ten může být výhodnější ostřejší kernel — třeba proto, že FFN pak snáze provede korekci odpovídající efektu K⁻¹.

Entropie attention roste přes vrstvy — rané vrstvy mají nižší entropii (ostřejší, soustředěnější), poslední vrstva nejvyšší (distribuovanější přes train body). To je konzistentní s tím, že poslední vrstva provádí samotnou GP inferenci, kde je potřeba vzít v úvahu všechny train body.

### Závěr

PFN se nenaučil RBF kernel. Naučil se vlastní implicitní kernel optimalizovaný pro to, aby celý transformer jako celek dal správný GP posterior. Tento nález platí konzistentně pro oba modely.

---

## Experiment 4: PFN vs Nadaraya-Watson vs pravý GP

### Myšlenka

Nadaraya-Watson (NW) estimátor je nejjednodušší forma kernel smoothingu — predikce je vážený průměr trénovacích y hodnot, kde váhy jsou dané RBF kernelem. Celý GP posterior je složitější — zahrnuje inverzi kernel matice K⁻¹, která dekoreluje blízké trénovací body. Chtěli jsme kvantitativně změřit, ke které metodě má PFN blíž.

### Postup

Pro stejná data jsme spočítali predikce tří metod: PFN (forward pass transformeru), Nadaraya-Watson (RBF kernel smoothing s LS=0.3), a pravý GP posterior (se znalostí správných hyperparametrů). Porovnali jsme je pomocí MSE.

### Výsledky a interpretace

Výsledky jsou jednoznačné. MSE mezi PFN a GP je řádově 10⁻⁵, zatímco MSE mezi NW a GP je řádově 10⁻². To je rozdíl čtyř řádů. PFN predikce se prakticky neliší od GP posterioru, zatímco NW vykazuje systematické odchylky — je příliš hladký, nedostatečně reaguje na lokální variace v datech, a v oblastech s hustými trénovacími body nevyužívá informaci o tom, že body jsou korelované.

Tento nález je zásadní: PFN neimplementuje prostý kernel averaging. Aproximuje kompletní GP posterior formuli μ(x*) = k(x*, X) · K⁻¹ · y, kde klíčovou roli hraje inverze kernel matice K⁻¹, která odlišuje GP od NW.

### Závěr

PFN provádí plnou GP inferenci, ne kernel smoothing. Rozdíl čtyř řádů v MSE to potvrzuje bez pochybností.

---

## Experiment 5: Vliv počtu context bodů

### Myšlenka

GP inference se stává obtížnější s rostoucím počtem trénovacích bodů — inverze kernel matice je O(n³). Chtěli jsme zjistit, jak PFN zvládá různé velikosti kontextu a zda jeho přesnost závisí na počtu bodů.

### Postup

Měřili jsme MSE(PFN, GP) pro 5, 10, 20, 30 a 50 context bodů, průměrováno přes 20 opakování.

### Výsledky a interpretace

MSE klesá s rostoucím počtem context bodů u obou modelů. To je intuitivní — s více daty je GP posterior lépe definovaný a PFN ho snáze aproximuje. Velký model má konzistentně nižší MSE pro všechny velikosti kontextu. Zajímavé je, že model nevyžaduje přetrénování pro různé velikosti kontextu — zvládá je přirozeně díky mechanismu `BatchShapeSampler`, který během tréninku náhodně variuje počet bodů.

### Závěr

PFN je robustní vůči velikosti kontextu. Přesnost se zlepšuje s více daty, což odpovídá chování pravého GP posterioru.

---

## Experiment 6: Jedna attention hlava vs Nadaraya-Watson

### Myšlenka

Toto byl nejambicióznější experiment a jádro celé analýzy. Pokud celý PFN aproximuje GP posterior a ten se dá rozepsat jako k(x*, X) · K⁻¹ · y, nabízí se hypotéza o dekompozici: rané vrstvy by mohly počítat kernel similarity (analogie k NW), zatímco pozdější vrstvy by přidávaly korekci odpovídající K⁻¹. Konkrétně: chová se jednotlivá attention hlava jako Nadaraya-Watson estimátor?

### Postup

Pro každou hlavu v každé vrstvě jsme extrahovali attention váhy v kvadrantu Test→Train, spočítali predikci jako `attn_weights @ y_train` (přesně NW operace), a porovnali ji s NW predikcí (RBF kernel) a s pravým GP. Měřili jsme korelaci attention vah s RBF kernelem a MSE predikce jedné hlavy vůči NW a GP. Analýzu jsme provedli systematicky přes všech 48 hlav (6 vrstev × 8 hlav) s průměrováním přes 5 realizací dat.

### Výsledky — Malý model

Na malém modelu jsme pozorovali zajímavý gradient přes vrstvy. Korelace s RBF kernelem je slabá až střední všude (0.3–0.6) — žádná hlava nepoužívá čistý RBF kernel. Ale klíčový nález je v MSE(Head, NW): vrstvy 1–3 mají nízké MSE (0.03–0.15), zatímco vrstvy 0 a 5 mají vysoké (0.5+).

Tady jsme narazili na důležitou otázku: jak může být korelace nízká, ale MSE nízké? Odpověď spočívá v hladkosti GP dat. Když je y_train generované z GP s lengthscale 0.3, sousední body mají podobné y hodnoty. Pak mnoho různých váhových distribucí dá podobný vážený průměr — protože ať dáš váhu bodu 5 nebo bodu 6, y₅ ≈ y₆. Ale pravděpodobnější vysvětlení je, že hlavy se naučily jiný kernel (ne RBF), který ale pro účely kernel smoothingu dává podobné predikce.

Vrstva 5 se chová úplně jinak — korelace ~0, vysoké MSE. Tato vrstva nedělá kernel smoothing vůbec, provádí kvalitativně odlišnou operaci.

### Výsledky — Velký model

U velkého modelu se obraz zásadně změnil. MSE(Head, NW) je teď vysoké ve všech vrstvách — 0.2–0.8 všude. Žádná hlava v žádné vrstvě nedává predikci blízkou NW, ani s RBF, ani s jiným kernelem. Větší model distribuoval výpočet jemněji — každá hlava dělá jen malý příspěvek k celkové predikci a teprve součet všech hlav přes všechny vrstvy a FFN dá GP posterior.

### Interpretace a syntéza obou modelů

Tento kontrast mezi modely je jedním z nejzajímavějších nálezů celé analýzy. Malý (podtrénovaný) model implementuje GP inferenci rozložením na interpretovatelné kroky: rané vrstvy dělají kernel smoothing s naučenými kernely (NW-like), poslední vrstva přidává korekci (efekt K⁻¹). Je to jako student, který řeší úlohu krok po kroku.

Velký model stejnou úlohu řeší jinak — distribuovaně, bez čitelné dekompozice. Výsledek je přesnější, ale postup nelze rozložit na jednoduché kroky. Je to jako expert, který vidí řešení rovnou.

Důležité je, co zůstalo stejné: celková struktura attention (distribuovaná → sparse → Test→Train) je stabilní u obou modelů. Liší se jen míra, v jaké lze jednotlivé komponenty interpretovat jako známé statistické operace.

### Závěr

Hypotéza „jedna attention hlava ≈ Nadaraya-Watson estimátor" je vyvrácena pro oba modely, ale z různých důvodů. U malého modelu hlavy dávají podobné predikce jako NW, ale s jiným kernelem (ne RBF). U velkého modelu ani to neplatí. Dekompozice „rané vrstvy = kernel smoothing, poslední = korekce" je vlastnost podtrénovaného modelu, ne fundamentální mechanismus PFN.

---

## Poznámka o vlivu architektury

Během práce jsme narazili na důležitý metodologický problém. Při tréninku nového modelu bez parametrů `features_per_group=1` a `attention_between_features=False` jsme pozorovali kvalitativně odlišné attention patterny — výrazné vertikální pruhy v první vrstvě a extrémně sparse attention v prostředních vrstvách. Zpočátku jsme to interpretovali jako jinou strategii většího modelu, ale ukázalo se, že jde o artefakt jiné architektury.

Bez těchto parametrů má každý transformer blok dva attention moduly místo jednoho (včetně cross-feature attention), což mění interní tok dat. Pro 1D data (num_features=1) nemá cross-feature attention co mixovat, ale mění se modul, na který hooky zachytávají attention váhy. To znemožňuje srovnání attention patternů mezi architekturami.

Z toho plyne pravidlo: výstupní metriky (MSE, uncertainty kalibrace) lze porovnávat mezi jakýmikoli architekturami, ale attention analýza vyžaduje identickou konfiguraci. Všechny výsledky v tomto dokumentu pocházejí z modelů se shodnou architekturou.

---

## Souhrnné závěry

PFN aproximuje GP posterior s přesností čtyři řády lepší než Nadaraya-Watson. To je nejrobustnější nález celé analýzy — platí konzistentně pro oba modely, všechny velikosti kontextu a opakovaná měření. PFN neimplementuje prostý kernel averaging, ale plnou GP inferenci zahrnující efekt inverze kernel matice.

Model se autonomně naučil správnou kauzální strukturu. Test body attendují na trénovací data, trénovací data ignorují testovací body. Tato asymetrie, viditelná především v poslední vrstvě, odpovídá matematice GP posterioru a je důkazem, že PFN provádí Bayesovskou inferenci.

PFN nepoužívá RBF kernel. Přestože GP prior má RBF kernel, attention váhy s ním korelují jen slabě (0.04–0.65). Model si naučil vlastní implicitní kernel, optimalizovaný pro to, aby celý transformer jako celek dal správný výsledek. To je konzistentní nález u obou modelů.

Delší trénink výrazně zlepšuje kvalitu aproximace, zejména uncertainty kalibraci. MSE škály uncertainty kleslo 5–50× u většího modelu, korelace tvaru vzrostla z 0.95 na 0.99. Přetrvávající gap v extrapolačním ratiu (~50% GP) může být limitací BarDistribution nebo potřebou ještě delšího tréninku.

Interpretovatelnost závisí na míře natrénování. Malý model má čitelnou dekompozici — rané vrstvy provádějí kernel smoothing s naučenými kernely, poslední vrstva přidává korekci na plný GP posterior. Větší model tuto dekompozici nemá — výpočet je distribuovaný přes všechny hlavy a vrstvy, žádná komponenta sama nedělá interpretovatelnou operaci. Ale celek funguje přesněji. To naznačuje, že čitelná dekompozice je vlastnost podtrénovaných modelů, ne fundamentální mechanismus PFN architektury.

Celková struktura attention je stabilní vlastnost architektury. Pattern „vrstva 0 distribuovaná, vrstvy 1–4 sparse, vrstva 5 Test→Train" se opakuje u obou modelů. Mění se jen ostrost a čistota patternů — větší model má čistší strukturu. Toto je nejspolehlivější strukturální nález, který přetrvává napříč různými délkami tréninku.
