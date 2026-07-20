# Pravidla psaní práce

Souhrn všech stylistických, typografických a strukturních pravidel, na kterých jsme se
domluvili při psaní textu práce. Platí pro všechny soubory `text/Kapitola*.tex` a pro každý
nový text. Cílem je konzistence napříč celou prací a zachování autorova hlasu.

---

## 1. Jazyk a tón

- **Čeština.** Text práce, popisky, komentáře.
- **Didaktický, výkladový tón.** Píše se, jako by autor vedl čtenáře za ruku. Inkluzivní
  imperativy a autorský plurál: „Ukážeme", „Představme si", „Zavedeme", „Označme", „Uveďme",
  „Ukažme", „Položme", „Mějme", „Poznamenejme", „Připomeňme".
- **Autorský plurál (my).** Vždy „ukážeme, provedeme, zjistíme". Nikdy 1. osoba jednotného
  čísla ani neosobní pasivum tam, kde jde použít „my".
- **Postup od obecného ke konkrétnímu.** Nejdřív problém a princip, pak konkrétní příklad.
- **Jednoduchý jazyk pro složité věci.** Podstata stylu je vysvětlit složité jednoduše. Ale
  jednoduchý **není** hravý ani básnický. Žádné metafory a expresivní obraty v odborném textu
  (ne „nejistota žije na hranici", „netáhne to novost domény" apod.) — piš doslovně a jednoznačně.
- **Krátké věty, jedna myšlenka na větu.** Skoro každé souvětí rozděl.
- **Hedging.** Silná tvrzení změkčuj: „může prozrazovat", „dá se říct, že", „tímto usuzujeme",
  „se může zdát paradoxní".
- **Konektivy.** „totiž", „Současně" (preferuj před „Zároveň"), „opět", „již", „tedy", „tzv.",
  „Nejprve / Dále / Následně / Nakonec", „Za prvé / Za druhé", „Avšak", „Ovšem", „Přesto".

---

## 2. Struktura výkladu

- **Roadmap odstavec u každé kapitoly i sekce.** U kapitoly krátká narativní roadmapa (co a
  proč se ukáže). U sekce **jedna až dvě věty**, ne víc.
- **Žádné druhé roadmapy a sliby dopředu.** Nevypisuj obsah („Nejprve popíšeme… poté… Na konci
  shrneme…") a neslibuj věci, které ještě nebyly zavedeny.
- **Žádné prázdné meta-věty.** Vynech „Nyní se dostáváme k hlavnímu měření.", velké závěrečné
  fráze („Hlavní linie práce tak zůstává stejná.") apod.
- **Terminologii zaváděj explicitně.** Nový pojem při prvním výskytu: „tzv." + `\emph{}` +
  okamžité vysvětlení. Anglické termíny nenechávej bez uvození (závorka s českým protějškem).
- **Jedna věta = jeden řádek zdroje.** Kvůli čitelným git diffům. Odstavce odděluj `\par`.
- **Každý blok interpretace výsledku začíná pseudo-hlavičkou `\textbf{Diskuze:}`** (prázdný
  řádek za ní, pak „Z tabulky/obrázku plyne…").
- **Složité obrázky dostanou reader-guide odstavec** — před náročným grafem vysvětli krok za
  krokem, co znázorňuje a jak ho číst (osy, barvy, co který vzor znamená).
- **Shrnutí sekce se jmenuje vždy `\subsection{Shrnutí}`** — nikdy s podtitulem.
- **Ohlášené vynechávky.** Když něco vynecháváš, řekni to nahlas i s důvodem a odkazem
  („bez důkazu, viz [X]", „pouhá analogie, proto neuvádíme").
- **Kotvi vágní odkazy na konkrétní místa.** „Zatím" → „V předchozí kapitole"; „co už umíme" →
  „co jsme viděli v kapitole~\ref{...}".
- **Nikdy neodkazuj v shrnutí na experiment, který sekce nepředstavila.** Čtenář se nesmí ztratit.

---

## 3. Zvýrazňování (bold / emph / italic)

- **Tučně (`\textbf`) jen dvě věci:**
  1. pseudo-hlavičky odstavců („\textbf{Diskuze:}", „\textbf{Model se fixními hyperparametry:}",
     „\textbf{Návaznost na Naglerovu teorii:}"),
  2. paragraph-initial definiční labely a záměrné tučné odrážky (styl `BarDistribution`).
- **Termíny NIKDY tučně — ani při prvním výskytu.** Nový termín = `\emph{}` (nebo `\textit{}`
  pro české coinages) + vysvětlení.
- **Tučná slova uprostřed vět jsou zakázaná.** Žádné `\textbf{nemá}`-style zvýraznění; místo
  toho prostý text.
- **`\mathbf{}` na hodnotách v tabulkách NIKDY** — ani pro označení nejlepší hodnoty v řádku.
  `\mathbf{x}` pro vektory je v pořádku.
- **Žádný `\textbf` v popiscích obrázků** (panelové značky „Nahoře:/Vlevo:" plain).
- **Žádný `\textbf` v hlavičkách tabulek** — všechny tabulky plain headers.
- **Kurziva (`\emph{}`) pro anglické termíny** — attention, seed, kernel, oracle, token,
  support set, in-context, Bayes floor, Dice, reliability, ECE, encoder/decoder, TabPFN,
  kernel-averaging… — **každý výskyt, všechny tvary**.
- **Složené termíny do JEDNOHO `\emph{}`** — `\emph{attention hlavy}`, `\emph{attention váhy}`,
  `\emph{value vektory}` — nikdy `\emph{attention} hlavy`.
- **`\textit{}` pro české zdůraznění a kontrastní dvojice** (tvar/velikost, správný/špatný,
  kde/kolik). Dekorativní kurzivu na běžných českých slovech netěž.
- Metriky `\emph{labelová korelace}` / `\emph{kernelová korelace}` kurzivou celé, včetně
  podstatného jména. „kernelová matice" a „RBF kernel(em)" zůstávají plain.

---

## 4. Rod vybraných pojmů

- **„attention" = střední rod:** „vysoké attention", „své attention", „attention je ostré",
  „s lineárním attention", „globálním attention".
- **„PFN" = střední rod:** „PFN je založeno / trénováno / konkurenceschopné / kalibrované",
  „jeho prior", „PFN dostávalo / předpovídalo". Ale predikátová jména si drží vlastní rod:
  „PFN je architektura / transformer / model", „PFN transformer" skloňuj podle „transformer".
- Složené termíny berou rod hlavního jména (attention matice → ženský, attention mechanismus →
  mužský).

---

## 5. Slovník (preferované výrazy)

| Používej | Nepoužívej |
|---|---|
| rozptyl | variance |
| rozdělení | distribuce |
| vzorkované | nasamplované |
| oblast | region |
| vzor (pro patterns) | vzorec |
| sekce / část | oddíl |
| stejný | týž / tentýž / tatáž |
| fixní model / náhodný model | „100-epoch model" |
| korektní, česky opsané | „worst-case", „fér" |
| ale, a to, a tou je | nýbrž |
| zcela nová úloha | odbočka |
| Současně | (méně) Zároveň |

- Nezaváděj hravá/anglická slova, i když pocházejí z tvých neformálních poznámek („blob" →
  „útvar", „defaultní" → „výchozí", „centrovaný" → „uprostřed / vystředěný", „rozčepýřené" →
  „ostré", „venku" → „mimo", „šablona" → „obrys").

---

## 6. Pomlčky a interpunkce

- **Žádné em-pomlčky v běžném textu.** `X – Y` nahraď větným předělem nebo „, a to", „,
  konkrétně", „, tj.", „, tedy", „, a tou je". Pomlčky tolerované jen v popiscích a hned za
  tučným právě definovaným termínem.
- **Dvojtečka uprostřed věty → „, a to" nebo předěl.** Dvojtečka zůstává jen před vzorcem,
  výčtem nebo otázkou.
- **České uvozovky „…".**
- Za jednopísmennými předložkami v nových větách `~` nemusí být; u `~\ref` ho drž.

---

## 7. Čísla a citace

- **Konkrétní efekt-size čísla nepatří do prózy.** Píše se „chyba se zvedne, jak je vidět
  v tabulce~\ref{...}", ne „na 10 až 25násobek". Čísla žijí v tabulkách a jejich popiscích;
  próza uvádí kvalitativní závěr a odkaz.
- **Design hodnoty jsou v pořádku** (n, ℓ, σ², počet epoch, počet parametrů, rozsahy).
- **Neodvozuj čísla z obrázků** (žádné „prior jí přiřazuje ~3,4 %").
- **Čísla ber z měřených tabulek v `background/*_experiments_article.md`, ne z paměti.**
  Side-derivace a poznámky v těchto souborech **nejsou** autoritativní, jen měřené tabulky.
- **Citace jsou placeholdery** `[Rasmussen]` nebo `(Müller et al., 2022)`; bibliografie ručně
  v `thebibliography`. TODO citaci nech jako „(zde přidat reference)".
- **České desetinné číslo v math módu jako `{,}`:** `$0{,}3$`, `$14{,}14$`.

---

## 8. Obrázky a tabulky

- **Odstraň pythonovské nadpisy (suptitle) z obrázků** — house-style. Panelové titulky a osy
  se ponechávají; hodnotící/anglické suptitles se ořezávají (popisek nese vysvětlení).
- **Popisek nejdřív popisuje prózou, co na obrázku je, pak teprve interpretuje.**
- **Každý obrázek a tabulka MUSÍ mít aspoň jeden odkaz v textu.** Objekt bez odkazu je hrubá
  chyba (a plave po sazbě).
- **Žádné velké svislé mezery** kolem obrázků/tabulek. Umístění `[!tb]` / `[!htb]`, v preambuli
  `\raggedbottom`.
- **Tabulka + obrázek, které mají držet spolu na jedné straně**, slučuj do jednoho
  celostránkového floatu `figure[p]` s `\captionof{table}{...}` a `\vfill` mezi bloky.
- Obrázky vždy z `figures/<experiment>/...` (relativní cesta z kořene).
- Jediný zdroj obrázků je `figures/` — needituj obrázky na dvou místech; vše plyne z notebooků
  přes `background/extract_figures.py`. Když upravíš PNG ručně, oprav i generující notebook.

---

## 9. LaTeX a sazba

- **Kompilace z adresáře `text/`:** `cd text && latexmk -pdf VU_RG.tex`.
- **Kapitoly přes `\input`, ne `\include`** (`\include` vynucuje `\clearpage`).
- **Nové sekce ani podsekce nezačínají na nové straně** — jen nové kapitoly. Žádný
  `\clearpage`/`\newpage` před (pod)sekcemi.
- **Názvy kapitol sjednocené na `Kapitola*.tex`** (velké K), aby fungovaly na case-sensitive
  systémech.
- **Křížové odkazy s prefixy** `chap:`, `sec:`, `subsec:`, `fig:`, `tab:`, `eq:`; před `\ref`
  nezlomitelná mezera `~`.
- **Vektory a matice tučně** `\mathbf{x}`, `\boldsymbol{\theta}`; prostory/množiny `\mathcal{}`.
- **Rovnice** číslované přes `equation` + `\label{eq:...}`; nečíslované jednorázové `$$…$$`, ale
  **ne uprostřed věty** (to dá ošklivou mezeru — použij inline `$…$`).
- **Kód a identifikátory** `\texttt{}` nebo `\lstinline|...|`; delší ukázky `listings`.
- **Theorem prostředí** `\newtheorem{definition}{Definice}[chapter]`,
  `\newtheorem{theorem}{Věta}[section]`.

---

## 10. Nadpisy sekcí

- **Věcné, ne tázací.** Ne „Dělá PFN něco podobného jako GP?" → „Srovnání s RBF kernelem".
- **Nepojmenovávej podsekce písmeny variant** („Varianta A/C") — v nadpisech ani v próze;
  popiš, o co jde.
- Ustálené odborné pojmy (např. „kolaps k prioru") se v nadpisu používat smí, pokud jsou v textu
  zavedené. „kolaps" není povinný překlad `prior collapse`, ale rozhodli jsme se ho ponechat a
  při prvním výskytu (v úvodu kapitoly) rozepsat, co znamená.

---

*Tato pravidla jsou destilát autorových vlastních úprav a jeho bakalářské práce. Když se dva
body dostanou do sporu, vyhrává konkrétní autorova ruční úprava — pravidla se pak aktualizují
podle ní.*
