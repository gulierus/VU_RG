# Varianta C — odborně a matematicky

*Rigorózní doprovod k [VariantC_vysvetleni_jednoduse.md](VariantC_vysvetleni_jednoduse.md).
Čísla ber z [VariantC_experiments_article.md](VariantC_experiments_article.md), kód z
[../experiments/Experiment_C_PFN_segmentation.ipynb](../experiments/Experiment_C_PFN_segmentation.ipynb).*

---

## 0. Notace a přehled

Pracujeme na diskrétní mřížce $\mathcal{G}=\mathbb{Z}_H\times\mathbb{Z}_W$ (torus), $N=HW$ pixelů;
v experimentech $H=W=64$. Pixely indexujeme $ij$ nebo lineárně $1..N$. Označme:

- $f\in\mathbb{R}^{\mathcal G}$ — latentní pole (Gaussovský proces),
- $x=f+\varepsilon$ — pozorovaný „obrázek", $\varepsilon\sim\mathcal N(0,\sigma^2 I)$,
- $y_{ij}=\mathbf 1[f_{ij}>\tau]\in\{0,1\}$ — binární maska (segmentace),
- $\theta=(\ell,\tau,\sigma^2)$ — hyperparametry jedné úlohy (task),
- $S=\{(x^{(k)},y^{(k)})\}_{k=1}^{n}$ — support set, $x_\star$ — query obrázek, $y_\star$ — jeho maska.

Cíl Varianty C: kvantifikovat, jak dobře **PFN** $\hat p_{\mathrm{PFN}}(x_\star,S)$ aproximuje
pravý per-pixel posterior, a to proti **exaktnímu oraclu** $p_{\mathrm{oracle}}$ dostupnému díky
tomu, že generativní model si volíme sami.

---

## 1. Generativní model úlohy (prior $\Pi$)

Jedna úloha vzniká hierarchicky:

$$
\theta=(\ell,\tau,\sigma^2)\sim\Pi,\qquad
f\mid\theta\sim\mathcal{GP}\!\big(0,\,k_\ell\big),\qquad
x=f+\varepsilon,\ \varepsilon\sim\mathcal N(0,\sigma^2 I),\qquad
y_{ij}=\mathbf 1[f_{ij}>\tau].
$$

Kernel je **stacionární, izotropní RBF na toru**:

$$
k_\ell(u,v)=\sigma_f^2\exp\!\Big(-\frac{d_{\mathrm{tor}}(u,v)^2}{2\ell^2}\Big),\qquad
d_{\mathrm{tor}}(u,v)^2=\sum_{a\in\{x,y\}}\min\!\big(|u_a-v_a|,\;L_a-|u_a-v_a|\big)^2,
$$

s $\sigma_f^2=\text{outputscale}=1$ (fixní). Prior $\Pi$ nad hyperparametry:

$$
\ell/L\sim\text{LogU}(0{,}05,\,0{,}40),\quad
\sigma\sim\text{LogU}(0{,}01,\,0{,}30),\quad
\text{fg}\sim\mathcal U(0{,}15,\,0{,}85),\quad
\tau=\sigma_f\,\Phi^{-1}(1-\text{fg}),
$$

kde $\Phi$ je CDF $\mathcal N(0,1)$. Volba $\tau$ zaručuje marginální foreground fraction
$\mathbb P(f_{ij}>\tau)=\Phi(-\tau/\sigma_f)=\text{fg}$; omezení fg brání degenerativním (celo-0 /
celo-1) maskám. **Support set komunikuje $\theta$** — hladkost $\ell$, práh $\tau$, šum $\sigma^2$ —
takže model musí provést **amortizovanou inferenci hyperparametrů**.

---

## 2. GP na toru: cirkulantní kovariance a spektrum

Protože $k_\ell$ je stacionární a periodické, kovarianční matice $K\in\mathbb R^{N\times N}$
(vektorizovaná mřížka) je **blokově-cirkulantní s cirkulantními bloky (BCCB)**. Takové matice
diagonalizuje 2D diskrétní Fourierova transformace (DFT) $F$ (unitární):

$$
K=F^{*}\,\Lambda\,F,\qquad \Lambda=\mathrm{diag}(\lambda),\qquad
\lambda=\underbrace{\mathrm{DFT}_2(c)}_{\text{vlastní čísla}},\quad c=\text{první „řádek" }K.
$$

Zde $c$ je jádro rozložené na mřížce, $c_{pq}=k_\ell(0,(p,q))$; vlastní čísla $\lambda_k\in\mathbb R_{\ge0}$
(v kódu `Lam = fft2(c).real.clamp_min(0)`). Spektrum $\lambda$ hraje roli **výkonového spektra**
pole.

**Sampling (circulant embedding).** Realizaci $f\sim\mathcal N(0,K)$ získáme spektrálně:

$$
f=\mathrm{Re}\,F^{*}\!\big(\lambda^{1/2}\odot\xi\big),\qquad \xi\ \text{komplexní bílý šum},
$$

což je $O(N\log N)$ (v kódu `_sample_fields` přes `fft2`). Ověření korektnosti (notebook):
$\mathrm{Var}(f)\approx1{,}003$ (cíl $\sigma_f^2=1$) a empirická autokovariance sedí na $c$.

---

## 3. Oracle: exaktní posterior

### 3.1 Posterior nad latentním polem

Model $x=f+\varepsilon$ s $f\sim\mathcal N(0,K)$, $\varepsilon\sim\mathcal N(0,\sigma^2 I)$
nezávislé je společně Gaussovský. Standardní podmiňování dává posterior nad latentním polem:

$$
f\mid x\sim\mathcal N(\mu_{\mathrm{post}},\Sigma_{\mathrm{post}}),\qquad
\mu_{\mathrm{post}}=K(K+\sigma^2 I)^{-1}x,\qquad
\Sigma_{\mathrm{post}}=K-K(K+\sigma^2 I)^{-1}K.
$$

Kovarianci lze zjednodušit:
$\Sigma_{\mathrm{post}}=\sigma^2 K(K+\sigma^2 I)^{-1}=(K^{-1}+\sigma^{-2}I)^{-1}$.

### 3.2 Spektrální forma posteriorní střední hodnoty (Wienerův filtr)

V bázi $F$ je vše diagonální; per frekvenci $k$:

$$
\widehat{\mu}_{\mathrm{post},k}=\underbrace{\frac{\lambda_k}{\lambda_k+\sigma^2}}_{\text{Wiener gain}}\widehat{x}_k
\;\Longrightarrow\;
\mu_{\mathrm{post}}=F^{*}\!\Big(\tfrac{\lambda}{\lambda+\sigma^2}\odot Fx\Big)
=\texttt{ifft2}\!\big(\tfrac{\lambda}{\lambda+\sigma^2}\cdot\texttt{fft2}(x)\big).
$$

### 3.3 Posteriorní rozptyl (konstantní přes pixely)

Protože $\Sigma_{\mathrm{post}}$ je také BCCB (cirkulantní), její diagonála je konstantní přes
pixely (důsledek stacionarity) a rovná se průměru vlastních čísel:

$$
s_{\mathrm{post}}^2=(\Sigma_{\mathrm{post}})_{ii}=\frac1N\sum_{k}\frac{\sigma^2\lambda_k}{\lambda_k+\sigma^2}
\qquad(\forall i).
$$

### 3.4 Posterior predictive labelu

Maska je deterministická funkce pole, takže per-pixel posterior predictive je exaktně

$$
\boxed{\,p_{\mathrm{oracle},ij}=\mathbb P(y_{ij}=1\mid x)=\mathbb P(f_{ij}>\tau\mid x)
=\Phi\!\Big(\frac{\mu_{\mathrm{post},ij}-\tau}{s_{\mathrm{post}}}\Big).\,}
$$

(Marginála $f_{ij}\mid x$ je $\mathcal N(\mu_{\mathrm{post},ij},s_{\mathrm{post}}^2)$; per-pixel
pravděpodobnost potřebuje jen tuto marginálu, ne celou $\Sigma_{\mathrm{post}}$.)

### 3.5 Explicitní prior maska

Bez pozorování je $f_{ij}\sim\mathcal N(0,\sigma_f^2)$, takže

$$
p_{\mathrm{prior},ij}=\mathbb P(f_{ij}>\tau)=\Phi(-\tau/\sigma_f)=\text{fg}\quad(\text{uniformní}).
$$

To je „defaultní" maska, ke které by amortizovaný prediktor kolaboval při neinformativním kontextu.

### 3.6 Výpočetní složitost

Naivně $(K+\sigma^2 I)^{-1}$ stojí $O(N^3)=O((HW)^3)$; cirkulantní/FFT forma je $O(N\log N)$. Proto
je oracle na $64\times64$ levný a exaktní (past C.8: pro krátké $\ell$ je $K$ špatně podmíněná,
$\sigma^2$ působí jako nugget/jitter a stabilizuje inverzi).

### 3.7 Shrnutí

Pro známé $\theta=(\ell,\tau,\sigma^2)$ a pozorovaný query $x$ je per-pixel posterior predictive
labelu **exaktně** $p_{\mathrm{oracle},ij}=\Phi\big((\mu_{\mathrm{post},ij}-\tau)/s_{\mathrm{post}}\big)$,
kde $\mu_{\mathrm{post}}$ je Wienerův filtr $x$ (spektrální gain $\lambda/(\lambda+\sigma^2)$) a
$s_{\mathrm{post}}$ je konstantní posteriorní směrodatná odchylka daná spektrem. Cirkulantní
struktura činí celý výpočet $O(N\log N)$ a numericky stabilní ($\sigma^2$ jako jitter). Marginální
prior maska je uniformní $\Phi(-\tau/\sigma_f)=\text{fg}$. Toto $p_{\mathrm{oracle}}$ je referenční
„pravda", proti níž měříme PFN v §5–§7.

---

## 4. PFN: architektura, trénovací objektiv a referenční veličina

### 4.1 Architektura

UniverSeg (encoder–decoder), random init; **CrossBlock** vyměňuje featury mezi
query a každým support párem (globální feature-averaging). Výstup: 1 logit/pixel
$g_\phi(x_\star,S)\in\mathbb R^{\mathcal G}$; predikce $\hat p_{\mathrm{PFN}}=\varsigma(g_\phi)$,
$\varsigma$ = sigmoid.

### 4.2 Trénovací objektiv

Jeden krok: $\theta\sim\Pi$; nasampluj $n+1$ polí; $S$ z $n$ dvojic, query $(x_\star,y_\star)$.
Ztráta = **per-pixel binary cross-entropy**

$$
\mathcal L(\phi)=\mathbb E_{\theta\sim\Pi}\,\mathbb E_{S,x_\star,y_\star}\,
\frac1N\sum_{ij}\mathrm{BCE}\big(\varsigma(g_\phi)_{ij},\,y_{\star,ij}\big),\qquad
\mathrm{BCE}(q,y)=-y\log q-(1-y)\log(1-q).
$$

### 4.3 Konzistence (sigmoid, BCE) s posteriorem

BCE je **striktně vlastní skórovací pravidlo** (strictly proper scoring rule). Pro pevný vstup je
populační minimalizátor

$$
\arg\min_{q}\ \mathbb E_{y}\big[\mathrm{BCE}(q,y)\mid x_\star,S\big]=\mathbb E[y_\star\mid x_\star,S]
=\mathbb P(y_\star=1\mid x_\star,S).
$$

Perfektně natrénovaný PFN tedy realizuje **amortizovaný posterior predictive**

$$
p_{\mathrm{PFN}}^{\star}(x_\star,S)=\mathbb P(y_\star=1\mid x_\star,S)
=\int \mathbb P(y_\star=1\mid x_\star,\theta)\;\underbrace{p(\theta\mid S)}_{\text{HP posterior}}\,d\theta .
$$

### 4.4 Dvě referenční veličiny: oracle a amortizovaný posterior predictive

- **Oracle** $p_{\mathrm{oracle}}=\mathbb P(y_\star=1\mid x_\star,\theta_{\mathrm{true}})$ — podmiňuje
  na *známé* $\theta$.
- **Populační optimum PFN** $p_{\mathrm{PFN}}^{\star}=\mathbb P(y_\star=1\mid x_\star,S)$ —
  marginalizuje $\theta$ přes jeho posterior daný $S$.

Liší se o **nejistotu v hyperparametrech**: $p_{\mathrm{PFN}}^{\star}\to p_{\mathrm{oracle}}$ když
$p(\theta\mid S)\to\delta_{\theta_{\mathrm{true}}}$, tj. $n\to\infty$. Měřit PFN proti oraclu proto
mírně „účtuje" i tuto neredukovatelnou HP-nejistotu. Empiricky je ale rozdíl malý, protože **query
$x_\star$ je husté pozorování** a $\theta$ je z něj + z pár support párů skoro identifikovatelné
(viz slabá závislost na $n_{\mathrm{supp}}$, §7). Oracle proto bereme jako čistou referenci a tento
caveat explicitně přiznáváme.

---

## 5. Metriky (formálně)

Nechť $p:=p_{\mathrm{oracle}}$, $q:=\hat p_{\mathrm{PFN}}$; průměry přes pixely a query.

**Fidelita (L1 / Brier).** $\ \mathrm{Fid}_{L_1}=\frac1N\sum_{ij}|q_{ij}-p_{ij}|$. Čistá míra
amortizační chyby (jinde referenční posterior chybí).

**KL vůči oraclu.** $\ \mathrm{KL}(p\Vert q)=\frac1N\sum_{ij}\big[p_{ij}\log\frac{p_{ij}}{q_{ij}}
+(1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}\big]$ (s clampem $q\in[\epsilon,1-\epsilon]$).

**Prediktivní entropie.** $\ H_b(p)=-p\log p-(1-p)\log(1-p)$; sledujeme $\overline{H_b(q)}$ vs
$\overline{H_b(p)}$ (míra „ostrosti").

**Kalibrace (ECE) vůči oraclu.** Binujeme podle $p_{\mathrm{oracle}}$; v binu $B_m$
$\ \mathrm{ECE}=\sum_m\frac{|B_m|}{N}\,\big|\overline{q}_{B_m}-\overline{p}_{B_m}\big|$.

### 5.1 Bayes floor a excess risk (klíčová identita)

Pro cíl = **pravá tvrdá maska** $y$ a libovolný prediktor $q$ platí (rozklad vlastního
skórovacího pravidla):

$$
\mathbb E_{y\sim\mathrm{Ber}(p)}[\mathrm{BCE}(q,y)]=\underbrace{H_b(p)}_{\text{entropie}}+\mathrm{KL}(p\Vert q),
\qquad p=\mathbb P(y=1\mid x).
$$

Zprůměrováním přes $x$:

$$
\underbrace{\mathrm{BCE}(q)}_{\text{expected loss}}
=\underbrace{\mathbb E_x[H_b(p_{\mathrm{oracle}})]}_{\textbf{Bayes floor}\ =\ H(y\mid x)}
+\underbrace{\mathbb E_x[\mathrm{KL}(p_{\mathrm{oracle}}\Vert q)]}_{\textbf{excess risk}\ \ge 0}.
$$

**Interpretace.** *Bayes floor* je **podmíněná entropie** $H(y\mid x)$ — neredukovatelná chyba
daná pozorovacím šumem, jíž dosahuje právě oracle ($q=p_{\mathrm{oracle}}\Rightarrow\mathrm{KL}=0$).
**Excess risk** libovolného prediktoru je $\ge0$ a rovná se přesně **očekávané KL divergenci od
pravého posterioru**. Proto:

$$
\boxed{\ \text{excess risk}_{\mathrm{PFN}}=\mathrm{BCE}_{\mathrm{PFN}}-\mathrm{BCE}_{\mathrm{oracle}}
=\mathbb E_x\big[\mathrm{KL}(p_{\mathrm{oracle}}\Vert\hat p_{\mathrm{PFN}})\big]\ge 0.\ }
$$

Naměřený „excess" sloupec (Krok 6) je tedy odhad $\mathbb E[\mathrm{KL}(\text{oracle}\Vert\text{PFN})]$
a jeho nezápornost je matematická nutnost (pozorovaná ve všech režimech).

### 5.2 Bias–variance rozklad (per pixel)

Fixujeme úlohu $\theta$ i query $x_\star$; jediná náhoda je **tah support setu** $S$. Per pixel:

$$
\mathrm{bias}_{ij}=\mathbb E_S[\hat p_{ij}]-p_{\mathrm{oracle},ij},\qquad
\mathrm{var}_{ij}=\mathrm{Var}_S[\hat p_{ij}],
$$

$$
\mathbb E_S\big[(\hat p_{ij}-p_{\mathrm{oracle},ij})^2\big]=\mathrm{bias}_{ij}^2+\mathrm{var}_{ij}.
$$

Reportujeme $\overline{\mathrm{bias}^2}$ a $\overline{\mathrm{var}}$ vs $n_{\mathrm{supp}}$. (Past
z Ch.3: nefitujeme degenerovaný joint $\mathrm{bias}^2+c/n$ s hranicemi — sklon variance čteme
přímo v log-log a $\mathrm{bias}^2$ jako plateau.)

---

## 6. Teoretické zakotvení (Nagler)

- **Locality condition (Thm 5.4).** Amortizovaný prediktor s **globální** (softmax-like) attention
  má tzv. tilted-measure limit, který se nelokalizuje kolem $x_\star$ → **neredukovatelný bias**.
  V našem 2D CrossBlocku má feature-averaging přes support globální dosah, takže očekáváme
  strukturální bias soustředěný na hranicích masky.
- **Variance (Thm 6.2).** Nagler bounduje **absolutní deviaci** predikce (doslovné znění, str. 6):
  $$\big|q_\theta(y\mid x,D_n)-\mathbb E[q_\theta(y\mid x,D_n)]\big|\lesssim n^{-1/2}\quad\text{s vysokou pravděpodobností.}$$
  Bounded je tedy **magnituda fluktuace** (std-škála), ne přímo rozptyl; „the variance vanishes"
  je Naglerův prozaický popisek. Protože $q_\theta\in[0,1]$ je omezené, deviace $\sim n^{-1/2}$
  implikuje **rozptyl** $\mathrm{Var}\sim n^{-1}$. Formálně (Appendix A.8 / Thm A.1) síť splňuje
  podmínku (5) s $\alpha=1$, což dává právě $n^{1/2-\alpha}=n^{-1/2}$. Mizení variance je dle
  Naglera *strukturální* (vlastnost architektury, nezávislá na $\theta$; attention dává každému
  vzorku váhu řádu $1/n$).

  *Nezávislý argument (proč $-1$ dává smysl i bez Naglera):* prediktor je v jádru vážený průměr
  přes $\sim n$ support příspěvků s vahami $O(1/n)$; rozptyl takového průměru je $\mathrm{Var}[\bar\cdot]\sim1/n$
  elementárně (jako CLT-škála). Sklon $-1$ tedy potvrzují dvě nezávislé úvahy.

Naše měření (§7): log-log sklon $\overline{\mathrm{var}}$ je $\approx-1$, což je **přesná shoda**
s Thm 6.2 (deviace $\lesssim n^{-1/2}\Leftrightarrow\mathrm{Var}\sim n^{-1}$) — ne překonání teorie.
$\overline{\mathrm{bias}^2}$ drží malé kladné plateau, konzistentní s Naglerovou predikcí
neredukovatelného biasu. Že plateau odpovídá **strukturálnímu** biasu (a ne optimalizačnímu
reziduu jednoho běhu), je ověřeno přes **3 nezávislé seedy**: plateau se shoduje na jednotky
procent (CV 1,6 % Easy / 4,4 % Hard, §7f).

---

## 7. Výsledky (rigorózně, s čísly)

**(a) Fidelita a její rozpad.** $\mathrm{Fid}_{L_1}$ vs $n_{\mathrm{supp}}$ přes režimy tvrdosti
(řízené $\ell/L$): OOD-short ($0{,}03$) nejhorší (~$0{,}034$, s kontextem neklesá); Hard/Medium
nejnižší (~$0{,}005$–$0{,}008$); extrapolace **asymetrická** — delší $\ell$ (OOD-long) model dožene,
kratší (OOD-short) ne. Závislost na $n_{\mathrm{supp}}$ je **slabá** (hustý query → $\theta$ skoro
identifikovatelné z jednoho páru; strukturní rozdíl vůči řídké GP regresi).

**(b) PFN vs Bayes floor (Krok 6).** BCE vůči pravé masce (S=16, $\sigma$ mix):

| režim | $\mathrm{BCE}_{\mathrm{PFN}}$ | Bayes floor $H(y\mid x)$ | excess $=\mathbb E[\mathrm{KL}]$ |
|---|---|---|---|
| OOD-short | 0,1164 | 0,0586 | **+0,0577** |
| Hard | 0,0463 | 0,0420 | **+0,0044** |
| Medium | 0,0315 | 0,0250 | **+0,0065** |
| Easy | 0,0480 | 0,0327 | +0,0153 |
| OOD-long | 0,0403 | 0,0272 | +0,0131 |

In-distribution je excess risk $\approx0{,}004$–$0{,}006$ nat/pixel → v čistém matched režimu je PFN
**skoro Bayes-optimální**. OOD-short: excess $\approx9\times$ vyšší. Easy $>$ Hard, protože hladká
pole mají široké nejisté hranice (velké $\overline{H_b(p_{\mathrm{oracle}})}$), kde mírná
over-sharpness stojí nejvíc KL.

> **Pozor na čtení tabulky vs. tréninkové plató.** Tréninkové plató $0{,}0194$ je nižší než každý
> floor v tabulce (min $0{,}025$), což zdánlivě porušuje excess $\ge0$. Není tomu tak: tabulkové
> floory jsou **tvrdší řezy** (fixní fg$=0{,}5$, dané režimy), kdežto plató je průměr přes **celé**
> $\Pi$. Log-uniformní $\sigma\in[0{,}01,0{,}30]$ klade většinu hmoty na malý šum (medián
> $\sigma\approx0{,}055$), kde jsou masky skoro bezšumové a $H_b\approx0$. Nezávisle spočtený
> full-prior floor je $\mathbb E_\Pi[H_b(p_{\mathrm{oracle}})]=0{,}0133$ (medián $0{,}0095$, 5–95 %
> $[0{,}0020,\,0{,}0420]$), takže $0{,}0194\ge0{,}0133$ a excess $\approx0{,}006$ — konzistentní.
> Plató tedy sedí těsně nad *full-prior* floorem, ne nad tabulkovými řezy.

**(c) Kolaps k prioru — negativní.** Průkazná veličina je $d_{\mathrm{oracle}}=|\hat p-p_{\mathrm{oracle}}|$:
zůstává malé a ploché i při $n_{\mathrm{supp}}=1$, takže kolaps ($\hat p\to\text{fg}$)
**nenastává**. *(Doplňkové $d_{\mathrm{prior}}=|\hat p-\text{fg}|\approx0{,}47$ je téměř neinformativní
diagnostik: pro libovolný sebejistý model, $\hat p\approx\{0,1\}$, platí
$\mathbb E|\hat p-\text{fg}|\approx2\,\text{fg}(1-\text{fg})\approx0{,}47$ při fg$\approx0{,}5$
nezávisle na kolapsu; proto argumentujeme $d_{\mathrm{oracle}}$, ne $d_{\mathrm{prior}}$.)* Kolaps by
vyžadoval řídký neinformativní kontext (Varianta A); není to univerzálie amortizace, ale důsledek
řídkosti dat.

**(d) Bias–variance.** $\overline{\mathrm{var}}$ klesá s $n_{\mathrm{supp}}$ k $\approx0$ s log-log
sklonem $\approx-1$ — tj. $\mathrm{Var}\sim n^{-1}$, **přesná shoda s Naglerovým Thm 6.2**.
$\overline{\mathrm{bias}^2}$ drží malé plateau (~$0{,}0015$ Hard, ~$0{,}0031$ Easy): variance mizí,
bias přetrvává.

**(e) Kalibrace vs oracle.** Rozdělení $\hat p_{\mathrm{PFN}}$ a $p_{\mathrm{oracle}}$ se téměř
překrývají; $\overline{H_b}$ srovnatelné i na nejisté množině (0,522 vs 0,552). ECE(oracle)
$=0{,}0049$ celkově, $0{,}0369$ na nejisté množině ($\sim5\%$ pixelů). Tj. **věrná reprezentace
nejistoty** s jen mírnou zbytkovou over-sharpness na hranicích.

**(f) Strukturálnost biasu — 3 seedy.** Aby šlo plateau prohlásit za *strukturální* (vlastnost
architektury), a ne za optimalizační reziduum jednoho běhu, natrénovali jsme 3 modely (seed 0/1/2;
shodný config, mění se init i realizace dat) a změřili plateau na **stejné** fixní sadě úloh
(`background/variant_c_seed_aggregation.py`):

| režim | plateau (seed 0/1/2) | mean ± std | CV | sklon var. |
|---|---|---|---|---|
| Easy | 0,00311 / 0,00302 / 0,00309 | 0,00307 ± 0,00005 | 1,6 % | −1,30 |
| Hard | 0,00147 / 0,00144 / 0,00156 | 0,00149 ± 0,00007 | 4,4 % | −0,96 |

Plateau je napříč seedy shodné (CV $\ll$ 25 %) → **bias je strukturální**, ne artefakt běhu; sklon
variance je taktéž stabilní. Naglerova predikce neredukovatelného biasu je tím potvrzena empiricky.

---

## 8. Souhrn a limity

**Hlavní teze (opatrně formulovaná).** C **neukazuje, že PFN nemá patologie** — ukazuje, že se
v **čistém, hustém, matched režimu neprojevují**. Slabá závislost na $n_{\mathrm{supp}}$ (§7a),
chybějící kolaps (§7c) i malý excess risk (§7b) mají **společnou příčinu**: úloha je skoro
identifikovatelná i bez support setu. Hustý query $x_\star$ sám prozradí $\ell$ (hladkost) a
$\sigma$ (lokální rozptyl); jediné, co z něj nevyčteš, je práh $\tau$, a ten pinne už **jediná**
support maska. Proto (i) neexistuje neinformativní režim, ve kterém by kolaps mohl nastat, a (ii)
„near-Bayes-optimalita" je zčásti tím, že testujeme **snadný inferenční problém**. Patologie
předpovězené teorií (kolaps, velký amortizační bias) proto vyžadují **řídký kontext** (Varianta A)
nebo **OOD** — a přesně tam se v C i objevují (excess risk $\sim10\times$ u OOD-short, asymetricky
v $\ell$, křížově konzistentní s 1D nálezem z GP2).

**Proč to tedy měřit (C jako kontrolní podmínka).** C není „nepovedlo se ukázat patologii", ale
**clean baseline**: dokazuje, že v matched/hustém režimu je PFN *prokazatelně* skoro
Bayes-optimální (excess $\approx0{,}006$ ověřený proti exaktnímu oraclu). Tím **vylučuje**, že by
kolaps ve Variantě A nebo selhání v OOD byly dílem vadné architektury či špatného tréninku — model
umí být optimální, když je úloha identifikovatelná. Patologie jsou tedy **podmíněné režimem**
(řídkost / OOD), ne defektem modelu. Bez C by tenhle rozdíl nešel odlišit.

**Souhrn měření (formálně).** PFN realizuje amortizovaný posterior predictive; proti exaktnímu
oraclu je $\mathbb E[\mathrm{KL}(\text{oracle}\Vert\text{PFN})]$ řádu $10^{-3}$ in-distribution
a roste $\sim10\times$ OOD (asymetricky v $\ell$). Rozptyl $\to0$ jako $n^{-1}$ (přesně Thm 6.2),
zbytkový bias přetrvává, kalibrace věrná s mírnou over-sharpness na hranicích.

**Limity (poctivě).**
1. **Strukturálnost biasu — VYŘEŠENO (3 seedy).** Původní výtka „jeden seed → nelze prohlásit za
   strukturální" je zodpovězena: plateau se přes seedy 0/1/2 shoduje (CV 1,6 %/4,4 %, §7f), takže
   „strukturální bias" je oprávněné psát natvrdo. (Zůstává rozlišené jen na dvou režimech Easy/Hard;
   plný $\ell$-profil biasu je možné rozšíření.)
2. **Snadný inferenční problém (matched prior + husté pozorování).** „Near-optimal" je zčásti
   vlastnost setupu, ne jen modelu; $\theta$ je skoro identifikovatelné z $x_\star$. Praktické
   riziko a zajímavý režim je **OOD** a **řídký kontext**, ne in-distribution.
3. **Oracle vs $p_{\mathrm{PFN}}^{\star}$.** Měříme proti $\theta$-known oraclu, ne proti
   $\theta$-marginalizovanému populačnímu optimu PFN; rozdíl je HP-nejistota (empiricky malá, §4),
   ale znaménkově „přičítá" PFN chybu, za kterou nemůže.
4. **Torus** je zjednodušení kvůli exaktnímu $O(N\log N)$ oraclu; neperiodická/reálná data =
   další krok.
5. **Trénink** (log `pfn_seg.o235933`): konvergoval, loss $0{,}15\to0{,}0194$, pix-acc $0{,}991$,
   grad-clip pohltil přechodné výkyvy gradientu (ep. 76–77); `best` = epocha 250.
