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

Model $x=f+\varepsilon$ s $f\sim\mathcal N(0,K)$, $\varepsilon\sim\mathcal N(0,\sigma^2 I)$
nezávislé je společně Gaussovský. Standardní podmiňování dává **posterior nad latentním polem**:

$$
f\mid x\sim\mathcal N(\mu_{\mathrm{post}},\Sigma_{\mathrm{post}}),\qquad
\mu_{\mathrm{post}}=K(K+\sigma^2 I)^{-1}x,\qquad
\Sigma_{\mathrm{post}}=K-K(K+\sigma^2 I)^{-1}K.
$$

Kovarianci lze zjednodušit:
$\Sigma_{\mathrm{post}}=\sigma^2 K(K+\sigma^2 I)^{-1}=(K^{-1}+\sigma^{-2}I)^{-1}$.

**Spektrální (FFT) forma — Wienerův filtr.** V bázi $F$ je vše diagonální; per frekvenci $k$:

$$
\widehat{\mu}_{\mathrm{post},k}=\underbrace{\frac{\lambda_k}{\lambda_k+\sigma^2}}_{\text{Wiener gain}}\widehat{x}_k
\;\Longrightarrow\;
\mu_{\mathrm{post}}=F^{*}\!\Big(\tfrac{\lambda}{\lambda+\sigma^2}\odot Fx\Big)
=\texttt{ifft2}\!\big(\tfrac{\lambda}{\lambda+\sigma^2}\cdot\texttt{fft2}(x)\big).
$$

Protože $\Sigma_{\mathrm{post}}$ je také BCCB (cirkulantní), její **diagonála je konstantní přes
pixely** (stacionarita) a rovná se průměru vlastních čísel:

$$
s_{\mathrm{post}}^2=(\Sigma_{\mathrm{post}})_{ii}=\frac1N\sum_{k}\frac{\sigma^2\lambda_k}{\lambda_k+\sigma^2}
\qquad(\forall i).
$$

**Pushforward na label.** Maska je deterministická funkce pole, takže per-pixel posterior
predictive je exaktně

$$
\boxed{\,p_{\mathrm{oracle},ij}=\mathbb P(y_{ij}=1\mid x)=\mathbb P(f_{ij}>\tau\mid x)
=\Phi\!\Big(\frac{\mu_{\mathrm{post},ij}-\tau}{s_{\mathrm{post}}}\Big).\,}
$$

(Marginála $f_{ij}\mid x$ je $\mathcal N(\mu_{\mathrm{post},ij},s_{\mathrm{post}}^2)$; per-pixel
pravděpodobnost potřebuje jen tuto marginálu, ne celou $\Sigma_{\mathrm{post}}$.)

**Explicitní prior maska.** Bez pozorování je $f_{ij}\sim\mathcal N(0,\sigma_f^2)$, takže

$$
p_{\mathrm{prior},ij}=\mathbb P(f_{ij}>\tau)=\Phi(-\tau/\sigma_f)=\text{fg}\quad(\text{uniformní}).
$$

To je „defaultní" maska, ke které by amortizovaný prediktor kolaboval při neinformativním kontextu.

**Výpočetní složitost.** Naivně $(K+\sigma^2 I)^{-1}$ stojí $O(N^3)=O((HW)^3)$; cirkulantní/FFT
forma je $O(N\log N)$. Proto je oracle na $64\times64$ levný a exaktní (past C.8: pro krátké $\ell$
je $K$ špatně podmíněná, $\sigma^2$ působí jako nugget/jitter a stabilizuje inverzi).

---

## 4. PFN: architektura, objektiv a co je vlastně „pravda"

**Architektura.** UniverSeg (encoder–decoder), random init; **CrossBlock** vyměňuje featury mezi
query a každým support párem (globální feature-averaging). Výstup: 1 logit/pixel
$g_\phi(x_\star,S)\in\mathbb R^{\mathcal G}$; predikce $\hat p_{\mathrm{PFN}}=\varsigma(g_\phi)$,
$\varsigma$ = sigmoid.

**Trénovací objektiv.** Jeden krok: $\theta\sim\Pi$; nasampluj $n+1$ polí; $S$ z $n$ dvojic,
query $(x_\star,y_\star)$. Ztráta = **per-pixel binary cross-entropy**

$$
\mathcal L(\phi)=\mathbb E_{\theta\sim\Pi}\,\mathbb E_{S,x_\star,y_\star}\,
\frac1N\sum_{ij}\mathrm{BCE}\big(\varsigma(g_\phi)_{ij},\,y_{\star,ij}\big),\qquad
\mathrm{BCE}(q,y)=-y\log q-(1-y)\log(1-q).
$$

**Proč sigmoid + BCE dává posterior.** BCE je **striktně vlastní skórovací pravidlo** (strictly
proper scoring rule). Pro pevný vstup je populační minimalizátor

$$
\arg\min_{q}\ \mathbb E_{y}\big[\mathrm{BCE}(q,y)\mid x_\star,S\big]=\mathbb E[y_\star\mid x_\star,S]
=\mathbb P(y_\star=1\mid x_\star,S).
$$

Perfektně natrénovaný PFN tedy realizuje **amortizovaný posterior predictive**

$$
p_{\mathrm{PFN}}^{\star}(x_\star,S)=\mathbb P(y_\star=1\mid x_\star,S)
=\int \mathbb P(y_\star=1\mid x_\star,\theta)\;\underbrace{p(\theta\mid S)}_{\text{HP posterior}}\,d\theta .
$$

**Dvě „pravdy" (důležité rozlišení).**

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
- **Variance (Thm 6.2, analog).** Variance z konečného kontextu klesá $\sim O(n^{-1/2})$
  (softmax → diminishing sensitivity).

Naše měření (§7): variance klesá (empirický log-log sklon $\approx-1$, tj. i strměji než
$-\tfrac12$), $\mathrm{bias}^2$ drží malé kladné plateau. Protože je model jinak *near-optimal*
(§7b), tento zbytkový bias **není optimalizační artefakt, ale strukturální** — přesně predikce
locality condition.

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

In-distribution je excess risk $\approx0{,}004$–$0{,}006$ nat/pixel → PFN je **skoro
Bayes-optimální** (tréninkové plató na loss $\approx0{,}019$ = sezení těsně nad $H(y\mid x)$, ne
zaseknutí). OOD-short: excess $\approx9\times$ vyšší. Easy $>$ Hard, protože hladká pole mají široké
nejisté hranice (velké $\overline{H_b(p_{\mathrm{oracle}})}$), kde mírná over-sharpness stojí nejvíc
KL.

**(c) Kolaps k prioru — negativní.** $d_{\mathrm{oracle}}=|\hat p-p_{\mathrm{oracle}}|$ zůstává
malé a ploché i při $n_{\mathrm{supp}}=1$; $d_{\mathrm{prior}}=|\hat p-\text{fg}|\approx0{,}47$
napříč. Kolaps ($\hat p\to\text{fg}$) tedy **nenastává** — vyžadoval by řídký neinformativní
kontext (Varianta A). Není to univerzálie amortizace, ale důsledek řídkosti dat.

**(d) Bias–variance.** $\overline{\mathrm{var}}$ klesá s $n_{\mathrm{supp}}$ k $\approx0$ (sklon
$\approx-1$); $\overline{\mathrm{bias}^2}$ plateau (~$0{,}0016$ Hard, ~$0{,}0035$ Easy). Variance
mizí, bias strukturálně přetrvává.

**(e) Kalibrace vs oracle.** Rozdělení $\hat p_{\mathrm{PFN}}$ a $p_{\mathrm{oracle}}$ se téměř
překrývají; $\overline{H_b}$ srovnatelné i na nejisté množině (0,522 vs 0,552). ECE(oracle)
$=0{,}0049$ celkově, $0{,}0369$ na nejisté množině ($\sim5\%$ pixelů). Tj. **věrná reprezentace
nejistoty** s jen mírnou zbytkovou over-sharpness na hranicích.

---

## 8. Souhrn a limity

**Souhrn (formálně).** Skutečný PFN natrénovaný minimalizací per-pixel BCE na tazích z $\Pi$
realizuje amortizovaný posterior predictive; měřeno proti exaktnímu oraclu je jeho
$\mathbb E[\mathrm{KL}(\text{oracle}\Vert\text{PFN})]$ řádu $10^{-3}$ in-distribution
(near-Bayes-optimal) a roste $\sim10\times$ OOD, asymetricky v $\ell$. Variance $\to0$ s kontextem,
bias strukturálně přetrvává (locality condition), kalibrace věrná s mírnou over-sharpness na
hranicích. Kolaps k prioru se u husté segmentace neprojevuje.

**Limity (poctivě).**
1. **Matched prior** (train $\Pi$ = test $\Pi$) a **husté** pozorování činí úlohu skoro
   identifikovatelnou; „near-optimal" je proto částečně vlastnost setupu. Praktické riziko = OOD.
2. **Oracle vs $p_{\mathrm{PFN}}^{\star}$**: měříme proti $\theta$-known oraclu, ne proti
   $\theta$-marginalizovanému optimu; rozdíl je HP-nejistota, empiricky malá (§4).
3. **Torus** je zjednodušení kvůli exaktnímu $O(N\log N)$ oraclu; neperiodická/reálná data =
   další krok.
4. **Trénink** (log `pfn_seg.o235933`): konvergoval, loss $0{,}15\to0{,}0194$, pix-acc $0{,}991$,
   grad-clip pohltil přechodné výkyvy gradientu (ep. 76–77); `best` = epocha 250.
