# Kapitola 6.3–6.4: Transformery a Localized PFN (Nagler, 2023)

## 6.3 Transformer Networks

Uvažujme transformer s jednou vrstvou. Datasety $D_n = \{V_i\}_{i=1}^n$, kde $V_i = (Y_i, X_i) \in \{0,1\} \times \mathbb{R}^d$, a testovací vektor $v = (0, x)$. Síť je definována následujícími operacemi:

$$a^{(h)}_j = \mathrm{SoftMax}\!\left(v^\top W^{(h)}_q V_1, \ldots, v^\top W^{(h)}_q V_n\right)_j,$$

$$u' = \sum_{h=1}^{H} \sum_{j=1}^{n} a^{(h)}_j W^{(h)}_v V_j,$$

$$u = \mathrm{LayerNorm}(v + u';\, \gamma),$$

$$z' = W_{r,2}\, \mathrm{ReLU}(W_{r,1} u;\, \gamma),$$

$$z = \mathrm{LayerNorm}(u + z';\, \gamma),$$

$$q_\theta(\cdot \mid x, D_n) = \mathrm{SoftMax}(W_o z),$$

kde matice vah jsou $W^{(h)}_q, W^{(h)}_v \in \mathbb{R}^{(d+1)\times(d+1)}$, $W_{r,1}, W_{r,2}^\top \in \mathbb{R}^{m\times(d+1)}$, $W_o \in \mathbb{R}^{|\mathcal{Y}|\times(d+1)}$. Parametr $\theta$ sbírá všechny tyto matice. Operace SoftMax, LayerNorm a ReLU jsou definovány jako:

$$\mathrm{SoftMax}(v) = \frac{\exp(v)}{\sum_j \exp(v_j)}, \qquad \mathrm{LayerNorm}(v;\,\gamma) = \gamma_1 \frac{v - \mathrm{avg}(v)}{\|v - \mathrm{avg}(v)\| + |\gamma_2|} + \gamma_3,$$

$$\mathrm{ReLU}(v) = \max(0, v),$$

kde $\exp$ a $\max$ působí po složkách. Norma $\|\cdot\|$ je všude chápána jako $\|\cdot\|_2$.

### Attention mechanismus — intuice

První dvě rovnice definují **attention mechanismus s $H$ hlavami** (Vaswani et al., 2017). Ze součtu $a^{(h)}_1 + \cdots + a^{(h)}_n = 1$ plyne, že attention váhy tvoří pravděpodobnostní distribuci přes trénovací vzorky. Myšlenka je, že v každé hlavě attention váhy $a^{(h)}_j$ zdůrazňují konkrétní vzorky $V_j \in D_n$ — ty, které jsou "podobné" testovacímu vektoru $v$ ve smyslu měřeném skalárním součinem $v^\top W^{(h)}_q V_j$. Každá attention hlava dovoluje jinou definici podobnosti prostřednictvím matice $W^{(h)}_q$.

Intuitivně: transformer si při predikci pro $x$ "vyhledá" v trénovacích datech vzorky, které považuje za relevantní, a jejich labely agreguje. Různé hlavy mohou sledovat různé aspekty podobnosti — jedna hlava třeba srovnává hodnoty $X_i$ s $x$, jiná sleduje jiný rys vstupních vektorů.

### Variance transformeru

**Theorem 6.2.** Pro $\mathcal{X} = \{x : \|x\| \leq K\}$ a omezené matice vah ($\|W^{(h)}_q\|, \|W^{(h)}_v\|, \|W_{r,1}\|, \|W_{r,2}\|, \|W_o\| < \infty$) platí

$$\left|q_\theta(y \mid x, D_n) - \mathbb{E}[q_\theta(y \mid x, D_n)]\right| \lesssim n^{-1/2}$$

s vysokou pravděpodobností.

Variance mizí **bez ohledu na parametr $\theta$** — jde o strukturální vlastnost architektury, nikoliv výsledek tréninku. Důvodem je, že attention mechanismus nutně přiděluje každému vzorku váhu řádu $1/n$, takže vliv libovolného jednotlivého vzorku klesá s $n$ (podmínka (5) je splněna s $\alpha = 1$, viz důkaz v Appendix A.8).

### Limitní bias transformeru a exponenciálně tiltovaná míra

Bias závisí silně na volbě $\theta$. Označme $\bar{q}_\theta(y \mid x)$ limitní hodnotu predikce pro $n \to \infty$.

**Theorem 6.3.** Za předpokladů Theorem 6.2 platí

$$\mathbb{E}[q_\theta(\cdot \mid x, D_n)] \xrightarrow{n \to \infty} \bar{q}_\theta(\cdot \mid x),$$

kde $\bar{q}_\theta(\cdot \mid x)$ je definováno stejně jako $q_\theta(\cdot \mid x, D_n)$, ale s $u'$ nahrazeným výrazem

$$\bar{u}' = \sum_{h=1}^{H} W^{(h)}_v \, \mathbb{E}_{V \sim g_h}[V],$$

kde

$$g_h(s) = \frac{\exp(v^\top W^{(h)}_q s)}{\mathbb{E}_{V \sim p_0}[\exp(v^\top W^{(h)}_q V)]} \cdot p_0(s).$$

**Intuice k $g_h$:** Míra $g_h$ je **exponenciálně tiltovaná verze** $p_0$ (Siegmund, 1976). Relativně vůči $p_0$ zvyšuje pravděpodobnost těm hodnotám $s$, které jsou "podobné" testovacímu vektoru $v$ ve smyslu $v^\top W^{(h)}_q s$, a snižuje pravděpodobnost nepodobných hodnot. Jde o infinitezimální (tj. asymptotickou) analogii attention mechanismu: místo váhování konkrétních vzorků $V_j$ se váhuje celá distribuce $p_0$.

Každá attention hlava $h$ tak hodnotí určitý **aspekt** neznámé distribuce $p_0$ (charakterizovaný maticí $W^{(h)}_q$). Pokud jsou matice $(W^{(h)}_q)_{h=1}^H$ dobře nastaveny, jednotlivé aspektové pohledy rozlišují různé hodnoty příznaků.

### Proč bias transformeru nemizí

Tiltovaná míra $g_h$ **lokalizuje** prediktor do jisté míry — zvyšuje váhu vzorků podobných $v$. Avšak **ne v smyslu Theorem 5.4**: vzorky vzdálené od $x$ stále nenulově přispívají, protože SoftMax nikdy nevytváří přesně nulové váhy. Přepsání labelů vzdálených vzorků proto vždy predikci změní, a lokalita ve smyslu podmínky (7) není splněna. Z Theorem 5.4 proto plyne, že **bias transformeru obecně nemizí**.

Limitní bias $\bar{q}_\theta(y \mid x) - p_0(y \mid x)$ přesto může být malý, pokud síť za attention vrstvou dokáže ze součtu aspektových souhrnů $W^{(h)}_v \mathbb{E}_{V \sim g_h}[V]$ zkonstruovat dobrou aproximaci $p_0$. Relevance jednotlivých aspektů závisí na pravé distribuci $p_0$; méně relevantní aspekty mohou přispívat méně. Na malých vzorcích ($n = 1$) jsou všechny attention váhy $a^{(h)}_j = 1$, takže všechny aspekty přispívají stejnou měrou. To naznačuje, že bias může klesat v rozsahu velikostí, na které byl $\theta$ natrénován — nikoliv však za jejich hranicí, kde není důvod očekávat pokles biasu.

Klíčovou roli hraje přítomnost **více attention hlav** ($H > 1$): spolupráce více hlav může efektivně napodobit mechanismus průměrování modelů analogický sekci 6.2.

---

## 6.4 Localized PFN

Podle Theorem 5.4 je lokalita nutnou podmínkou pro mizení biasu. Protože transformer ji nezaručuje, Nagler navrhuje jednoduchou **post-hoc lokalizaci** aplikovatelnou na libovolnou pre-trénovanou síť $q_\theta$:

Pro predikci labelu v testovacím bodě $x$:

1. Sestrojí se redukovaná trénovací množina $D_n(x)$ odstraněním všech vzorků kromě $k_n$ nejbližších sousedů bodu $x$ z $D_n$.
2. Predikuje se label maximalizující $q_\theta(\cdot \mid x, D_n(x))$.

**Intuice:** Omezení na okolí $x$ je ekvivalentní "roztažení" cílové funkce $p(y \mid \cdot)$ v okolí $x$ — lokálně hladká funkce se chová přibližně jako konstantní. Konstantní funkce jsou pro $q_\theta$ snazší k aproximaci. Jedná se o stejný mechanismus, jaký stojí za window smootherem ze sekce 6.1. Lokalizace tedy zlepšuje bias za cenu mírného zvýšení variance — prediktor pracuje s menší efektivní velikostí datasetu $k_n \ll n$.

## 6.5 Numerická validace

Teoretické předpovědi jsou empiricky ověřeny na modelu $p_0(1 \mid X) = 1/2 + \sin(\mathbf{1}^\top X)/2$ s $Y \in \{0,1\}$, $X \sim \mathcal{N}(0, I_5)$, na 500 simulovaných datasetech a pre-trénovaném TabPFN (Hollmann et al., 2022). Výsledky (Figure 1 v článku) potvrzují:

- Variance klesá rychlostí $1/n$ — v souladu s Theorem 6.2.
- Bias klesá do $n \approx 1000$, poté se stabilizuje a nemizí — v souladu s absencí lokality.
- Lokalizovaný TabPFN ($k_n = \min\{500, \lceil n^{4/(d+4)} \rceil\}$) vykazuje bias klesající i za $n = 1000$, za cenu mírně větší variance.

Analýza ukazuje, že TabPFN při větších $n$ učí hlavně snižováním variance — ta klesá bez ohledu na natrénované parametry $\hat\theta$, jako přímý důsledek transformer architektury.
