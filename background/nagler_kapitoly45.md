# Kapitola 4: PFN jako aproximace PPD (Nagler, 2023)

Na kvalitu PFN aproximace, definované optimalizačním problémem

$$\hat{\theta} = \arg\max_{\theta} \sum_{j=1}^{m} \log q_\theta(Y_j \mid X_j, D^{(j)}), \tag{4}$$

mají vliv čtyři faktory: datový prior $\Pi$, size prior $\Pi_N$, model $q_\theta$ a počet Monte Carlo vzorků $m$. Protože PFN je pre-trénován offline, třídu modelů $\{q_\theta : \theta \in \Theta\}$ lze považovat za fixní vůči $m$. Přesnost $\hat{\theta}$ jako aproximace $\theta^*$ pak plyne ze standardních výsledků empirické minimalizace rizika: $\hat{\theta} = \theta^* + O_p(m^{-1/2})$ (viz Appendix B). Zbývající tři faktory jsou zajímavější.

## Kapacita modelu a prioru

Pokud $\Pi$ obsahuje pouze jednoduché modely, bude i optimální PFN $q_{\theta^*}$ produkovat jednoduché funkce $(y, x)$. Naopak, jednoduchá architektura $\{q_\theta : \theta \in \Theta\}$ nemůže těžit ze složitého prioru $\Pi$. Aby PFN dobře fungoval na různorodých úlohách, musí mít dostatečnou kapacitu **jak architektura $q_\theta$, tak prior $\Pi$**.

## Size prior $\Pi_N$ jako regularizátor

Při pre-tréninku dle (4) se vzorkují datasety $D^{(j)}$ s náhodnou velikostí $N_j \sim \Pi_N$. Definujme KL-optimální parametr pro danou velikost vzorku:

$$\theta^*_n = \arg\max_{\theta} \, \mathbb{E}_\Pi[\log q_\theta(Y \mid X, D_n)].$$

PPD $\pi(y \mid x, D_n)$, kterou se snažíme aproximovat, se mění s $n$. Jako funkce $(y, x)$ roste její složitost s $n$ — pro $n = 1$ je PPD blízká průměrnému modelu v prioru a typicky téměř konstantní; pro velká $n$ je stále komplexnější. Analogicky $\theta^*_n$ preferuje složitější modely s rostoucím $n$.

Optimalizací přes náhodné velikosti $N_j$ výsledný $\theta^*$ průměruje $\theta^*_{N_j}$. Distribuce $\Pi_N$ tak určuje, které velikosti datasetů jsou zdůrazněny. To lze chápat jako **regularizaci na komplexitu modelu**: malé $N_j$ nutí model k jednoduchým predikcím, velké $N_j$ k složitým. TabPFN (Hollmann et al., 2022) byl trénován s $\Pi_N = \mathrm{Uniform}\{1, \ldots, 1023\}$ — omezení na malé velikosti má výpočetní důvod: transformer se škáluje kvadraticky v $N_j$.

## Extrapolace za hranici pre-tréninku

Protože TabPFN nikdy neviděl při pre-tréninku datasety větší než $n \approx 1000$, je překvapivé, že jeho predikce se při inferenci zlepšují i pro větší $n$. To závisí netriviálně na struktuře rodiny $\{q_{\hat\theta}(\cdot \mid \cdot, D_n),\, n \in \mathbb{N}\}$. Zřejmě se model naučil strukturu, která umožňuje rozumnou extrapolaci na větší $n$ — ať už jde o vlastnost architektury $q_\theta$, nebo výsledek učení $\theta^*$ pro dané $\Pi$. Mechanismy, které to umožňují, jsou předmětem kapitoly 5.

---

# Kapitola 5: Proč se PFN umí učit in-context? (Nagler, 2023)

Theorem 3.1 garantuje konvergenci PPD $\pi$, ale PFN $q_{\hat\theta}$ PPD není — je to pouze její aproximace trénovaná na omezených velikostech datasetů. Proč tedy $q_{\hat\theta}$ pre-trénovaný na $n \leq 1000$ vzorcích zlepšuje predikce, když mu při inferenci dáme větší datasety?

## 5.1 PFN jako frekventistický prediktor

Pro pochopení je užitečné opustit bayesovskou perspektivu. Pro libovolnou velikost $n$ při inferenci nahlížíme na pre-trénovanou síť $q_{\hat\theta}(y \mid x, \cdot)$ jako na **nenatrénovaný frekventistický prediktor** pro $p_0(y \mid x)$ — funkci $(\mathcal{Y} \times \mathcal{X})^n \to \mathcal{P}_{\mathcal{Y}|\mathcal{X}}$, která mapuje dataset $D_n$ na podmíněnou distribuci. Parametry $\theta$ jsou pak jen hyperparametry tohoto prediktoru, vyladěné při pre-tréninku. Prior $\Pi$ a $\Pi_N$ jsou jednoduše distribuce nad úlohami, na kterých chceme, aby prediktor fungoval dobře.

Chybu predikce lze rozložit na bias a varianci:

$$q_\theta(y \mid x, D_n) - p_0(y \mid x) = \underbrace{q_\theta(y \mid x, D_n) - \mathbb{E}_{D_n \sim p_0^n}[q_\theta(y \mid x, D_n)]}_{\text{variance}} + \underbrace{\mathbb{E}_{D_n \sim p_0^n}[q_\theta(y \mid x, D_n)] - p_0(y \mid x)}_{\text{bias}}.$$

Empiricky celková chyba klesá s $n$. Jaké strukturální vlastnosti PFN to vysvětlují?

## 5.2 Symetrie

Standardní transformery jsou symetrickými funkcemi jednotlivých vzorků v $D_n$, což je přirozené, jsou-li vzorky iid.

**Lemma 5.1.** Nechť $f : (\mathcal{Y} \times \mathcal{X})^n \to \mathcal{P}_{\mathcal{Y}|\mathcal{X}}$ je libovolný prediktor. Pak existuje jeho symetrizovaná verze $\tilde{f}$ taková, že pro každé pravděpodobnostní míry $P$:

$$\mathbb{E}_{D_n \sim P^n}[\tilde{f}(D_n)] = \mathbb{E}_{D_n \sim P^n}[f(D_n)], \qquad \mathrm{Var}_{D_n \sim P^n}[\tilde{f}(D_n)] \leq \mathrm{Var}_{D_n \sim P^n}[f(D_n)].$$

Symetrické prediktory jsou tedy v MSE smyslu optimální. Symetrie sama o sobě ale pro učení nic neznamená — například $q(y \mid x, D_n) = 1/|\mathcal{Y}|$ je symetrická funkce, která se neumí nic naučit.

## 5.3 Variance a diminishing sensitivity

Rozumnou vlastností prediktoru $q_\theta$ je, že s větší $D_n$ klesá vliv jednotlivých vzorků. Formálně předpokládáme, že existují $\alpha > 0$ a $L < \infty$ takové, že pro dostatečně velká $n$ a skoro všechny datasety $D_n$, $D'_n$ lišící se v jediném vzorku:

$$\left| q_\theta(y \mid x, D_n) - q_\theta(y \mid x, D'_n) \right| \leq L n^{-\alpha}. \tag{5}$$

**Theorem 5.2.** Pokud platí (5), pak

$$\left| q_\theta(y \mid x, D_n) - \mathbb{E}[q_\theta(y \mid x, D_n)] \right| \lesssim n^{1/2 - \alpha}$$

s vysokou pravděpodobností. Pro $\alpha > 1/2$ platí $\lim_{n \to \infty} n^{1/2-\alpha} = 0$, takže variance mizí.

**Lemma 5.3.** Pokud platí (5) s $\alpha > 1/2$, pak

$$q_\theta(y \mid x, D_n) - \mathbb{E}[q_\theta(y \mid x, D_n)] \xrightarrow{n \to \infty} 0 \quad \text{almost surely.}$$

Mizení variance tedy vysvětluje část toho, jak PFN učí při inferenci. Zbývající chyba pochází z biasu.

## 5.4 Bias a nutnost lokality

Bias je určen chováním posloupnosti $\mathbb{E}[q_\theta(y \mid x, D_n)]$. Je rozumné předpokládat, že

$$\mathbb{E}[q_\theta(y \mid x, D_n)] \xrightarrow{n \to \infty} \bar{q}_\theta(y \mid x)$$

pro nějakou limitní funkci $\bar{q}_\theta$. Bez znalosti konkrétní architektury nelze říci více — v kapitole 6 jsou ukázány příklady, kde bias je konstantní, klesající i rostoucí s $n$.

Lze však dát **nutnou podmínku** pro mizení biasu. Prediktor, který má mizející bias na dostatečně bohaté třídě funkcí, musí být **lokální**: asymptoticky by měly k $q_\theta(y \mid x, D_n)$ přispívat pouze vzorky $(Y_i, X_i) \in D_n$ s $X_i$ blízko $x$.

**Theorem 5.4.** Nechť $\mathcal{P}$ je třída distribucí taková, že pro každé $p \in \mathcal{P}$:

$$\mathbb{E}_{D_n \sim p^n}[q_\theta(y \mid x, D_n)] \xrightarrow{n \to \infty} p(y \mid x). \tag{6}$$

Pokud platí (5), pak existuje posloupnost $\epsilon_n \to 0$ taková, že pro každé $\tilde{p} \in \mathcal{P}$ platí almost surely:

$$\left| q_\theta(y \mid x, D_n) - q_\theta(y \mid x, \tilde{D}_n) \right| \xrightarrow{n \to \infty} 0, \tag{7}$$

kde $D_n = (Y_i, X_i)_{i=1}^n$ a $\tilde{D}_n = (Y'_i, X_i)_{i=1}^n$ s

$$Y'_i = \begin{cases} Y_i & \text{pokud } \|X_i - x\| \leq \epsilon_n, \\ \sim \tilde{p}(\cdot \mid X_i) & \text{pokud } \|X_i - x\| > \epsilon_n. \end{cases}$$

Pokud je tedy $q_\theta$ nestranný pro dostatečně bohatou $\mathcal{P}$, lze libovolně přepsat labely vzorků vzdálených od $x$, aniž by se predikce změnila — záleží pouze na vzorcích s $X_i$ blízko $x$.

### Diskuze k Theorem 5.4

Výsledek postrádá smysl, je-li $\mathcal{P}$ příliš malá (a je triviální, obsahuje-li $\mathcal{P}$ jediné $p$). I pro bohatou $\mathcal{P}$ jde jen o nutnou, nikoliv postačující podmínku — konstantní prediktor $q_\theta = 1$ je lokální ve smyslu (7), ale jeho bias se s $n$ nemění.

Důležitý je ale důsledek pro bias-variance tradeoff: pokud bias mizí pro bohatou $\mathcal{P}$, prediktor může efektivně využít jen $\epsilon_n n = o(n)$ vzorků, takže podmínka (5) pravděpodobně neplatí pro $\alpha = 1$. To je v souladu se spodními mezemi bias-variance tradeoffu (Derumigny & Schmidt-Hieber, 2020).

### Proč transformery lokalitu nesplňují

Attention váhy $a^{(h)}_j$ jsou výstupem SoftMax — jsou vždy kladné pro všechny vzorky $j$, nikdy přesně nulové. Transformer tedy **vždy nenulově váží i vzdálené vzorky**. Přepis labelů vzdálených bodů proto vždy predikci ovlivní, a lokalita ve smyslu (7) není splněna. Transformer tak garantuje mizení variance (Theorem 6.2 v článku), ale nikoli mizení biasu.
