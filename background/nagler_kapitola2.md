# Kapitola 2: Teoretický rámec (Nagler, 2023)

## 2.1 Statistický model

Uvažujme klasifikační úlohu se třídovým labelem $Y \in \mathcal{Y}$ a příznaky (features) $X \in \mathcal{X} \subseteq \mathbb{R}^d$. Předpokládáme, že máme iid trénovací data

$$D_n = (Y_i, X_i)_{i=1}^n$$

pocházející z nějaké distribuce $p_0$. Cílem je predikovat podmíněné třídové pravděpodobnosti

$$p_0(y \mid x) = P(Y = y \mid X = x).$$

Z pohledu bayesovské neparametriky nahlížíme na $p_0$ jako na realizaci náhodného, nekonečně-dimenzionálního parametru $p \in \mathcal{P}$, kde $\mathcal{P}$ je prostor podmíněných pravděpodobnostních distribucí na $\mathcal{Y} \times \mathcal{X}$. Tento parametr má distribuci $\Pi$, zvanou **prior** — vyjadřuje naše přesvědčení o tom, jaké modely $p$ jsou apriori pravděpodobné, ještě předtím než vidíme jakákoliv data.

Celý generativní mechanismus, ze kterého data pocházejí, je:

1. Vygeneruj $p \sim \Pi$.
2. Vygeneruj iid vzorky $D_n = (Y_i, X_i)_{i=1}^n$ a jeden dodatečný pár $(Y, X)$ z modelu $p$.

Tento mechanismus definuje sdílenou distribuci nad $(D_n \cup (Y, X), p)$ pro každé $n$. Skutečná distribuce $p_0$ je ta realizace $p$, ze které skutečně pocházejí naše data — zpravidla je $p_0$ neznámá.

## 2.2 Posterior Predictive Distribution (PPD)

Pro každé $n$ poskytuje výše popsaný statistický model dobře definovanou sdruženou distribuci n-tice $(D_n \cup (Y, X), p)$. Na základě tohoto modelu lze $p_0(y \mid x)$ aproximovat pomocí **posterior predictive distribution (PPD)**:

$$\pi(y \mid x, D_n) = P(Y = y \mid X = x, D_n).$$

Tím vzniká celá rodina PPD indexovaná $n$. Pokud prior $\Pi$ faktorizuje nezávisle pro marginální složky $p(y \mid x)$ a $p(x)$ — tj. distribuce příznaků a distribuce labelů jsou v prioru nezávislé — lze PPD zapsat jako

$$\pi(y \mid x, D_n) = \int p(y \mid x) \, d\Pi(p \mid D_n), \tag{1}$$

kde $\Pi(\cdot \mid D_n)$ je **posterior** — podmíněná distribuce $p$ daná pozorovanými daty $D_n$, získaná Bayesovým pravidlem. PPD $\pi(y \mid x, D_n)$ je tedy **posteriorní střední hodnota** podmíněných distribucí $p(y \mid x)$, vážená tím, jak jsou jednotlivé modely $p$ konzistentní s $D_n$.

> **Poznámka 2.1.** Müller et al. (2022) a Hollmann et al. (2022) používají priory, které výše popsaným způsobem faktorizují, avšak tuto podmínku explicitně nezmiňují jako zdůvodnění vzorce (1). Priory, které takovýmto způsobem nefaktorizují, by vedly k odlišné formě PPD:
> $$\pi(y \mid x, D_n) = \int p(y \mid x) \, d\Pi(p \mid x, D_n),$$
> kde by pozorování testovacího příznaku $x$ bylo informativní o podmíněné distribuci $p(y \mid x)$. To je neintuitivní — testovací příznak by ovlivňoval samotný model $p$, nikoliv jen predikci.

## 2.3 PFN jako aproximace PPD

**Prior-Data Fitted Network (PFN)** je numerická aproximace celé rodiny PPD $\{\pi(\cdot \mid \cdot, D_n),\, n \in \mathbb{N}\}$.

Základním teoretickým poznatkem je, že PPD sama o sobě je pro každé $n$ **optimálním prediktorem** v následujícím smyslu. Definujme třídu všech podmíněných pravděpodobnostních funkcí

$$\mathcal{Q} = \left\{ q : (\mathcal{Y} \times \mathcal{X})^{n+1} \to [0,1] \;\Big|\; \sum_{y \in \mathcal{Y}} q(y \mid \cdot, \cdot) = 1 \right\}.$$

Každé $q \in \mathcal{Q}$ je tedy funkce, která dostane $n$ trénovacích párů $(Y_i, X_i)$ a jeden testovací vstup $X$, a vrátí distribuci přes $\mathcal{Y}$.

**Theorem 2.2.** $\pi$ z rovnice (1) splňuje

$$\pi = \arg\max_{q \in \mathcal{Q}} \, \mathbb{E}_\Pi[\log q(Y \mid X, D_n)],$$

kde $\mathbb{E}_\Pi$ je střední hodnota přes $(Y, X) \cup D_n$ generované dle mechanismu v sekci 2.1.

Jinak řečeno: PPD $\pi$ maximalizuje očekávanou podmíněnou log-likelihood přes všechny možné prediktory z $\mathcal{Q}$. Je to nejlepší možný prediktor za daného prioru $\Pi$.

> **Poznámka 2.3.** Maximalizaci výrazu $\mathbb{E}_\Pi[\log q(Y \mid X, D_n)]$ lze ekvivalentně chápat jako minimalizaci očekávané KL divergence
> $$\mathbb{E}\!\left[\mathrm{KL}\!\left(q(\cdot \mid X, D_n) \,\Big\|\, \pi(\cdot \mid X, D_n)\right)\right].$$
> PPD $\pi$ je tedy KL-optimálním prediktorem v $\mathcal{Q}$.

### Trénování PFN: hledání parametrů

Abychom PPD aproximovali v praxi, trénujeme model $q_\theta$ parametrizovaný $\theta \in \Theta$ (typicky váhy neuronové sítě). Přesněji, pro každou hodnotu $\theta$ existuje celá rodina funkcí

$$\left\{q_{\theta} : (\mathcal{Y} \times \mathcal{X})^{n+1} \to [0,1],\; n \in \mathbb{N}\right\},$$

ale závislost na $n$ v notaci explicitně neuvádíme. KL-optimální parametry jsou definovány jako

$$\theta^* = \arg\max_{\theta \in \Theta} \, \mathbb{E}_{\Pi_N} \mathbb{E}_\Pi[\log q_\theta(Y \mid X, D_N)], \tag{3}$$

kde $\Pi_N$ je **size prior** — distribuce nad velikostí trénovací množiny $N$. Střední hodnota přes $N$ zajišťuje, že $q_{\theta^*}$ napodobuje **celou rodinu** PPD pro různá $n$, nikoliv jen její $n$-tý prvek.

Model $q_\theta$ je zpravidla **misspecifikovaný** — neexistuje $\theta$ takové, že $q_\theta = \pi$ přesně. V tom případě rovnice (3) definuje **KL-optimální aproximaci** $\pi$ ve třídě $\{q_\theta : \theta \in \Theta\}$, tj. nejlepší aproximaci, které je daná architektura schopna.

### Monte Carlo aproximace při trénování

Střední hodnota v (3) se v praxi aproximuje průměrováním přes $m$ iid datasetů. Konkrétně generujeme datasety $(Y_j, X_j) \cup D^{(j)}$ velikosti $N_j + 1$ s $N_j \sim \Pi_N$ dle generativního mechanismu ze sekce 2.1. Empirická verze optimalizačního problému je

$$\hat{\theta} = \arg\max_{\theta \in \Theta} \sum_{j=1}^{m} \log q_\theta(Y_j \mid X_j, D^{(j)}). \tag{4}$$

Přesnost $\hat{\theta}$ jako aproximace $\theta^*$ závisí na počtu Monte Carlo vzorků $m$; lze ukázat, že $\hat{\theta} = \theta^* + O_p(m^{-1/2})$.

Toto je idealizace skutečného trénování — sofistikované PFN jsou velké modely trénované zpravidla v jediné epoše a maximum v (4) není nikdy přesně dosaženo. Hlavní teoretické výsledky v dalších sekcích jsou nicméně formulovány pro libovolné $\theta$, takže to jejich platnost neovlivňuje.

> **Poznámka 2.4.** Hollmann et al. (2022) používají jako $q_\theta$ transformerovou síť (Vaswani et al., 2017). Taková architektura přijímá dataset $D_n$ libovolné délky a jeden nebo více testovacích bodů $x_1, \ldots, x_{n_\text{test}}$, a vrací vektory podmíněných třídových pravděpodobností $q_\theta(\cdot \mid x_j, D_n)$ pro každý $x_j$. Počet testovacích bodů $n_\text{test}$ je pro teorii nepodstatný; WLOG uvažujeme $n_\text{test} = 1$.
