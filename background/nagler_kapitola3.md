# Kapitola 3: Kdy se PPD umí učit? (Nagler, 2023)

PPD je definována jako

$$\pi(y \mid x, D_n) = \int p(y \mid x) \, d\Pi(p \mid D_n).$$

Posterior $\Pi(\cdot \mid D_n)$ je plně určen priorem $\Pi$ — jde o standardní Bayesovo pravidlo aplikované na nekonečně-dimenzionální parametr $p$. Pokud jsou data $D_n$ generována z $p_0$, doufáme, že se posterior $\Pi(p \mid D_n)$ s rostoucím $n$ soustředí okolo $p_0$, a tím pádem $\pi(y \mid x, D_n) \to p_0(y \mid x)$.

Nastavení dobrého prioru v neparametrickém kontextu je však netriviální. Je třeba, aby $\Pi$ měl dostatečně velký support — pokrýval dostatečně bohatou třídu funkcí. Ale i to nestačí: prior může sice $p_0$ obsahovat ve svém supportu, ale dávat příliš velkou hmotnost nepříznivým oblastem prostoru $\mathcal{P}$, čímž může konvergence posteriorů probíhat velmi pomalu nebo vůbec (Ghosal & van der Vaart, 2017, Sections 1.2–1.3).

Klíčovým výsledkem kapitoly 3 je, že PPD se umí učit i v situaci, kdy $p_0$ leží **mimo support prioru** $\mathcal{P} = \{p : \Pi(p) > 0\}$, pokud je prior dostatečně "dobře chovaný":

## Podmínky (A1) a (A2)

Theorem 3.1 předpokládá dvě podmínky na prior $\Pi$:

**(A1)** Existuje jedinečné $p^* \in \mathcal{P}$ takové, že

$$p^* = \arg\min_{p \in \mathcal{P}} \, \mathrm{KL}(p \mid p_0), \qquad \mathrm{KL}(p^* \mid p_0) < \infty.$$

Jinými slovy, v supportu prioru existuje jednoznačný KL-optimální aproximant $p^*$ pravého modelu $p_0$, a KL divergence $p^*$ vůči $p_0$ je konečná.

**(A2)** Pro každé $\alpha \in (0, 1/2)$ existují množiny $B_1, \ldots, B_{J(\alpha)}$ takové, že

$$\mathcal{P} \subseteq \bigcup_{j=1}^{J(\alpha)} B_j, \qquad \sup_{p, p' \in B_j} H(p, p') \leq 4(\alpha^2/2)^{1/\alpha}, \qquad \sum_{j=1}^{J(\alpha)} \Pi(B_j)^\alpha < \infty,$$

kde $H(p, p')$ je Hellingerova vzdálenost mezi $p$ a $p'$. Tato podmínka je technickým požadavkem na to, aby prior nepřiděloval příliš velkou hmotnost oblastem daleko od $p^*$ — prior nesmí být "příliš rozptýlený" v nepříznivých směrech.

## Theorem 3.1

**Theorem 3.1.** Za podmínek (A1) a (A2) existuje $p^* \in \mathcal{P}$ takové, že

$$\pi(y \mid x, D_n) \xrightarrow{n \to \infty} p^*(y \mid x) \quad \text{almost surely},$$

pro $P_0$-almost every $(y, x)$. Navíc $p^*$ je KL-optimální aproximace $p_0$ v $\mathcal{P}$:

$$p^* = \arg\min_{p \in \mathcal{P}} \, \mathrm{KL}(p \mid p_0).$$

Přesné podmínky a důkaz jsou uvedeny v Appendix A.2 článku.

## Interpretace

Theorem 3.1 říká, že PPD se s přibývajícími daty vždy naučí **nejlepší možnou aproximaci** $p_0$ dosažitelnou v $\mathcal{P}$:

- Pokud $p_0 \in \mathcal{P}$ (prior obsahuje pravý model): $p^* = p_0$ a PPD konverguje přímo k $p_0$.
- Pokud $p_0 \notin \mathcal{P}$ (prior pravý model neobsahuje): PPD stále konverguje, ale k nejbližšímu $p^* \in \mathcal{P}$ v KL smyslu. Čím větší je $\mathcal{P}$, tím blíže je $p^*$ pravému $p_0$.

Výsledek je tedy silný: prior nemusí být perfektní. Stačí, aby byl "dostatečně rozumný" ve smyslu podmínek (A1) a (A2), a data sama posterior opraví. To je teoretickým zdůvodněním toho, proč PPD vůbec funguje jako smysluplný cíl pro trénink PFN — a zároveň motivací pro volbu co nejbohatšího prioru $\Pi$: čím bohatší $\mathcal{P}$, tím kvalitnější je $p^*$ jako aproximace $p_0$.

Theorem 3.1 se ovšem vztahuje na ideální PPD $\pi$, nikoliv na PFN $q_{\hat\theta}$ samotný. PFN je pouze aproximací PPD trénovanou na omezených velikostech datasetů, takže Theorem 3.1 na $q_{\hat\theta}$ přímo nelze aplikovat — tím se zabývají kapitoly 4 a 5.
