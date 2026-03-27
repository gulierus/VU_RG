# PFN and GP: Moving to Open Research Questions

*This document continues from `PFN-GP.md` and builds on the experiments you have already run. The goal is to connect your empirical findings to the theoretical framework and to frame the open research questions you should be working towards.*

---

## Part 1: Theoretical Background

### 1.1 Why Attention Weights Are Not Kernel Weights

`PFN-GP.md` introduced the analogy: one attention head ≈ Nadaraya-Watson (NW) kernel smoother. Your experiments already showed this is wrong — NW correlation was low in all heads even after 100 epochs of training. Here is why it *must* be wrong, from the architecture itself.

In the PFN transformer, each training point is packed into a single token:

$$V_j = (Y_j,\; X_j)^\top$$

The test point is the token $v = (0, x_*)^\top$. For head $h$, the attention score between the test token and training token $j$ is:

$$a_j^{(h)} \propto \exp\!\bigl(v^\top W_Q^{(h)} V_j\bigr)$$

Expanding this for 1-D inputs:

$$v^\top W_Q^{(h)} V_j = \underbrace{x_* \cdot [W_Q]_{22}}_{\alpha} \cdot X_j \;+\; \underbrace{x_* \cdot [W_Q]_{21}}_{\beta} \cdot Y_j$$

The weight on training point $j$ is a **linear combination of $X_j$ and $Y_j$** — both feature and label contribute. The GP weight $w_j = k(x_*, X_j) / \sum_k k(x_*, X_k)$ depends only on $X_j$. Unless the network sets $\beta = 0$ (which it has no incentive to do), the two quantities measure structurally different things.

**Nagler (2023) Theorem 6.3** makes this precise for the large-$n$ limit: the bias of a one-layer PFN converges to an integral under a *tilted* version of the data distribution, $g_h(s) \propto \exp(v^\top W_Q^{(h)} s)\, p_0(s)$. This is a distribution whose iso-density lines in the $(X_j, Y_j)$ plane are **straight parallel lines** — not circles centred on $x_*$. The attention is doing something bilinear in feature and label space, not localised kernel smoothing.

Furthermore, Theorem 5.4 shows that for a one-layer network the bias cannot vanish: vanishing bias requires the estimator to be *local* (ignore far-away training points), which the tilted exponential measure never achieves. The one-layer model is theoretically limited regardless of how long you train it.

> **Reading.** Nagler (2023), *Statistical Foundations of Prior-Data Fitted Networks*, ICML 2023. The key results are Theorem 6.2 (variance vanishes at rate $n^{-1/2}$) and Theorem 6.3 (bias converges to the tilted-measure limit). Theorem 5.4 on page 7 is the clearest statement of why one layer is not enough: "we should not expect the bias to vanish." Read Sections 5 and 6.

---

### 1.2 Multi-Layer Theory: The Neumann Series

`PFN-GP.md` stated "Layers 2+ approximate matrix inversion." This is approximately right but the precision matters, because it determines when the approximation works and how many layers are needed.

The GP posterior mean requires $(K + \sigma^2 I)^{-1} y$. This can be computed iteratively via gradient descent on $F(\alpha) = \tfrac{1}{2}\alpha^\top (K+\sigma^2 I)\alpha - y^\top \alpha$, whose minimiser is $(K+\sigma^2 I)^{-1} y$:

$$\alpha^{(0)} = 0, \qquad \alpha^{(t+1)} = \alpha^{(t)} + \eta\bigl(y - (K+\sigma^2 I)\alpha^{(t)}\bigr)$$

Unrolling gives the **Neumann series**:

$$(K+\sigma^2 I)^{-1} = \frac{1}{\eta}\sum_{t=0}^\infty \bigl(I - \eta(K+\sigma^2 I)\bigr)^t$$

which converges for any positive definite matrix with $0 < \eta < 2/\lambda_{\max}$.

The first term ($t=0$) gives $\hat{f}^{(1)}(x_*) = \eta \sum_j k(x_*, X_j) Y_j$ — the Nadaraya-Watson estimator. Each subsequent term adds a correction involving the off-diagonal structure of $K$ (the correlations between training points).

**How many terms (layers) are needed?** The series reaches error $\varepsilon$ in $O(\kappa \log 1/\varepsilon)$ steps, where the **condition number** is:

$$\kappa = \frac{\lambda_{\max}(K) + \sigma^2}{\lambda_{\min}(K) + \sigma^2}$$

| Setting | $\ell$ | $\sigma^2$ | $\kappa$ | Layers needed |
|---------|--------|------------|----------|---------------|
| Easy    | 0.1    | 0.5        | ~2       | 1–2           |
| Hard    | 1.0    | 0.01       | ~100     | many          |

A 6-layer model is a practical compromise over a distribution of hyperparameters. It will approximate the GP well when the kernel is easy to invert and underfit when the system is ill-conditioned.

**Important caveat.** The Neumann series picture is exact under *linear* attention (no softmax). With softmax, the denominator couples all points and the one-step = one-gradient-step correspondence breaks. The real network has freedom to deviate from this picture — the question is whether it does, and in which direction.

> **Reading.** von Oswald et al. (2022), *Transformers Learn In-Context by Gradient Descent*, arXiv:2212.07677. Read Sections 1–3 for the precise hand-construction of a linear-attention transformer implementing gradient descent. Akyürek et al. (2023), *What Learning Algorithm is In-Context Learning?*, ICLR 2023 (arXiv:2211.15661) provides empirical evidence that trained (not hand-constructed) transformers implement similar algorithms.


Nás by zajímalo, jestli opravdu existuje korelace mezi složitostí problému a počtem vrstev v PFN.
Platí totiž, že transformer umí napodobit gradient descent, tudíž je schopen řešit stejné problémy, které by mohl teoreticky vyřešit  GD, s tím, že jedna vrstva by měla odpovídat jednomu kroku GD.
Zkusit vzít různé modely s fixními hyperparametry a podívat se, jestli opravdu dělají něco jako Nuemannova řada. 

---

### 1.3 The Key Advantage: Amortised Bayesian Model Averaging

Standard GP workflow: choose hyperparameters $\theta = (\ell, \sigma_f^2, \sigma_n^2)$ by maximising the marginal likelihood (Type-II ML):

$$\hat\theta = \arg\max_\theta \log p(y \mid X, \theta)$$

This gives a **point estimate** of $\theta$. The fully Bayesian predictive distribution marginalises over $\theta$:

$$p(y_* \mid x_*, X, y) = \int p(y_* \mid x_*, X, y, \theta)\; p(\theta \mid X, y)\; d\theta$$

This integral is intractable and approximated by MCMC (as in your `Příklady_ML_Type_II_a_NUTS.ipynb`).

**What the PFN does:** it is meta-trained over a prior $p(\theta)$. After training, a single forward pass approximates the above integral — not by iterating over $\theta$ values, but by having encoded the average behaviour across many $(X, y, \theta)$ triples during pre-training. Inference cost is $O(n^2)$, independent of how many hyperparameter configurations were seen during training.

Your experiments confirmed the key prediction directly:

> **PFN wins at $n < 10$.** With very few points, $p(\theta \mid X, y)$ is broad and ML-II overfits $\theta$. The PFN carries a useful prior over $\theta$ from meta-training. At larger $n$, the marginal likelihood surface sharpens and ML-II converges to the true $\theta$, eliminating the advantage.

The crossover $n^* \approx 10$ is informative: it tells you approximately how many points are needed for ML-II to reliably identify the hyperparameters of the prior you used in training. This crossover depends on the prior width — a wider prior over $\theta$ means more uncertainty, a later crossover.

**The mixture-of-GPs view.** Making this precise reveals something important about the shape of the predictive distribution. For a fixed $\theta$, the GP predictive is Gaussian: $p(y_* \mid x_*, X, y, \theta) = \mathcal{N}(\mu_\theta, \sigma^2_\theta)$. Marginalising over $\theta$ gives a **mixture of Gaussians**:

$$p(y_* \mid x_*, X, y) = \int \mathcal{N}(\mu_\theta, \sigma^2_\theta)\; p(\theta \mid X, y)\; d\theta$$

This mixture is generally non-Gaussian. Its total variance decomposes as:

$$\text{Var}[y_*] = \underbrace{\mathbb{E}_\theta[\sigma^2_\theta]}_{\text{noise}} + \underbrace{\text{Var}_\theta[\mu_\theta]}_{\text{uncertainty about }\theta}$$

The second term — variance of the GP posterior mean across hyperparameter settings — is what ML-II discards when it plugs in $\hat\theta$. It is large exactly when $n$ is small and $p(\theta \mid X, y)$ is broad, which is precisely the regime where the PFN has an advantage. This gives a sharper theoretical explanation for the $n^*$ crossover: the PFN advantage is driven by the second variance term, which vanishes as $n \to \infty$ and the posterior over $\theta$ concentrates.

When the posterior $p(\theta \mid X, y)$ is **bimodal** — for example, when the observations are consistent with two very different lengthscales — the mixture predictive distribution can be bimodal or heavily non-Gaussian. A Gaussian output head cannot represent this. This is the theoretical justification for why the `BarDistribution` criterion used in your training setup is the right choice: it discretises the output into bins and can represent any shape, including the non-Gaussian mixtures that arise when hyperparameter uncertainty is large.

> **Reading.** Müller et al. (2022), *Transformers Can Do Bayesian Inference*, ICLR 2022. Sections 1–3 and the GP experiments in Section 5 are most relevant. For a related architecture that separates "summarise the training data" from "predict at the test point" more explicitly, see Garnelo et al. (2018), *Neural Processes* (arXiv:1807.01622). Rothfuss et al. (2021), *PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees*, ICML 2021, shows the theoretical basis for placing a prior over GP hyperparameters and optimising it over tasks — the PFN can be seen as amortising this into network weights.

---

### 1.4 Synthesis: Three Lenses on the Same Architecture

The three theoretical perspectives above each illuminate a different aspect of what the PFN is doing. None of them alone is complete.

| | **Nagler (2023)** | **von Oswald et al.** | **This project** |
|---|---|---|---|
| Attention type | Softmax | Linear (no softmax) | Softmax |
| Kernel | Fixed (any) | Fixed (linear) | Data-adaptive |
| Labels in attn. score | Yes ($\beta Y_j$) | No | Yes, but structured? |
| Hyperparameter tuning | Not modelled | Not modelled | Central topic |
| Multi-layer theory | Open | Yes ($L$ GD steps) | To derive |
| Bias vanishes? | No (Thm 5.4) | Yes (linear model) | Under conditions? |

**How to read the table.** Nagler tells you what one layer *does* and proves the bias cannot vanish. Oswald tells you what many layers *could* do — approximate the GP posterior — but only under linear attention and a fixed kernel. Neither framework accounts for the fact that a trained PFN adapts its effective kernel to the data it sees. That adaptation is what makes the PFN most useful in practice (the small-$n$ advantage), and it is the central object of study in this project.

The research questions in Part 2 are precisely the gaps in the "This project" column.

---

## Part 2: Research Objectives

**Q1 — Label mixing vs. kernel computation: layer-by-layer structure**

The attention score in each layer mixes $X_j$ and $Y_j$. Is this uniform across all layers, or does the network specialise early layers for kernel-like feature comparison ($\beta \approx 0$) and later layers for label propagation?

*Hypothesis:* $\text{corr}(a_j^{(\ell)}, Y_j)$ decreases with depth $\ell$ and $\text{corr}(a_j^{(\ell)}, k(x_*, X_j))$ increases. Meta-training on GP data gives the network an incentive to localise attention in feature space progressively.

**Q2 — How many layers for a given condition number?**

The Neumann series predicts that ill-conditioned systems (large $\ell$, small $\sigma^2$) need more correction terms. Does the empirical MSE-vs-depth curve match this prediction? Does adding a layer help more in the hard-$\kappa$ regime than in the easy one?

**Q3 — Minimum depth for hyperparameter identification**

Identifying the lengthscale requires computing a summary of the training set (e.g. pairwise distances). A 1-layer model can only compute weighted averages of individual tokens; it cannot compare tokens to each other without a second layer. Does the 1-layer model fail to adapt its bandwidth when $\ell$ changes, while a 2-layer model succeeds?

**Q4 — Computational crossover**

At equal wall-clock time, at what problem size $n$ and hyperparameter prior width does a PFN with $L$ layers achieve lower expected loss than GP-ML? This requires accounting for $O(n^2 L)$ PFN inference vs $O(n^3 T)$ GP-ML optimisation ($T$ gradient steps for the marginal likelihood). The answer depends on how well-conditioned the kernel matrix is, linking back to Q2.

**Q5 — Does the PFN's predictive distribution reflect hyperparameter uncertainty?**

The mixture-of-GPs view predicts that the PFN's output should be non-Gaussian when $p(\theta \mid X, y)$ is broad, and potentially bimodal when two hyperparameter settings are equally consistent with the data. Does the trained model's `BarDistribution` output actually exhibit this behaviour? When does it collapse to a near-Gaussian (suggesting the model has effectively identified $\theta$), and when does it spread out or develop multiple modes?

---

## Part 3: Guiding Instructions

### Step 1: The One-Layer Failure

Before extending to multiple layers, establish clearly that one layer is not sufficient. Train a 1-layer PFN with 8 heads on the same GP-RBF prior you used for the 6-layer model (fixed or variable lengthscale — either works). Evaluate it against the exact GP posterior on a test set of 200 instances with $n = 40$ context points.

Look for two distinct effects that Nagler's theorems predict:

1. **Variance** ($\approx$ predictive uncertainty): should decrease as $n$ grows, roughly as $1/n$. This is Theorem 6.2 and the model should satisfy it — the softmax normalisation guarantees it regardless of the weights.

2. **Bias** ($\approx$ systematic prediction error): should plateau at a non-zero level and *not* decrease further as $n$ increases beyond some moderate value. This is the consequence of Theorem 6.3 — the tilted measure $g_h$ never localises around $x_*$, so the bias cannot vanish.

To make this visible, plot the MSE decomposition as a function of $n$: fit a curve $\text{MSE}(n) \approx \text{bias}^2 + c/n$ and check whether bias$^2 > 0$ is statistically significant at large $n$.

Additionally, for a single test instance with $n = 40$, make the scatter plot in the $(X_j, Y_j)$ plane where each training point is coloured by its Layer 1, Head 0 attention weight. If the iso-weight contours are diagonal lines (not vertical bands), label mixing is operating as predicted by the $\alpha X_j + \beta Y_j$ formula.

### Step 2: Layer-Count Sweep (Q2)

Train four models with $L \in \{1, 2, 4, 8\}$ transformer layers, all with 8 heads, same embedding dimension, same training prior. For each model, compute the test NLL gap relative to the exact GP posterior:

$$\Delta_{\text{NLL}}(L) = \text{NLL}_{\text{PFN}}(L) - \text{NLL}_{\text{GP}}$$

averaged over 200 test instances with $n = 40$ context points. Plot $\Delta_{\text{NLL}}$ vs $L$.

Run this for **two** GP configurations:

| Config | $\ell$ | $\sigma^2$ | Expected $\kappa$ |
|--------|--------|------------|-------------------|
| Easy   | 0.1    | 0.5        | ~2                |
| Hard   | 0.8    | 0.05       | ~80               |

The Neumann series prediction: in the Easy config, 1–2 layers should nearly close the gap; in the Hard config, the gap should decrease more slowly with $L$ and a 6-layer model may still show significant residual bias.

If both curves converge at the same rate, the Neumann series picture is not capturing the dominant effect. If the Hard curve saturates while the Easy curve reaches zero, it is.

### Step 3: Label-Mixing Structure Per Layer (Q1)

Using your trained 6-layer model, register forward hooks to capture the attention weights at each layer:

```python
layer_attn_weights = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # output[1]: attn_weights, shape (batch, heads, seq_len, seq_len)
        layer_attn_weights[layer_idx] = output[1].detach().cpu()
    return hook

hooks = []
for i, layer in enumerate(model.encoder.layers):
    hooks.append(layer.self_attn.register_forward_hook(make_hook(i)))

with torch.no_grad():
    _ = model(x_train, y_train, x_test)

for h in hooks:
    h.remove()
```

For each layer $\ell$ and head $h$, extract the row of the attention matrix corresponding to the test token $x_*$ (the weights over the training context) and compute:

```python
for ell in range(n_layers):
    for head in range(n_heads):
        a = layer_attn_weights[ell][0, head, test_idx, :n_train].numpy()
        nw = compute_nw_weights(x_test_val, x_train, lengthscale)
        corr_kernel = np.corrcoef(a, nw)[0, 1]
        corr_label  = np.corrcoef(a, y_train.numpy())[0, 1]
```

Average over 200 test instances. Plot `corr_kernel` and `corr_label` as a function of layer index, separately for each head. The question is whether any layer shows a trend toward feature-based weighting.

### Step 4: Minimum Depth for Hyperparameter Identification (Q3)

Use the models trained in Step 2 with $L = 1$ and $L = 2$, trained on a variable-lengthscale prior $\ell \sim \text{LogNormal}(-1, 0.7)$.

For each model, evaluate on test sets generated at five fixed lengthscales $\ell \in \{0.05, 0.1, 0.3, 0.8, 2.0\}$ with $n = 20$ context points. Compute the **effective attention bandwidth** — the expected distance between the test point and attended training points:

```python
bandwidth = np.sum(attn_weights * np.abs(x_train - x_test_val))
```

For each model, plot bandwidth vs the true $\ell$ used to generate the test data. A model that identifies the lengthscale should show bandwidth increasing with $\ell$. A model that ignores it will show a flat line.

If the 1-layer model's bandwidth is flat and the 2-layer model's bandwidth tracks $\ell$, this is evidence that at least two layers are required for in-context hyperparameter adaptation.

### Step 5: Crossover as a Function of Prior Width (Q4)

You already know the crossover $n^* \approx 10$ for your current training prior. To turn this into a quantitative finding, vary the prior width:

- **Narrow prior**: $\ell \sim \text{LogNormal}(-1, 0.2)$ (concentrated around $\ell \approx 0.37$)
- **Wide prior**: $\ell \sim \text{LogNormal}(-1, 1.5)$ (spanning roughly $\ell \in [0.01, 10]$)

Train one PFN on each prior. For each model and for $n \in \{2, 5, 10, 20, 50, 100\}$, compute the NLL of the PFN and of GP-ML (with the true $\theta$ unknown, so it must estimate $\theta$ from $n$ points). Find the crossover $n^*$ for each prior width.

The prediction: $n^*$ is smaller for the narrow prior (less hyperparameter uncertainty means ML-II identifies $\theta$ reliably with fewer points) and larger for the wide prior. A clean plot of $n^*$ vs prior width $\sigma_\ell$ — estimated from just two or three training runs — would be an original contribution.

### Step 6: Predictive Distribution Shape under Hyperparameter Ambiguity (Q5)

Design a prior where hyperparameter ambiguity is maximised: at each meta-training step, draw $\ell$ from a **bimodal mixture**:

$$\ell \sim \tfrac{1}{2}\,\mathcal{N}(0.1, 0.01^2) + \tfrac{1}{2}\,\mathcal{N}(0.8, 0.05^2)$$

(truncated to positive values). Train a PFN on this prior. Then construct a test context that is deliberately ambiguous — for example, $n = 5$ observations drawn from the $\ell = 0.8$ GP but spaced such that they are also consistent with $\ell = 0.1$ (smooth observations at widely spaced $X_j$).

For this context, extract the full `BarDistribution` output at a test point $x_*$ and visualise it as a histogram. Compare it to:

1. The GP posterior with $\ell = 0.1$ (narrow, wiggly prior)
2. The GP posterior with $\ell = 0.8$ (wide, smooth prior)
3. The oracle mixture: $\tfrac{1}{2}\mathcal{N}(\mu_{0.1}, \sigma^2_{0.1}) + \tfrac{1}{2}\mathcal{N}(\mu_{0.8}, \sigma^2_{0.8})$

```python
# After a forward pass, extract the BarDistribution output
logits = model(x_train, y_train, x_test)  # raw logits over bins
probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

bin_edges = model.criterion.borders.cpu().numpy()
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(bin_centers, probs, width=np.diff(bin_edges), alpha=0.6, label='PFN')
# overlay the oracle mixture as a curve
ax.plot(bin_centers, oracle_mixture_pdf(bin_centers), 'r-', label='Oracle mixture')
ax.plot(bin_centers, gp_pdf(bin_centers, ell=0.1), 'b--', label='GP $\\ell=0.1$')
ax.plot(bin_centers, gp_pdf(bin_centers, ell=0.8), 'g--', label='GP $\\ell=0.8$')
ax.legend()
```

**What to look for:**
- If the PFN output is bimodal or has heavier tails than either single-GP posterior → the model has captured hyperparameter uncertainty.
- If it closely matches one of the single-GP posteriors → the model has effectively committed to one lengthscale from context.
- If it is wider than both → the model is uncertain but not in a structured way (this would suggest the network encodes uncertainty as scale rather than mixture shape).

Repeat this for an unambiguous context (many observations clearly from the $\ell = 0.8$ GP). The PFN output should narrow towards the $\ell = 0.8$ posterior in that case — if it does not, the BarDistribution is not being used to represent hyperparameter uncertainty effectively.

---

## Summary

| Experiment | Tests | Key prediction |
|---|---|---|
| Step 1: One-layer failure | Nagler Thm 5.4 + 6.3 | Bias plateaus; iso-weight contours are diagonal |
| Step 2: Layer-count sweep | Q2, Neumann series | Hard-$\kappa$ gap closes slower with $L$ than easy-$\kappa$ |
| Step 3: Label-mixing per layer | Q1 | `corr(NW)` increases, `corr(Y)` decreases with depth |
| Step 4: Bandwidth vs $\ell$ | Q3 | Only $L \geq 2$ tracks the true lengthscale |
| Step 5: Crossover vs prior width | Q4 | $n^*$ grows with prior width |
| Step 6: BarDistribution shape, bimodal prior | Q5 | PFN output is bimodal for ambiguous contexts, unimodal when $\theta$ is identifiable |

Working through these in order will give you both the empirical results and the theoretical framing needed to connect them to the Nagler / Oswald literature.
