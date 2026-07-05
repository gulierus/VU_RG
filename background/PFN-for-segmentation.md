# PFNs and amortized inference for image segmentation — a survey

*A possible real-world application of your project (analysis of how PFNs approximate GP
inference — attention structure, kernel comparison, and uncertainty calibration): where
amortized / in-context inference already lives in image segmentation, what is genuinely
open, and how your results plug in.*

---

## 1. The one-line answer

- **Amortized / in-context inference is heavily used in segmentation.** A whole family of
  models conditions on a *support set* of labeled examples and segments a new image with
  no retraining — exactly the PFN *idea* (amortize over tasks, one forward pass), built
  for pixels.
- **But nobody has trained a literal PFN for segmentation** — synthetic draws from an
  explicit prior → posterior-predictive density. That is genuine white space.
- **Your transferable asset is your uncertainty-calibration result, not "attention is a
  kernel smoother."** Your own experiments show the PFN does *more* than smoothing (full
  GP inference), yet its uncertainty is shape-right/scale-wrong. That calibration finding
  is what lifts to segmentation.

## 2. The landscape you'd be entering

Two lineages, converging on one paper.

**In-context (set-based) segmentation** — your amortization idea, for images:
- **UniverSeg** (Butoi et al., ICCV 2023) — query image + support set of image/label
  pairs → mask, no fine-tuning; a "CrossBlock" exchanges features between query and each
  support entry. Performance rises with support-set size and then plateaus — the same
  context-size saturation you measured in Experiment 5 (MSE↓ with more context points).
- **SegGPT** (Wang et al., ICCV 2023) — decoder-only transformer, "paint the mask in
  context" from a prompt image+mask; ≈56 mIoU one-shot on COCO-20i.
- SEEM, Painter — related promptable / in-context segmenters.

**Amortized-posterior uncertainty** — the Bayesian half:
- **Probabilistic U-Net** (Kohl et al., NeurIPS 2018) — U-Net + conditional VAE that
  amortizes a *posterior over plausible masks* and samples many hypotheses; scored by
  generalized energy distance. Amortized *per image*, not over a task set.
- PhiSeg (Baumgartner et al., MICCAI 2019), Stochastic Segmentation Networks
  (Monteiro et al., NeurIPS 2020) — hierarchical / covariance variants.

**Where they meet — your target:**
- **Tyche** (Rakic et al., CVPR 2024) — **stochastic in-context** segmentation: a context
  set defines the task *and* the model emits a diverse *set* of candidate masks. This is
  the closest thing that exists to "amortized-Bayesian in-context segmentation." Crucially
  it is **still not a PFN**: its spread comes from a best-candidate training loss, not a
  calibrated predictive density. Whether that spread is *calibrated* is exactly the
  question your methods answer — see §4.

## 3. What your experiments actually established (and why it matters here)

Be precise about your own findings — they are stronger and more specific than a generic
"attention = kernel smoother" claim, and they set up the segmentation angle:

- **The PFN performs full GP inference, not kernel smoothing.** MSE(PFN, GP) ≈ 10⁻⁵ vs
  MSE(Nadaraya–Watson, GP) ≈ 10⁻² — four orders of magnitude. The "single attention head ≈
  NW estimator" hypothesis is **refuted**; the model learns an implicit, non-RBF kernel
  (attention–RBF correlation only 0.04–0.65) and captures the K⁻¹ (decorrelation) effect
  that separates GP from NW.
- **Interpretability depends on training.** The undertrained model decomposes into
  readable steps (NW-like early layers + a final-layer correction); the fully-trained model
  distributes the computation and is not decomposable — the accuracy is higher, the story
  less legible.
- **Uncertainty is shape-right, scale-wrong.** The PFN gets *where* to be uncertain nearly
  perfectly (PFN–GP σ shape correlation 0.94 → 0.99 with more training) but the *scale* is
  off: inflated baseline and a compressed dynamic range (far/near σ ratio ≈ 50% of the GP's),
  and this gap **persists in the fully-trained model** — likely a BarDistribution
  discretization limit and/or a train-distribution coverage gap.

**Why the naive framing is a trap.** "The segmentation transformer's attention is a kernel
smoother over the support set" is (a) *prior art* — Tsai et al. 2019 ("Transformer
Dissection") proved attention is a Nadaraya–Watson smoother in general, and few-shot
segmentation already matches query↔support via correlation/prototype kernels (HSNet, Min et
al. ICCV 2021); and (b) *empirically incomplete* — your own results show a competent PFN
does strictly more than smoothing. So don't pitch "attention = smoother." Pitch the
calibration/bias-variance characterization, which is genuinely yours.

## 4. What IS open — your actual angle

Your uncertainty result answers questions nobody has answered for in-context segmenters:

**(a) Is the shape-right/scale-wrong calibration failure a general property of amortized
in-context models?** You showed it for PFN-as-GP in 1-D. Test the same on UniverSeg/Tyche:
does the segmenter know *where* it is uncertain (high σ on ambiguous boundaries / far from
any support example) while getting the *magnitude* wrong — and does the miscalibration grow
as the support set shrinks or goes out-of-distribution (unseen anatomy/modality)? This is
your Experiment 1 finding, in 2-D, on a real model.

**(b) Does the segmenter do full posterior inference, or only support-set smoothing?**
Your PFN-vs-NW result gives you the tool to ask whether Tyche's mask distribution is a
genuine calibrated posterior or merely diverse candidates around a smoothed mean — the
segmentation analogue of your "full GP vs NW" separation.

**(c) [most ambitious] Train a real PFN segmenter.** Define an explicit prior over
segmentation tasks, train posterior-predictive, and compare its calibration to
UniverSeg/Tyche. High risk, but it is the untried white space and would make the thesis a
genuine "PFN for segmentation" first.

## 5. The diagnostic toolkit (what "calibration / bias–variance analysis" concretely means)

These are the measurements to run on any amortized/in-context segmenter — self-contained,
and a direct generalization of the metrics you already used in 1-D:

1. **Uncertainty shape vs scale.** Separate the two the way you did for PFN-vs-GP σ:
   correlation of the model's σ with the actual error map (does it know *where*), and the
   ratio of far-from-support to near-support σ against a reference (does it know *how much*).
2. **Error-awareness.** Spearman correlation between per-pixel/per-region σ and actual
   error |μ − y|.
3. **Coverage + recalibration.** Empirical vs nominal interval coverage; then a single-
   parameter σ-scaling fix (scale factor κ = std of standardized residuals; κ > 1 ⇒
   overconfident, κ < 1 ⇒ underconfident) — both a diagnostic and a cheap correction, and
   the natural remedy for a shape-right/scale-wrong model like yours.
4. **Full-density NLL**, computed separately *inside* vs *outside* the ambiguous / hard
   region — rewards a calibrated mean+variance, not just accuracy.
5. **Epistemic vs aleatoric split.** Separate reducible uncertainty (more/closer support
   examples lower it) from irreducible ambiguity (inherent grader disagreement).

## 6. Concrete first experiment (low-risk, high-signal)

Take a **pretrained UniverSeg or Tyche** (public weights), no training needed:
1. Sweep support-set size and support-set OOD-ness; plot Dice **and** the toolkit metrics
   (§5) against both.
2. Track the **shape-vs-scale** split: does the model stay shape-right (σ high on true
   error regions) while scale drifts as the support degrades — the 2-D echo of your
   Experiment 1?
3. Test the **smoothing-vs-inference** question: compare the segmenter's mask to a simple
   support-set label smoother (a prototype/correlation baseline). Where the segmenter beats
   it is where it is doing "more than smoothing" — the segmentation analogue of your
   PFN-vs-NW four-orders result.

Deliverable: "in-context segmenters inherit the PFN shape-right/scale-wrong calibration
failure and it worsens under support-set shift; here is the bias–variance account and a
σ-scaling fix." Reuses your existing analysis notebooks almost verbatim; no new
segmentation infrastructure.

## 7. Honest ranking

| Path | Reuses your work | New infra | Novelty | Risk |
|---|---|---|---|---|
| Diagnose pretrained UniverSeg/Tyche (§6) | high | none | medium–high | low |
| Train a real PFN segmenter (§4c) | medium | high | high | high |
| "Attention = smoother" claim | — | none | **none — prior art *and* refuted by your own data** | — |

Start at §6. It turns your GP-approximation calibration result into a real-image finding in
weeks, and it de-risks the more ambitious PFN-segmenter idea by first showing the failure
mode exists.

## References

- Butoi, Gonzalez Ortiz, Ma, Sabuncu, Guttag, Dalca. *UniverSeg: Universal Medical Image Segmentation.* ICCV 2023. arXiv:2304.06131.
- Wang, Zhang, Cao, Wang, Shen, Huang. *SegGPT: Segmenting Everything In Context.* ICCV 2023. arXiv:2304.03284.
- Rakic, Wong, Gonzalez Ortiz, Cimini, Guttag, Dalca. *Tyche: Stochastic In-Context Learning for Medical Image Segmentation.* CVPR 2024. arXiv:2401.13650.
- Kohl, Romera-Paredes, Meyer, De Fauw, Ledsam, Maier-Hein, Eslami, Rezende, Ronneberger. *A Probabilistic U-Net for Segmentation of Ambiguous Images.* NeurIPS 2018. arXiv:1806.05034.
- Tsai, Bai, Yamada, Morency, Salakhutdinov. *Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel.* EMNLP-IJCNLP 2019. arXiv:1908.11775.
- Min, Kang, Cho. *Hypercorrelation Squeeze for Few-Shot Segmentation.* ICCV 2021. arXiv:2104.01538.
