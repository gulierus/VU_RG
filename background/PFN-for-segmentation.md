# PFNs and amortized inference for image segmentation — a survey

*A possible real-world application of your project (bias–variance and internal-representation
analysis of PFNs as GP approximators): where amortized / in-context inference already lives
in image segmentation, what is genuinely open, and how your results plug in.*

---

## 1. The one-line answer

- **Amortized / in-context inference is heavily used in segmentation.** A whole family of
  models conditions on a *support set* of labeled examples and segments a new image with
  no retraining — exactly the PFN *idea* (amortize over tasks, one forward pass), built
  for pixels.
- **But nobody has trained a literal PFN for segmentation** — synthetic draws from an
  explicit prior → posterior-predictive density. That is genuine white space.
- **Your "attention = kernel smoother over the support set" framing is *not* novel by
  itself** — it has two independent precedents. Your contribution has to be the
  *diagnostic* you bring on top of it (bias–variance, calibration under shift), which
  *is* open.

## 2. The landscape you'd be entering

Two lineages, converging on one paper.

**In-context (set-based) segmentation** — your amortization idea, for images:
- **UniverSeg** (Butoi et al., ICCV 2023) — query image + support set of image/label
  pairs → mask, no fine-tuning; a "CrossBlock" exchanges features between query and each
  support entry. Performance rises with support-set size and then plateaus. *This plateau
  is literally your Experiment 5 (context size → error) in 2D* — the same context-size
  saturation seen in PFN regression.
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
  calibrated predictive density. That gap is your opening.

## 3. Why your "attention = kernel smoother over the support set" is prior art

Be honest about this up front — a reviewer will know both of these:
- **Tsai et al. 2019, "Transformer Dissection"** already proved, in general, that
  attention *is* a Nadaraya–Watson kernel smoother (kernel score = query–key similarity) —
  the same lens the PFN-as-GP analysis uses.
- **Few-shot segmentation** already matches query↔support with correlation/prototype
  kernels (Hypercorrelation Squeeze / HSNet, Min et al. ICCV 2021; prototype networks).

So "the segmentation transformer's attention is a kernel smoother over the support set"
would be *re-deriving* known results. Don't pitch that as the contribution.

## 4. What IS open — your actual angle

Your two skills answer questions nobody has answered for in-context segmenters:

**(a) Does the amortized mask posterior collapse toward a prior under support-set shift?**
Your Experiment 1 result — "the mean reverts to the prior far from data, and the variance
is under-estimated" — lifts directly. For UniverSeg/Tyche: shrink the support set, push it
out-of-distribution (unseen anatomy/modality), and measure whether predictions regress to
a majority/prior mask and whether uncertainty stays calibrated. This is your
bias-toward-prior finding, in 2D, on a real model.

**(b) Is the uncertainty calibrated, and how does it split into bias vs variance?**
Tyche's diversity ≠ calibration. Bring a proper diagnostic toolkit (below) and a
bias/variance decomposition of the mask posterior as a function of support-set size and
shift.

**(c) [most ambitious] Train a real PFN segmenter.** Define an explicit prior over
segmentation tasks, train posterior-predictive, and compare its calibration to
UniverSeg/Tyche. High risk, but it is the untried white space and would make the thesis a
genuine "PFN for segmentation" first.

## 5. The diagnostic toolkit (what "calibration / bias–variance analysis" concretely means)

These are the measurements to run on any amortized/in-context segmenter — self-contained,
no new architecture needed:

1. **Error-awareness.** Spearman correlation between the model's per-pixel (or per-region)
   uncertainty σ and its actual error |μ − y| — "does the model know where it errs?"
2. **Full-density NLL**, computed separately *inside* vs *outside* the ambiguous / hard
   region — rewards a calibrated mean+variance, not just accuracy.
3. **Posterior fidelity vs a reference.** Where a reference posterior exists (multiple
   human graders, or a GP/oracle on a synthetic prior), measure ρ(model-mean, oracle-mean)
   and — the key plot — how that fidelity **decays** as the task hardens, the dimension
   grows, or the support set shrinks. This is the direct measurement of *amortization
   bias* (collapse toward a stationary prior).
4. **Coverage + recalibration.** Empirical vs nominal interval coverage; then a single-
   parameter σ-scaling fix (scale factor κ = std of standardized residuals; κ > 1 ⇒
   overconfident) — both a diagnostic and a cheap correction.
5. **Epistemic vs aleatoric split.** Separate reducible uncertainty (more/closer support
   examples lower it) from irreducible ambiguity (inherent grader disagreement) — this is
   the bias–variance story in uncertainty form.

## 6. Concrete first experiment (low-risk, high-signal)

Take a **pretrained UniverSeg or Tyche** (public weights), no training needed:
1. Sweep support-set size and support-set OOD-ness; plot Dice **and** the toolkit metrics
   (§5) against both.
2. Overlay: does the predicted mask move toward the dataset-prior / majority mask as the
   support degrades? (your "convergence to prior" result).
3. Read the CrossBlock / SetBlock attention as a kernel smoother over support pixels — and
   show *where* it stops being a faithful smoother under shift (the mechanism, not the
   equivalence).

Deliverable: "in-context segmenters inherit the PFN prior-collapse and mis-calibration
failure modes; here is the bias–variance account." Reuses your existing notebooks almost
verbatim and needs no new segmentation infrastructure.

## 7. Honest ranking

| Path | Reuses your work | New infra | Novelty | Risk |
|---|---|---|---|---|
| Diagnose pretrained UniverSeg/Tyche (§6) | high | none | medium–high | low |
| Train a real PFN segmenter (§4c) | medium | high | high | high |
| "Attention = smoother" alone | high | none | **none (prior art)** | — |

Start at §6. It turns your GP-approximation thesis into a real-image result in weeks, and
it de-risks the more ambitious PFN-segmenter idea by first showing the failure mode exists.

## References

- Butoi, Gonzalez Ortiz, Ma, Sabuncu, Guttag, Dalca. *UniverSeg: Universal Medical Image Segmentation.* ICCV 2023. arXiv:2304.06131.
- Wang, Zhang, Cao, Wang, Shen, Huang. *SegGPT: Segmenting Everything In Context.* ICCV 2023. arXiv:2304.03284.
- Rakic, Wong, Gonzalez Ortiz, Cimini, Guttag, Dalca. *Tyche: Stochastic In-Context Learning for Medical Image Segmentation.* CVPR 2024. arXiv:2401.13650.
- Kohl, Romera-Paredes, Meyer, De Fauw, Ledsam, Maier-Hein, Eslami, Rezende, Ronneberger. *A Probabilistic U-Net for Segmentation of Ambiguous Images.* NeurIPS 2018. arXiv:1806.05034.
- Tsai, Bai, Yamada, Morency, Salakhutdinov. *Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel.* EMNLP-IJCNLP 2019. arXiv:1908.11775.
- Min, Kang, Cho. *Hypercorrelation Squeeze for Few-Shot Segmentation.* ICCV 2021. arXiv:2104.01538.
