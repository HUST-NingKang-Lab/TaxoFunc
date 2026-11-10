# Method

This document describes the released model, **TaxoFunc-P2FM-ens-v2**. It covers
the problem formulation, the two-stage training procedure, the flow-matching
objective, the PICRUSt2 prior branch, and the fixed latent fusion.

---

## 1. Problem formulation

Given a prokaryotic taxonomic abundance profile

```text
x_tax ∈ R^{8234},   x_tax ≥ 0,   sum(x_tax) ≈ 1
```

predict the KEGG Ortholog relative-abundance profile

```text
y_ko ∈ R^{13945},   y_ko ≥ 0,   sum(y_ko) = 1
```

This is treated as **cross-modal conditional generation**, not as 13,945
independent regressions. KO abundances are strongly correlated through pathways,
modules, and genome adjacency; modelling their joint structure in a low-
dimensional latent space is the central design choice.

---

## 2. Stage 1 — independent autoencoders

Two autoencoders are trained independently on the training split. They do not
share latents and have no cross-modal alignment loss.

```text
E_tax : 8,234 → 2,048 → 1,024 → 200        z_tax
D_tax : 200 → 512 → 1,024 → 8,234          taxonomy reconstruction
E_ko  : 13,945 → 1,024 → 512 → 100         z_ko
D_ko  : 100 → 256 → 512 → 13,945           KO reconstruction
```

The decoder width is decoupled from the encoder mirror. For a sample with size
factor `s`:

```text
μ_tax = softmax(D_tax(z_tax)) · s_tax
μ_ko  = softmax(D_ko(z_ko))  · s_ko

L_AE = − log Poisson(x_tax | μ_tax)
       − log NegativeBinomial(x_ko | μ_ko, θ)
```

The size factor is passed separately, so sequencing depth does not need to be
represented in the latent. The encoder sees raw counts (`AE_INPUT=raw`), matching
the inference path.

Both likelihoods are floored at the float32 epsilon before entering the
likelihood (`models/scdiffusion_x/py/mgm_patch.py`). This is a numerical guard,
not a model change: taxonomy depth spans several orders of magnitude, and a
softmax rate can underflow to exactly zero, making the Poisson/NB gradient
undefined.

After training, latent scales are computed on the training split:

```text
rna_std  = mean_j std_train(z_ko[:, j])  × 10
atac_std = mean_j std_train(z_tax[:, j]) × 10
```

The flow model and the prior encoder operate on the standardised latents
`z_ko / rna_std` and `z_tax / atac_std`.

---

## 3. Stage 2 — conditional flow matching

### 3.1 Objective

The flow model learns a velocity field that transports a Gaussian base
distribution to the conditional distribution of the standardised KO latent.

```text
u    = 10 · (z_ko / rna_std)
u_t  = (1 − t) · ε + t · u,       ε ~ N(0, I),   t ~ U[0, 1)
loss = || (u − ε) − v_θ(u_t, t | z_tax / atac_std, biome) ||²
```

The target scale `10` brings the per-dimension standard deviation of `u` close
to one. The condition is the **clean** taxonomic latent at every flow time; it is
not re-noised. This is the key difference from a naive conditional diffusion
objective in which the condition is corrupted and the network can ignore it.

### 3.2 Network

The velocity network is the `MultimodalUNet` from scDiffusion-X with its dual
cross-attention (DCA) blocks:

- 128 channels, one residual block per resolution.
- Cross-attention at resolutions 2, 4, and 8 with windows 1, 4, and 8.
- DCA performs attention in both directions: KO tokens attend to taxonomy
  tokens, and taxonomy tokens attend to KO tokens.
- The biome label (84 training classes) is embedded and combined with the time
  embedding, then used to modulate the residual blocks.

The network architecture itself is unchanged from upstream; the project-specific
change is the objective and the clean-condition sampling convention.

### 3.3 Sampling

At inference, the learned velocity field is integrated with 1,000 Euler steps:

```text
u_0 ~ N(0, I)
du/dt = v_θ(u_t, t | z_tax / atac_std, biome),   t: 0 → 1
z_FM  = u_1 / 10
```

The condition is kept clean at every step. Four independent sampling seeds are
averaged in latent space (`avg4`) to reduce sampling noise. This is an inference
variance-reduction step, not a separate model.

---

## 4. PICRUSt2 prior branch

A fixed PICRUSt2 taxon-to-KO table provides an external functional prior:

```text
P_ext = row-normalised PICRUSt2 taxon → KO table       (8,234 × 13,945)
P     = A_rel @ P_ext,          A_rel = x_tax / sum(x_tax)
L     = log(P + 1e-6)
L    -= mean(L)                                         # CLR along the KO axis
P2    = ((L − TRAIN_mean) @ TRAIN_components) / TRAIN_scale
```

`TRAIN_mean`, `TRAIN_components` (128 directions), and `TRAIN_scale` are fitted
on the training split only. The resulting 128-dimensional block is multiplied by
`0.1` to match the scale of the other model inputs.

The prior encoder is a small MLP:

```text
h(P2) = Linear(128 → 256) → GELU → LayerNorm → Dropout(0.1) → Linear(256 → 100)
```

It is trained on the training split to predict the standardised KO latent. The
released configuration (`cfg11`) uses:

- learning rate `5e-4`
- 200 epochs
- batch size 256
- weight decay 0
- dropout 0.1
- z-scored P2 input

Five training seeds (`42, 7, 123, 2024, 99`) are averaged.

`P2` is a **functional prior input to the model**, not a separately reported
prediction arm. It is one of the two estimators in the released ensemble.

---

## 5. Fixed latent fusion

The two 100-dimensional estimates are fused with a fixed weight that is not
learned inside the network:

```text
z* = 0.65 · h(P2) + 0.35 · z_FM
```

The KO decoder then produces the output profile:

```text
y_ko = normalize( softmax(D_ko(z* · rna_std)) )
```

The fusion weight is fixed in the released model and is not learned inside the
network, nor is it adapted per sample. However, the cfg11 prior-encoder
configuration and `α=0.35` were selected on the validation split during model
development. The validation metrics therefore include a hyperparameter-selection
bias and are optimistic relative to an unbiased estimate on a new cohort. This
is why the released model is described as an **explicit two-estimator ensemble**,
not a single end-to-end graph.

---

## 6. What is project-specific vs upstream

| Component | Source |
|---|---|
| `MultimodalUNet`, DCA blocks, autoencoder modules | scDiffusion-X (MIT top-level; BSD-3-Clause package), vendored; see `THIRD_PARTY_NOTICES.md` |
| Numerical rate-floor guard | project patch (`mgm_patch.py`) |
| Decoupled decoder and encoder input parameterization | project patch (`ae_v2_patch.py`) |
| Flow-matching objective with clean condition | project patch (`flow_objective_patch.py`) |
| Euler sampler with clean condition | project patch (`flow_sampler_patch.py`) |
| P2 block construction | project code (`taxofunc/build_p2_block.py`) |
| PriorEncoder (`h_model.py`) | project code |
| Fixed fusion | project code (reference assembly) |

The project-specific changes are deliberately implemented as monkey-patches and
small standalone modules so that the project-specific patches do not modify the
upstream network code.

---

## 7. Scope and caveats

- The model is a statistical translation model, not a mechanistic model of
  metabolism.
- The autoencoder latents are not identifiable; the flow model is tied to the
  released AE space.
- The `final_s2` AE was pretrained on the 17,959 L2 training rows, which include
  the validation studies. Absolute validation scores are optimistic.
- The cfg11 prior-encoder configuration and the fusion weight `α=0.35` were
  selected on the validation split; validation metrics therefore include a
  hyperparameter-selection bias.
- No held-out or cross-cohort evaluation is reported in this release.
- The repository does not include the data or weights required for end-to-end
  reproduction; see `REPRODUCIBILITY.md`.
