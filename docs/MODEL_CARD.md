# Model card — TaxoFunc-P2FM-ens-v2

## Model details

| Field | Value |
|---|---|
| Model name | `TaxoFunc-P2FM-ens-v2` |
| Task | Prokaryotic taxonomy → KEGG KO relative abundance |
| Input | 8,234 multi-rank prokaryotic taxon features |
| Output | 13,945 KEGG KO relative abundances (scored on 5,610 KOs) |
| Architecture | Independent AE latents + conditional flow-matching UNet + PICRUSt2 prior encoder + fixed latent fusion |
| Fusion | `z* = 0.65 · h(P2) + 0.35 · z_FM` |
| Training data | MGnify paired taxonomic and KO profiles; 14,861 training rows for flow/prior |
| Validation data | 3,098 samples from 99 held-out studies |
| Release date | 2026-11-06 (v2 configuration) |
| License | TaxoFunc-specific code: no license granted yet; reuse rights pending |
| Third-party backbone | scDiffusion-X (MIT top-level; BSD-3-Clause package; see `THIRD_PARTY_NOTICES.md`) |

## Intended use

TaxoFunc is a **research model** for studying whether a conditional generative
model can learn the joint structure of KO profiles from taxonomic composition.
It is intended for:

- methodological research on cross-modal generation for microbiome data;
- benchmarking taxonomy-to-function prediction pipelines;
- inspecting the contribution of a PICRUSt2 prior versus a learned generative
  branch in a controlled latent ensemble.

It is **not** intended for clinical diagnosis, therapeutic decision-making,
industrial process control, or any application where incorrect functional
predictions could cause harm.

## Out-of-scope use

- Predicting functional profiles for samples whose taxonomy is outside the
  prokaryotic panel or whose biome is not represented in training.
- Claiming causal or mechanistic relationships between taxa and KO functions.
- Using validation scores as an estimate of performance on a new cohort.
- Redistributing the model as a licensed product; the license is pending.

## Architecture

```text
x_tax (8,234)
 ├─ AE_tax encoder: 8,234 → 2,048 → 1,024 → 200
 │   └─ flow-matching UNet (DCA, 128 channels, 180k steps)
 │       └─ 1,000-step Euler ODE, four sampling seeds averaged
 │           └─ z_FM (100)
 │
 └─ P_ext (8,234 × 13,945) → A_rel @ P_ext
     └─ CLR → TRAIN-only PCA-128 → × 0.1 → P2 (128)
         └─ PriorEncoder 128 → 256 → 100, five seeds averaged
             └─ h(P2) (100)

z* = 0.65 · h(P2) + 0.35 · z_FM
z* → AE_ko decoder (100 → 256 → 512 → 13,945) → KO relative abundance
```

The model is an **explicit two-estimator ensemble**, not a single end-to-end
graph. The fusion weight is fixed and is not learned inside the network. The
cfg11 prior-encoder configuration and `α=0.35` were selected on the validation
split during model development; the validation metrics therefore include a
hyperparameter-selection bias and are optimistic relative to an unbiased estimate
on a new cohort.

## Training data

- Source: MGnify paired taxonomic and KO functional profiles.
- Scope: Bacteria and Archaea only; 8,234 multi-rank taxon features.
- KO space: 13,945 KOs common to the truth matrix and the PICRUSt2 reference.
- Samples: 23,951 after removing degenerate rows; 619 studies.
- Split: study-grouped (L2). 17,959 L2 training rows, of which 3,098 validation
  samples from 99 studies are held out from flow/prior training.
- Flow/prior training: 14,861 rows after removing the 99 validation studies.
- Scoring mask: 5,610 KOs with training prevalence ≥ 10%.

The 3,098 validation rows are a subset of the 17,959 L2 training rows; the
flow/prior models use the remaining 14,861 rows. The total of 23,951 also
includes a further partition that is not described in this release.

Raw data are not redistributed. See `DATA.md`.

## Evaluation

All reported metrics are on the validation split:

- Primary metric: feature-level Spearman correlation on raw truth.
- Pooled: per-KO Spearman across samples, median over 5,610 KOs.
- Per-study median: per-study per-KO Spearman, median over KOs, then median over
  the 64 studies with at least two samples.
- `n≥4`: subset of studies with at least four samples.

| Method | Pooled | Per-study median (64 studies) | Per-study median (n≥4) |
|---|---:|---:|---:|
| **TaxoFunc-P2FM-ens-v2** | **0.6952** | **0.3197** | **0.3188** |
| Ridge (raw taxa) | 0.4740 | 0.2320 | 0.2236 |
| RandomForest 300 | 0.5100 | 0.2024 | 0.2047 |
| MLP 256 (raw taxa) | 0.3462 | 0.1096 | 0.1074 |
| PICRUSt2 prior (B4) | 0.4292 | 0.1997 | 0.1853 |
| MGFunc (B5) | 0.4094 | 0.1845 | 0.1772 |

MGFunc and PICRUSt2 are validation-benchmark comparators, not external
validation of TaxoFunc. Per-study paired test statistics are withheld from this
release.

## Caveats

- **Validation-only evidence.** No held-out or cross-cohort results are included.
- **Optimistic absolute scores.** The `final_s2` AE was pretrained on the 17,959
  L2 training rows, which include the validation studies. Absolute validation
  scores are optimistic by roughly 0.03 Spearman.
- **Hyperparameter selection on validation.** The cfg11 configuration and the
  fusion weight `α=0.35` were selected on the validation split. Validation
  metrics include this selection bias and are optimistic for a new cohort.
- **Explicit ensemble.** The released model fuses two estimators; it is not a
  single end-to-end graph.
- **No generalization guarantee.** Performance varies by study and biome.
- **Asset dependence.** The data and weights required for end-to-end inference
  are not included in this repository.
- **Statistical association only.** The model does not infer metabolic flux,
  interactions, or causality.

## Ethical and societal considerations

The model operates on environmental and host-associated microbiome profiles.
Taxonomic and functional profiles can be sensitive; users should follow the
consent and data-use terms of the original MGnify studies. The model should not
be used to make claims about individuals or groups. No human-subject data are
redistributed in this repository.

## Citation

Citation information for the accompanying manuscript will be added when
available. If you use the vendored backbone, please also cite scDiffusion-X (see
`THIRD_PARTY_NOTICES.md`).
