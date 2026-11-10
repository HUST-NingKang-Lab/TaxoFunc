# Released validation metrics

`val_benchmark_20261106_public.tsv` contains the basic validation-split metrics
for `TaxoFunc-P2FM-ens-v2` and the released comparators.

## Protocol

- Validation split: 3,098 samples from 99 held-out studies.
- 64 studies have at least two samples and contribute to per-study medians.
- Scoring mask: 5,610 KOs with training prevalence ≥ 10%.
- Primary metric: feature-level Spearman correlation on raw truth.
- `pooled`: per-KO Spearman across samples, median over the 5,610 KOs.
- `ps_med_raw`: per-study median Spearman across the 64 scoreable studies.
- `ps_med_n4`: per-study median over studies with at least four samples.

## Columns

| Column | Meaning |
|---|---|
| `method` | Method name |
| `family` | Predictor family |
| `input` | Input features |
| `pooled` | Pooled feature-level Spearman |
| `ps_med_raw` | Per-study median Spearman, all 64 scoreable studies |
| `ps_med_n4` | Per-study median Spearman, studies with n ≥ 4 |

## What is withheld

Per-study paired test statistics (paired medians, sign-test p-values, and
per-study deltas) are withheld from this release pending authorization. Their
omission does not change the metric definitions or the released point estimates.

## Scope

All numbers are validation-split results for `TaxoFunc-P2FM-ens-v2`. No held-out
evaluation or external-cohort result is included.

MGFunc and PICRUSt2 are **validation-benchmark comparators**, not external
validation of TaxoFunc. The traditional-ML rows use raw taxon relative abundance
only and do not use the AE, P2 block, or model decoder.

The `final_s2` autoencoder was pretrained on the 17,959 L2 training rows, which
include the validation studies. Absolute TaxoFunc validation scores are therefore
optimistic by roughly 0.03 Spearman.

## Provenance

The table is a public-facing extraction of the project's internal validation
metrics. Metric values are unchanged. Method labels were normalized for public
readability:

- internal `Ridge F0 a=1` → `Ridge (raw taxa)`
- internal `MLP 256 (raw F0)` → `MLP 256 (raw taxa)`
- internal `MLP 256 (z-scored F0)` → `MLP 256 (z-scored taxa)`
- internal `RF shuffled-label control` → `RandomForest shuffled-label control`
- internal `PICRUSt2 B4 raw prior` → `PICRUSt2 prior (B4)`
- internal `MGFunc B5` → `MGFunc (B5)`
- internal family `external/prior` → `validation benchmark baseline`
