# Evaluation

## Split

All numbers in this release are computed on the validation split:

- 3,098 samples from 99 held-out studies.
- 64 studies have at least two samples and contribute to per-study medians.
- The validation studies are excluded from flow-matching and prior-encoder
  training.
- The `final_s2` autoencoder was pretrained on the 17,959 L2 training rows,
  which include the validation studies. Absolute validation scores are therefore
  optimistic.
- The cfg11 prior-encoder configuration and the fusion weight `α=0.35` were
  selected on the validation split during model development. The validation
  metrics therefore include a hyperparameter-selection bias and are optimistic.

The 3,098 validation rows are a subset of the 17,959 L2 training rows; the
flow/prior models use the remaining 14,861 rows. The total of 23,951 also
includes a further partition that is not described in this release. No other
evaluation partitions or their metrics are included.

## Metric

The primary metric is **feature-level Spearman rank correlation** on raw truth.
Spearman is used instead of Pearson because KO profiles are compositional,
zero-inflated, and heavy-tailed; Pearson is dominated by a few high-abundance
KOs and by sequencing depth.

For a prediction matrix `P` and truth matrix `T`, both `(n_samples, n_ko)`:

### Pooled score

For each KO `j`, compute the Spearman correlation between `P[:, j]` and `T[:, j]`
across all validation samples. The pooled score is the median over the 5,610
scored KOs.

### Per-study median

For each study `s` with at least two samples:

1. Compute the Spearman correlation between `P[:, j]` and `T[:, j]` across the
   samples of that study, for every KO `j`.
2. Take the median over KOs.
3. Take the median of these per-study medians over all scoreable studies.

The `n≥4` column restricts step 3 to studies with at least four samples.

### Scoring mask

The 5,610-KO mask is defined by training prevalence ≥ 10%. KOs that are almost
always zero in training are not scored, because their rank correlation is
undefined or unstable.

## Comparators

| Comparator | Family | Input |
|---|---|---|
| Ridge (raw taxa) | traditional ML, end-to-end | raw taxon relative abundance |
| RandomForest 300 / 100 | traditional ML, end-to-end | raw taxon relative abundance |
| MLP 256 (raw taxa) | traditional ML, end-to-end | raw taxon relative abundance |
| PICRUSt2 prior (B4) | validation benchmark baseline | PICRUSt2 taxon→KO prior |
| MGFunc (B5) | validation benchmark baseline | MGFunc tool output |

The traditional-ML comparators do not use the AE, the P2 block, or the model
decoder. PICRUSt2 and MGFunc are reference tools/priors. These are different
families of predictors, so cross-family comparisons are descriptive.

MGFunc and PICRUSt2 are **validation-benchmark comparators**, not external
validation of TaxoFunc. No cross-cohort evaluation is reported.

## Released table

`results/val_benchmark_20261106_public.tsv` contains the basic validation
metrics:

| Column | Meaning |
|---|---|
| `method` | Method name |
| `family` | Predictor family |
| `input` | Input features |
| `pooled` | Pooled feature-level Spearman |
| `ps_med_raw` | Per-study median Spearman, all 64 scoreable studies |
| `ps_med_n4` | Per-study median Spearman, studies with n ≥ 4 |

Per-study paired test statistics (paired medians, sign-test p-values, and
per-study deltas) are withheld from this release pending authorization. Their
omission does not change the metric definitions or the released point estimates.

## What is not reported

- Held-out evaluation results.
- External-cohort or cross-study generalization results.
- Per-study paired test statistics.
- Model weights, predictions, or latent caches.

## Reproducing the metrics

The released code includes the model, training, inference, and P2 construction
logic, but not the data or weights. See `REPRODUCIBILITY.md` for the required
assets and the explicit limitations on end-to-end reproduction.
