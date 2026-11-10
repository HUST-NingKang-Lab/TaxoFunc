# Data

## Source

The project uses paired prokaryotic taxonomic and KEGG KO functional profiles
from **MGnify** (<https://www.ebi.ac.uk/metagenomics/>). MGnify provides
metagenomic analysis outputs, including taxonomic abundance profiles and
functional annotations, for public metagenomic studies.

The PICRUSt2 prior branch uses a fixed taxon-to-KO reference table derived from
PICRUSt2 (<https://github.com/picrust/picrust2>). The table is an external
functional prior, not a learned component of the generative model.

KEGG Ortholog identifiers and annotations are from KEGG
(<https://www.genome.jp/kegg/>).

## Scope

| Item | Value |
|---|---|
| Taxonomic domain | Bacteria + Archaea only |
| Taxon features | 8,234 multi-rank taxon features |
| KO features | 13,945 KOs in the common KO space |
| Samples | 23,951 paired samples after removing degenerate rows |
| Studies | 619 |
| Validation studies | 99 |
| Validation samples | 3,098 (subset of the 17,959 L2 training rows) |
| Scoring mask | 5,610 KOs with training prevalence ≥ 10% |

The taxon panel is **prokaryotic-only**. Eukaryotic, viral, and unclassified
features are excluded. The KO space is the intersection of the truth matrix and
the PICRUSt2 reference (13,945 KOs). The 5,610-KO scoring mask applies an
additional training-prevalence filter (≥ 10%); it is not part of the 13,945-KO
definition.

## Splits

The main split is **study-grouped (L2)**. The flow/prior training set (14,861
rows) and the 99 validation studies (3,098 rows) are study-disjoint, which
prevents study-level leakage from shared protocols, sites, depths, or laboratory
effects. The AE pretraining set (17,959 rows) intentionally includes the
validation studies; that is why absolute validation scores are optimistic.

```text
L2 training rows            17,959   (includes the 99 validation studies)
Validation studies              99
Validation rows              3,098   (subset of the L2 training rows)
Scoreable validation studies    64   (at least two samples)
Flow/prior training rows    14,861   (training rows minus the 99 validation studies)
```

The validation studies are held out from flow-matching and prior-encoder
training. The `final_s2` autoencoder was pretrained on the 17,959 L2 training
rows, which include the validation studies; absolute validation scores are
therefore optimistic. This is stated in the model card and evaluation notes.

The 3,098 validation rows are a subset of the 17,959 L2 training rows; the
flow/prior models use the remaining 14,861 rows. The total of 23,951 also
includes a further partition that is not described in this release.

Only the training and validation partitions are described in this release. No
additional evaluation partitions or their metrics are included.

## Preprocessing

- Taxon counts and KO counts are stored as raw counts.
- Degenerate samples with zero total counts in either modality are removed.
- Missing biome labels are mapped to an `unknown` category.
- KO scoring uses the 5,610-KO mask defined by training prevalence ≥ 10%.
- The P2 PCA basis is fitted on the training split only.

## What is not redistributed

This repository does **not** contain:

- MGnify raw downloads or processed abundance matrices.
- The paired h5mu files used for training.
- Split manifests, sample tables, or study-level metadata.
- The PICRUSt2 reference table (`P_ext`) or its source profiles.
- Model weights, latent caches, or generated samples.

Users who need to reproduce training must obtain the source data from MGnify and
construct the paired matrices following the description in this document and the
released code. The repository does not provide a data downloader or a
reproduction script for the full data assembly, and it does not claim that the
reported numbers can be reproduced from the repository alone.

## Data-use considerations

MGnify data are public, but individual studies may carry their own consent,
ethics, and data-use terms. Users are responsible for checking the terms of the
studies they use. No human-subject data are redistributed here.
