# Vendored scDiffusion-X

This directory contains vendored source code from **scDiffusion-X**, used as
the network backbone for TaxoFunc. It includes one non-functional relative-path
change (see below).

- Repository: <https://github.com/EperLuo/scDiffusion-X>
- Commit: `d60d928b635ab7a52ace030a641efd02137c193f`
- Paper: Luo et al., *Nature Communications* 17:5196 (2026),
  DOI [10.1038/s41467-026-71744-x](https://doi.org/10.1038/s41467-026-71744-x)
- Top-level license: **MIT License**, Copyright (c) 2026 Erpai Luo (`LICENSE`).
- Package license: **BSD 3-Clause License**, Copyright (c) 2024, Erpai Luo
  (`scdiffusionX/LICENSE`).

Both license texts are retained unchanged. The exact file-by-file coverage
boundary is **pending confirmation** with the upstream project; the
directory-based mapping is not a legal conclusion.

## What is vendored

- `scdiffusionX/src/scdiffusionX/` — package source.
- `scdiffusionX/pyproject.toml`.
- `script/training_autoencoder/` — training entry and Hydra configs.
- `script/training_diffusion/py_scripts/` — upstream training and translation
  entry points.

## What is not vendored

Upstream notebooks, generated outputs, logs, caches, documentation builds, and
figures are not included. One cluster-specific default path in
`DiffusionBackbone/multimodal_datasets.py` was made relative; no other source
change was made.

## Boundary

The upstream license texts are retained for the vendored files in this
directory. The exact coverage boundary is pending confirmation. The
TaxoFunc-specific code elsewhere in the repository is covered by the project's
pending license status; see the repository-level `LICENSE_STATUS.md` and
`THIRD_PARTY_NOTICES.md`.
