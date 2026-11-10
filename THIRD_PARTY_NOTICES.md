# Third-party notices

## scDiffusion-X

This repository vendors source code from **scDiffusion-X** as the network
backbone used by TaxoFunc:

- Upstream repository: <https://github.com/EperLuo/scDiffusion-X>
- Upstream commit: `d60d928b635ab7a52ace030a641efd02137c193f`
- Upstream paper: Luo et al., *Nature Communications* 17:5196 (2026),
  DOI [10.1038/s41467-026-71744-x](https://doi.org/10.1038/s41467-026-71744-x)
- Top-level license: **MIT License**, Copyright (c) 2026 Erpai Luo
  (`models/scdiffusion_x/upstream/LICENSE`).
- Package license: **BSD 3-Clause License**, Copyright (c) 2024, Erpai Luo
  (`models/scdiffusion_x/upstream/scdiffusionX/LICENSE`).

Both license texts are retained unchanged. The exact file-by-file coverage
boundary is **pending confirmation** with the upstream project. Based on the
directory layout, the top-level `LICENSE` appears to cover repository-level files
(for example `script/`), and the package `LICENSE` appears to cover
`scdiffusionX/src`. This is not a legal conclusion.

Vendored paths:

- `models/scdiffusion_x/upstream/scdiffusionX/src/scdiffusionX/`
- `models/scdiffusion_x/upstream/scdiffusionX/pyproject.toml`
- `models/scdiffusion_x/upstream/script/training_autoencoder/`
- `models/scdiffusion_x/upstream/script/training_diffusion/py_scripts/`

The vendored files are upstream source with one non-functional change: a
cluster-specific default path in
`DiffusionBackbone/multimodal_datasets.py` was made relative. Generated outputs,
caches, logs, notebooks, and upstream figures were not copied.

## Project-specific modifications

The TaxoFunc-specific modifications are separate monkey-patch modules and entry
points under `models/scdiffusion_x/py/` (for example `ae_v2_patch.py`,
`flow_objective_patch.py`, `flow_sampler_patch.py`, and the training/inference
entry scripts). They are not part of the upstream MIT- or BSD-3-Clause-licensed
work and are covered by the project's pending license status; see
`LICENSE_STATUS.md`.

## Other third-party material

No other third-party source code, images, notebooks, PDFs, or database dumps are
vendored in this repository. The Python dependencies listed in
`requirements.txt` are installed separately and are governed by their own
licenses.
