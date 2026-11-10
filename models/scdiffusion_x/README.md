# scDiffusion-X fork — curated code

This directory contains the project-specific code built around the
scDiffusion-X network backbone, plus a vendored copy of the upstream source.

## Layout

```text
models/scdiffusion_x/
├── py/          # project patches and training/inference entries
├── configs/     # AE and flow-matching shell/YAML configurations
└── upstream/    # vendored scDiffusion-X source (MIT + BSD-3-Clause)
```

## Project-specific code

- `py/mgm_patch.py` — numerical rate-floor guard for Poisson / negative-binomial
  likelihoods.
- `py/ae_v2_patch.py` — encoder input parameterization and decoupled decoder.
- `py/flow_objective_patch.py` — linear flow-matching objective with a clean
  taxonomic condition.
- `py/flow_sampler_patch.py` — 1,000-step Euler sampler that keeps the condition
  clean.
- `py/train_encoder_v2.py`, `py/train_df_flow.py` — training entry points.
- `py/translate_mgm.py`, `py/translate_v4_flow.py` — inference entry points.

The project-specific modules are monkey-patches or thin entry points; the
upstream network code itself is not modified.

## Upstream code

`upstream/` contains vendored scDiffusion-X source with one relative-path change.
It retains two upstream license texts: **MIT** (top-level repository,
Copyright (c) 2026 Erpai Luo) and **BSD 3-Clause** (`scdiffusionX` package,
Copyright (c) 2024, Erpai Luo). The exact coverage boundary is pending
confirmation. See `upstream/README.md` and `THIRD_PARTY_NOTICES.md`.

## Not included

Model weights, checkpoints, generated samples, training logs, experiment logs,
and data caches are not included. See `../../docs/REPRODUCIBILITY.md`.
