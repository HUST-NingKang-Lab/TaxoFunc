# Reproducibility notes

## Important scope statement

This repository is a **code and validation-results release**, not a turnkey
end-to-end reproduction package. The MGnify-derived data and the model weights
are not redistributed. The commands below are the released training and
inference recipes; they require the external assets listed in this document.

**The repository alone does not reproduce the reported numbers.**

## Environment

The original research environment used:

- Python 3.8.20
- PyTorch 2.4.1 (CUDA 12.1 build)
- PyTorch Lightning 1.8.6
- NumPy 1.24.4, SciPy 1.10.1, pandas 2.0.3
- scvi-tools 0.12.0, muon 0.1.6, scanpy 1.9.8
- mpi4py 3.1.x and Pillow 9.5–10.x (runtime requirements of the vendored code)

The dependency constraints are in `requirements.txt`. `mpi4py` requires a working MPI
implementation (OpenMPI or MPICH). Install the CUDA 12.1 PyTorch wheel for GPU
training/inference. Activate the environment before sourcing the shell configs.

## Required external assets (not included)

The code expects the original project layout under the repository root. The
following paths are required for the corresponding stages:

### Data and caches

```text
models/scdiffusion_x/data/mgm/cache/
├── A_tax.npz                 # taxon count cache (8,234 × 23,972)
├── X_ko.npz                  # KO count cache (13,945 × 23,972)
├── taxids.npy
├── ko_names.npy
├── design.tsv                # sample / study / biome table
├── tr_idx.npy                # training row indices
├── p2_pca128.npz             # TRAIN-only P2 PCA basis
└── ...

models/scdiffusion_x/data/mgm/df_pilot/
├── mgm_train_trn.h5mu        # 14,861 flow/prior training rows
├── mgm_val99.h5mu            # 3,098 validation rows
├── biome_map_trn.json
└── split_manifest.json
```

The cache matrices contain 23,972 columns. The h5mu training/validation files
contain 23,951 samples after removing 21 degenerate training rows; see `DATA.md`
for the used-sample counts.

### Model assets

```text
models/scdiffusion_x/runs/mgm/df_full/ae/
├── checkpoints/last.ckpt     # final_s2 AE
└── norm_factor.npz           # rna_std / atac_std

models/scdiffusion_x/runs/mgm/df_full_flow/df/
└── ema_0.9999_180000.pt      # flow-matching UNet, 180k steps

models/scdiffusion_x/samples/df_full_flow/
├── flow_180000_seed0/RNA_0.npz
├── flow_180000_seed1/RNA_0.npz
├── flow_180000_seed2/RNA_0.npz
└── flow_180000_seed3/RNA_0.npz
```

### P2 assets

```text
taxofunc/cache/side_p2/
├── side_blocks.npz           # p2 block (N, 128)
├── P_ext.npz                 # row-normalised PICRUSt2 taxon→KO table
└── sample_accession.npy
```

The P2 block and `P_ext` can be rebuilt from the PICRUSt2 reference model at:

```text
baselines/picrust2_full/model/picrust2_full_KO_prokaryotic/
├── feature_index.txt
├── function_index.txt
└── feature_profiles.tsv.gz
```

## Commands

### 1. Train the autoencoder

```bash
bash models/scdiffusion_x/configs/ae_final_train.sh 0 2
```

This writes under `models/scdiffusion_x/runs/mgm/ae_v2/final_s2/` and expects
`mgm_train.h5mu` at `models/scdiffusion_x/data/mgm/`.

### 2. Train the flow-matching model

```bash
bash models/scdiffusion_x/configs/df_flow_full.sh 0
```

This copies the frozen `final_s2` AE into the flow run directory and trains for
180,000 steps. It expects `mgm_train_trn.h5mu`.

### 3. Build the P2 block

```bash
python taxofunc/build_p2_block.py --out-dir taxofunc/cache/side_p2
```

This rebuilds `P_ext.npz` from the PICRUSt2 reference model and writes the P2
block. It reads taxonomy counts only; no KO truth is used.

### 4. Run flow-matching inference

```bash
STEP=180000 SAMPLE_SEED=0 bash models/scdiffusion_x/configs/translate_flow.sh
```

Repeat with `SAMPLE_SEED=1,2,3` to reproduce the four-seed average. The outputs
are written under `models/scdiffusion_x/samples/df_full_flow/`.

## What can and cannot be checked from this repository

| Check | Possible with this repository? |
|---|---|
| Inspect the model and training objectives | Yes |
| Inspect the P2 block construction | Yes |
| Inspect the released validation metrics | Yes |
| Rebuild the P2 block | Only with the PICRUSt2 reference assets |
| Train the AE / flow model | Only with the MGnify-derived data |
| Run inference | Only with the model weights and data |
| Reproduce the reported validation numbers end to end | No |

## Code-documentation consistency

The commands above use the released configs and entry points:

- `models/scdiffusion_x/configs/ae_final_train.sh`
- `models/scdiffusion_x/configs/mgm_train_ae_v2.sh`
- `models/scdiffusion_x/configs/df_flow_full.sh`
- `models/scdiffusion_x/configs/translate_flow.sh`
- `models/scdiffusion_x/py/train_encoder_v2.py`
- `models/scdiffusion_x/py/train_df_flow.py`
- `models/scdiffusion_x/py/translate_v4_flow.py`
- `taxofunc/build_p2_block.py`

The shell configs resolve `SCX_ROOT` relative to the repository root and do not
contain cluster-specific absolute paths.
