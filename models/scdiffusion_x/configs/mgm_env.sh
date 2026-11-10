#!/bin/bash
# Paths/config for the MGnifyMultiOmics scDiffusion-X run.
#   modality "rna"  <- KO profile        (13,945 common KOs, counts, NegativeBinomial)
#   modality "atac" <- prokaryotic taxa  (8,234 taxids, counts, Poisson)
#   condition       <- biome             (84 classes in the L2 training split)
#   split           <- L2 (group by study): 17,959 training samples
# BASH_SOURCE (not $0) so this works both when executed and when sourced from a shell
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"

export MGM_ROOT=$SCX_ROOT
export MGM_DATA=$SCX_ROOT/data/mgm
export MGM_TRAIN=$MGM_DATA/mgm_train.h5mu
export MGM_BIOME_MAP=$MGM_DATA/biome_map.json
export MGM_RUNS=$SCX_ROOT/runs/mgm
export MGM_CONDITION=biome
export MGM_NUM_CLASS=84
export MGM_LATENT=100

# dataset config overrides shared by every stage.
# NOTE: dataset.is_binarized=True is intentional even though the taxonomy data is NOT
# binary -- that flag is what makes RNAseqLoader skip its log1p branch, so BOTH modalities
# are fed raw counts (matching encode_raw_data / translation / evaluation). The model's
# own count likelihood is selected separately by encoder.is_binarized=False.
export MGM_DATASET_OVERRIDES="dataset.layer_key=X_counts dataset.covariate_keys=[$MGM_CONDITION] dataset.conditioning_covariate=$MGM_CONDITION dataset.theta_covariate=$MGM_CONDITION dataset.multimodal=True dataset.is_binarized=True"
