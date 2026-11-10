#!/bin/bash
# AE v2 candidate training launcher (fully parameterized, outputs isolated under runs/mgm/ae_v2/).
#
# Usage:
#   RUN_NAME=a1_logcpm AE_INPUT=logcpm \
#   RNA_DIMS='[1024,512,150]' ATAC_DIMS='[2048,1024,200]' EPOCHS=150 \
#   bash configs/mgm_train_ae_v2.sh
#
# Everything is passed as a hydra override, so no upstream file is modified.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "$HERE/mgm_env.sh"

RUN_NAME=${RUN_NAME:?set RUN_NAME}
AE_INPUT=${AE_INPUT:-raw}
RNA_DIMS=${RNA_DIMS:-'[1024,512,150]'}
ATAC_DIMS=${ATAC_DIMS:-'[2048,1024,200]'}
NORM_TYPE=${NORM_TYPE:-layernorm}
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-256}
LR=${LR:-0.001}
WD=${WD:-0.0001}
DATA=${DATA:-$MGM_TRAIN}
CKPT_ROOT=${CKPT_ROOT:-$MGM_RUNS/ae_v2}
EXTRA=${EXTRA:-}                 # extra hydra overrides, e.g. 'encoder.covariate_specific_theta=True'

export AE_INPUT
# AE_EXT_RENORM=1 normalises each modality's loss by its own carrier count, which is what makes
# "paired + unpaired rows in one loop" numerically equivalent to training the two AEs separately
# (verified to float32 precision in eval/ae_zero_fill_equivalence.json).
export AE_EXT_RENORM=${AE_EXT_RENORM:-0}
# AE_LATENT_NOISE=sigma perturbs the KO latent (relative to its per-dim batch std) before decoding,
# training only. This is the lever on the reconstruction <-> generation trade-off: the generation
# pipeline always decodes a PREDICTED latent, never a true one.
export AE_LATENT_NOISE=${AE_LATENT_NOISE:-0}
# AE_DROPOUT_SIDE=both|decoder|encoder -- the config's dropout flag hits BOTH MLPs, so this switch
# isolates which side the regularisation acts on (implemented as identity-swaps post-construction,
# no parameter change).
export AE_DROPOUT_SIDE=${AE_DROPOUT_SIDE:-both}
# AE_SEED=N makes the run a deterministic function of (config, seed); distinct N = genuine replicate.
export AE_SEED=${AE_SEED:-}
# AE_DEC_DIMS='{"rna":[...],"atac":[...]}' decouples the DECODER width/depth from the
# encoder (upstream forces dims[::-1]). Same depth => same state_dict layout.
export AE_DEC_DIMS=${AE_DEC_DIMS:-}

cd "$SCX_REPO/script/training_autoencoder"
echo "[mgm_train_ae_v2] run=$RUN_NAME input=$AE_INPUT epochs=$EPOCHS"
echo "[mgm_train_ae_v2] rna $RNA_DIMS / atac $ATAC_DIMS / norm=$NORM_TYPE / lr=$LR wd=$WD"

python "$SCX_ROOT/py/train_encoder_v2.py" \
  dataset=openproblem \
  dataset.dataset_path="$DATA" \
  dataset.split_rates=[0.90,0.10] \
  $MGM_DATASET_OVERRIDES \
  +dataset.encoder_type=learnt_autoencoder \
  encoder=encoder_multimodal \
  encoder.is_binarized=False \
  encoder.encoder_kwargs.rna.dims="$RNA_DIMS" \
  encoder.encoder_kwargs.atac.dims="$ATAC_DIMS" \
  encoder.encoder_kwargs.rna.norm_type="$NORM_TYPE" \
  encoder.encoder_kwargs.atac.norm_type="$NORM_TYPE" \
  encoder.learning_rate="$LR" \
  encoder.weight_decay="$WD" \
  $EXTRA \
  training_config=default \
  training_config.chekpoint_path="$CKPT_ROOT" \
  training_config.batch_size="$BATCH" \
  logger.project="$RUN_NAME" \
  trainer.max_epochs="$EPOCHS"
