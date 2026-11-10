#!/bin/bash
# AE TRAINING -- released final configuration.
#
#   architecture : enc rna [1024,512,100] / atac [2048,1024,200]   (3 layers, LayerNorm, raw counts)
#                  dec rna [100,256,512,13945] / atac [200,512,1024,8234]   (ASYMMETRIC)
#   latent       : KO 100 / taxonomy 200            params 50.4 M
#   optimisation : lr 3e-4 / wd 1e-4 / batch 256 / AdamW / constant LR / **300 epochs**
#   data         : L2 train split only (17,959 rows; study-grouped split)
#
# Seeds are explicit (AE_SEED) so the three runs are genuine independent replicates.
# Selection uses the TRAIN-internal validation split only.
#
# Usage: bash configs/ae_final_train.sh <gpu> <seed>
set -eu
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "$HERE/env.sh"
GPU=$1; SEED=$2
SCX="$SCX_ROOT"
cd "$SCX"
LOGD="$SCX/logs"
mkdir -p "$LOGD"
export CUDA_VISIBLE_DEVICES=$GPU
export AE_INPUT=raw EPOCHS=300 AE_SEED=$SEED
export AE_DEC_DIMS='{"rna":[100,256,512,13945],"atac":[200,512,1024,8234]}'
export RNA_DIMS='[1024,512,100]' ATAC_DIMS='[2048,1024,200]'
export LR=0.0003 WD=0.0001 AE_DROPOUT_SIDE=both
unset EXTRA
export RUN_NAME="final_s$SEED"
echo "[final] $RUN_NAME on GPU $GPU (seed $SEED) start $(date -Is)"
bash configs/mgm_train_ae_v2.sh > "$LOGD/ae_v2_$RUN_NAME.log" 2>&1
echo "$RUN_NAME done $(date -Is)" >> "$LOGD/ae_final_train.progress"
echo "[final] $RUN_NAME done $(date -Is)"
