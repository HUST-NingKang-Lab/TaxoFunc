#!/bin/bash
# Flow-matching training (same UNet/AE/data/budget as the reference AE run).
#
# RECIPE (released):
#   space     : final_s2 AE (rna latent 100 / atac latent 200). The AE is copied into the
#               NEW run directory so the datamodule never writes into the reference AE directory.
#   data      : data/mgm/df_pilot/mgm_train_trn.h5mu (14,861; L2 train minus 99 val studies)
#   objective : linear flow matching (`py/train_df_flow.py`:
#               u = 10 * (z_ko/rna_std); u_t=(1-t)eps+t*u; velocity-MSE;
#               condition = CLEAN z_tax/atac_std at every t).
#   sampler   : Euler ODE, 1000 steps (FM_STEPS=1000), at translate time.
#   budget    : 180,000 steps, batch 128, lr 1e-4 linear anneal, wd 1e-4, EMA 0.9999.
#   seed      : FM_SEED=42 (explicit; upstream seeding is disabled).
#
# Usage:
#   bash configs/df_flow_full.sh 0                 # 180k full
#   STEPS=20000 SAVE_INTERVAL=10000 bash configs/df_flow_full.sh 0   # pilot
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "$HERE/mgm_env.sh"

STEPS=${STEPS:-180000}
SAVE_INTERVAL=${SAVE_INTERVAL:-30000}
LOG_INTERVAL=${LOG_INTERVAL:-500}
DEVICES=${DEVICES:-0}
FM_TARGET_SCALE=${FM_TARGET_SCALE:-10.0}
FM_SEED=${FM_SEED:-42}
RUN_TAG=${RUN_TAG:-}          # e.g. _pilot; empty for the 180k full run

ROOT="$SCX_ROOT/runs/mgm/df_full_flow${RUN_TAG}"
AE_DIR="$ROOT/ae"; OUT="$ROOT/df"
REF_AE_DIR="$SCX_ROOT/runs/mgm/df_full/ae"
mkdir -p "$AE_DIR/checkpoints" "$OUT"
if compgen -G "$OUT/model*.pt" >/dev/null; then
  echo "[df_flow_full] refusing to overwrite existing checkpoints in $OUT"; exit 1
fi

# Copy (do not move/overwrite) the reference AE and its norm_factor into the new run dir.
if [ ! -f "$AE_DIR/checkpoints/last.ckpt" ]; then
  cp "$REF_AE_DIR/checkpoints/last.ckpt" "$AE_DIR/checkpoints/last.ckpt"
  echo "[df_flow_full] AE copy -> $AE_DIR/checkpoints/last.ckpt (md5 $(md5sum "$REF_AE_DIR/checkpoints/last.ckpt" | cut -c1-16))"
fi
if [ ! -f "$AE_DIR/norm_factor.npz" ]; then
  cp "$REF_AE_DIR/norm_factor.npz" "$AE_DIR/norm_factor.npz"
  echo "[df_flow_full] norm_factor copy -> $AE_DIR/norm_factor.npz"
fi
AE_PATH="$AE_DIR/checkpoints/last.ckpt"

export AE_DEC_DIMS='{"rna":[100,256,512,13945],"atac":[200,512,1024,8234]}'
export AE_INPUT=raw PYTHONDONTWRITEBYTECODE=1
export FM_TARGET_SCALE FM_SEED
ENC_CFG="$SCX_ROOT/configs/enc_arch_lrna100.yaml"
DATA="$MGM_DATA/df_pilot/mgm_train_trn.h5mu"

cd "$SCX_REPO/script/training_diffusion"
echo "[df_flow_full] steps=$STEPS save=$SAVE_INTERVAL anneal=$STEPS dev=$DEVICES seed=$FM_SEED scale=$FM_TARGET_SCALE tag=${RUN_TAG:-<none>}"
echo "[df_flow_full] data=$DATA"; echo "[df_flow_full] ae=$AE_PATH out=$OUT"
echo "[df_flow_full] start $(date -Is)"

python3 "$SCX_ROOT/py/train_df_flow.py" \
  --data_dir "$DATA" --output_dir "$OUT" \
  --cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8 \
  --cross_attention_shift True --dropout 0.0 \
  --rna_dim 1,100 --atac_dim 1,200 --learn_sigma False --num_channels 128 \
  --num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False \
  --use_scale_shift_norm True --num_workers 2 --condition "$MGM_CONDITION" \
  --encoder_config "$ENC_CFG" \
  --num_class "$MGM_NUM_CLASS" --weight_decay 0.0001 \
  --ae_path "$AE_PATH" \
  --lr 0.0001 --batch_size 128 --devices "$DEVICES" \
  --log_interval "$LOG_INTERVAL" --save_interval "$SAVE_INTERVAL" --use_db False \
  --lr_anneal_steps "$STEPS" \
  --noise_schedule linear --diffusion_steps 1000 --sample_fn ddpm

echo "[df_flow_full] done $(date -Is)"
