#!/bin/bash
# Flow-matching VAL translation (Euler ODE, 1000 steps, clean condition).
#
#   STEP=<int> SAMPLE_SEED=<int> [OUT=...] [DEVICES=0] [SHUF=1] bash configs/translate_flow.sh
#
# SHUF=1 -> data_dir = mgm_val99_shufcond.h5mu (wrong-condition control).
# The model is the same MultimodalUNet; only the sampler is replaced by
# py/flow_sampler_patch.py. The output RNA_0.npz is in the same latent space as the AE
# (z_ko / rna_std), so downstream scoring uses the same latent convention.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "$HERE/mgm_env.sh"

STEP=${STEP:?must set STEP (e.g. 180000)}
SAMPLE_SEED=${SAMPLE_SEED:-0}
SHUF=${SHUF:-0}
DEVICES=${DEVICES:-0}
FM_STEPS=${FM_STEPS:-1000}
FM_TARGET_SCALE=${FM_TARGET_SCALE:-10.0}
SUF=$([ "$SHUF" = "1" ] && echo "_shuf" || echo "")
RUN_TAG=${RUN_TAG:-}          # e.g. _pilot; empty for the 180k full run
TAG="${TAG:-flow_$(printf %06d "$STEP")_seed${SAMPLE_SEED}${SUF}}"

MODEL="${MODEL:-$SCX_ROOT/runs/mgm/df_full_flow${RUN_TAG}/df/ema_0.9999_$(printf %06d "$STEP").pt}"
AE_PATH="${AE_PATH:-$SCX_ROOT/runs/mgm/df_full_flow${RUN_TAG}/ae/checkpoints/last.ckpt}"
DATA="${DATA_OVERRIDE:-$MGM_DATA/df_pilot/mgm_val99$([ "$SHUF" = "1" ] && echo "_shufcond" || echo "").h5mu}"
OUT="${OUT:-$SCX_ROOT/samples/df_full_flow${RUN_TAG}/$TAG}"
ENC_CFG="$SCX_ROOT/configs/enc_arch_lrna100.yaml"

[ -f "$MODEL" ] || { echo "checkpoint not found: $MODEL"; exit 1; }
[ -f "$AE_PATH" ] || { echo "AE not found: $AE_PATH"; exit 1; }
if [ -s "$OUT/RNA_0.npz" ]; then
  echo "[translate_flow] refusing to overwrite existing $OUT/RNA_0.npz"; exit 1
fi
mkdir -p "$OUT"
export AE_DEC_DIMS='{"rna":[100,256,512,13945],"atac":[200,512,1024,8234]}'
export AE_INPUT=raw PYTHONDONTWRITEBYTECODE=1
export FM_STEPS FM_TARGET_SCALE SAMPLE_SEED

cd "$SCX_REPO/script/training_diffusion"
echo "[translate_flow] tag=$TAG run_tag=${RUN_TAG:-<none>} model=$MODEL data=$DATA out=$OUT"
echo "[translate_flow] seed=$SAMPLE_SEED steps=$FM_STEPS scale=$FM_TARGET_SCALE"
python3 "${TRANSLATE_ENTRY:-$SCX_ROOT/py/translate_v4_flow.py}" \
  --cross_attention_resolutions 2,4,8 --cross_attention_windows 1,4,8 --cross_attention_shift True \
  --rna_dim 1,100 --atac_dim 1,200 --learn_sigma False --num_channels 128 \
  --num_head_channels -1 --num_res_blocks 1 --resblock_updown True --use_fp16 False \
  --use_scale_shift_norm True --class_cond True --condition "$MGM_CONDITION" \
  --num_class "$MGM_NUM_CLASS" \
  --diffusion_steps 1000 --noise_schedule linear \
  --devices "$DEVICES" --batch_size 2000 --is_strict True \
  --sample_fn ddpm --classifier_scale 0 --gen_times 1 \
  --output_dir "$OUT" \
  --multimodal_model_path "$MODEL" \
  --data_dir "$DATA" \
  --gen_mode rna \
  --condition "$MGM_CONDITION" \
  --encoder_config "$ENC_CFG" \
  --ae_path "$AE_PATH" \
  --biome_map "${BIOME_MAP:-$MGM_DATA/df_pilot/biome_map_trn.json}"
echo "[translate_flow] done $(date -Is) -> $OUT"
