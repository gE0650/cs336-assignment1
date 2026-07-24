#!/usr/bin/env bash
set -euo pipefail

TOKENIZED_INPUT="ts_smoke.bin"
VOCAB_SIZE=10000
BATCH_SIZE=16
ITER_NUM=500
CONTEXT_LEN=128
D_MODEL=128
NUM_HEADS=4
NUM_LAYER=2
THETA=10000
DEVICE="cpu"

WEIGHT_DECAY=0.1
B1=0.9
B2=0.999
MAX_GRAD=1.0
A_MIN=1e-4
T_W=50
T_C=500

for LR in 3e-4 1e-3 3e-3 1e-2 3e-2; do
  RUN_NAME="lr_${LR}"
  CKPT_NAME="ts_lr_${LR}_ckpt"

  uv run python scripts/training_loop.py \
    --input "${TOKENIZED_INPUT}" \
    --output "${CKPT_NAME}" \
    --log-dir "${RUN_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --vocab-size "${VOCAB_SIZE}" \
    --iter-num "${ITER_NUM}" \
    --context-len "${CONTEXT_LEN}" \
    --d-model "${D_MODEL}" \
    --num-heads "${NUM_HEADS}" \
    --theta "${THETA}" \
    --num-layer "${NUM_LAYER}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --b1 "${B1}" \
    --b2 "${B2}" \
    --max-grad "${MAX_GRAD}" \
    --a-max "${LR}" \
    --a-min "${A_MIN}" \
    --T-w "${T_W}" \
    --T-c "${T_C}" \
    --device "${DEVICE}"
done

# result: best choice: 1e-2 ~ 1e-3