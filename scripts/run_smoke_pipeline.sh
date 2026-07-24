#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/train_bpe.py \
  --input TinyStoriesV2-GPT4-train.txt \
  --output ts_tokenizer.pt \
  --vocab-size 10000

uv run python scripts/encode.py \
  --file-input TinyStoriesV2-GPT4-valid.txt \
  --tokenizer-input ts_tokenizer.pt \
  --output ts_smoke.bin

uv run python scripts/training_loop.py \
  --input ts_smoke.bin \
  --output ts_smoke_ckpt \
  --log-dir ts_smoke \
  --batch-size 16 \
  --vocab-size 10000 \
  --iter-num 500 \
  --context-len 128 \
  --d-model 128 \
  --num-heads 4 \
  --theta 10000 \
  --num-layer 2 \
  --weight-decay 0.1 \
  --b1 0.9 \
  --b2 0.999 \
  --max-grad 1.0 \
  --a-max 1e-3 \
  --a-min 1e-4 \
  --T-w 50 \
  --T-c 500 \
  --device cpu

uv run python scripts/training_loop.py \
  --input ts_smoke.bin \
  --output ts_smoke_ckpt \
  --resume-from ts_smoke_ckpt_251 \
  --log-dir ts_smoke \
  --batch-size 16 \
  --vocab-size 10000 \
  --iter-num 500 \
  --context-len 128 \
  --d-model 128 \
  --num-heads 4 \
  --theta 10000 \
  --num-layer 2 \
  --weight-decay 0.1 \
  --b1 0.9 \
  --b2 0.999 \
  --max-grad 1.0 \
  --a-max 1e-3 \
  --a-min 1e-4 \
  --T-w 50 \
  --T-c 500 \
  --device cpu

uv run python scripts/inference.py \
  --model ts_smoke_ckpt_451 \
  --output generated.txt \
  --prefix baseline.txt \
  --tokenizer-input ts_tokenizer.pt \
  --max-length 100 \
  --vocab-size 10000 \
  --context-len 128 \
  --d-model 128 \
  --num-heads 4 \
  --theta 10000 \
  --num-layer 2 \
  --temperature 1.0 \
  --top-num 50 \
  --device cpu