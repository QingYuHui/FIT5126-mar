#!/bin/bash

# MAR Base 64x64 Training Script
# Date: $(date)

torchrun --nproc_per_node=1 main_mar.py \
  --use_cached \
  --cached_path "output_cache/class0" \
  --vae_path "vqgan/stage1.ckpt" \
  --vae_embed_dim 4 \
  --vae_stride 4 \
  --output_dir "output_run_64_patch1" \
  --model mar_base \
  --batch_size 8 \
  --img_size 64 \
  --patch_size 1 \
  --save_last_freq 50 \
  --num_workers 8 \
  --resume "output_run_64_patch1" \
  --epochs 2000