#!/bin/bash

# Train to reproduce
python puzzle_diff/train_script.py \
  -dataset celeba \
  -puzzle_sizes 6 8 10 12 \
  -batch_size 8 \
  -gpus 1 \
  -steps 300 \
  -sampling DDIM \
  -inference_ratio 10 \
  --rotation True \
  --degree 100% \
  --backbone resnet18equiv \
  --architecture transformer

# CPU
CUDA_VISIBLE_DEVICES="" python puzzle_diff/train_script_cpu.py \
    -dataset celeba \
    -puzzle_sizes 6 8 10 12 \
    -batch_size 2 \
    -num_workers 2 \
    -gpus 0 \
    -steps 300 \
    -inference_ratio 5 \
    --rotation True \
    --degree 100% \
    --backbone resnet18equiv \
    --architecture transformer \
    --evaluate True \
    --checkpoint_path "Puzzle-Diff/99qcofwy/checkpoints/last.ckpt"
