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

# ("-num_workers", type=int, default=8)
# Train to reproduce - ak
python puzzle_diff/train_script.py \
  -dataset jpwleg3 \
  -puzzle_sizes 6 \
  -batch_size 16 \
  -gpus 2 \
  -steps 300 \
  -sampling DDIM \
  -inference_ratio 10 \
  --rotation True \
  --degree 100% \
  --backbone resnet18equiv \
  --architecture transformer

# Train to reproduce imagenet
python puzzle_diff/train_script.py \
  -dataset imagenet \
  -puzzle_sizes 6 \
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

python puzzle_diff/train_script.py \
  -dataset jpwleg3 \
  -puzzle_sizes 6 \
  -batch_size 16 \
  -gpus 2 \
  -steps 300 \
  -sampling DDIM \
  -inference_ratio 10 \
  --rotation True \
  --degree 100% \
  --backbone resnet18equiv \
  --architecture transformer \
  --evaluate True \
  --offline \
  --checkpoint_path /cluster/home/akmarala/DiffAssemble/Puzzle-Diff/jk8jhzia/checkpoints/last.ckpt

#SETUP (if you want to start from scratch)
conda env remove --prefix /cluster/home/akmarala/envs/diffassemble
rm -rf /cluster/home/akmarala/envs/diffassemble
module load Anaconda3/2024.02-1
conda create --prefix /cluster/home/akmarala/envs/diffassemble python=3.9 -y
conda activate /cluster/home/akmarala/envs/diffassemble
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-sparse==0.6.15 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric==2.1.0
conda install -c conda-forge cudatoolkit=11.3
find $CONDA_PREFIX -name "libcusparse.so*" 2>/dev/null
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -m pip install "pip<24.1"
pip install pytorch-lightning==1.7.7 torchmetrics==0.11.4 lightning-utilities==0.9.0
pip install einops black pre-commit matplotlib wandb transformers timm kornia trimesh
pip install networkx pandas scikit-learn opencv-python imageio seaborn plotly
pip uninstall -y numpy
pip install "numpy<2"
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
pip install transformers==4.25.1

#IDUN:
python puzzle_diff/train_script.py \
  -dataset imagenet \
  -puzzle_sizes 6 \
  -batch_size 64 \
  -gpus 4 \
  -steps 300 \
  -sampling DDIM \
  -inference_ratio 10 \
  -num_workers 16 \
  --rotation True \
  --degree 100% \
  --backbone resnet18equiv \
  --architecture transformer \
  --acc_grad 2

# 4090 local machine missing inference
python puzzle_diff/train_script.py \
    -dataset celeba \
    -puzzle_sizes 6 8 10 12 \
    -batch_size 8 \
    -gpus 1 \
    -steps 300 \
    --rotation True \
    --degree 100% \
    --backbone resnet18equiv \
    --architecture transformer \
    --evaluate True \
    --checkpoint_path "Puzzle-Diff/99qcofwy/checkpoints/last.ckpt" \
    --missing 30 \
    --save_eval_images True

# 4090 local machine imagenet 4x4
python puzzle_diff/train_script.py \
    -dataset imagenet \
    -puzzle_sizes 4 \
    -batch_size 8 \
    -gpus 1 \
    -steps 300 \
    --degree 100% \
    --architecture transformer \
    --evaluate True \
    --checkpoint_path "Puzzle-Diff/8fuyrpvq/checkpoints/last.ckpt" \
    --save_eval_images True \
    --offline

# 4090 local machine texmet
python puzzle_diff/train_script.py \
    -dataset texmet \
    -puzzle_sizes 6 \
    -batch_size 8 \
    -gpus 1 \
    -steps 300 \
    --degree 100% \
    --architecture transformer \

python puzzle_diff/train_script.py   -dataset texmet   -puzzle_sizes 3  
 -batch_size 128   -gpus 4

# Test all irregular fragment types
python visuals4x4imagenetirregular.py --fragment_type all --save_images --dataset imagenet

# Test specific fragment types
python visuals4x4imagenetirregular.py --fragment_type jigsaw --save_images --dataset texmet
python visuals4x4imagenetirregular.py --fragment_type torn --save_images --dataset texmet
python visuals4x4imagenetirregular.py --fragment_type geometric --save_images --dataset texmet

# Test only irregular fragments (skip regular)
python visuals4x4imagenetirregular.py --irregular_only --save_images --dataset texmet

# Test only first 50 images with graph nodes
python visuals4x4imagenetirregular.py --irregular_only --save_images --test_limit 50 --show_graph_nodes

# Test first 20 images, jigsaw fragments only, with graph visualization
python visuals4x4imagenetirregular.py --fragment_type jigsaw --save_images --test_limit 20 --show_graph_nodes

# Quick test with first 10 images
python visuals4x4imagenetirregular.py --irregular_only --save_images --test_limit 10 --num_examples 2
# Default 10% erosion
python visuals4x4imagenetirregular.py --irregular_only --save_images --test_limit 20

# More aggressive 15% erosion
python visuals4x4imagenetirregular.py --irregular_only --save_images --test_limit 20 --erosion_percent 15

# No erosion (clean cut fragments)
python visuals4x4imagenetirregular.py --irregular_only --save_images --test_limit 20 --erosion_percent 0

# Heavy erosion for very weathered fragments
python visuals4x4imagenetirregular.py --irregular_only --save_images --test_limit 20 --erosion_percent 20

tmux attach -t puzzle
python puzzle_diff/train_script.py \
    -dataset imagenet \
    -puzzle_sizes 3 \
    -batch_size 8 \
    -gpus 1 --evaluate True --checkpoint_path "Puzzle-Diff/l4moa60r/checkpoints/last.ckpt" --offline
