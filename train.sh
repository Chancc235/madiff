#!/bin/bash

# Simple training script for VAE Diffusion RL
# Usage: bash train.sh [gpu] [config]

GPU=${1:-0}
CONFIG=${2:-backbone_training}

echo "Training on GPU $GPU with config $CONFIG"

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU

# Update config
python -c "
with open('config/vae_diffusion_rl.py', 'r') as f:
    content = f.read()
content = content.replace(\"SELECTED_CONFIG = 'backbone_training'\", \"SELECTED_CONFIG = '$CONFIG'\")
content = content.replace(\"'device': 'cuda:0'\", \"'device': 'cuda:$GPU'\")
with open('config/vae_diffusion_rl.py', 'w') as f:
    f.write(content)
"

# Start training
python run_scripts/train_vae_diffusion.py 