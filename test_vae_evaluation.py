#!/usr/bin/env python3

import os
import sys
import torch
import yaml
from diffuser.utils.launcher_util import build_config_from_dict
import diffuser.utils as utils

def test_vae_evaluation():
    """Test VAE evaluation functionality"""
    
    # Load config
    exp_file = "exp_specs/smac/3m/vae_diffusion_rl_quick_test.yaml"
    with open(exp_file, "r") as f:
        spec_string = f.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
    
    if "constants" in exp_specs:
        config_dict = exp_specs["constants"]
    else:
        config_dict = exp_specs
    
    if "meta_data" in exp_specs:
        config_dict.update(exp_specs["meta_data"])
    
    Config = build_config_from_dict(config_dict)
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find the latest checkpoint
    log_dirs = []
    for root, dirs, files in os.walk("logs"):
        if "checkpoint" in dirs and "vae_diffusion_rl" in root:
            log_dirs.append(root)
    
    if not log_dirs:
        print("No log directories found")
        return
    
    # Use the most recent log directory
    log_dir = max(log_dirs, key=lambda x: os.path.getmtime(x))
    print(f"Using log directory: {log_dir}")
    
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("state_") and f.endswith(".pt")]
    
    if not checkpoint_files:
        print("No checkpoint files found")
        return
    
    # Get the latest step
    latest_step = max([int(f.split("_")[1].split(".")[0]) for f in checkpoint_files])
    print(f"Found latest checkpoint at step: {latest_step}")
    
    # Test VAE evaluator
    try:
        evaluator = utils.VAEDiffusionEvaluator(verbose=True)
        print("Successfully created VAE evaluator")
        
        evaluator.init(log_dir=log_dir)
        print("Successfully initialized VAE evaluator")
        
        print(f"Starting evaluation at step {latest_step}...")
        results = evaluator.evaluate(load_step=latest_step)
        print(f"Evaluation completed! Results: {results}")
        
    except Exception as e:
        print(f"Error during VAE evaluation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vae_evaluation() 