#!/usr/bin/env python3

import os
import sys
import torch
import yaml
from ml_logger import logger
from diffuser.utils.launcher_util import build_config_from_dict
import diffuser.utils as utils

def test_evaluation():
    """Test evaluation functionality"""
    
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
    log_dir = "logs/vae_diffusion_rl_quick_test/3m-Medium/h_8-hh_2-models.DiffusionBackbone-r_20-guidew_1.2/100"
    
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return
    
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # List checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("state_") and f.endswith(".pt")]
    if not checkpoint_files:
        print("No checkpoint files found")
        return
    
    # Get the latest step
    latest_step = max([int(f.split("_")[1].split(".")[0]) for f in checkpoint_files])
    print(f"Found latest checkpoint at step: {latest_step}")
    
    # Test if we can load the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"state_{latest_step}.pt")
    try:
        state_dict = torch.load(checkpoint_path, map_location=Config.device)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
        print(f"Checkpoint contains keys: {list(state_dict.keys())}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    # Test basic config parameters for evaluation
    print(f"Config.num_eval: {getattr(Config, 'num_eval', 'Not set')}")
    print(f"Config.env_type: {getattr(Config, 'env_type', 'Not set')}")
    print(f"Config.max_path_length: {getattr(Config, 'max_path_length', 'Not set')}")
    
    # Try to create evaluator with verbose output
    try:
        evaluator_config = utils.Config(
            Config.evaluator,
            verbose=True,
        )
        evaluator = evaluator_config()
        print("Successfully created evaluator")
        
        # Try to initialize evaluator
        evaluator.init(log_dir=log_dir)
        print("Successfully initialized evaluator")
        
        # Try evaluation
        print(f"Starting evaluation at step {latest_step}...")
        evaluator.evaluate(load_step=latest_step)
        print("Evaluation command sent")
        
        # Wait a bit to see if there are any outputs
        import time
        time.sleep(5)
        print("Evaluation test completed")
        
    except Exception as e:
        print(f"Error during evaluation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation() 