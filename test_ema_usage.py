#!/usr/bin/env python3

import sys
import re

def check_ema_usage_in_vae_evaluator():
    """检查 vae_evaluator.py 中是否正确使用了 ema_model"""
    
    file_path = "diffuser/utils/vae_evaluator.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 查找所有 trainer.model 的使用
    trainer_model_matches = re.findall(r'self\.trainer\.model\.[a-zA-Z_]+', content)
    trainer_ema_model_matches = re.findall(r'self\.trainer\.ema_model\.[a-zA-Z_]+', content)
    
    print("=== VAE Evaluator EMA Usage Check ===")
    print(f"Found {len(trainer_model_matches)} trainer.model method calls:")
    for match in set(trainer_model_matches):
        print(f"  - {match}")
    
    print(f"\nFound {len(trainer_ema_model_matches)} trainer.ema_model method calls:")
    for match in set(trainer_ema_model_matches):
        print(f"  - {match}")
    
    # 检查推理相关的方法是否使用了 ema_model
    inference_methods = ['get_action', 'conditional_sample']
    
    print(f"\n=== Inference Method Usage ===")
    for method in inference_methods:
        model_usage = len(re.findall(rf'self\.trainer\.model\.{method}', content))
        ema_model_usage = len(re.findall(rf'self\.trainer\.ema_model\.{method}', content))
        
        print(f"{method}:")
        print(f"  - trainer.model usage: {model_usage}")
        print(f"  - trainer.ema_model usage: {ema_model_usage}")
        
        if ema_model_usage > 0 and model_usage == 0:
            print(f"  ✅ Correctly using EMA model for {method}")
        elif model_usage > 0:
            print(f"  ❌ Still using regular model for {method}")
        else:
            print(f"  ℹ️  No usage of {method} found")
    
    # 检查属性访问
    print(f"\n=== Property Access ===")
    model_action_dim = len(re.findall(r'self\.trainer\.model\.action_dim', content))
    ema_model_action_dim = len(re.findall(r'self\.trainer\.ema_model\.action_dim', content))
    
    print(f"action_dim property access:")
    print(f"  - trainer.model.action_dim: {model_action_dim}")
    print(f"  - trainer.ema_model.action_dim: {ema_model_action_dim}")
    
    if ema_model_action_dim > model_action_dim:
        print(f"  ✅ Mostly using EMA model for action_dim")
    elif model_action_dim > 0:
        print(f"  ⚠️  Still some usage of regular model for action_dim")
    
    # 总结
    print(f"\n=== Summary ===")
    if ema_model_action_dim > 0 and len([m for m in trainer_ema_model_matches if 'get_action' in m]) > 0:
        print("✅ VAE Evaluator is correctly configured to use EMA model for inference!")
    else:
        print("❌ VAE Evaluator may not be using EMA model properly for inference.")
    
    return True

if __name__ == "__main__":
    check_ema_usage_in_vae_evaluator() 