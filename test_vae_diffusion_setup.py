#!/usr/bin/env python3
"""
Quick test script to verify VAE+Diffusion RL setup
"""

import torch
import numpy as np
from diffuser.models.diffusion_rl import DiffusionRL, VAE, TrajectoryEncoder
from diffuser.models.ma_temporal import SharedConvAttentionDeconv
from diffuser.datasets.sequence import VAESequenceDataset
import diffuser.utils as utils

def test_vae():
    """Test VAE component"""
    print("Testing VAE...")
    
    action_dim = 4
    latent_dim = 64
    batch_size = 8
    n_agents = 3
    
    vae = VAE(action_dim=action_dim, latent_dim=latent_dim, n_agents=n_agents)
    
    # Test encoding
    actions = torch.randn(batch_size, 24, n_agents, action_dim)
    mu, log_var = vae.encode(actions.reshape(-1, action_dim))
    print(f"  Encoded actions shape: {mu.shape}, {log_var.shape}")
    
    # Test reparameterization
    z = vae.reparameterize(mu, log_var)
    print(f"  Latent samples shape: {z.shape}")
    
    # Test decoding
    decoded = vae.decode(z)
    print(f"  Decoded actions shape: {decoded.shape}")
    
    # Test full forward pass
    reconstructed, mu_full, log_var_full = vae(actions.reshape(-1, action_dim))
    print(f"  Full forward pass: {reconstructed.shape}")
    print("  ✓ VAE test passed")

def test_trajectory_encoder():
    """Test trajectory encoder"""
    print("Testing TrajectoryEncoder...")
    
    obs_dim = 10
    action_dim = 4
    history_horizon = 4
    n_agents = 3
    batch_size = 8
    
    encoder = TrajectoryEncoder(
        observation_dim=obs_dim,
        action_dim=action_dim,
        history_horizon=history_horizon,
        n_agents=n_agents,
    )
    
    # Test trajectory encoding
    trajectory = torch.randn(batch_size, history_horizon, n_agents, obs_dim + action_dim)
    compressed = encoder(trajectory)
    print(f"  Compressed trajectory shape: {compressed.shape}")
    print("  ✓ TrajectoryEncoder test passed")

def test_diffusion_rl():
    """Test full DiffusionRL model"""
    print("Testing DiffusionRL...")
    
    # Parameters
    n_agents = 3
    horizon = 24
    history_horizon = 4
    obs_dim = 10
    action_dim = 4
    vae_latent_dim = 32  # Smaller for testing
    batch_size = 4
    
    # Create underlying diffusion model
    model = SharedConvAttentionDeconv(
        horizon=horizon + history_horizon,
        transition_dim=obs_dim + action_dim,
        n_agents=n_agents,
        dim=64,
        dim_mults=[1, 2, 4],
    )
    
    # Create DiffusionRL
    diffusion_rl = DiffusionRL(
        model=model,
        n_agents=n_agents,
        horizon=horizon,
        history_horizon=history_horizon,
        observation_dim=obs_dim,
        action_dim=action_dim,
        vae_latent_dim=vae_latent_dim,
        trajectory_latent_dim=64,
        hidden_dim=128,
        n_timesteps=100,  # Fewer steps for testing
    )
    
    # Test data
    trajectories = torch.randn(batch_size, horizon, n_agents, obs_dim + action_dim)
    actions = torch.randn(batch_size, horizon, n_agents, action_dim)
    history_trajectory = torch.randn(batch_size, history_horizon, n_agents, obs_dim + action_dim)
    # Fix: loss_masks should match the shape of latent loss
    loss_masks = torch.ones(batch_size, horizon, n_agents, vae_latent_dim)
    
    # Test loss computation
    cond = {"history_trajectory": history_trajectory}
    loss, info = diffusion_rl.loss(
        x=trajectories,
        actions=actions,
        history_trajectory=history_trajectory,
        cond=cond,
        loss_masks=loss_masks,
    )
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Info keys: {list(info.keys())}")
    
    # Test action generation
    with torch.no_grad():
        generated_actions = diffusion_rl.conditional_sample(cond, verbose=False)
        print(f"  Generated actions shape: {generated_actions.shape}")
    
    print("  ✓ DiffusionRL test passed")

def test_vae_dataset():
    """Test VAE dataset (mock test)"""
    print("Testing VAESequenceDataset...")
    
    try:
        # This is a mock test since we don't have actual data
        print("  VAESequenceDataset class exists and is importable")
        print("  ✓ VAESequenceDataset test passed (mock)")
    except Exception as e:
        print(f"  ✗ VAESequenceDataset test failed: {e}")

def main():
    """Run all tests"""
    print("=" * 50)
    print("VAE+Diffusion RL Setup Test")
    print("=" * 50)
    
    try:
        test_vae()
        print()
        test_trajectory_encoder()
        print()
        test_diffusion_rl()
        print()
        test_vae_dataset()
        print()
        
        print("=" * 50)
        print("✓ All tests passed! Setup is working correctly.")
        print("=" * 50)
        
        print("\nNext steps:")
        print("1. Prepare your dataset")
        print("2. Configure exp_specs/vae_diffusion_rl_example.yaml")
        print("3. Run: python run_experiment.py -e exp_specs/vae_diffusion_rl_example.yaml -g 0")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error and fix any issues before running experiments.")

if __name__ == "__main__":
    main() 