import torch
import torch.nn as nn
from diffuser.models.backbone import OfflineDiffusionRL, DiffusionBackbone, VAE, TrajectoryEncoder


def test_vae():
    """Test VAE component"""
    print("Testing VAE...")
    
    vae = VAE(
        action_dim=4,
        latent_dim=16,
        n_agents=2
    )
    
    # Test data
    batch_size = 3
    actions = torch.randn(batch_size, 2, 4)  # [batch, n_agents, action_dim]
    
    # Forward pass
    reconstructed, mu, log_var = vae(actions)
    
    # Check shapes
    assert reconstructed.shape == actions.shape, f"Reconstruction shape mismatch: {reconstructed.shape} vs {actions.shape}"
    assert mu.shape == (batch_size, 2, 16), f"Mu shape mismatch: {mu.shape}"
    assert log_var.shape == (batch_size, 2, 16), f"Log_var shape mismatch: {log_var.shape}"
    
    print("VAE test passed!\n")


def test_trajectory_encoder():
    """Test trajectory encoder"""
    print("Testing TrajectoryEncoder...")
    
    encoder = TrajectoryEncoder(
        observation_dim=6,
        action_dim=4,
        history_horizon=5,
        n_agents=2,
        latent_dim=32
    )
    
    # Test data
    batch_size = 3
    trajectory = torch.randn(batch_size, 5, 2, 10)  # [batch, horizon, n_agents, obs_dim + action_dim]
    
    # Forward pass
    encoded = encoder(trajectory)
    
    # Check shape
    assert encoded.shape == (batch_size, 32), f"Encoded shape mismatch: {encoded.shape}"
    
    print("TrajectoryEncoder test passed!\n")


def test_diffusion_backbone():
    """Test diffusion backbone model"""
    print("Testing DiffusionBackbone...")
    
    backbone = DiffusionBackbone(
        latent_dim=16,
        trajectory_latent_dim=32,
        horizon=8,
        n_agents=2,
        hidden_dim=64,
        n_layers=2
    )
    
    # Test data
    batch_size = 3
    x = torch.randn(batch_size, 8, 2, 16 + 32)  # [batch, horizon, n_agents, latent_dim + trajectory_latent_dim]
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = backbone(x, t)
    
    # Check shape
    assert output.shape == (batch_size, 8, 2, 16), f"Output shape mismatch: {output.shape}"
    
    print("DiffusionBackbone test passed!\n")


def test_offline_diffusion_rl():
    """Test complete OfflineDiffusionRL model"""
    print("Testing OfflineDiffusionRL...")
    
    model = OfflineDiffusionRL(
        n_agents=2,
        horizon=8,
        history_horizon=3,
        observation_dim=6,
        action_dim=4,
        vae_latent_dim=16,
        trajectory_latent_dim=32,
        hidden_dim=64,
        n_timesteps=100,
        backbone_layers=2,
        backbone_hidden_dim=64,
        use_conservative_loss=False,
        use_behavior_cloning=False,
    )
    
    # Test data
    batch_size = 3
    x = torch.randn(batch_size, 8, 2, 10)  # [batch, horizon, n_agents, obs_dim + action_dim]
    actions = torch.randn(batch_size, 8, 2, 4)  # [batch, horizon, n_agents, action_dim]
    history_trajectory = torch.randn(batch_size, 3, 2, 10)  # [batch, history_horizon, n_agents, obs_dim + action_dim]
    loss_masks = torch.ones(batch_size, 8, 2, 16)  # [batch, horizon, n_agents, vae_latent_dim]
    cond = {'x': torch.randn(batch_size, 1, 2, 10)}
    
    # Forward pass
    loss, info = model.loss(
        x=x,
        actions=actions,
        history_trajectory=history_trajectory,
        cond=cond,
        loss_masks=loss_masks
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Info keys: {list(info.keys())}")
    
    # Test sampling
    sample_cond = {"history_trajectory": history_trajectory[:2]}
    with torch.no_grad():
        generated_actions = model.conditional_sample(sample_cond, horizon=8)
    
    print(f"Generated actions shape: {generated_actions.shape}")
    print("OfflineDiffusionRL test passed!\n")


if __name__ == "__main__":
    print("Testing VAE-based Diffusion RL backbone model implementation\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        test_vae()
        test_trajectory_encoder()
        test_diffusion_backbone()
        test_offline_diffusion_rl()
        
        print("All tests passed! The backbone model implementation is working correctly.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 