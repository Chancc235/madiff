import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple
import numpy as np

import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning
from diffuser.models.temporal import SinusoidalPosEmb, TemporalMlpBlock


class VAE(nn.Module):
    """VAE for encoding/decoding actions"""
    
    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        n_agents: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def encode(self, x):
        """Encode action to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent to action"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class TrajectoryEncoder(nn.Module):
    """Sequence model for encoding historical trajectory sequences"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        history_horizon: int,
        n_agents: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        decentralized: bool = False,
        sequence_model: str = "transformer",  # "lstm", "gru", "transformer"
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.history_horizon = history_horizon
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.decentralized = decentralized
        self.sequence_model = sequence_model

        input_dim = observation_dim + action_dim

        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Sequence model
        if sequence_model == "lstm":
            self.sequence_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            # LSTM is bidirectional, so output size is 2 * hidden_dim
            sequence_output_dim = hidden_dim * 2
            
        elif sequence_model == "gru":
            self.sequence_encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            # GRU is bidirectional, so output size is 2 * hidden_dim
            sequence_output_dim = hidden_dim * 2
            
        elif sequence_model == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
            sequence_output_dim = hidden_dim
            
        else:
            raise ValueError(f"Unsupported sequence model: {sequence_model}")
        
        # Output projection to latent dimension
        self.output_proj = nn.Sequential(
            nn.Linear(sequence_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Positional encoding for transformer
        if sequence_model == "transformer":
            self.pos_encoding = nn.Parameter(
                torch.randn(1, history_horizon, hidden_dim) * 0.02
            )
        
    def forward(self, trajectory, agent_idx=None):
        """
        Encode historical trajectory sequence to compressed representation
        trajectory: [batch, history_horizon, n_agents, obs_dim + action_dim] (centralized)
                   or [batch, history_horizon, obs_dim + action_dim] (decentralized)
        agent_idx: Index of agent for decentralized mode
        """
        batch_size = trajectory.shape[0]
        sequence_input = trajectory[:, :, agent_idx, :]
        
        # Input projection: [batch, seq_len, hidden_dim]
        x = self.input_proj(sequence_input)
        if self.sequence_model == "transformer":
            # Add positional encoding
            seq_len = x.shape[1]
            if seq_len <= self.history_horizon:
                x = x + self.pos_encoding[:, :seq_len, :]
            else:
                # Handle sequences longer than expected
                pos_enc_extended = self.pos_encoding.repeat(1, (seq_len // self.history_horizon) + 1, 1)
                x = x + pos_enc_extended[:, :seq_len, :]
            
            # Transformer encoding
            encoded = self.sequence_encoder(x)  # [batch, seq_len, hidden_dim]
            
            # Global average pooling over sequence dimension
            pooled = encoded.mean(dim=1)  # [batch, hidden_dim]
            
        elif self.sequence_model in ["lstm", "gru"]:
            # RNN encoding
            encoded, _ = self.sequence_encoder(x)  # [batch, seq_len, hidden_dim * 2]
            
            # Use the last hidden state (or could use mean/max pooling)
            pooled = encoded[:, -1, :]  # [batch, hidden_dim * 2]
        
        # Project to latent dimension
        latent = self.output_proj(pooled)  # [batch, latent_dim]
        
        return latent


class DiffusionBackbone(nn.Module):
    """
    Simplified diffusion backbone model for generating action latents
    This is a lightweight version that works directly with latent space
    """
    agent_share_parameters = True
    
    def __init__(
        self,
        latent_dim: int,
        trajectory_latent_dim: int,
        horizon: int,
        n_agents: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_embed_dim: int = 32,
        decentralized: bool = False,  # New parameter for decentralized mode
        returns_condition: bool = False,  # Add returns conditioning support
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.trajectory_latent_dim = trajectory_latent_dim
        self.horizon = horizon
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.decentralized = decentralized
        self.returns_condition = returns_condition
        
        # Input dimension: action latents + trajectory condition
        input_dim = latent_dim + trajectory_latent_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim // 4),
            nn.Linear(time_embed_dim // 4, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Returns conditioning
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, time_embed_dim // 2),
                nn.ReLU(),
                nn.Linear(time_embed_dim // 2, time_embed_dim),
                nn.ReLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            self.mask_dist = torch.distributions.Bernoulli(probs=0.1)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal processing layers
        self.temporal_layers = nn.ModuleList([
            TemporalMlpBlock(
                dim_in=hidden_dim,
                dim_out=hidden_dim,
                embed_dim=time_embed_dim,
                act_fn=nn.ReLU(),
                out_act_fn=nn.ReLU()
            ) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(
        self, 
        x, 
        t, 
        returns=None, 
        env_timestep=None, 
        attention_masks=None,
        use_dropout=True,
        force_dropout=False,
        agent_idx=None,  # New parameter for decentralized execution
        **kwargs
    ):
        """
        Forward pass through diffusion backbone
        x: [batch, horizon, n_agents, latent_dim + trajectory_latent_dim] (centralized)
           or [batch, horizon, 1, latent_dim + trajectory_latent_dim] (decentralized)
        t: [batch] timestep
        returns: [batch, 1] return values for conditioning (optional)
        agent_idx: Index of agent for decentralized mode
        """
        if len(x.shape) == 4:
            batch_size, horizon, n_agents, input_dim = x.shape
        else:
            batch_size, input_dim = x.shape
        
        # Time embedding
        t_emb = self.time_mlp(t)  # [batch, time_embed_dim]
        
        # Returns conditioning
        if self.returns_condition:
            if returns is not None and not force_dropout:
                # Handle different return shapes
                if len(returns.shape) == 2 and returns.shape[-1] > 1:
                    # Multi-agent returns [batch, n_agents] -> take mean
                    returns_scalar = returns.mean(dim=-1, keepdim=True)  # [batch, 1]
                elif len(returns.shape) == 1:
                    # Single return per batch [batch] -> add dimension
                    returns_scalar = returns.unsqueeze(-1)  # [batch, 1]
                elif len(returns.shape) == 2 and returns.shape[-1] == 1:
                    # Already [batch, 1] format
                    returns_scalar = returns
                else:
                    # Fallback: flatten and take mean for any other shapes
                    returns_scalar = returns.view(returns.shape[0], -1).mean(dim=-1, keepdim=True)  # [batch, 1]
                
                # Ensure returns_scalar is always [batch, 1]
                if returns_scalar.shape[-1] != 1:
                    returns_scalar = returns_scalar.mean(dim=-1, keepdim=True)
                
                # Compute returns embedding
                returns_emb = self.returns_mlp(returns_scalar.float())
                
                # During training, randomly drop out returns with probability 0.1 for CFG
                if self.training:
                    mask = self.mask_dist.sample((batch_size,)).to(returns.device)
                    # mask is 1 with prob 0.9 (keep returns), 0 with prob 0.1 (drop returns)
                    returns_emb = returns_emb * mask.unsqueeze(-1)
            else:
                # Zero out returns embedding when force_dropout or returns is None
                returns_emb = torch.zeros(batch_size, self.time_embed_dim, device=x.device)
            
            # Combine time and returns embeddings
            t_emb = t_emb + returns_emb
        
        # Input projection
        x = self.input_proj(x)  # [batch, hidden_dim]
        
        # For TemporalMlpBlock, we need to reshape to [batch * n_agents * horizon, hidden_dim]
        # and then reshape back after processing
        if len(x.shape) == 4:
            x_reshaped = x.reshape(batch_size * n_agents * horizon, self.hidden_dim)  # [batch * n_agents * horizon, hidden_dim]
        else:
            x_reshaped = x.reshape(batch_size, self.hidden_dim)
        
        # Expand time embedding for all agents and timesteps
        if len(x.shape) == 4:
            t_emb_expanded = t_emb.unsqueeze(1).unsqueeze(1).expand(batch_size, n_agents, horizon, -1)
            t_emb_expanded = t_emb_expanded.reshape(batch_size * n_agents * horizon, -1)  # [batch * n_agents * horizon, time_embed_dim]
        else:
            t_emb_expanded = t_emb.unsqueeze(1).reshape(batch_size, -1)  # [batch, time_embed_dim]
        
        # Apply temporal layers
        for i, temp_layer in enumerate(self.temporal_layers):
            x_reshaped = temp_layer(x_reshaped, t_emb_expanded)
        
        # Reshape back to [batch, horizon, n_agents, hidden_dim] after temporal processing
        if len(x.shape) == 4:
            x = x_reshaped.reshape(batch_size, n_agents, horizon, self.hidden_dim).permute(0, 2, 1, 3)
        else:
            x = x_reshaped.reshape(batch_size, self.hidden_dim)
        
        # Output projection
        output = self.output_proj(x)  # [batch, horizon, n_agents, latent_dim]
        
        return output 