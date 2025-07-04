import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple
import numpy as np

import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning
from diffuser.models.temporal import SinusoidalPosEmb, TemporalMlpBlock

class PatternEncoder(nn.Module):
    """LSTM-based sequence model for encoding action patterns"""
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension depends on bidirectional setting
        lstm_output_dim = hidden_dim
        
        # Output projection to latent dimension
        self.output_proj = nn.Linear(lstm_output_dim, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, action_dim] - sequence of actions
            
        Returns:
            latent: [batch_size, latent_dim] - encoded pattern representation
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim * (2 if bidirectional else 1)]
        final_hidden = h_n[-1]  # [batch_size, hidden_dim]
        
        # Project to latent dimension
        latent = self.output_proj(final_hidden)  # [batch_size, latent_dim]
        
        return latent


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
        action_dim: int,
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
        self.action_dim = action_dim
        self.trajectory_latent_dim = trajectory_latent_dim
        self.horizon = horizon
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.decentralized = decentralized
        self.returns_condition = returns_condition
        
        # Input dimension: action latents + trajectory condition
        input_dim = action_dim + trajectory_latent_dim
        
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
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        
    def forward(
        self, 
        x, 
        t, 
        returns=None, 
        env_timestep=None, 
        attention_masks=None,
        use_dropout=True,
        force_dropout=False,
        **kwargs
    ):
        """
        Forward pass through diffusion backbone
        x: [batch, n_agents, action_dim + trajectory_latent_dim] for independent agent denoising
        t: [batch] timestep
        returns: [batch, 1] return values for conditioning (optional)
        agent_idx: Index of agent for decentralized mode
        """
        batch_size, n_agents, input_dim = x.shape
        
        # Time embedding
        t_emb = self.time_mlp(t)  # [batch, time_embed_dim]
        
        # Returns conditioning
        if self.returns_condition:
            if returns is not None and not force_dropout:
                returns_scalar = returns.float().view(returns.shape[0], -1).mean(dim=-1, keepdim=True)  # [batch, 1]
                
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
        x = self.input_proj(x)  # [batch, n_agents, hidden_dim]
        
        # Expand time embedding for all agents
        # t_emb: [batch, n_agents, time_embed_dim] -> [batch * n_agents, time_embed_dim]
        t_emb_expanded = t_emb.unsqueeze(1).expand(batch_size, n_agents, -1)
        
        # Apply temporal layers (each agent processed independently)
        for i, temp_layer in enumerate(self.temporal_layers):
            x = temp_layer(x, t_emb_expanded)
        
        # Output projection
        output = self.output_proj(x)  # [batch, n_agents, action_dim]
        
        # Reshape back to [batch, n_agents, action_dim]
        output = output.view(batch_size, n_agents, -1)
        
        return output 