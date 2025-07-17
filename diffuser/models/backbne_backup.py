
class DiffusionBackbone0(nn.Module):
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
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
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
            # Reshape for BatchNorm1d: [batch, n_agents, hidden_dim] -> [batch*n_agents, hidden_dim]
            batch_size, n_agents, hidden_dim = x.shape
            x_reshaped = x.view(batch_size * n_agents, hidden_dim)
            x_reshaped = self.batch_norm[i](x_reshaped)
            # Reshape back: [batch*n_agents, hidden_dim] -> [batch, n_agents, hidden_dim]
            x = x_reshaped.view(batch_size, n_agents, hidden_dim)
        x = self.layer_norm(x)
        
        # Output projection
        output = self.output_proj(x)  # [batch, n_agents, action_dim]
        
        # Reshape back to [batch, n_agents, action_dim]
        output = output.view(batch_size, n_agents, -1)
        
        return output 


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
        
        # Agent interaction layers with self-attention
        self.interaction_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=4,
                    batch_first=True,
                    dropout=0.1
                ),
                'mlp': TemporalMlpBlock(
                    dim_in=hidden_dim,
                    dim_out=hidden_dim,
                    embed_dim=time_embed_dim,
                    act_fn=nn.ReLU(),
                    out_act_fn=nn.ReLU()
                ),
                'layer_norm1': nn.LayerNorm(hidden_dim),
                'layer_norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(0.1)
            }) for _ in range(n_layers)
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
        x: [batch, n_agents, action_dim + trajectory_latent_dim] for agent interaction
        t: [batch] timestep
        returns: [batch, 1] return values for conditioning (optional)
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
        t_emb_expanded = t_emb.unsqueeze(1).expand(batch_size, n_agents, -1)
        # Apply interaction layers with self-attention for agent communication
        for layer in self.interaction_layers:
            # Self-attention for agent interaction
            attn_out, _ = layer['self_attention'](
                query=x,
                key=x,
                value=x,
                attn_mask=None
            )
            x = layer['layer_norm1'](x + layer['dropout'](attn_out))
            
            # MLP processing with time embedding
            x = layer['layer_norm2'](x + layer['dropout'](layer['mlp'](x, t_emb_expanded)))
        
        # Output projection
        output = self.output_proj(x)  # [batch, n_agents, action_dim]
        
        return output 