import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import diffuser.utils as utils
from diffuser.models.helpers import Losses, apply_conditioning
from diffuser.models.backbone import VAE, TrajectoryEncoder


class OfflineDiffusionRL(nn.Module):
    """Offline Diffusion-based RL with VAE for action generation"""
    agent_share_parameters = True
    
    def __init__(
        self,
        model,  # Pre-built diffusion backbone model
        n_agents: int,
        horizon: int,
        history_horizon: int,
        observation_dim: int,
        action_dim: int,
        vae_latent_dim: int = 64,
        trajectory_latent_dim: int = 128,
        hidden_dim: int = 256,
        n_timesteps: int = 1000,
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        vae_weight: float = 1.0,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        # Offline RL specific parameters
        conservative_weight: float = 1.0,
        awac_weight: float = 1.0,
        use_conservative_loss: bool = True,
        use_behavior_cloning: bool = True,
        bc_weight: float = 1.0,
        q_weight: float = 1.0,  # Weight for Q-learning loss
        policy_weight: float = 0.1,  # Weight for policy optimization loss
        # Model architecture parameters (for backward compatibility)
        backbone_layers: int = 4,
        backbone_hidden_dim: int = 256,
        data_encoder: utils.Encoder = utils.IdentityEncoder(),
        # New parameter for decentralized execution
        decentralized: bool = True,
        use_ddim_sample: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.horizon = horizon
        self.history_horizon = history_horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.vae_latent_dim = vae_latent_dim
        self.trajectory_latent_dim = trajectory_latent_dim
        self.vae_weight = vae_weight
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.decentralized = decentralized
        
        # Store discrete_action flag for proper action handling
        self.discrete_action = kwargs.get('discrete_action', False)
        self.discount = kwargs.get('discount', 0.99)
        self.n_ddim_steps = kwargs.get('n_ddim_steps', 15)
        self.target_update_freq = kwargs.get('target_update_freq', 10)
        self.target_update_tau = kwargs.get('target_update_tau', 0.5)
        # Offline RL parameters

        self.awac_weight = awac_weight

        self.use_behavior_cloning = use_behavior_cloning
        self.bc_weight = bc_weight
        self.q_weight = q_weight
        self.policy_weight = policy_weight
        
        # VAE for action encoding/decoding
        self.vae = VAE(
            action_dim=action_dim,
            latent_dim=vae_latent_dim,
            hidden_dim=hidden_dim,
            n_agents=n_agents,
        )
        
        # Trajectory encoder for conditioning
        # For trajectory encoding, always use the full action space dimension
        # even for discrete actions, since we'll handle the conversion inside
        # In training, we always pass full trajectory with all agents, so use centralized input_dim
        self.trajectory_encoder = TrajectoryEncoder(
            observation_dim=observation_dim,
            action_dim=action_dim,  # Always use full action space
            history_horizon=max(history_horizon, 1),
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            latent_dim=trajectory_latent_dim,
            decentralized=False,  # Always use centralized mode for training
            # Sequence model parameters
            sequence_model=kwargs.get('trajectory_sequence_model', 'transformer'),
            num_layers=kwargs.get('trajectory_num_layers', 2),
            num_heads=kwargs.get('trajectory_num_heads', 8),
            dropout=kwargs.get('trajectory_dropout', 0.1),
        )
        
        # Store discrete_action flag for proper action handling
        self.discrete_action = kwargs.get('discrete_action', False)
        
        # Use the provided diffusion backbone model
        self.model = model
        
        # Set returns_condition on the backbone model
        if hasattr(self.model, 'returns_condition'):
            self.model.returns_condition = returns_condition
        
        # Calculate proper input dimensions for Q-function
        # Q-function always sees all agents' data for proper value evaluation
        # even in decentralized execution mode
        q_input_dim = observation_dim * n_agents + vae_latent_dim * n_agents + trajectory_latent_dim * n_agents
        
        # Value function for offline RL (Q-function)
        self.q_function = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Target Q-function for stable training
        self.target_q_function = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize target network
        self.target_q_function.load_state_dict(self.q_function.state_dict())
        
        # Noise scheduler
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.use_ddim_sample = use_ddim_sample
        if self.use_ddim_sample:
            self.set_ddim_scheduler(n_ddim_steps=self.n_ddim_steps)
            
        # Loss functions
        self.data_encoder = data_encoder
        
    
    def encode_actions(self, actions):
        """Encode actions to latent space"""
        if len(actions.shape) == 4:
            batch_size, horizon, n_agents, action_dim_input = actions.shape
        else:
            batch_size, action_dim_input = actions.shape
        
        # Handle discrete actions: convert scalar indices to one-hot
        if hasattr(self, 'discrete_action') and self.discrete_action and action_dim_input == 1:
            # Convert scalar action indices to one-hot (vectorized - much faster!)
            action_indices = actions.squeeze(-1).long()  # [batch, horizon, n_agents]
            # Clamp indices to valid range to prevent out-of-bounds errors
            action_indices = torch.clamp(action_indices, 0, self.action_dim - 1)
            # Use PyTorch's built-in one_hot function - extremely fast vectorized operation
            actions_onehot = F.one_hot(action_indices, num_classes=self.action_dim).float()  # [batch, horizon, n_agents, action_dim] or [batch, action_dim]
            actions_for_vae = actions_onehot
        else:
            # Continuous actions - use as is
            actions_for_vae = actions
        
        actions_flat = actions_for_vae.reshape(-1, self.action_dim)
        mu, log_var = self.vae.encode(actions_flat)
        latents = self.vae.reparameterize(mu, log_var)
        if len(actions.shape) == 4:
            latents = latents.reshape(batch_size, horizon, n_agents, self.vae_latent_dim)
        else:
            latents = latents.reshape(batch_size, self.vae_latent_dim)
        return latents, mu, log_var
    
    def decode_actions(self, latents):
        """Decode latents to actions"""
        if len(latents.shape) == 4:
            batch_size, horizon, n_agents, latent_dim = latents.shape
        else:
            batch_size, latent_dim = latents.shape
        latents_flat = latents.reshape(-1, latent_dim)
        actions_logits = self.vae.decode(latents_flat)
        
        if hasattr(self, 'discrete_action') and self.discrete_action:
            # For discrete actions, return the full logits and let downstream handle conversion
            # This preserves the action_dim dimension for proper compatibility
            if len(latents.shape) == 4:
                actions = actions_logits.reshape(batch_size, horizon, n_agents, self.action_dim)
            else:
                actions = actions_logits.reshape(batch_size, self.action_dim)
        else:
            # For continuous actions, use as is
            if len(latents.shape) == 4:
                actions = actions_logits.reshape(batch_size, horizon, n_agents, self.action_dim)
            else:
                actions = actions_logits.reshape(batch_size, self.action_dim)
        
        return actions
    
    def encode_trajectory(self, trajectory, agent_idx=None):
        return self.trajectory_encoder(trajectory, agent_idx=agent_idx)


    def get_model_output(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        trajectory_condition: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_timestep: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
        agent_idx: Optional[int] = None,  # New parameter for decentralized mode
    ):
        """Get diffusion model output"""
        batch_size, latent_dim = x.shape
        
        # Expand trajectory condition to match x dimensions
        trajectory_condition = trajectory_condition.expand(
            batch_size, self.trajectory_latent_dim
        )
        
        # Concatenate latents with trajectory condition
        model_input = torch.cat([x, trajectory_condition], dim=-1)
        
        # Get model output
        if self.returns_condition and returns is not None:
            epsilon_cond = self.model(
                model_input, t, returns=returns, env_timestep=env_timestep,
                attention_masks=attention_masks, use_dropout=False,
                agent_idx=agent_idx,  # Pass agent_idx for decentralized mode
            )
            epsilon_uncond = self.model(
                model_input, t, returns=returns, env_timestep=env_timestep,
                attention_masks=attention_masks, force_dropout=True,
                agent_idx=agent_idx,  # Pass agent_idx for decentralized mode
            )
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(
                model_input, t, env_timestep=env_timestep,
                attention_masks=attention_masks, use_dropout=use_dropout,
                force_dropout=force_dropout,
                agent_idx=agent_idx,  # Pass agent_idx for decentralized mode
            )
        
        return epsilon
    
    def conditional_sample(
        self,
        cond: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        verbose: bool = True,
        return_diffusion: bool = False,
        agent_idx: Optional[int] = None,  # New parameter for decentralized mode
        enable_grad: bool = False,  # New parameter to control gradients
    ):
        """Sample action latents conditioned on trajectory"""
        
        # Use context manager to control gradients
        context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
        
        with context_manager:
            if "history_trajectory" in cond:
                if self.decentralized and agent_idx is not None:
                    # In decentralized mode, encode only specified agent's trajectory
                    trajectory_condition = self.encode_trajectory(cond["history_trajectory"], agent_idx=agent_idx)
                batch_size = cond["history_trajectory"].shape[0]
            
            if self.decentralized and agent_idx is not None:
                # In decentralized mode, generate actions for single agent
                shape = (batch_size, self.vae_latent_dim)

            device = trajectory_condition.device
            if self.use_ddim_sample:
                scheduler = self.ddim_noise_scheduler
            else:
                scheduler = self.noise_scheduler
                
            x = torch.randn(shape, device=device)
            
            if return_diffusion:
                diffusion = [x]
                
            timesteps = scheduler.timesteps
            
            progress = utils.Progress(len(timesteps)) if verbose else utils.Silent()
            for t in timesteps:
                ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = self.get_model_output(
                    x, ts, trajectory_condition, returns, env_ts, attention_masks, agent_idx=agent_idx
                )
                
                x = scheduler.step(model_output, t, x).prev_sample
                
                progress.update({"t": t})
                if return_diffusion:
                    diffusion.append(x)
                    
            progress.close()
            
            # Decode action latents to actions
            actions = self.decode_actions(x)
            
            if return_diffusion:
                diffusion_mean = torch.mean(torch.stack(diffusion, dim=0), dim=0)
                return actions, diffusion_mean
            else:
                return actions
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        trajectory_condition: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
    ):
        """Compute diffusion losses"""
        noise = torch.randn_like(x_start)
        
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, t)
        x_noisy = self.data_encoder(x_noisy)
        
        epsilon = self.get_model_output(
            x_noisy, t, trajectory_condition, returns, env_ts, attention_masks
        )
        
        if not self.predict_epsilon:
            epsilon = self.data_encoder(epsilon)
            
        assert noise.shape == epsilon.shape
        
        # 直接计算diffusion loss，不使用复杂的权重和a0_loss
        if self.predict_epsilon:
            loss = F.mse_loss(epsilon, noise)
        else:
            loss = F.mse_loss(epsilon, x_start)
        
        info = {}  # 空的info字典，不包含a0_loss
            
        return loss, info
    
    def vae_loss(self, actions, mu, log_var):
        """Compute VAE reconstruction and KL divergence losses"""
        if len(actions.shape) == 4:
            batch_size, horizon, n_agents, action_dim_input = actions.shape
        else:
            batch_size, action_dim_input = actions.shape
        
        # Handle discrete actions: convert scalar indices to one-hot
        if hasattr(self, 'discrete_action') and self.discrete_action and action_dim_input == 1:
            # Convert scalar action indices to one-hot (vectorized - much faster!)
            action_indices = actions.squeeze(-1).long()  # [batch, horizon, n_agents]
            # Clamp indices to valid range to prevent out-of-bounds errors
            action_indices = torch.clamp(action_indices, 0, self.action_dim - 1)
            # Use PyTorch's built-in one_hot function - extremely fast vectorized operation
            actions_onehot = F.one_hot(action_indices, num_classes=self.action_dim).float()  # [batch, horizon, n_agents, action_dim]
            actions_for_vae = actions_onehot
        else:
            # Continuous actions - use as is
            actions_for_vae = actions
        
        actions_flat = actions_for_vae.reshape(-1, self.action_dim)
        mu_flat = mu.reshape(-1, self.vae_latent_dim)
        log_var_flat = log_var.reshape(-1, self.vae_latent_dim)
        
        z = self.vae.reparameterize(mu_flat, log_var_flat)
        recon_actions = self.vae.decode(z)
        
        if hasattr(self, 'discrete_action') and self.discrete_action and action_dim_input == 1:
            # For discrete actions, use cross-entropy loss
            # action_indices was computed above when converting to one-hot
            action_indices_flat = action_indices.reshape(-1)
            recon_loss = F.cross_entropy(recon_actions, action_indices_flat, reduction='mean')
        else:
            # For continuous actions, use MSE loss
            recon_loss = F.mse_loss(recon_actions, actions_flat, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var_flat - mu_flat.pow(2) - log_var_flat.exp())
        kl_loss = kl_loss.mean()
        
        return recon_loss, kl_loss 

  
    
    def behavioral_cloning_loss(self, obs, actions, trajectory_condition):
        """Behavioral cloning loss to stay close to data distribution"""
        # Generate actions from current policy (with gradients enabled)
        # Since we already have encoded trajectory_condition, we'll do the sampling manually
        batch_size = actions.shape[0]
        device = actions.device
        
        target_actions = actions[:, -1]  # [batch, 1, action_dim]
        
        # Handle discrete actions: convert to one-hot if needed
        if hasattr(self, 'discrete_action') and self.discrete_action and target_actions.shape[-1] == 1:
            # Convert scalar action indices to one-hot
            action_indices = target_actions.squeeze(-1).long()  # [batch, n_agents]
            action_indices = torch.clamp(action_indices, 0, self.action_dim - 1)
            target_actions = F.one_hot(action_indices, num_classes=self.action_dim).float().squeeze(1)  # [batch, action_dim]

        # Enable gradients for BC loss
        with torch.enable_grad():
            # Only generate single timestep actions for BC loss
            if self.decentralized:
                # In decentralized mode, generate actions for single agent
                shape = (batch_size, self.vae_latent_dim)

            
            if self.use_ddim_sample:
                scheduler = self.ddim_noise_scheduler
            else:
                scheduler = self.noise_scheduler
                
            x = torch.randn(shape, device=device)
            timesteps = scheduler.timesteps
            
            for t in timesteps:
                ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
                model_output = self.get_model_output(
                    x, ts, trajectory_condition, None, None, None
                )
                x = scheduler.step(model_output, t, x).prev_sample
            
            # Decode action latents to actions
            generated_actions = self.decode_actions(x)  # [batch, 1, n_agents, action_dim]
            if self.discrete_action:
                generated_actions = F.softmax(generated_actions, dim=-1)
        # Use appropriate loss based on action type
        if hasattr(self, 'discrete_action') and self.discrete_action:
            # For discrete actions, use cross entropy loss
            # generated_actions should be logits, target_actions should be class indices
            target_actions = target_actions.long()
            bc_loss = F.cross_entropy(generated_actions, target_actions)
        else:
            # For continuous actions, use MSE loss
            bc_loss = F.mse_loss(generated_actions, target_actions)
        
        return bc_loss
    
    def q_learning_loss(self, obs, actions, rewards, next_obs, dones, history_trajectory):
        """Q-learning loss for value function training"""
        
        batch_size = obs.shape[0]
        device = obs.device
        
        # Encode trajectory for each agent and average the conditions
        trajectory_conditions = []
        action_latents_list = []
        for agent_idx in range(self.n_agents):
            agent_condition = self.encode_trajectory(history_trajectory, agent_idx=agent_idx)
            trajectory_conditions.append(agent_condition)
            action_latents, mu, log_var = self.encode_actions(actions[:, agent_idx])
            action_latents_list.append(action_latents)

        action_latents = torch.stack(action_latents_list, dim=1).view(batch_size, -1)
        trajectory_condition = torch.stack(trajectory_conditions, dim=1).view(batch_size, -1)
    
        obs_last = obs
        actions_last = actions
        next_obs_last = next_obs
        rewards_last = rewards
        dones_last = dones
        action_latents_last = action_latents
        
        # Use all agents data directly
        # obs_last is already [batch, n_agents, obs_dim]
        # actions_last is already [batch, n_agents, action_dim]
        # next_obs_last is already [batch, n_agents, obs_dim]
        # rewards_last is already [batch, n_agents] or [batch, n_agents, 1]
        # dones_last is already [batch, n_agents] or [batch, n_agents, 1]
        # Flatten for Q-function
        obs_flat = obs_last.view(batch_size, -1)
        actions_flat = actions_last.view(batch_size, -1)
        next_obs_flat = next_obs_last.view(batch_size, -1)
        action_latents_flat = action_latents_last.view(batch_size, -1)

        # Compute current Q-values
        q_input = torch.cat([obs_flat, action_latents_flat, trajectory_condition], dim=-1)
        q_current = self.q_function(q_input)
        
        # Compute target Q-values
        with torch.no_grad():
            # Create updated trajectory for next state
            # Append current observation and action to history
            # Handle discrete actions - convert to one-hot before concatenating
            if hasattr(self, 'discrete_action') and self.discrete_action and actions_last.shape[-1] == 1:
                action_indices = actions_last.squeeze(-1).long()
                action_indices = torch.clamp(action_indices, 0, self.action_dim - 1)
                actions_onehot = F.one_hot(action_indices, num_classes=self.action_dim).float()
                current_step = torch.cat([obs_last, actions_onehot], dim=-1)  # [batch, n_agents, obs_dim + action_dim]
            else:
                current_step = torch.cat([obs_last, actions_last], dim=-1)  # [batch, n_agents, obs_dim + action_dim]
            current_step = current_step.unsqueeze(1)  # [batch, 1, n_agents, obs_dim + action_dim]
            
            # Update trajectory by appending current step and removing oldest if needed
            updated_trajectory = torch.cat([history_trajectory, current_step], dim=1)  # [batch, horizon+1, n_agents, obs_dim + action_dim]
            if updated_trajectory.shape[1] > self.history_horizon:
                updated_trajectory = updated_trajectory[:, -self.history_horizon:]  # Keep only recent history
            
            # Generate next actions for each agent
            next_actions_latents_list = []
            next_conditions_list = []
            
            for agent_idx in range(self.n_agents):
                # Encode next trajectory condition for this agent
                next_agent_condition = self.encode_trajectory(updated_trajectory, agent_idx=agent_idx)
                next_conditions_list.append(next_agent_condition)
                
                # Create condition dict for sampling with updated trajectory
                next_cond = {"history_trajectory": updated_trajectory}
                
                _, agent_next_actions_latents = self.conditional_sample(next_cond, agent_idx=agent_idx, return_diffusion=True, verbose=False)
                next_actions_latents_list.append(agent_next_actions_latents.squeeze(1))

            # Concatenate actions and average pool conditions
            next_actions_latents_flat = torch.stack(next_actions_latents_list, dim=1).view(batch_size, -1)
            next_trajectory_condition = torch.stack(next_conditions_list, dim=1).view(batch_size, -1)
            
            next_q_input = torch.cat([next_obs_flat, next_actions_latents_flat, next_trajectory_condition], dim=1)
            next_q = self.target_q_function(next_q_input)
            
            # Process rewards and dones
            reward_scalar = rewards_last.view(batch_size, -1).mean(dim=-1, keepdim=True)  # (batch_size, 1)
            done_scalar = dones_last.view(batch_size, -1).any(dim=-1, keepdim=True)       # (batch_size, 1)
            
            target_q = reward_scalar + self.discount * next_q * (1 - done_scalar.float())
        
        q_loss = F.mse_loss(q_current, target_q)
        return q_loss 

    def policy_optimization_loss(self, obs, actions, history_trajectory, agent_idx=None):
        """Policy optimization loss: maximize Q(s, π(s))"""
        
        batch_size = obs.shape[0]
        device = obs.device
        
        # Extract single timestep data for policy optimization
        if len(obs.shape) == 4:  # [batch, horizon, n_agents, obs_dim]
            obs_last = obs[:, -1]  # [batch, n_agents, obs_dim]
        else:
            obs_last = obs
        
        # In decentralized mode, compute loss for each agent separately
        if self.decentralized:
            # Collect all agent actions first
            all_agent_actions_latents = []
            agent_actions_list = []
            for agent_i in range(self.n_agents):
                # Generate actions using current policy with gradients enabled
                with torch.enable_grad():
                    current_obs = obs_last[:, agent_i:agent_i+1, :]

                    cond = {"history_trajectory": history_trajectory}

                    agent_actions, agent_actions_latents = self.conditional_sample(cond, agent_idx=agent_i, return_diffusion=True)
                    if self.discrete_action:
                        agent_actions = F.softmax(agent_actions, dim=-1)
                    agent_actions_list.append(agent_actions)
                    all_agent_actions_latents.append(agent_actions_latents)
            
            # Concatenate all agent actions: [batch, n_agents, action_latent_dim]
            policy_actions_latents = torch.stack(all_agent_actions_latents, dim=1)
            policy_actions = torch.stack(agent_actions_list, dim=1)
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.n_agents)
            agent_indices = torch.arange(self.n_agents, device=device).unsqueeze(0).expand(batch_size, -1)
            
            
            # Extract the probability values for the actual actions
            action_probs = policy_actions[batch_indices, agent_indices, actions.squeeze(-1).long()]  # [batch, n_agents]
            log_action_probs = torch.log(action_probs + 1e-8)  # Add small epsilon to avoid log(0)

            # Flatten for Q-function input  
            obs_flat = obs_last.view(batch_size, -1)
            actions_latents_flat = policy_actions_latents.view(batch_size, -1)

            trajectory_conditions = []
            for agent_idx in range(self.n_agents):
                agent_condition = self.encode_trajectory(history_trajectory, agent_idx=agent_idx)
                trajectory_conditions.append(agent_condition)

            trajectory_condition_flat = torch.stack(trajectory_conditions, dim=1).view(batch_size, -1)

            # Compute current Q-values
            q_input = torch.cat([obs_flat, actions_latents_flat, trajectory_condition_flat], dim=1)
            q_value = self.q_function(q_input)
                
            # Policy loss: negative Q-value (we want to maximize Q, so minimize -Q)
            policy_loss = -(q_value.expand_as(log_action_probs) * log_action_probs).mean()

        return policy_loss

    def loss(
        self,
        x: torch.Tensor,  # Full trajectory
        actions: torch.Tensor,  # Actions only
        history_trajectory: torch.Tensor,  # Historical trajectory
        cond: Dict[str, torch.Tensor],
        observations: Optional[torch.Tensor] = None,  # Observations only
        rewards: Optional[torch.Tensor] = None,
        next_observations: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        env_ts: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        agent_idx: Optional[int] = None,  # New parameter for decentralized training
        loss_masks: Optional[torch.Tensor] = None,  # 保持兼容性，但不使用
        **kwargs,
    ):
        """Compute total offline RL loss"""
        
        batch_size = len(x)
        
        # Encode actions to latents
        if self.decentralized and agent_idx is not None:
            # In decentralized mode: process single agent's actions
            if len(actions.shape) == 4 and actions.shape[2] > 1:  # [batch, horizon, n_agents, action_dim]
                single_agent_actions = actions[:, -1, agent_idx:agent_idx+1, :].squeeze(1)  # [batch, action_dim]
            else:
                single_agent_actions = actions  # Already single agent format
            action_latents, mu, log_var = self.encode_actions(single_agent_actions)
        
        # Encode historical trajectory for conditioning
        trajectory_condition = self.encode_trajectory(history_trajectory, agent_idx=agent_idx)

        
        # Compute diffusion loss
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=x.device,
        ).long()
        
        diffusion_loss, info = self.p_losses(
            action_latents, trajectory_condition, t,
            returns, env_ts, attention_masks,
        )
        
        # Compute VAE loss
        if self.decentralized and agent_idx is not None:
            recon_loss, kl_loss = self.vae_loss(single_agent_actions, mu, log_var)
        else:
            recon_loss, kl_loss = self.vae_loss(actions, mu, log_var)
        vae_loss = recon_loss + 0.1 * kl_loss
        
        # Use provided observations or extract from full trajectory
        if observations is None:
            observations = x[..., self.action_dim:]
        
        # In decentralized mode, extract single agent's observations
        if self.decentralized and agent_idx is not None and len(observations.shape) == 4:
            if observations.shape[2] > 1:  # [batch, horizon, n_agents, obs_dim]
                observations = observations[:, :, agent_idx:agent_idx+1, :]  # [batch, horizon, 1, obs_dim]
        
        # Initialize total loss
        total_loss = diffusion_loss + self.vae_weight * vae_loss
        
        if self.use_behavior_cloning:
            if self.decentralized and agent_idx is not None:
                bc_loss = self.behavioral_cloning_loss(observations, single_agent_actions, trajectory_condition)
            else:
                bc_loss = self.behavioral_cloning_loss(observations, actions, trajectory_condition)
            total_loss += self.bc_weight * bc_loss
            info["bc_loss"] = bc_loss
        
        info.update({
            "diffusion_loss": diffusion_loss,
            "vae_loss": vae_loss,
        })
        
        return total_loss, info
    
    def update_target_network(self, tau: float = 0.005):
        """Soft update of target Q-network"""
        for target_param, param in zip(self.target_q_function.parameters(), self.q_function.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_action(self, obs, history_trajectory=None, deterministic=False, agent_idx=None, returns=None):
        """Get action from current policy
        
        Args:
            obs: [batch_size, n_agents, obs_dim] (centralized) or [batch_size, obs_dim] (decentralized)
            history_trajectory: Historical trajectory for conditioning  
            deterministic: Whether to use deterministic sampling
            agent_idx: If provided, only return action for this agent (decentralized execution)
            returns: [batch_size, 1] return values for conditioning (optional)
        """
        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device

            # Handle history trajectory creation
            if history_trajectory is None:
                # Create empty history for single agent
                history_trajectory = torch.zeros(
                    batch_size, max(self.history_horizon, 1), self.n_agents,
                    self.observation_dim + self.action_dim, device=device
                )
            
            # Create condition dictionary
            cond = {"history_trajectory": history_trajectory}

            # Use conditional_sample with appropriate agent_idx
            if self.decentralized and agent_idx is not None:
                # In decentralized mode, generate actions for single agent
                actions = self.conditional_sample(cond, returns=returns, verbose=False, agent_idx=agent_idx)
            return actions

    
    def forward(self, cond, *args, **kwargs):
        """Generate actions using the complete pipeline"""
        return self.conditional_sample(cond, *args, **kwargs)
    
    def set_ddim_scheduler(self, n_ddim_steps: int = 15):
        """Set DDIM scheduler for faster sampling"""
        self.ddim_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.n_timesteps,
            clip_sample=True,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
        )
        self.ddim_noise_scheduler.set_timesteps(n_ddim_steps)
        self.use_ddim_sample = True

    def pretrain_vae(
        self,
        dataloader,
        n_vae_steps: int = 2000,
        vae_lr: float = 1e-3,
        device: str = "cuda",
        log_freq: int = 100,
    ):
        """Pre-train VAE before main training"""
        print(f"Pre-training VAE for {n_vae_steps} steps...")
        
        # Create optimizer for VAE only
        vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)
        
        self.vae.train()
        
        for step in range(n_vae_steps):
            try:
                batch = next(dataloader)
            except StopIteration:
                # Reset dataloader if exhausted
                dataloader = iter(dataloader)
                batch = next(dataloader)
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                actions = batch.get('actions')
            else:
                actions = batch.to(device)
            
            if actions is None:
                print("Warning: No actions found in batch, skipping...")
                continue
            
            vae_optimizer.zero_grad()
            
            # VAE forward pass
            action_latents, mu, log_var = self.encode_actions(actions)
            reconstructed_actions = self.decode_actions(action_latents)
            if self.discrete_action:
                reconstructed_actions = F.softmax(reconstructed_actions, dim=-1)
            # Compute VAE loss
            recon_loss, kl_loss = self.vae_loss(actions, mu, log_var)
            vae_loss = recon_loss + 0.1 * kl_loss
            
            vae_loss.backward()
            vae_optimizer.step()
            
            if step % log_freq == 0:
                print(f"VAE Step {step}/{n_vae_steps}: "
                      f"Total: {vae_loss.item():.4f} | "
                      f"Recon: {recon_loss.item():.4f} | "
                      f"KL: {kl_loss.item():.4f}")
        
        print("VAE pre-training completed!")
        return self

    def compute_q_and_policy_losses(self, x, actions, history_trajectory, cond, observations=None, 
                                   rewards=None, next_observations=None, dones=None, returns=None, 
                                   env_ts=None, attention_masks=None, **kwargs):
        """Compute Q-learning and policy optimization losses for all agents together (centralized)"""
        
        batch_size = len(x)
        
        # Use provided observations or extract from full trajectory
        if observations is None:
            observations = x[..., self.action_dim:]
        
        # Extract last timestep data (not t0)
        if len(observations.shape) == 4:  # [batch, horizon, n_agents, obs_dim]
            obs_last = observations[:, -1]  # [batch, n_agents, obs_dim]
        else:
            obs_last = observations
        
        # Q-learning loss if we have next states and rewards
        q_loss = torch.tensor(0.0, device=x.device)
        policy_loss = torch.tensor(0.0, device=x.device)
        info = {}
        
        if next_observations is not None and rewards is not None and dones is not None:
            next_obs_last = next_observations[:, -1]
                
            rewards_last = rewards[:, -1]  # [batch, n_agents, 1]

            dones_last = dones[:, -1]  # [batch, n_agents, 1]

            actions_last = actions[:, -1]  # [batch, n_agents, action_dim]

            # Compute Q-learning loss for all agents (centralized)
            q_loss = self.q_learning_loss(obs_last, actions_last, rewards_last, next_obs_last, dones_last, history_trajectory)
            
            # Compute policy optimization loss for all agents (centralized)
            policy_loss = self.policy_optimization_loss(obs_last, actions_last, history_trajectory)
            
            info["q_loss"] = q_loss
            info["policy_loss"] = policy_loss
        else:
            info["q_loss"] = torch.tensor(0.0, device=x.device)
            info["policy_loss"] = torch.tensor(0.0, device=x.device)
        
        return self.q_weight * q_loss + self.policy_weight * policy_loss, info