import os
import numpy as np
import torch
from ml_logger import logger
import importlib
from collections import deque
from copy import deepcopy
import pickle
from diffuser.utils.launcher_util import build_config_from_dict
from diffuser.utils.arrays import to_device, to_np, to_torch
import diffuser.utils as utils
import gc


class VAEDiffusionEvaluator:
    """
    Simple evaluator for VAE+Diffusion RL models that runs in the main process
    and doesn't rely on inverse models.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.initialized = False
        
    def init(self, log_dir, **kwargs):
        """Initialize the evaluator"""
        if self.initialized:
            return
            
        self.log_dir = log_dir
        
        # Load config
        with open(os.path.join(log_dir, "parameters.pkl"), "rb") as f:
            params = pickle.load(f)
        
        Config = build_config_from_dict(params["Config"])
        self.Config = Config = build_config_from_dict(kwargs, Config)
        self.Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model configurations
        with open(os.path.join(log_dir, "model_config.pkl"), "rb") as f:
            model_config = pickle.load(f)
        
        with open(os.path.join(log_dir, "diffusion_config.pkl"), "rb") as f:
            diffusion_config = pickle.load(f)
        
        with open(os.path.join(log_dir, "trainer_config.pkl"), "rb") as f:
            trainer_config = pickle.load(f)
        
        with open(os.path.join(log_dir, "dataset_config.pkl"), "rb") as f:
            dataset_config = pickle.load(f)
        
        with open(os.path.join(log_dir, "render_config.pkl"), "rb") as f:
            render_config = pickle.load(f)
        
        # Initialize components
        dataset = dataset_config()
        self.normalizer = dataset.normalizer
        del dataset
        gc.collect()
        
        renderer = render_config()
        model = model_config()
        diffusion = diffusion_config(model)
        self.trainer = trainer_config(diffusion, None, renderer)
        
        # Load environment
        env_mod_name = {
            "d4rl": "diffuser.datasets.d4rl",
            "mahalfcheetah": "diffuser.datasets.mahalfcheetah", 
            "mamujoco": "diffuser.datasets.mamujoco",
            "mpe": "diffuser.datasets.mpe",
            "smac": "diffuser.datasets.smac_env",
            "smacv2": "diffuser.datasets.smacv2_env",
        }[Config.env_type]
        env_mod = importlib.import_module(env_mod_name)
        
        Config.num_envs = getattr(Config, "num_envs", Config.num_eval)
        self.env_list = [
            env_mod.load_environment(Config.dataset) for _ in range(Config.num_envs)
        ]
        
        self.discrete_action = False
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            self.discrete_action = True
        
        # Ensure models are on the correct device
        self.trainer.model = self.trainer.model.to(Config.device)
        self.trainer.ema_model = self.trainer.ema_model.to(Config.device)
            
        self.initialized = True
        
        if self.verbose:
            print(f"VAE Diffusion Evaluator initialized for {Config.env_type}")
        
    def evaluate(self, load_step):
        """Evaluate the model at given step with parallel environments"""
        if not self.initialized:
            print("Evaluator not initialized!")
            return
            
        Config = self.Config
        device = Config.device
        observation_dim = self.normalizer.observation_dim
        
        loadpath = os.path.join(self.log_dir, "checkpoint")
        
        if Config.save_checkpoints:
            loadpath = os.path.join(loadpath, f"state_{load_step}.pt")
        else:
            loadpath = os.path.join(loadpath, "state.pt")
            
        if not os.path.exists(loadpath):
            print(f"Checkpoint not found: {loadpath}")
            return
            
        # Load model weights
        state_dict = torch.load(loadpath, map_location=Config.device)
        self.trainer.step = state_dict["step"]
        self.trainer.model.load_state_dict(state_dict["model"])
        self.trainer.ema_model.load_state_dict(state_dict["ema"])
        
        # Ensure models are on the correct device after loading
        self.trainer.model = self.trainer.model.to(Config.device)
        self.trainer.ema_model = self.trainer.ema_model.to(Config.device)
        
        if self.verbose:
            print(f"Loaded checkpoint from step {load_step} and moved to {Config.device}")
            
        num_eval = getattr(Config, 'num_eval', 5)
        num_envs = getattr(Config, 'num_envs', num_eval)
        
        episode_rewards = []
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = []
            
        # Parallel evaluation similar to original evaluator
        cur_num_eval = 0
        while cur_num_eval < num_eval:
            num_episodes = min(num_eval - cur_num_eval, num_envs)
            if self.verbose:
                print(f"Running {num_episodes} episodes in parallel (batch {cur_num_eval//num_envs + 1})")
            
            rets = self._episodic_eval_parallel(num_episodes=num_episodes)
            episode_rewards.append(rets[0])
            if Config.env_type == "smac" or Config.env_type == "smacv2":
                episode_wins.append(rets[1])
                
            cur_num_eval += num_episodes
        
        # Concatenate results from all batches
        episode_rewards = np.concatenate(episode_rewards, axis=0)
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = np.concatenate(episode_wins, axis=0)
        
        # Calculate metrics
        metrics_dict = {
            "average_ep_reward": np.mean(episode_rewards, axis=0),
            "std_ep_reward": np.std(episode_rewards, axis=0),
        }
        
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            metrics_dict["win_rate"] = np.mean(episode_wins)
            
        if self.verbose:
            print("=" * 50)
            print(f"Evaluation Results (Step {load_step}):")
            for k, v in metrics_dict.items():
                print(f"  {k}: {v}")
            print("=" * 50)
            
        # Save results
        os.makedirs(os.path.join(self.log_dir, "results"), exist_ok=True)
        save_file_path = f"results/step_{load_step}-ep_{num_eval}-vae.json"
        
        import json
        with open(os.path.join(self.log_dir, save_file_path), 'w') as f:
            json.dump({
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics_dict.items()
            }, f, indent=2)
            
        return metrics_dict
    
    def _episodic_eval_parallel(self, num_episodes: int):
        """Evaluate multiple episodes in parallel"""
        Config = self.Config
        device = Config.device
        observation_dim = self.normalizer.observation_dim
        
        # Initialize parallel environments
        dones = [0 for _ in range(num_episodes)]
        episode_rewards = [np.zeros(Config.n_agents) for _ in range(num_episodes)]
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            episode_wins = np.zeros(num_episodes)
        
        # Reset all environments in parallel
        obs_list = [env.reset()[None] for env in self.env_list[:num_episodes]]
        obs = np.concatenate(obs_list, axis=0)  # [num_episodes, n_agents, obs_dim]
        
        # Initialize history for all environments
        history_horizon = getattr(Config, 'history_horizon', 1)
        obs_queues = [deque(maxlen=history_horizon + 1) for _ in range(num_episodes)]
        
        # Initialize history with current observations (normalized)
        for i in range(num_episodes):
            normed_obs = self.normalizer.normalize(obs[i], "observations")
            obs_queues[i].extend([normed_obs for _ in range(history_horizon)])
        
        # Initialize variables to track history for trajectory construction
        prev_obs_list = [None for _ in range(num_episodes)]
        prev_actions_list = [None for _ in range(num_episodes)]
        history_trajectory_queues = [deque(maxlen=history_horizon) for _ in range(num_episodes)]
        
        step = 0
        model_action_dim = self.trainer.ema_model.action_dim
        
        if self.verbose:
            print(f"Starting parallel evaluation with {num_episodes} environments")
            print(f"Model action_dim: {model_action_dim}")
            print(f"Observation_dim: {observation_dim}")
            print(f"History_horizon: {history_horizon}")
        
        while sum(dones) < num_episodes and step < Config.max_path_length:
            # Prepare batch observations for all active environments
            batch_obs_history = []
            batch_history_trajectory = []
            active_env_indices = []
            
            for i in range(num_episodes):
                if dones[i] == 1:
                    continue
                    
                active_env_indices.append(i)
                
                # Normalize current observation and add to queue
                obs_normalized = self.normalizer.normalize(obs[i], "observations")
                obs_queues[i].append(obs_normalized)
                
                # Stack observations for model input
                obs_history = np.stack(list(obs_queues[i]), axis=0)  # [history_horizon+1, n_agents, obs_dim]
                batch_obs_history.append(obs_history)
                
                # Build history trajectory for this environment
                if step > 0 and prev_obs_list[i] is not None and prev_actions_list[i] is not None:
                    # Normalize previous observations
                    prev_obs_normalized = self.normalizer.normalize(prev_obs_list[i], "observations")
                    
                    # For discrete actions, we need to handle the action format
                    if self.discrete_action:
                        if len(prev_actions_list[i].shape) == 1:  # [n_agents]
                            prev_actions_formatted = prev_actions_list[i].reshape(Config.n_agents, 1)
                        else:
                            prev_actions_formatted = prev_actions_list[i]
                    else:
                        # For continuous actions, normalize them
                        prev_actions_formatted = self.normalizer.normalize(prev_actions_list[i], "actions")
                    
                    # Combine previous observation and action
                    prev_trajectory_step = np.concatenate([prev_actions_formatted, prev_obs_normalized], axis=-1)
                    prev_trajectory_step = prev_trajectory_step.reshape(Config.n_agents, -1)
                    history_trajectory_queues[i].append(prev_trajectory_step)
                
                # Build history trajectory tensor from queue
                if len(history_trajectory_queues[i]) > 0:
                    history_array = np.stack(list(history_trajectory_queues[i]), axis=0)
                    batch_history_trajectory.append(history_array)
                else:
                    # Create minimal history using current observation
                    minimal_history = np.zeros((max(history_horizon, 1), Config.n_agents, observation_dim + model_action_dim))
                    # Fill the last timestep with current observation
                    minimal_history[-1, :, :observation_dim] = obs_normalized
                    batch_history_trajectory.append(minimal_history)
            
            if not active_env_indices:
                break
                
            # Convert to batch tensors
            batch_obs_history = np.stack(batch_obs_history, axis=0)  # [batch_size, history_horizon+1, n_agents, obs_dim]
            batch_history_trajectory = np.stack(batch_history_trajectory, axis=0)  # [batch_size, history_horizon, n_agents, obs_dim+action_dim]
            
            obs_tensor = to_torch(batch_obs_history, device=Config.device)
            history_trajectory_tensor = to_torch(batch_history_trajectory, device=Config.device)
            
            # Generate actions for all active environments in batch
            with torch.no_grad():
                if hasattr(self.trainer.ema_model, 'get_action'):
                    batch_size = obs_tensor.shape[0]
                    
                    # Handle returns conditioning
                    kwargs = {
                        'obs': obs_tensor[:, -1],  # Use latest observation [batch_size, n_agents, obs_dim]
                        'history_trajectory': history_trajectory_tensor,
                        'deterministic': True
                    }
                    
                    # Add returns conditioning if enabled
                    if hasattr(self.trainer.ema_model, 'returns_condition') and self.trainer.ema_model.returns_condition:
                        return_scale = getattr(Config, 'returns_scale', 600.0)
                        target_return = torch.full((batch_size, 1), return_scale, device=Config.device)
                        kwargs['returns'] = target_return
                    
                    # Generate actions for all agents
                    if hasattr(self.trainer.ema_model, 'decentralized') and self.trainer.ema_model.decentralized:
                        # In decentralized mode, generate actions for each agent separately
                        all_agent_actions = []
                        for agent_idx in range(Config.n_agents):
                            agent_kwargs = kwargs.copy()
                            agent_action = self.trainer.ema_model.get_action(agent_idx=agent_idx, **agent_kwargs)
                            all_agent_actions.append(agent_action)
                        
                        # Concatenate all agent actions
                        actions_tensor = torch.cat(all_agent_actions, dim=0)  # [batch, n_agents * action_dim]
                        # Reshape to [batch, n_agents, action_dim]
                        actions_tensor = actions_tensor.reshape(batch_size, Config.n_agents, -1)
                    
                    actions_batch = to_np(actions_tensor)  # [batch_size, n_agents, action_dim]
            
            # Process actions for each environment
            for batch_idx, env_idx in enumerate(active_env_indices):
                actions = actions_batch[batch_idx]  # [n_agents, action_dim]
                
                # Handle discrete vs continuous actions
                if self.discrete_action:
                    legal_actions = self.env_list[env_idx].get_legal_actions()  # [n_agents, action_dim]
                    actions_masked = actions.copy()
                    actions_masked[np.where(legal_actions.astype(int) == 0)] = -np.inf
                    actions = np.argmax(actions_masked, axis=-1)  # [n_agents]
                else:
                    actions = self.normalizer.unnormalize(actions, "actions")
                
                # Execute action in environment
                next_obs, reward, done, info = self.env_list[env_idx].step(actions)
                episode_rewards[env_idx] += reward
                
                # Save current state for next step's history
                prev_obs_list[env_idx] = obs[env_idx].copy()
                if hasattr(self.trainer.ema_model, 'get_action'):
                    if self.discrete_action:
                        # Convert discrete actions to one-hot for history trajectory
                        actions_onehot = np.zeros((Config.n_agents, model_action_dim))
                        for agent_idx in range(Config.n_agents):
                            action_idx = actions[agent_idx]
                            if 0 <= action_idx < model_action_dim:
                                actions_onehot[agent_idx, action_idx] = 1.0
                        prev_actions_list[env_idx] = actions_onehot
                    else:
                        prev_actions_list[env_idx] = actions.copy()
                
                # Update observation and check if done
                obs[env_idx] = next_obs
                if done.all():
                    dones[env_idx] = 1
                    if Config.env_type == "smac" or Config.env_type == "smacv2" and "battle_won" in info:
                        episode_wins[env_idx] = info["battle_won"]
                        
                    if self.verbose:
                        if Config.env_type == "smac" or Config.env_type == "smacv2":
                            print(f"Episode {env_idx}: reward={episode_rewards[env_idx]}, win={episode_wins[env_idx] if Config.env_type in ['smac', 'smacv2'] else 'N/A'}")
                        else:
                            print(f"Episode {env_idx}: reward={episode_rewards[env_idx]}")
            
            step += 1
        
        # Return results
        episode_rewards = np.array(episode_rewards)
        if Config.env_type == "smac" or Config.env_type == "smacv2":
            return episode_rewards, episode_wins
        else:
            return episode_rewards, 