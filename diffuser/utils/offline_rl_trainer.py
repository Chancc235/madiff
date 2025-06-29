import torch
import numpy as np
from ml_logger import logger
from .training import Trainer


class OfflineRLTrainer(Trainer):
    """
    Trainer for offline reinforcement learning using diffusion models
    This trainer only uses static datasets without environment interaction
    """
    
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer=None,
        target_update_freq=100,    # Target network update frequency
        **kwargs
    ):
        super().__init__(diffusion_model, dataset, renderer, **kwargs)

        self.target_update_freq = target_update_freq
        
        # Separate optimizers for different components
        if hasattr(self.model, 'q_function'):
            self.q_optimizer = torch.optim.Adam(
                self.model.q_function.parameters(), 
                lr=self.optimizer.defaults['lr']
            )
        else:
            self.q_optimizer = None
            
            
    def pretrain_vae(self, n_vae_steps=2000, vae_lr=1e-3):
        """Pre-train VAE before main training"""
        if hasattr(self.model, 'pretrain_vae'):
            # Create a separate dataloader for VAE pretraining
            def cycle_dataloader():
                while True:
                    for data in torch.utils.data.DataLoader(
                        self.dataset,
                        batch_size=self.batch_size,
                        num_workers=0,
                        shuffle=True,
                        pin_memory=True,
                    ):
                        yield data
            
            vae_dataloader = cycle_dataloader()
            
            self.model.pretrain_vae(
                dataloader=vae_dataloader,
                n_vae_steps=n_vae_steps,
                vae_lr=vae_lr,
                device=self.device,
                log_freq=100
            )
        else:
            print("Model does not support VAE pretraining")
    
    def train(self, n_train_steps):
        """Training loop for offline RL"""
        for step_idx in range(n_train_steps):
            # Initialize for this training step
            step_losses = []  # Store loss for each gradient accumulation step
            step_infos = {}   # Store all metrics for this step
            
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = self.batch_to_device(batch)
                
                # Check if model supports decentralized training
                if hasattr(self.model, 'decentralized') and self.model.decentralized:
                    # Decentralized training: iterate over all agents for individual losses
                    agent_loss_tensors = []
                    batch_infos = {}  # Info for this batch
                    
                    for agent_idx in range(self.model.n_agents):
                        # Compute loss for specific agent (without Q-learning and policy optimization)
                        agent_loss, agent_info = self.model.loss(agent_idx=agent_idx, **batch)
                        agent_loss = agent_loss / self.gradient_accumulate_every
                        agent_loss_tensors.append(agent_loss)
                        
                        # Accumulate info for this agent
                        for key, val in agent_info.items():
                            if key not in batch_infos:
                                batch_infos[key] = []
                            batch_infos[key].append(val.item() if torch.is_tensor(val) else val)
                    
                    # Compute average agent loss for this batch
                    avg_agent_loss = sum(agent_loss_tensors) / len(agent_loss_tensors)
                    
                    # Then, compute Q-learning and policy optimization losses for all agents together
                    if hasattr(self.model, 'q_function') and self.model.q_function is not None:
                        # Compute Q-learning and policy losses for all agents (centralized)
                        q_p_loss, q_info = self.model.compute_q_and_policy_losses(**batch)
                        q_p_loss = q_p_loss / self.gradient_accumulate_every
                        
                        # Total loss for this batch
                        batch_total_loss = avg_agent_loss + q_p_loss
                        
                        # Accumulate Q and policy infos
                        for key, val in q_info.items():
                            if key not in batch_infos:
                                batch_infos[key] = []
                            batch_infos[key].append(val.item() if torch.is_tensor(val) else val)
                    else:
                        batch_total_loss = avg_agent_loss
                    
                    # Single backward pass
                    batch_total_loss.backward()
                    
                    # Store loss for this batch
                    step_losses.append(batch_total_loss.item())
                    
                    # Average batch infos across agents and accumulate for this step
                    for key, val_list in batch_infos.items():
                        if key not in step_infos:
                            step_infos[key] = []
                        # Average the values across agents for this batch
                        avg_val = np.mean(val_list)
                        step_infos[key].append(avg_val)

            # Update main model parameters
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Calculate final metrics for this step
            step_loss = np.mean(step_losses)  # Average loss across gradient accumulation steps
            
            # Average all accumulated infos across gradient accumulation steps
            final_infos = {}
            for key, val_list in step_infos.items():
                final_infos[key] = np.mean(val_list)
            
            # Add total loss to final infos
            final_infos['total_loss'] = step_loss
            
            # Target network update
            if self.target_update_freq > 0 and self.step % self.target_update_freq == 0:
                if hasattr(self.model, 'update_target_network'):
                    self.model.update_target_network(tau=self.model.target_update_tau)
                    logger.print(f"[ OfflineRLTrainer ] Updated target network at step {self.step}", color="blue")
            
            # EMA update
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Checkpointing
            if self.step % self.save_freq == 0:
                self.save()

            # Evaluation
            if self.eval_freq > 0 and self.step % self.eval_freq == 0:
                logger.print(f"[ OfflineRLTrainer ] Running evaluation at step {self.step}", color="green")
                if self.evaluator is not None:
                    self.evaluate()
                    logger.print(f"[ OfflineRLTrainer ] Evaluation completed at step {self.step}", color="green")
                else:
                    logger.print(f"[ OfflineRLTrainer ] Warning: evaluator is None, skipping evaluation", color="yellow")

            # Logging
            if self.step % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in final_infos.items()]
                )
                
                full_log = f"{self.step}: {step_loss:8.4f} | {infos_str}"
                    
                logger.print(full_log)
                
                # Log metrics
                metrics = {k: v.detach().item() if torch.is_tensor(v) else v for k, v in final_infos.items()}
                
                logger.log(
                    step=self.step, 
                    loss=step_loss, 
                    **metrics,
                    flush=True
                )

            # Sampling and visualization
            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()

            self.step += 1
    
    def batch_to_device(self, batch):
        """Convert batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        else:
            return batch.to(self.device)
    