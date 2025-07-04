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
                lr=self.q_lr
            )
        else:
            self.q_optimizer = None

    
    def train(self, n_train_steps):
        """Training loop for offline RL"""
        for step_idx in range(n_train_steps):
            # Initialize for this training step
            step_losses = []  # Store loss for each gradient accumulation step
            step_infos = {}   # Store all metrics for this step
            # Q-function warm-up for the first step
            if step_idx == 0 and self.q_optimizer is not None:
                # Perform Q-function warm-up iterations
                q_warmup_steps = n_train_steps
                for warmup_i in range(q_warmup_steps):
                    warmup_batch = next(self.dataloader)
                    warmup_batch = self.batch_to_device(warmup_batch)
                    
                    # Only train Q-function during warm-up
                    q_loss, _ = self.model.compute_q_loss(**warmup_batch)
                    
                    self.q_optimizer.zero_grad()
                    q_loss.backward()
                    self.q_optimizer.step()
                    
                    if warmup_i % 100 == 0:
                        print(f"Q warm-up step {warmup_i}/{q_warmup_steps}, Q loss: {q_loss.item():.4f}")
                
                print(f"Q-function warm-up completed after {q_warmup_steps} steps")
                self.model.target_q_function.load_state_dict(self.model.q_function.state_dict())
                print("Target Q-function copied from Q-function")
                self.q_optimizer.zero_grad()
            scale_loss_sum = 0    
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = self.batch_to_device(batch)
                
                # Check if model supports decentralized training
                if hasattr(self.model, 'decentralized') and self.model.decentralized:
                    # Decentralized training: iterate over all agents for individual losses
                    agent_loss_tensors = []
                    batch_infos = {}  # Info for this batch

                    # Compute loss for specific agent (without Q-learning and policy optimization)
                    agent_loss, agent_info = self.model.loss(**batch)
                    scale_agent_loss = agent_loss / self.gradient_accumulate_every
                    scale_loss_sum += scale_agent_loss
                    
                    batch_infos = agent_info
                    batch_total_loss = agent_loss
                    
                    critic_loss, critic_info = self.model.compute_critic_loss(**batch)
                    scale_critic_loss = critic_loss / self.gradient_accumulate_every
                    scale_loss_sum += scale_critic_loss
                    batch_total_loss += critic_loss
                    batch_critic_loss = critic_loss
                    for key, val in critic_info.items():
                        if key not in batch_infos:
                            batch_infos[key] = []
                        batch_infos[key].append(val)
                    
                    # Compute Q-learning and policy losses for all agents (centralized)
                    # q_loss, policy_loss, q_p_info = self.model.compute_q_and_policy_losses(**batch)
                    # scale_q_loss = q_loss / self.gradient_accumulate_every
                    # scale_policy_loss = policy_loss / self.gradient_accumulate_every
                    # scale_loss_sum += scale_policy_loss

                    # # Total loss for this batch
                    # batch_total_loss += policy_loss
                    # batch_q_loss = q_loss
                    # Accumulate Q and policy infos
                    # for key, val in q_p_info.items():
                    #     if key not in batch_infos:
                    #         batch_infos[key] = []
                    #     batch_infos[key].append(val)
                    
                    # Store loss for this batch
                    step_losses.append(batch_total_loss.item())
                    
                    # Average batch infos across agents and accumulate for this step
                    for key, val_list in batch_infos.items():
                        if key not in step_infos:
                            step_infos[key] = []
                        avg_val = np.mean(val_list)
                        
                        step_infos[key].append(avg_val)
            scale_loss_sum.backward()
            # scale_q_loss.backward()
            # Update main model parameters
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # self.q_optimizer.step()
            self.optimizer.zero_grad()
            # self.q_optimizer.zero_grad()
            
            # Calculate final metrics for this step
            step_loss = np.mean(step_losses)  # Average loss across gradient accumulation steps
            
            # Average all accumulated infos across gradient accumulation steps
            final_infos = {}
            for key, val_list in step_infos.items():
                final_infos[key] = np.mean(val_list)
            
            # Add total loss to final infos
            final_infos['total_loss'] = step_loss
            
            # Target network update
            # if self.target_update_freq > 0 and self.step % self.target_update_freq == 0:
            #     if hasattr(self.model, 'update_target_network'):
            #         self.model.update_target_network(tau=self.model.target_update_tau)
            #         # logger.print(f"[ OfflineRLTrainer ] Updated target network at step {self.step}", color="blue")
            
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
    