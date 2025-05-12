import torch
from torch import Tensor
from typing import Optional

import gymnasium as gym
from torch.nn.utils.rnn import pad_sequence
from lm_human_preferences.rl.policy import LanguageAgent
import numpy as np
from tqdm import tqdm

import os
from torch.utils.tensorboard import SummaryWriter
import datetime

class PPOTrainer:
    def __init__(self, envs: gym.vector.SyncVectorEnv, policy: LanguageAgent,
                 gamma: float = 0.99, lam: float = 0.95, update_epochs: int = 4, batch_size: int = 32,
                 clip_coef: float = 0.2, value_loss_coef: float = 0.5, entropy_loss_coef: float = 0.01,
                 max_grad_norm: float = 0.5, kl_target: Optional[float] = None):
        self.envs = envs
        self.policy = policy
        self.value_model = policy.value_model
        self.device = policy.device
        self.tokenizer = policy.tokenizer
        self.pad_token_id = policy.pad_token_id
        self.eos_token_id = policy.eos_token_id
        self.gamma = gamma
        self.lam = lam
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.clip_coef = clip_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.max_grad_norm = max_grad_norm
        self.kl_target = kl_target

    def train(self, num_iter: int, num_steps: int):
        # Initialize TensorBoard writer for metric tracking
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=os.path.join("runs", f"policy_rl_{timestamp}"))

        optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value_model.parameters()), lr=3e-4) # TODO: parameterize lr

        # These keep track of vectorized environment
        # First dim is time step, second is env index
        obs: list[list[int]] = [] # dynamic len
        actions: list[int] = []
        dones: list[list[bool]] = [] # fixed len
        values: list[list[float]] = [] # fixed len
        rewards: list[list[float]] = [] # fixed len

        global_step = 0
        next_obs, _ = self.envs.reset()
        next_done = np.zeros(self.envs.num_envs, dtype=bool)

        for iteration in tqdm(range(1, num_iter+1)):
            for step in range(1, num_steps+1):
                global_step += self.envs.num_envs
                
                obs.append(next_obs)
                dones.append(next_done)

                with torch.no_grad():
                    action, state_value = self.policy.get_action_and_value(next_obs)
                
                actions.append(action)
                values.append(state_value.tolist())

                next_obs, reward, termination, truncation, _ = self.envs.step(action)
                
                rewards.append(reward)
                next_done = np.logical_or(termination, truncation)
            
            tensor_values: Tensor = torch.tensor(np.array(values), dtype=torch.float32).to(self.device)
            tensor_rewards: Tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
            tensor_dones: Tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
            assert tensor_values.shape == tensor_rewards.shape == tensor_dones.shape, f"value, reward, done shapes do not match: {tensor_values.shape}, {tensor_rewards.shape}, {tensor_dones.shape}"
            assert tensor_values.shape[1] == self.envs.num_envs, f"values shape does not match num_envs: {tensor_values.shape}, [1] should {self.envs.num_envs}"
            
            # 5. Compute advantages and returns
            # advantage = reward + self.gamma * state_value - state_value
            # returns = advantage + value
            with torch.no_grad():
                # TODO: last_hidden_state computation duplication
                last_next_value = self.policy.get_state_value(state=next_obs).to(self.device)
                advantages = torch.zeros_like(tensor_values, dtype=torch.float32).to(self.device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = torch.tensor(1 - next_done).to(self.device)
                        next_value = last_next_value
                    else:
                        nextnonterminal = (1 - tensor_dones[t+1]).detach().to(self.device)
                        next_value = tensor_values[t+1]
                    
                    delta = tensor_rewards[t] + self.gamma * next_value * nextnonterminal - tensor_values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
                returns = advantages + tensor_values
            
            # flatten everything
            flat_obs: Tensor = pad_sequence(
                [torch.tensor(item) for timeobs in obs for item in timeobs], 
                batch_first=True, 
                padding_value=self.policy.tokenizer.pad_token_id
            ).to(self.device)
            # assert len(flat_obs) == num_steps * self.envs.num_envs, f"flat_obs length does not match num_steps * num_envs: {len(flat_obs)} should {num_steps}, {self.envs.num_envs}"
            flat_actions = np.array([item for timeactions in actions for item in timeactions])
            flat_values = tensor_values.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_advantages = advantages.reshape(-1)

            # print('=== FLATTENED ===')
            # print(flat_obs.shape)
            # print(flat_actions.shape)
            # print(flat_values.shape)
            # print(rewards.shape)
            # print(flat_returns.shape)

            # 6. Policy gradient optimization
            # TODO: batch is dynamic in reference
            n_size = num_steps * self.envs.num_envs
            indices = np.arange(n_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, n_size, self.batch_size):
                    end = min(start + self.batch_size, n_size)
                    minibatch_indices = indices[start:end]
                    # print("=== MINIBATCH ===")
                    # print(len(minibatch_indices))
                    
                    new_actions, new_values = self.policy.get_action_and_value(state=flat_obs[minibatch_indices])
                    new_logprobs = torch.stack([ action.logprobs[action.action] for action in new_actions ])
                    new_entropy = torch.stack([ action.entropy for action in new_actions ])

                    # print("=== NEW ACTION")
                    # print(len(new_actions))
                    # print(new_logprobs.shape)
                    # print(new_entropy.shape)

                    old_actions = flat_actions[minibatch_indices]
                    old_logprobs = torch.stack([ action.logprobs[action.action] for action in old_actions ])

                    # Compute r: ratio
                    logratio = new_logprobs - old_logprobs
                    ratio = torch.exp(logratio)

                    # print("=== RATIO")
                    # print(ratio.shape)

                    # Compute KL
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1).abs() > self.clip_coef).float().mean().item()
                        )
                    
                    batch_advantage = flat_advantages[minibatch_indices]
                    # if norm_adv: # TODO: Understand
                    batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

                    # print("=== BATCH ADVANTAGE")
                    # print(batch_advantage.shape)
                    
                    # Policy Loss
                    pg_loss1 = -batch_advantage * ratio
                    pg_loss2 = -batch_advantage * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # print("=== PG LOSS")
                    # print(pg_loss.shape)

                    # Value Loss
                    new_values = new_values.reshape(-1)
                    v_loss_unclipped = (new_values - flat_returns[minibatch_indices]) ** 2
                    v_clipped = flat_values[minibatch_indices] + torch.clamp(
                        new_values - flat_values[minibatch_indices],
                        -self.clip_coef, 
                        self.clip_coef
                    )
                    v_loss_clipped = (v_clipped - flat_returns[minibatch_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    v_loss = v_loss_max.mean() * 0.5

                    # print("=== VALUE LOSS")
                    # print(v_loss.shape)

                    # Entropy
                    entropy_loss = new_entropy.mean()

                    # print("=== ENTROPY LOSS")
                    # print(entropy_loss.shape)

                    loss = pg_loss + self.value_loss_coef * v_loss + self.entropy_loss_coef * entropy_loss
                    
                    # print('=== LOSS ===')
                    # print(loss.shape)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value_model.parameters()), self.max_grad_norm)
                    optimizer.step()

                if self.kl_target is not None and old_approx_kl > self.kl_target:
                    break

            # Log metrics per iteration
            avg_reward = np.array(rewards).flatten().mean()
            writer.add_scalar("metrics/avg_reward", avg_reward, iteration)
            writer.add_scalar("loss/policy", pg_loss.item(), iteration)
            writer.add_scalar("loss/value", v_loss.mean().item(), iteration)
            writer.add_scalar("loss/entropy", entropy_loss.item(), iteration)
            writer.add_scalar("metrics/approx_kl", approx_kl.item(), iteration)
            writer.add_scalar("metrics/clip_fraction", float(sum(clipfracs)/len(clipfracs)), iteration)
            writer.close()
        
        self.envs.close()