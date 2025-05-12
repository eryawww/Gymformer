import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Callable
from torch.distributions import Categorical
from gymformer.utils import RolloutBuffer, DeviceManager

class PPOConfig:
    # TODO: Investigate the following hyperparameter clip_coef, vf_coef, ent_coef, norm_adv, max_grad_norm
    def __init__(
        self,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        update_epochs: int = 10,
        eps_clip: float = 0.2,
        minibatch_size: int = 8,
        pad_obs: bool = False,
        int_obs: bool = False,
    ):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.update_epochs = update_epochs
        self.eps_clip = eps_clip
        self.minibatch_size = minibatch_size
        self.pad_obs = pad_obs
        self.int_obs = int_obs

class PPO:
    """
    Proximal Policy Optimization algorithm with clipped objective.
    """
    def __init__(
        self, 
        actor: nn.Module,
        critic: nn.Module,
        config: PPOConfig = PPOConfig(), 
        device: str = 'cuda',
    ):
        """
        Initialize the PPO agent.
        
        Args:
            config: PPOConfig, 
            device: str = 'cuda',
        """
        self.config = config
        self.device_manager = DeviceManager()
        self.device = self.device_manager.set_device(device)
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.update_epochs = config.update_epochs
        self.minibatch_size = config.minibatch_size
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic
        self.pad_obs = config.pad_obs
        self.int_obs = config.int_obs
        
        self.buffer = RolloutBuffer()
        
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        self.actor_old = copy.deepcopy(self.actor)
        self.actor_old.requires_grad_(False)
        
        self.mse_loss = nn.MSELoss()
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            # TODO: Dynamically determine the dtype based on the observation space
            state = torch.tensor(state, dtype=torch.int32 if self.int_obs else torch.float32).to(self.device)            
            action_logits = self.actor_old(state)
            action_dist = Categorical(logits=action_logits)
            
            action = action_dist.sample()
            action_logprob = action_dist.log_prob(action)
            state_val = self.critic(state)
            
            # Store experience in buffer
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            
            return action.item()
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given states using the current policy.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            log_probs, state_values, dist_entropy
        """
        action_logits = self.actor(states)
        action_dist = Categorical(logits=action_logits)
        
        action_log_probs = action_dist.log_prob(actions)
        dist_entropy = action_dist.entropy().mean()
        
        state_values = self.critic(states)
        
        return action_log_probs, state_values, dist_entropy
    
    def update(self):
        """
        Update policy and value networks using PPO algorithm.
        """
        # Calculate discounted rewards and advantages
        rewards = []
        discounted_reward = 0
        
        # Compute GAE
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        _, old_states, old_actions, old_logprobs, old_state_values = self._to_tensor(
            self.buffer.rewards, 
            self.buffer.states, 
            self.buffer.actions, 
            self.buffer.logprobs, 
            self.buffer.state_values
        )
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # Calculate and normalize advantages
        advantages = rewards - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        T = rewards.size(0)
        for name, tensor in [
            ("rewards", rewards),
            ("old_actions", old_actions),
            ("old_logprobs", old_logprobs),
            ("old_state_values", old_state_values)
        ]:
            assert tensor.dim() == 1, (
                f"{name} must be 1-D of length {T}, but got shape {tuple(tensor.shape)}"
            )
            assert tensor.size(0) == T, (
                f"{name} length {tensor.size(0)} != expected {T}"
            )
        assert advantages.dim() == 1 and advantages.size(0) == T, (
            f"advantages must be 1-D of length {T}, got {tuple(advantages.shape)}"
        )

        # Optimize policy for K epochs
        for k in range(self.update_epochs):
            indexes = torch.randperm(len(rewards))
            if self.minibatch_size == -1:
                minibatch_size = len(rewards)
            else:
                minibatch_size = self.minibatch_size
                
            for i in range(0, len(rewards), minibatch_size):
                batch_indexes = indexes[i:i + minibatch_size]
                b_old_states = old_states[batch_indexes]
                b_old_actions = old_actions[batch_indexes]
                b_old_logprobs = old_logprobs[batch_indexes]
                b_rewards = rewards[batch_indexes]
                b_advantages = advantages[batch_indexes]
                # b_old_states = old_states
                # b_old_actions = old_actions
                # b_old_logprobs = old_logprobs
                # b_rewards = rewards
                # b_advantages = advantages

                # Evaluate old actions and values
                b_logprobs, b_state_values, dist_entropy = self.evaluate(b_old_states, b_old_actions)
                b_state_values = b_state_values.squeeze(-1)
                
                for name, tensor in [
                    ("b_logprobs", b_logprobs),
                    ("b_state_values", b_state_values)
                ]:
                    assert tensor.dim() == 1 and tensor.size(0) == minibatch_size or tensor.size(0) == len(rewards) % minibatch_size, (
                        f"{name} must be 1-D of length {minibatch_size} or {len(rewards) % minibatch_size}, got {tuple(tensor.shape)}"
                    )
                
                # Calculate ratios for importance sampling
                ratios = torch.exp(b_logprobs - b_old_logprobs)
                
                # Calculate surrogate losses
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * b_advantages
                
                # PPO clipped objective with value function loss and entropy bonus
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(b_state_values, b_rewards)
                entropy_loss = -0.01 * dist_entropy  # Entropy bonus for exploration
                
                # Combined loss
                loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                # Take gradient step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # Update old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear()
    
    def _to_tensor(self, rewards: List[float], states: List[np.ndarray], actions: List[np.ndarray], logprobs: List[np.ndarray], state_values: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if self.pad_obs:
            # Import pad_sequence if needed for padding observations
            from torch.nn.utils.rnn import pad_sequence
            old_states = pad_sequence(states, batch_first=True, padding_value=self.actor.pad_token_id, padding_side='left').to(self.device)
        else:
            old_states = torch.stack(states, dim=0).to(self.device)
        old_actions = torch.stack(actions, dim=0).detach().to(self.device)
        old_logprobs = torch.stack(logprobs, dim=0).detach().to(self.device)
        old_state_values = torch.stack(state_values, dim=0).squeeze().detach().to(self.device)
        return rewards, old_states, old_actions, old_logprobs, old_state_values
    
    def save(self, checkpoint_path: str):
        """
        Save model parameters.
        
        Args:
            checkpoint_path: Path to save the model
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_old': self.actor_old.state_dict(),
        }, checkpoint_path)
    
    @staticmethod
    def load(checkpoint_path: str, actor: nn.Module, critic: nn.Module, device: Optional[str] = None) -> Tuple[nn.Module, nn.Module]:
        """
        Load actor and critic networks from checkpoint.
        
        Args:
            checkpoint_path: Path to load the model from
            device: Device to load the model on. If None, uses DeviceManager's default.
            
        Returns:
            Tuple[nn.Module, nn.Module]: A tuple containing (actor_network, critic_network)
        """
        # Initialize device
        device_manager = DeviceManager()
        device_obj = device_manager.set_device(device) if device else device_manager.get_device()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device_obj)
        actor.load_state_dict(checkpoint['actor'])
        critic.load_state_dict(checkpoint['critic'])
        
        # Return the networks
        return PPO(actor, critic, device=device)