import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical, Normal
from typing import Tuple, List, Dict, Optional, Union, Callable

# Set device to cpu or cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RolloutBuffer:
    """
    Buffer to store experiences collected during rollouts.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logprobs.clear()
        self.state_values.clear()
        self.is_terminals.clear()

import copy

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
        config: PPOConfig, 
        device: str = 'cuda',
    ):
        """
        Initialize the PPO agent.
        
        Args:
            config: PPOConfig, 
            device: str = 'cuda',
        ):
        """
        self.config = config
        self.device = device
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.update_epochs = config.update_epochs
        self.minibatch_size = config.minibatch_size
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic
        self.pad_obs = config.pad_obs
        self.int_obs = config.int_obs
        
        self.buffer = RolloutBuffer()
        
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
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
            state = torch.tensor(state, dtype=torch.int32 if self.int_obs else torch.float32).to(device)            
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # Calculate and normalize advantages
        advantages = rewards - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Optimize policy for K epochs
        for k in range(self.update_epochs):
            # indexes = torch.randperm(len(rewards))
            if self.minibatch_size == -1:
                minibatch_size = len(rewards)
            else:
                minibatch_size = self.minibatch_size
                
            # for i in range(0, len(rewards), minibatch_size):
                # batch_indexes = indexes[i:i + minibatch_size]
            # b_old_states = old_states[i:i + minibatch_size]
            # b_old_actions = old_actions[i:i + minibatch_size]
            # b_old_logprobs = old_logprobs[i:i + minibatch_size]
            # b_rewards = rewards[i:i + minibatch_size]
            # b_advantages = advantages[i:i + minibatch_size]
            b_old_states = old_states
            b_old_actions = old_actions
            b_old_logprobs = old_logprobs
            b_rewards = rewards
            b_advantages = advantages

            # Evaluate old actions and values
            b_logprobs, b_state_values, dist_entropy = self.evaluate(b_old_states, b_old_actions)
            
            # Match dimensions with rewards
            if b_state_values.dim() > 1:
                b_state_values = b_state_values.squeeze(-1)
            
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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        if self.pad_obs:
            # Import pad_sequence if needed for padding observations
            from torch.nn.utils.rnn import pad_sequence
            old_states = pad_sequence(states, batch_first=True, padding_value=actor.pad_token_id, padding_side='left').to(device)
        else:
            old_states = torch.stack(states, dim=0).to(device)
        old_actions = torch.stack(actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(logprobs, dim=0).detach().to(device)
        old_state_values = torch.stack(state_values, dim=0).squeeze().detach().to(device)
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
    
    def load(self, checkpoint_path: str):
        """
        Load model parameters.
        
        Args:
            checkpoint_path: Path to load the model from
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_old.load_state_dict(checkpoint['actor_old'])

def train(env_name: str, actor: nn.Module, critic: nn.Module, ppo_config: PPOConfig, max_episodes: int = 1000, max_timesteps: int = 1000, 
        update_interval: int = 100, save_interval: int = 100, callback: Optional[Callable] = None,
        log_interval: int = 10, device='cuda') -> List[float]:
    """
    Train a PPO agent in a given environment.
    
    Args:
        env_name: Name of the gym environment
        max_episodes: Maximum number of episodes for training
        max_timesteps: Maximum timesteps in one episode
        update_interval: Update policy every n episodes
        save_interval: Save model in the interval
    
    Returns:
        List of average rewards per episode
    """
    # Create environment
    env = gym.make(env_name)
    
    # Initialize PPO agent
    ppo_agent = PPO(
        actor,
        critic,
        ppo_config,
        device,
    )
    
    # Logging variables
    running_reward = 0
    time_step = 0
    
    # Training loop
    episode_rewards = []
    
    for episode in range(1, max_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(1, max_timesteps+1):
            # Select action
            action = ppo_agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience in buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step += 1
            episode_reward += reward

            state = next_state
            
            if done or truncated:
                break
        
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        episode_rewards.append(running_reward)

        if episode % log_interval == 0:
            if callback is not None:
                callback(state, action, reward, done, truncated, episode, running_reward)
            print(f"Episode {episode}, Running Reward: {running_reward:.2f}")

        # Update if its time
        if episode % update_interval == 0:
            # Make sure we have enough data in the buffer before updating
            if len(ppo_agent.buffer.states) > 0:  # Only update if buffer contains experiences
                ppo_agent.update()
            else:
                print(f"Warning: Episode {episode} - Buffer is empty, skipping update")
        
        # Save model
        if episode % save_interval == 0:
            ppo_agent.save(f"./PPO_{env_name}_{episode}.pth")
    
    env.close()
    return episode_rewards

class MLPActorNetwork(nn.Module):
    """
    Actor network using a Multi-Layer Perceptron for continuous or discrete action spaces.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network. Return action logits"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # [batch_size, output_dim]
        if x.dim() == 1:
            return x[0, :]
        return x

class MLPCriticNetwork(nn.Module):
    """
    Critic network using a Multi-Layer Perceptron to estimate the value of a state.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(MLPCriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network. Return state value"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if x.dim() == 1:
            return x.item()
        return x[:, -1]

env_name = 'CartPole-v1'
env = gym.make(env_name)

actor = MLPActorNetwork(
    env.observation_space.shape[0],
    32,
    env.action_space.n,
)
critic = MLPCriticNetwork(
    env.observation_space.shape[0],
    32,
)

ppo_config = PPOConfig(
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    eps_clip=0.2,
    update_epochs=10,
    minibatch_size=-1
)
rewards_discrete = train(
    env_name=env_name,
    actor=actor,
    critic=critic,
    ppo_config=ppo_config,
    max_episodes=3_000, 
    max_timesteps=1000, 
    update_interval=80, 
    save_interval=1000,
    log_interval=100
)