"""
Proximal Policy Optimization (PPO) Implementation in PyTorch
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union

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

class ActorNetwork(nn.Module):
    """
    Actor network for continuous or discrete action spaces.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, 
                 continuous: bool = False, action_std_init: float = 0.6):
        super(ActorNetwork, self).__init__()
        
        self.continuous = continuous
        
        # Common layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers based on action space type
        if continuous:
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.action_std = nn.Parameter(torch.full((action_dim,), action_std_init))
        else:
            self.categorical = nn.Linear(hidden_dim, action_dim)
            
    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        if self.continuous:
            action_mean = self.mean_layer(x)
            action_std = self.action_std.expand_as(action_mean)
            return action_mean, action_std
        else:
            action_probs = torch.softmax(self.categorical(x), dim=-1)
            return action_probs
    
    def set_action_std(self, new_action_std: float):
        """Set the action standard deviation (for continuous action spaces)."""
        if self.continuous:
            self.action_std = nn.Parameter(torch.full(self.action_std.shape, new_action_std))
        else:
            print("WARNING: Calling set_action_std on discrete action space policy")

class CriticNetwork(nn.Module):
    """
    Critic network that estimates the value of a state.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        
        return value

class PPO:
    """
    Proximal Policy Optimization algorithm with clipped objective.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        K_epochs: int = 10,
        eps_clip: float = 0.2,
        has_continuous_action_space: bool = False,
        action_std_init: float = 0.6,
        hidden_dim: int = 64
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor
            K_epochs: Number of epochs to update policy
            eps_clip: Clip parameter for PPO
            has_continuous_action_space: Whether the action space is continuous
            action_std_init: Initial standard deviation for action distribution
            hidden_dim: Hidden dimension size for neural networks
        """
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Initialize buffer
        self.buffer = RolloutBuffer()
        
        # Initialize actor
        self.actor = ActorNetwork(
            state_dim, 
            action_dim, 
            hidden_dim=hidden_dim, 
            continuous=has_continuous_action_space, 
            action_std_init=action_std_init
        ).to(device)
        
        # Initialize critic
        self.critic = CriticNetwork(state_dim, hidden_dim=hidden_dim).to(device)
        
        # Set optimizers for both networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize old policy (for importance sampling)
        self.actor_old = ActorNetwork(
            state_dim, 
            action_dim, 
            hidden_dim=hidden_dim, 
            continuous=has_continuous_action_space, 
            action_std_init=action_std_init
        ).to(device)
        
        # Copy parameters from current policy to old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Loss function for value network
        self.MseLoss = nn.MSELoss()
        
        # Set initial action standard deviation
        if has_continuous_action_space:
            self.action_std = action_std_init
    
    def set_action_std(self, new_action_std: float):
        """
        Set the action standard deviation for continuous action spaces.
        
        Args:
            new_action_std: New standard deviation value
        """
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.actor.set_action_std(new_action_std)
            self.actor_old.set_action_std(new_action_std)
        else:
            print("WARNING: Calling set_action_std on discrete action space policy")
    
    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float):
        """
        Decay the action standard deviation based on the decay rate.
        
        Args:
            action_std_decay_rate: Rate at which to decay the std
            min_action_std: Minimum std value
        """
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = max(self.action_std, min_action_std)
            self.set_action_std(self.action_std)
            print(f"Setting actor output action_std to: {self.action_std}")
        else:
            print("WARNING: Calling decay_action_std on discrete action space policy")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            
            if self.has_continuous_action_space:
                action_mean, action_std = self.actor_old(state)
                action_dist = Normal(action_mean, action_std)
            else:
                action_probs = self.actor_old(state)
                action_dist = Categorical(action_probs)
            
            action = action_dist.sample()
            action_logprob = action_dist.log_prob(action)
            state_val = self.critic(state)
            
            # Store experience in buffer
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            
            if self.has_continuous_action_space:
                return action.detach().cpu().numpy().flatten()
            else:
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
        if self.has_continuous_action_space:
            action_mean, action_std = self.actor(states)
            action_dist = Normal(action_mean, action_std)
            
            # For continuous action spaces, calculate the log probability of each action dimension
            if actions.dim() > action_mean.dim():
                actions = actions.squeeze(-1)
            
            action_log_probs = action_dist.log_prob(actions).sum(dim=1)
            dist_entropy = action_dist.entropy().sum(dim=1).mean()
            
        else:
            action_probs = self.actor(states)
            action_dist = Categorical(action_probs)
            
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
        
        # Compute discounted rewards
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Convert lists to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).squeeze().detach().to(device)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Calculate advantages
        advantages = rewards - old_state_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            
            # Match dimensions with rewards
            state_values = state_values.squeeze()
            
            # Calculate ratios for importance sampling
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # PPO clipped objective with value function loss and entropy bonus
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards)
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
        
        # Clear buffer
        self.buffer.clear()
    
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

def train(env_name: str, max_episodes: int = 1000, max_timesteps: int = 1000, 
          update_timestep: int = 4000, log_interval: int = 20, save_interval: int = 100,
          has_continuous_action_space: bool = False, action_std_init: float = 0.6) -> List[float]:
    """
    Train a PPO agent in a given environment.
    
    Args:
        env_name: Name of the gym environment
        max_episodes: Maximum number of episodes for training
        max_timesteps: Maximum timesteps in one episode
        update_timestep: Update policy every n timesteps
        log_interval: Print avg reward in the interval
        save_interval: Save model in the interval
        has_continuous_action_space: Whether the environment has continuous action space
        action_std_init: Initial action standard deviation
    
    Returns:
        List of average rewards per episode
    """
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=action_std_init
    )
    
    # Logging variables
    running_reward = 0
    avg_length = 0
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
            
            # Update if its time
            if time_step % update_timestep == 0:
                ppo_agent.update()
            
            if done or truncated:
                break
        
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        episode_rewards.append(running_reward)
        
        # Logging
        if episode % log_interval == 0:
            print(f"Episode: {episode}, Avg. Reward: {running_reward:.2f}")
        
        # Save model
        if episode % save_interval == 0:
            ppo_agent.save(f"./PPO_{env_name}_{episode}.pth")
    
    env.close()
    return episode_rewards

def plot_learning_curve(rewards: List[float], title: str = "Learning Curve"):
    """
    Plot the learning curve based on rewards.
    
    Args:
        rewards: List of rewards
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    # For discrete action space (CartPole-v1)
    rewards_discrete = train(
        env_name="CartPole-v1",
        max_episodes=500,
        max_timesteps=1000,
        has_continuous_action_space=False
    )
    plot_learning_curve(rewards_discrete, "CartPole-v1 Learning Curve")
    
    # For continuous action space (Pendulum-v1)
    rewards_continuous = train(
        env_name="Pendulum-v1",
        max_episodes=1000,
        max_timesteps=1000,
        has_continuous_action_space=True,
        action_std_init=0.6
    )
    plot_learning_curve(rewards_continuous, "Pendulum-v1 Learning Curve")
