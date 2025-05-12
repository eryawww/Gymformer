import torch
import torch.nn as nn

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
        """Forward pass through the network. Return action probs"""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x) # [batch_size, output_dim]
        return logits

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
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value