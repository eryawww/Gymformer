import torch
import random
import numpy as np
import os

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
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx) -> tuple:
        self.__assert_all_equal_size()
        return self.states[idx], self.actions[idx], self.rewards[idx], self.logprobs[idx], self.state_values[idx], self.is_terminals[idx]
    
    def shuffle(self) -> None:
        self.__assert_all_equal_size()
        
        indices = list(range(len(self.states)))
        random.shuffle(indices)
        
        self.states = [self.states[i] for i in indices]
        self.actions = [self.actions[i] for i in indices]
        self.rewards = [self.rewards[i] for i in indices]
        self.logprobs = [self.logprobs[i] for i in indices]
        self.state_values = [self.state_values[i] for i in indices]
        self.is_terminals = [self.is_terminals[i] for i in indices]

    def sample(self, n: int) -> tuple:
        self.__assert_all_equal_size()
        
        indices = random.sample(list(range(len(self.states))), n)

        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        logprobs = [self.logprobs[i] for i in indices]
        state_values = [self.state_values[i] for i in indices]
        is_terminals = [self.is_terminals[i] for i in indices]
        
        return states, actions, rewards, logprobs, state_values, is_terminals
    
    def __assert_all_equal_size(self):
        assert len(self.states) == len(self.actions) == len(self.rewards) == len(self.logprobs) == len(self.state_values) == len(self.is_terminals), \
            "Buffer states, actions, rewards, logprobs, state_values, and is_terminals must have the same length got {}".format(
                len(self.states), 
                len(self.actions), 
                len(self.rewards), 
                len(self.logprobs), 
                len(self.state_values), 
                len(self.is_terminals)
            )

def gpt2_replace_unk_to_pad(query_tokens: torch.Tensor, pad_token: int) -> torch.Tensor:
    """
        Special handle function because GPT default doesn't have any pad tokens
        in this dataset, they pad with 50259 ('') which we need to remove for query before concatenating
    """
    query_tokens[query_tokens == 50259] = pad_token
    return query_tokens

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across all frameworks used in the project.
    
    Args:
        seed: Integer seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

class DeviceManager:
    """
    Singleton class to manage device selection and state.
    """
    _instance = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._device is None:
            self._device = self._initialize_device()

    def _initialize_device(self, device_str: str = 'cuda') -> torch.device:
        """
        Initialize the device based on availability and preference.
        """
        if device_str == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("CUDA is not available, falling back to CPU.")
                return torch.device('cpu')
        elif device_str == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("MPS is not available, falling back to CPU.")
                return torch.device('cpu')
        elif device_str == 'cpu':
            return torch.device('cpu')
        else:
            raise ValueError(f"Invalid device: {device_str}. Choose from 'cuda', 'cpu', or 'mps'.")

    def get_device(self) -> torch.device:
        """
        Get the current device.
        """
        return self._device

    def set_device(self, device_str: str) -> torch.device:
        """
        Set a new device and return it.
        """
        self._device = self._initialize_device(device_str)
        return self._device