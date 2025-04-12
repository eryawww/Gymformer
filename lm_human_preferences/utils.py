import torch
import random
import numpy as np
import os

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

def get_device(device_str: str = 'cuda') -> torch.device:
    """
    Get the appropriate torch device based on the provided string and availability.
    
    Args:
        device_str: String specifying the device ('cuda', 'cpu', or 'mps')
        
    Returns:
        torch.device: The appropriate torch device
        
    Raises:
        ValueError: If an invalid device string is provided
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