import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

class LMActorNetwork(nn.Module):
    """
    Actor network for continuous or discrete action spaces.
    """
    def __init__(self, model_name: str, pad_token_id: int):
        super(LMActorNetwork, self).__init__()
        
        self.lm_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.lm_model.config.pad_token_id = pad_token_id
        self.pad_token_id = pad_token_id
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network. Return action probs"""
        assert state.ndim in [1, 2], "State must be a 1D or 2D tensor got : {}".format(state)
        single = (state.ndim == 1)
        if single:
            state = state.unsqueeze(dim=0) # [1, seq_len]
        
        input_ids = state
        attention_mask = input_ids != self.pad_token_id
        logits = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).logits # [batch_size, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        if single:
            return next_token_logits[0] # [vocab_size]
        return next_token_logits # [batch_size, vocab_size]

class LMCriticNetwork(nn.Module):
    """
    Critic network that estimates the value of a state.
    """
    def __init__(self, model_name: str, pad_token_id: int):
        super(LMCriticNetwork, self).__init__()
        
        self.lm_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1
        )
        self.lm_model.config.pad_token_id = pad_token_id
        
        self.pad_token_id = pad_token_id
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network. Return tensor([1])"""
        assert state.ndim in [1, 2], "State must be a 1D or 2D tensor got : {}".format(state)
        single = (state.ndim == 1)
        if single:
            state = state.unsqueeze(dim=0) # [1, seq_len]
        
        input_ids = state
        attention_mask = input_ids != self.pad_token_id
        logits = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).logits # [batch_size, 1]
        logits = logits.squeeze(-1)
        
        if single:
            return logits[0] # 0-D tensor
        return logits # [batch_size]