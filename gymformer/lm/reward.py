import logging
from dataclasses import dataclass, asdict
from typing import override, Optional, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
from datasets import Dataset, concatenate_datasets
import pandas as pd
import swifter
from tqdm import tqdm
import wandb
import numpy as np
import os

from gymformer.utils import set_seed, DeviceManager

log = logging.getLogger(__name__)

@dataclass
class RewardOutput(ModelOutput):
    loss: torch.FloatTensor
    scores: torch.FloatTensor = None

class RewardModelWrapper(nn.Module):
    """
    Training wrapper for reward model that handles preference learning.
    The actual model is a standard transformers classification model.

    Implementation of : Fine-Tuning Language Models from Human Preferences (https://arxiv.org/pdf/1909.08593)
    
    Args:
        base_model: str, name of the base model to use
        trained_tokenizer: Optional[AutoTokenizer], pre-trained tokenizer
        trained_model: Optional[AutoModelForSequenceClassification], pre-trained model
        seed: int, seed for reproducibility
        device: str, device to use
    """
    def __init__(
        self, 
        base_model: str, 
        trained_tokenizer: Optional[AutoTokenizer] = None, 
        trained_model: Optional[AutoModelForSequenceClassification] = None, 
        seed: int = 42, 
        device: str = 'cuda'
    ):
        log.info('Instantiating RewardModel training wrapper')
        super().__init__()
        self.base_model = base_model
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Set device using DeviceManager
        self.device_manager = DeviceManager()
        self.device = self.device_manager.set_device(device)
        log.info(f'Using device: {self.device}')
        
        # Initialize tokenizer and model
        self.tokenizer = (
            trained_tokenizer if trained_tokenizer else 
            AutoTokenizer.from_pretrained(base_model)
        )
        self.model = (
            trained_model if trained_model else 
            AutoModelForSequenceClassification.from_pretrained(
                base_model,
                num_labels=1,
                problem_type='regression'
            )
        )
        
        # Configure model for training
        self._freeze_transformers()
        
        # Move model to device
        self.model.to(self.device)
    
    def _freeze_transformers(self):
        """Configure model for reward training."""
        # Freeze transformer layers
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        # Ensure head is trainable
        for param in self.model.score.parameters():
            param.requires_grad = True
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        

    def forward(
        self, 
        query: dict[str, torch.Tensor], 
        samples: tuple[dict[str, torch.Tensor]], 
        best: torch.Tensor
    ) -> RewardOutput:
        """
        This function computes the loss for one batch of data.
        
        Map x, y1, y2, y3, y4, best -> loss (real number)
        
        Algorithm:
            For each yi, compute real value logits r(x, yi) 
            loss = log frac{ exp(r(x, y_best)) }{ sum_i exp(r(x, y_i)) }
        
        Args:
            query: dict with 'input_ids' and 'attention_mask', shape (batch_size, seq_len)
            samples: tuple of dicts with 'input_ids' and 'attention_mask', shape (batch_size, seq_len)
            best: int tensor, shape (batch_size)
            
        Returns:
            RewardOutput with loss and scores
        """
        assert query["input_ids"].size(0) == best.size(0), 'Batch size must be same for all inputs'
        batch_size = query["input_ids"].size(0)
        sample_size = len(samples)

        # 1. Combine query and samples, flatten to generate (batch_size * sample_size, seq_len) so we could forward in paralel
        combined_query_samples_list, combined_attention_samples_list = [], []
        for i_batch in range(batch_size):
            for sample in samples:
                combined = self._combine_query_sample(query["input_ids"][i_batch], sample["input_ids"][i_batch])
                combined_query_samples_list.append(combined)
                
                combined_attention = (combined != self.tokenizer.pad_token_id).long()
                combined_attention_samples_list.append(combined_attention)

        try:
            flat_ids = torch.stack(combined_query_samples_list, dim=0)
            flat_att = torch.stack(combined_attention_samples_list, dim=0)
        except RuntimeError:
            # sequences have varying lengths; pad them
            flat_ids = pad_sequence(combined_query_samples_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            flat_att = pad_sequence(combined_attention_samples_list, batch_first=True, padding_value=0)
        
        flat_ids = flat_ids.to(self.device) # (batch_size * sample_size, seq_len)
        flat_att = flat_att.to(self.device) # (batch_size * sample_size, seq_len)

        # Forward to model
        scores = self.model(
            input_ids=flat_ids, 
            attention_mask=flat_att
        ).logits # (batch_size * sample_size, 1)

        scores = scores.view(batch_size, sample_size) # (batch_size, sample_size)
        
        selected = torch.gather(scores, 1, best.unsqueeze(1)) # (batch_size, 1)
        logsum = torch.logsumexp(scores, dim=1, keepdim=True) # (batch_size, 1)
        loss = -(selected - logsum).mean()
        
        return RewardOutput(loss=loss, scores=scores)
    
    @override
    def __getattr__(self, name: str):
        """Delegate attribute lookup to model for optimized implementation."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def _combine_query_sample(self, query: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
            Args:
                query: torch.tensor, shape (batch_size, seq_len)
                sample: torch.tensor, shape (batch_size, seq_len)
            Returns:
                combined: torch.tensor, shape (batch_size, seq_len)

            Combine query and samples prompt, remove pad inbetween
            Query format is [Q1, Q2, ..., PAD, PAD]
            Sample format is [S1, S2, ..., PAD, PAD]
            Output format is [Q1, Q2, ..., S1, S2, PAD, PAD]
        """
        assert query.dim() == 1 and sample.dim() == 1, 'Query and Sample must be 1D tensor'
        # BUG: On some datasets, there exists case where continuation is on subword level 
        #   Query: "... work" Sample: "ing ...", "working" should be single token instead of two tokens concatenated
        # in this case, decode into string and then re-encode is the general solution
        
        # Find the first padding token in query and sample
        concatenated_result = torch.full((len(query) + len(sample),), self.tokenizer.pad_token_id)
        q_valid = query[query != self.tokenizer.pad_token_id]
        s_valid = sample[sample != self.tokenizer.pad_token_id]
        
        # Concatenate valid tokens
        concatenated_result[:len(q_valid)] = q_valid
        concatenated_result[len(q_valid):len(q_valid) + len(s_valid)] = s_valid
        return concatenated_result
        
    def save(self, save_path: str):
        """Save the transformers model and tokenizer.
        
        Args:
            save_path: Path to save the model
        """
        log.info(f"Saving reward model to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save the transformers model and tokenizer only
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        log.info(f"Reward model saved successfully to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cuda') -> AutoModelForSequenceClassification:
        """Load a transformers model from the specified path.
        
        Args:
            load_path: Path to load the model from
            device: Device to load the model on
        
        Returns:
            AutoModelForSequenceClassification: The loaded transformers model
        """
        log.info(f"Loading reward model from {load_path}")
        
        # Load the transformers model
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        
        # Move to device using DeviceManager
        device_manager = DeviceManager()
        device_obj = device_manager.set_device(device)
        model.to(device_obj)
        
        log.info(f"Reward model loaded successfully from {load_path}")
        return model