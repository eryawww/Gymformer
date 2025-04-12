from typing import override
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import logging

from .utils import set_seed, get_device

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class RewardModel(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.1, seed: int = 42, device: str = 'cuda'):
        log.info('Instantiating RewardModel')
        super().__init__()
        self.base_model = base_model
        self.seed = seed
        self.device_str = device
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Set device
        self.device = get_device(self.device_str)
        log.info(f'Using device: {self.device}')
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)

        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_size, 1)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Move model components to the specified device
        self.model.to(self.device)
        self.score.to(self.device)

        # Accelerator support used in PPOTrainer
        # self.base_model_prefix = self.model.base_model_prefix
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.model, name): # Inherit everything from main
                return getattr(self.model, name)
            raise AttributeError(f"Attribute {name} not found in RewardModel or its submodules.")

    
    def forward(self, query: dict[str, torch.Tensor], sample0: dict[str, torch.Tensor], sample1: dict[str, torch.Tensor], sample2: dict[str, torch.Tensor], sample3: dict[str, torch.Tensor], best: torch.Tensor) -> torch.Tensor:
        """
            This function computes the loss for one batch of data.
            
            Map x, y1, y2, y3, y4, best -> loss (real number)
            
            Algorithm:
                For each yi, compute real value logits r(x, yi) 
                loss = log frac{ exp(r(x, y_best)) }{ sum_i exp(r(x, y_i)) }
        
            Args:
                query: dict[str, torch.Tensor] with keys 'input_ids' and 'attention_mask', shape (batch_size, seq_len)
                sample0: dict[str, torch.Tensor] with keys 'input_ids' and 'attention_mask', shape (batch_size, seq_len)
                sample1: dict[str, torch.Tensor] with keys 'input_ids' and 'attention_mask', shape (batch_size, seq_len)
                sample2: dict[str, torch.Tensor] with keys 'input_ids' and 'attention_mask', shape (batch_size, seq_len)
                sample3: dict[str, torch.Tensor] with keys 'input_ids' and 'attention_mask', shape (batch_size, seq_len)
                best: int, shape (batch_size)
            Returns:
                loss: torch.tensor, shape (1)
        """
        log.debug(f'Forward with Query: {query["input_ids"].shape} Samples: {[sample0["input_ids"].shape, sample1["input_ids"].shape, sample2["input_ids"].shape, sample3["input_ids"].shape]} Best: {best.shape}')
        assert query["input_ids"].size(0) == sample0["input_ids"].size(0) == sample1["input_ids"].size(0) == sample2["input_ids"].size(0) == sample3["input_ids"].size(0) == best.size(0), 'Batch size must be same for all inputs'
        batch_size = query["input_ids"].size(0)

        combined_query_samples_list = []
        combined_attention_samples_list = []
        for i_batch in range(batch_size):
            for sample in [sample0, sample1, sample2, sample3]:
                # print(query["input_ids"].shape, sample["input_ids"].shape)
                combined = self._combine_query_sample(query["input_ids"][i_batch], sample["input_ids"][i_batch])
                combined_query_samples_list.append(combined)
                
                combined_attention = self._combine_query_sample(query["attention_mask"][i_batch], sample["attention_mask"][i_batch])
                combined_attention_samples_list.append(combined_attention)

        combined_query_samples = torch.stack(combined_query_samples_list, dim=0) 
        combined_attention_samples = torch.stack(combined_attention_samples_list, dim=0)
        log.debug(f'Combined Query Padded Sequence: {combined_query_samples.shape}') # (batch_size * sample_size, seq_len)
        log.debug(f'Combined Attention Samples: {combined_attention_samples.shape}') # (batch_size * sample_size, seq_len)

        # Move combined tensors to device
        combined_query_samples = combined_query_samples.to(self.device)
        combined_attention_samples = combined_attention_samples.to(self.device)

        log.debug(f'Running Forward on Model')
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=combined_query_samples, attention_mask=combined_attention_samples, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :] # use vector from last token of last layer
        
        log.debug(f'Hidden States: {hidden_states.shape}') # (batch_size * sample_size, hidden_size)
        hidden_states = self.dropout(hidden_states)
        output = self.score(hidden_states) # (batch_size * sample_size, 1)
        log.debug(f'Output Flatten: {output.shape}')
        output = output.view(batch_size, 4) # (batch_size, 4)
        log.debug(f'Output: {output.shape}')
        
        best = best.unsqueeze(1) # (batch_size, 1)
        numerator_best_sample = torch.exp(torch.gather(output, 1, best)) # (batch_size, 1)
        denominator = torch.sum(torch.exp(output), dim=1, keepdim=True) # (batch_size, 1)

        criterion = torch.log(numerator_best_sample / denominator) # (batch_size, 1)
        loss = 1/criterion.size(0) * torch.sum(criterion)
        log.debug(f'loss: {loss.shape}')
        return loss
    
    def _combine_query_sample(self, query: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """
            Args:
                query: torch.tensor, shape (batch_size, seq_len)
                sample: torch.tensor, shape (batch_size, seq_len)
            Returns:
                combined: torch.tensor, shape (batch_size, seq_len)

            Combine query and samples prompt, remove pad between query and samples
            Query format is [Q1, Q2, ..., PAD, PAD]
            Sample format is [S1, S2, ..., PAD, PAD]
            Output format is [Q1, Q2, ..., S1, S2, ..., PAD, PAD]
        """
        assert query.dim() == 1 and sample.dim() == 1, 'Query and Sample must be 1D tensor'
        # BUG: On some datasets, there exists case where continuation is on subword level 
        #   Query: "... work" Sample: "ing ...", "working" should be single token instead of two tokens concatenated
        # in this case, decode into string and then re-encode is the general solution

        combined_query_sample = torch.full((query.size(0) + sample.size(0),), self.tokenizer.pad_token_id)
        # remove pad tokens on query
        new_query = query[query != self.tokenizer.pad_token_id] 
        combined_query_sample[:new_query.size(0)] = new_query
        
        combined_query_sample[new_query.size(0):new_query.size(0)+sample.size(0)] = sample
        return combined_query_sample
    
    @classmethod
    def load(cls, base_model: str, load_path: str, seed: int = 42, device: str = 'cuda') -> 'RewardModel':
        """
        Load a saved reward model from a checkpoint.
        
        Args:
            load_path (str): Path to the checkpoint file.
            seed (int): Random seed for reproducibility
            device (str): Device to load the model on ('cuda', 'cpu', or 'mps')
        
        Returns:
            RewardModel: The loaded reward model.
        """
        from transformers.models.gpt2.configuration_gpt2 import GPT2Config
        torch.serialization.safe_globals([GPT2Config]) # BUG: Dynamic workaround should be implemented
        
        train_reward_checkpoint = torch.load(load_path, weights_only=False, map_location='cpu')
        reward_model = train_reward_checkpoint['model_state_dict']
        # Get seed from checkpoint if available, otherwise use provided seed
        saved_seed = train_reward_checkpoint.get('seed', seed)
        # Get device from checkpoint if available, otherwise use provided device
        saved_device = train_reward_checkpoint.get('device', device)
        
        model = cls(base_model, seed=saved_seed, device=saved_device)
        model.load_state_dict(reward_model)
        
        # Move model to the specified device
        model.to(get_device(saved_device))
        
        return model
        
    def to(self, device):
        """
        Move the model to the specified device.
        
        Args:
            device: The device to move the model to
            
        Returns:
            self: The model instance
        """
        self.model = self.model.to(device)
        self.score = self.score.to(device)
        return super().to(device)