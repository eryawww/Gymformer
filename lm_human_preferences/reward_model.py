import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

log = logging.getLogger(__name__)

class RewardModel(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.1):
        log.info('Instantiating RewardModel')
        super().__init__()
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        
        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    
    def forward(self, query: torch.Tensor, samples: torch.Tensor, best: torch.Tensor) -> torch.Tensor:
        """
            This function computes the loss for one batch of data.
            
            Map x, y1, y2, y3, y4, best -> loss (real number)
            
            Algorithm:
                For each yi, compute real value logits r(x, yi) 
                loss = log frac{ exp(r(x, y_best)) }{ sum_i exp(r(x, y_i)) }
        
            Args:
                query: torch.tensor, shape (batch_size, seq_len)
                samples: torch.tensor, shape (batch_size, 4, seq_len)
                best: int, shape (batch_size)
            Returns:
                loss: torch.tensor, shape (1)
        """
        log.debug(f'Forward with Query: {query.shape} Samples: {samples.shape} Best: {best.shape}')
        assert query.size(0) == samples.size(0) == best.size(0), 'Batch size must be same for all inputs {}, {}, {}'.format(query.size(0), samples.size(0), best.size(0))
        batch_size, sample_size = query.size(0), samples.size(1)

        combined_query_samples_list = []
        for i_batch in range(batch_size):
            for j_sample in range(sample_size):
                combined = self._combine_query_sample(query[i_batch], samples[i_batch, j_sample, :])
                combined_query_samples_list.append(combined)
        combined_query_samples = pad_sequence(combined_query_samples_list, batch_first=True, padding_value=self.tokenizer.pad_token_id) 
        combined_attention_samples = (combined_query_samples != self.tokenizer.pad_token_id).long()
        log.debug(f'Combined Query Padded Sequence: {combined_query_samples.shape}') # (batch_size * sample_size, seq_len)
        log.debug(f'Combined Attention Samples: {combined_attention_samples.shape}') # (batch_size * sample_size, seq_len)

        log.debug(f'Running Forward on Model')
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=combined_query_samples, attention_mask=combined_attention_samples, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, -1, :] # use vector from last token of last layer
        
        log.debug(f'Hidden States: {hidden_states.shape}') # (batch_size * sample_size, hidden_size)
        hidden_states = self.dropout(hidden_states)
        output = self.linear(hidden_states) # (batch_size * sample_size, 1)
        log.debug(f'Output Flatten: {output.shape}')
        output = output.view(batch_size, sample_size) # (batch_size, sample_size)
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
        # TODO: On bad datasets, there exists case where continuation is on subword level
        # in this case, decode into string and then re-encode is the general solution

        combined_query_sample = torch.full((query.size(0) + sample.size(0),), self.tokenizer.pad_token_id)
        # remove pad tokens on query
        new_query = query[query != self.tokenizer.pad_token_id] 
        combined_query_sample[:new_query.size(0)] = new_query
        
        combined_query_sample[new_query.size(0):new_query.size(0)+sample.size(0)] = sample
        return combined_query_sample
    
    