import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import Optional
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from torch.nn.utils.rnn import pad_sequence
from lm_human_preferences.env.rlhf_env import Action

import os

class LanguageAgent(torch.nn.Module):

    def __init__(self, model_name: str, device: str = 'cuda'):
        super(LanguageAgent, self).__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.value_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
        self.backbone = getattr(self.policy, self.policy.base_model_prefix)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id # TODO: Generalize, force construction for GPT2
        self.eos_token_id = self.tokenizer.eos_token_id # TODO: Generalize, force construction for GPT2
    
    def get_state_value(self, state: tuple[list[int]] = None, last_hidden_states: Optional[Tensor] = None) -> Tensor:
        assert state is not None or last_hidden_states is not None, "One of state or last_hidden_states must be provided"
        if last_hidden_states is None:
            state_tensor = self._pad_state_tensor(state)
            last_hidden_states = self._get_backbone_last_hidden_states(state_tensor)
        # last_hidden_states [num_env, seq_len, hidden_size]
        all_token_score_value = self.value_model.score(last_hidden_states) # [num_env, seq_len, 1]
        last_token_state_score = all_token_score_value[:, -1, 0] # [num_env]
        return last_token_state_score
    
    def _get_backbone_last_hidden_states(self, state_tensor: Tensor) -> Tensor:
        attention_tensor = state_tensor != self.pad_token_id
        return self.backbone(
            input_ids=state_tensor,
            attention_mask=attention_tensor,
            output_hidden_states=True
        ).hidden_states[-1] # [num_env, seq_len, hidden_size]
    
    def get_action_and_value(self, state: tuple[list[int]]|Tensor) -> tuple[list[Action], Tensor]:
        # state : tuple (num_env,) of list[str] input_ids
        # state_tensor = torch.tensor(state).to(self.device)
        state_tensor = state
        if not isinstance(state_tensor, Tensor):
            state_tensor = self._pad_state_tensor(state)
            
        last_hidden_states = self._get_backbone_last_hidden_states(state_tensor)        

        state_value: Tensor = self.get_state_value(last_hidden_states=last_hidden_states) # 
        logits: Tensor = self.policy.lm_head(last_hidden_states)[:, -1, :] # [num_env, vocab_size]
        env_action_distribution = [ Categorical(logits=env_logits) for env_logits in logits ] 
        # convert action: ModelOutput back to iterable, gymnasium Vectorized expect iterable
        actions = [
            Action(
                action=env_distribution.sample(),
                logprobs=F.log_softmax(env_distribution.logits, dim=-1),
                entropy=env_distribution.entropy()
            ) for env_distribution in env_action_distribution
        ]
        return actions, state_value
    
    def _pad_state_tensor(self, state: tuple[list[int]]) -> Tensor:
        return pad_sequence(
            [torch.tensor(s) for s in state], 
            batch_first=True, 
            padding_value=self.pad_token_id, 
            padding_side='left'
        ).to(self.device)

    def rollout(self, state: tuple[list[int]]):
        pass
    
    def save(self, path: str):

        os.makedirs(path, exist_ok=True)

        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save({
            'base_model': self.model_name
        }, os.path.join(path, "policy_metadata.pt"))
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cuda'):
        metadata = torch.load(os.path.join(load_path, "policy_metadata.pt"))
        
        # recreate agent and load models
        agent = cls(model_name=metadata['base_model'], device=device)
        agent.policy = AutoModelForCausalLM.from_pretrained(load_path).to(device)
        agent.value_model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=1).to(device)
        agent.tokenizer = AutoTokenizer.from_pretrained(load_path)
        agent.backbone = getattr(agent.policy, agent.policy.base_model_prefix)
        agent.tokenizer.pad_token_id = agent.tokenizer.eos_token_id
        agent.pad_token_id = agent.tokenizer.pad_token_id
        agent.eos_token_id = agent.tokenizer.eos_token_id
        return agent
