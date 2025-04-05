import logging
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import swifter
from tqdm import tqdm
import wandb

from .reward_model import RewardModel

log = logging.getLogger(__name__)

@dataclass
class RewardData:
    query: list[str]
    sample0: list[str]
    sample1: list[str]
    sample2: list[str]
    sample3: list[str]
    best: int

    @staticmethod
    def from_openai(json_path: str, pad_token: int):        
        json_df = pd.read_json(json_path)
        # HACK: tokens 50259 is used as padding token, unknown for GPT2, replace with configured pad_token
        for col in ['query', 'sample0', 'sample1', 'sample2', 'sample3']:
            json_df[col] = json_df[col].swifter.apply(lambda lst: [pad_token if x == 50259 else x for x in lst])
        
        return RewardData(**{
            col: json_df[col] for col in json_df.columns
        })
    
    def to_dataset(self) -> Dataset:
        dataset = Dataset.from_dict(asdict(self))
        return dataset
    
class RewardTrainer:
    def __init__(self, base_model: str, datasets_path: list[str], batch_size: int, dry_run: bool):
        """
            Args: 
                datasets_path: list[str], convertable through RewardData.from_openai_format(path)
        """
        log.info('Instantiating RewardTrainer')
        self.base_model = base_model
        self.model = RewardModel(base_model)
        self.dry_run = dry_run
        self.batch_size = batch_size
        
        self.datasets_path = datasets_path
        
        # TODO: Support for another model, .from_openai is a tokenizer output for GPT2
        # by decoding the tokens to text using GPT2 tokenizer
        assert 'openai-community/gpt2' in base_model, 'Only GPT2 model is supported for now' 

        log.debug(f'Is Dry run: {self.dry_run}')
        if self.dry_run:
            datasets: list[Dataset] = [
                RewardData.from_openai(path, pad_token=self.model.tokenizer.pad_token_id).to_dataset().select(range(2*batch_size)) for path in datasets_path[:2]
            ]
        else:
            datasets: list[Dataset] = [
                RewardData.from_openai(path, pad_token=self.model.tokenizer.pad_token_id).to_dataset() for path in datasets_path
            ]
        self.dataset = concatenate_datasets(datasets)

    @staticmethod
    def train(base_model: str, datasets_path: list[str], batch_size: int, epochs: int, dry_run: bool) -> 'RewardTrainer':
        """
        Train a RewardModel on a given dataset.

        Args:
            base_model: str, the base model name to use for the reward model
            datasets_path: list[str], convertable through RewardData.from_openai_format(path)
            batch_size: int, number of samples to use for training
            epochs: int, number of epochs to train
            dry_run: bool, whether to do a dry run (only use a subset of the data)

        Returns:
            RewardModel: The trained reward model
        """
        trainer = RewardTrainer(base_model, datasets_path, batch_size, dry_run)
        model = trainer.model
        dataset = trainer.dataset
        log.debug(f'Training dataset size : {len(dataset)}')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        for epoch in tqdm(range(epochs)):
            log.debug(f'Running on epoch : {epoch}')
            for batch in dataset.iter(batch_size):
                batch_tensor = {
                    key: torch.tensor(batch[key]) for key in batch.keys()
                }
                
                query, best = batch_tensor['query'], batch_tensor['best']
                samples = torch.stack([batch_tensor[f'sample{i}'] for i in range(4)], dim=1)
                loss = model(query, samples, best)
                log.debug(f'Loss Value : {loss}')
                log.debug(f'Doing Backward')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({
                    'loss': loss.item(),
                    'epoch': epoch
                })
        
        return trainer
        
    def save(self, save_path: str):
        """Save the trainer state including model and configuration
        
        Args:
            save_path: str, path to save the model checkpoint
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'base_model': self.base_model,
            'config': self.model.model.config,
            'datasets_path': self.datasets_path,
            'batch_size': self.batch_size,
            'dry_run': self.dry_run
        }, save_path)
        
        log.info(f'Saved reward trainer state to {save_path}')
    
    @classmethod
    def load(cls, load_path: str) -> 'RewardTrainer':
        """Load a saved trainer state
        
        Args:
            load_path: str, path to the saved checkpoint
            
        Returns:
            RewardTrainer: Loaded trainer instance
        """
        checkpoint = torch.load(load_path)
        
        trainer = cls(
            base_model=checkpoint['base_model'],
            datasets_path=checkpoint['datasets_path'],
            batch_size=checkpoint['batch_size'],
            dry_run=checkpoint['dry_run']
        )
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        log.info(f'Loaded reward trainer state from {load_path}')
        
        return trainer