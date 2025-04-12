import logging
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import pandas as pd
import swifter
from tqdm import tqdm
import wandb
import numpy as np

from .reward_model import RewardModel
from .utils import set_seed, get_device

log = logging.getLogger(__name__)

class RewardData(torch.utils.data.Dataset):
    text_columns = ['query', 'sample0', 'sample1', 'sample2', 'sample3']

    def __init__(self, tokenizer, query: list[str], sample0: list[str], sample1: list[str], sample2: list[str], sample3: list[str], best: list[int]):
        self.tokenizer = tokenizer

        # HACK: CURRENTLY ONLY WORK FOR GPT2
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.query: list[str] = query
        self.sample0: list[str] = sample0
        self.sample1: list[str] = sample1
        self.sample2: list[str] = sample2
        self.sample3: list[str] = sample3
        self.best: list[int] = best

        self.collate_fn = lambda batch: RewardData.collate_fn(self.tokenizer.pad_token_id, batch)
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        single_data_tensor = {
            'query': self.query[idx],
            'sample0': self.sample0[idx],
            'sample1': self.sample1[idx],
            'sample2': self.sample2[idx],
            'sample3': self.sample3[idx],
            'best': torch.tensor(self.best[idx])
        }
        for key in self.text_columns:
            value = single_data_tensor[key]
            single_data_tensor[key] = self.tokenizer(value, return_tensors='pt')
    
        return single_data_tensor
    
    def __len__(self):
        return len(self.query)
    
    @staticmethod
    def from_dataset(tokenizer, dataset: Dataset) -> 'RewardData':
        return RewardData(tokenizer, **{col: dataset[col] for col in dataset.column_names})

    @staticmethod
    def from_openai_format(tokenizer, path: str) -> 'RewardData':
        dataset = Dataset.load_from_disk(path)
        return RewardData(tokenizer, **{col: dataset[col] for col in dataset.column_names})
    
    @staticmethod
    def collate_fn(pad_token_id, batch):
        keys_to_pad = ['query', 'sample0', 'sample1', 'sample2', 'sample3']
        num_batch = len(batch)
        
        # Determine the maximum sequence length
        max_length = 0
        for key in keys_to_pad:
            max_length = max(
                max_length, 
                max(len(batch[idx][key]['input_ids'][0]) for idx in range(num_batch))
            )
        
        # Pad each key to the maximum sequence length
        padded_batch = {
            'best': torch.tensor([batch[idx]['best'] for idx in range(num_batch)])
        }
        for key in keys_to_pad:
            key_input_ids_padded = []
            key_attention_mask_padded = []
            for idx in range(num_batch):
                input_ids = batch[idx][key]['input_ids'][0] # (seqlen)
                attention_mask = batch[idx][key]['attention_mask'][0] # (seqlen)

                needed_pad = max_length - len(input_ids)
                
                ids_pad_tensor = torch.full((needed_pad, ), pad_token_id)
                att_pad_tensor = torch.full((needed_pad, ), 0)

                input_ids_padded = torch.concat([input_ids, ids_pad_tensor], dim=0) # (maxlen)
                attention_mask_padded = torch.concat([attention_mask, att_pad_tensor], dim=0) # (maxlen)
                key_input_ids_padded.append(input_ids_padded)
                key_attention_mask_padded.append(attention_mask_padded)

            padded_batch[key] = {
                'input_ids': torch.stack(key_input_ids_padded, dim=0),
                'attention_mask': torch.stack(key_attention_mask_padded, dim=0)
            }
        
        # print(padded_batch['query']['input_ids'].shape, padded_batch['sample0']['input_ids'].shape, padded_batch['sample1']['input_ids'].shape, padded_batch['sample2']['input_ids'].shape, padded_batch['sample3']['input_ids'].shape, padded_batch['best'].shape)
        return padded_batch
    
    def to_dataset(self) -> Dataset:
        return Dataset.from_dict({
            'query': self.query,
            'sample0': self.sample0,
            'sample1': self.sample1,
            'sample2': self.sample2,
            'sample3': self.sample3,
            'best': self.best
        })
    
class RewardTrainer:
    def __init__(self, base_model: str, datasets_path: list[str], batch_size: int, dry_run: bool, seed: int = 42, device: str = 'cuda'):
        """
        Args: 
            datasets_path: list[str], convertable through RewardData.from_openai_format(path)
            seed: int, random seed for reproducibility
            device: str, device to run on ('cuda', 'cpu', or 'mps')
        """
        log.info('Instantiating RewardTrainer')
        self.base_model = base_model
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.seed = seed
        self.device_str = device
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Set device
        self.device = get_device(self.device_str)
        log.info(f'Using device: {self.device}')

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = RewardModel(base_model, device=self.device_str)
        
        # Move model to device
        self.model.to(self.device)
        
        self.datasets_path = datasets_path
        
        log.debug(f'Is Dry run: {self.dry_run}')
        if self.dry_run:
            datasets: list[Dataset] = [
                RewardData.from_openai_format(self.tokenizer, path).to_dataset().select(range(64)) for path in datasets_path[:2]
            ]
        else:
            datasets: list[Dataset] = [
                RewardData.from_openai_format(self.tokenizer, path).to_dataset() for path in datasets_path
            ]
        self.dataset = RewardData.from_dataset(self.tokenizer, concatenate_datasets(datasets))

    @staticmethod
    def train(base_model: str, datasets_path: list[str], batch_size: int, epochs: int, dry_run: bool, seed: int = 42, device: str = 'cuda') -> 'RewardTrainer':
        """
        Train a RewardModel on a given dataset.

        Args:
            base_model: str, the base model name to use for the reward model
            datasets_path: list[str], convertable through RewardData.from_openai_format(path)
            batch_size: int, number of samples to use for training
            epochs: int, number of epochs to train
            dry_run: bool, whether to do a dry run (only use a subset of the data)
            seed: int, random seed for reproducibility (default: 42)
            device: str, device to run on ('cuda', 'cpu', or 'mps')

        Returns:
            RewardModel: The trained reward model
        """
        trainer = RewardTrainer(base_model, datasets_path, batch_size, dry_run, seed, device)
        model = trainer.model
        dataset = trainer.dataset
        log.debug(f'Training dataset size : {len(dataset)}')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # Use seed for dataloader worker initialization
        g = torch.Generator()
        g.manual_seed(trainer.seed)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=trainer.dataset.collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(trainer.seed + worker_id),
            generator=g
        )

        for epoch in tqdm(range(epochs)):
            log.debug(f'Running on epoch : {epoch}')
            for batch in dataloader:
                # Move batch to device
                for key in ['query', 'sample0', 'sample1', 'sample2', 'sample3']:
                    if key in batch:
                        for tensor_key in batch[key]:
                            batch[key][tensor_key] = batch[key][tensor_key].to(trainer.device)
                if 'best' in batch:
                    batch['best'] = batch['best'].to(trainer.device)
                
                query, best = batch['query'], batch['best']
                sample0, sample1, sample2, sample3 = batch['sample0'], batch['sample1'], batch['sample2'], batch['sample3']
                loss = model(query, sample0, sample1, sample2, sample3, best)
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
        
        # Move model to CPU before saving
        self.model.to('cpu')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'base_model': self.base_model,
            'config': self.model.model.config,
            'datasets_path': self.datasets_path,
            'batch_size': self.batch_size,
            'dry_run': self.dry_run,
            'seed': self.seed,  # Save the seed for reproducibility
            'device': self.device_str  # Save the device
        }, save_path)
        
        # Move model back to original device
        self.model.to(self.device)
        
        log.info(f'Saved reward trainer state to {save_path}')
    
    @classmethod
    def load(cls, load_path: str, device: str = None) -> 'RewardTrainer':
        """Load a saved trainer state
        
        Args:
            load_path: str, path to the saved checkpoint
            device: str, device to load the model on (overrides saved device if provided)
            
        Returns:
            RewardTrainer: Loaded trainer instance
        """
        checkpoint = torch.load(load_path)
        
        # Use provided device if specified, otherwise use saved device or default to 'cuda'
        device_to_use = device if device is not None else checkpoint.get('device', 'cuda')
        
        trainer = cls(
            base_model=checkpoint['base_model'],
            datasets_path=checkpoint['datasets_path'],
            batch_size=checkpoint['batch_size'],
            dry_run=checkpoint['dry_run'],
            seed=checkpoint.get('seed', 42),  # Default to 42 if no seed was saved
            device=device_to_use
        )
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        log.info(f'Loaded reward trainer state from {load_path}')
        
        return trainer