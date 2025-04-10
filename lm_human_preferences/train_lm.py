import logging
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, AutoModelForSequenceClassification
from datasets import concatenate_datasets
import wandb

from trl import PPOConfig, PPOTrainer, create_reference_model
from trl.core import LengthSampler

from .reward_model import RewardModel
from .train_reward import RewardData

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class LMTrainer:
    def __init__(self, base_model: str, datasets_path: list[str], batch_size: int, epochs: int, reward_model_path: str, output_dir: str, dry_run: bool, config: dict):
        self.base_model_name = base_model
        self.dry_run = dry_run
        self.datasets_path = datasets_path
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
        )
        self.ref_model = create_reference_model(self.model)
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=1
        )
        self.reward_model = RewardModel.load(base_model, reward_model_path)

        training_data, test_datasets = self._prepare_dataset(datasets_path)
        
        self.ppo_config = PPOConfig(
            stop_token='eos',
            learning_rate=config['ppo']['learning_rate'],
            output_dir=output_dir,
            per_device_train_batch_size=config['batch_size'],
            # gradient_accumulation_steps=4,  # Effective batch size = batch_size * 4
            total_episodes=config['epochs'],
            cliprange=0.2,
            missing_eos_penalty=1.0,
            kl_coef=0.2
        )
        
        self.ppo_trainer = PPOTrainer(
            args=self.ppo_config,
            processing_class=self.tokenizer,
            value_model=self.value_model,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            train_dataset=training_data,
            # data_collator: Optional[DataCollatorWithPadding] = None,
            eval_dataset=test_datasets
        )
        
    def _prepare_dataset(self, datasets_path: list[str]):
        """Prepare dataset for PPO training"""
        if self.dry_run:
            datasets_path = self.datasets_path[:1]
            max_samples = 6 * self.batch_size
        else:
            datasets_path = self.datasets_path
            max_samples = None
        
        datasets = []
        for path in datasets_path:
            data = RewardData.from_openai_format(self.tokenizer, path)
            dataset = data.to_dataset()
            
            if max_samples:
                dataset = dataset.select(range(max_samples))
            
            def prepare_ppo_data(example) -> dict:
                encoded_query = self.tokenizer(example["query"], return_tensors="pt") # (1, seq_len)
                return {key: encoded_query[key][0] for key in encoded_query} # Make sure no batch dim returned
            
            dataset = dataset.map(prepare_ppo_data, remove_columns=dataset.column_names)

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size

            splitted_dataset = dataset.train_test_split(train_size, test_size)
            train_dataset, test_dataset = splitted_dataset['train'], splitted_dataset['test']
            datasets.append((train_dataset, test_dataset))
        
        train_datasets, test_datasets = zip(*datasets)
        return concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)
    
    @staticmethod
    def train(base_model: str, datasets_path: list[str], batch_size: int, epochs: int, reward_model_path: str, output_dir: str, dry_run: bool, config: dict) -> "LMTrainer":
        """Train language model using PPO with reward model feedback"""
        trainer = LMTrainer(base_model, datasets_path, batch_size, epochs, reward_model_path, output_dir, dry_run, config)
        
        log.info("Starting PPO training loop...")
        trainer.ppo_trainer.train()
        
        return trainer
    
    def save(self, save_path: str):
        self.ppo_trainer.save_model(save_path)
    
    def load(self, save_path: str):
        self.ppo_trainer.load_model(save_path)
        