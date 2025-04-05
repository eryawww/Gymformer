import logging

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from peft import LoraConfig
from reward_model import RewardModel
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

log = logging.getLogger(__name__)

class LMTrainer:
    def __init__(self, base_model: str, datasets_path: list[str], batch_size: int, epochs: int, reward_model_path: str, dry_run: bool):
        self.base_model_name = base_model
        self.dry_run = dry_run
        self.datasets_path = datasets_path
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize models and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize policy (actor) model with value head
        self.policy = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize reward model
        self.reward_model = RewardModel.load_state_dict(reward_model_path)
        
        # PPO config
        self.ppo_config = PPOConfig(
            batch_size=batch_size,
            learning_rate=1.4e-5,
            max_grad_norm=0.5,
            optimize_cuda_cache=True,
            target_kl=0.1,
            init_kl_coef=0.2,
            early_stopping=True,
            use_score_scaling=True,
            use_score_norm=True
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.policy,
            tokenizer=self.tokenizer,
            dataset=self._prepare_dataset()
        )
        
        # Response length sampler
        self.response_length_sampler = LengthSampler(32, 96)
    
    def _prepare_dataset(self):
        """Prepare dataset for PPO training"""
        if self.dry_run:
            # Use small subset of data for dry run
            datasets_path = self.datasets_path[:1]
            max_samples = 2 * self.batch_size
        else:
            datasets_path = self.datasets_path
            max_samples = None
        
        from datasets import load_dataset, Dataset, concatenate_datasets
        from dataclasses import asdict
        from train_reward import RewardData
        
        # Load and process each dataset
        datasets = []
        for path in datasets_path:
            # Load data using RewardData format
            data = RewardData.from_openai(path, pad_token=self.tokenizer.pad_token_id)
            dataset = data.to_dataset()
            
            if max_samples:
                dataset = dataset.select(range(max_samples))
            
            # Convert to format expected by PPOTrainer
            def prepare_ppo_data(example):
                return {
                    "input_ids": example["query"],
                    "attention_mask": (example["query"] != self.tokenizer.pad_token_id).long(),
                    "labels": example[f"sample{example['best']}"]  # Use the best sample as target
                }
            
            dataset = dataset.map(prepare_ppo_data)
            datasets.append(dataset)
        
        # Combine all datasets
        return concatenate_datasets(datasets)
    
    def _combine_query_response(self, query: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """
        Combine query and response tensors directly, similar to RewardModel._combine_query_sample
        
        Args:
            query: torch.tensor, shape (batch_size, seq_len)
            response: torch.tensor, shape (batch_size, seq_len)
        Returns:
            combined: torch.tensor, shape (batch_size, seq_len)
        """
        combined = torch.full((query.size(0) + response.size(0),), self.tokenizer.pad_token_id)
        # Remove pad tokens from query
        new_query = query[query != self.tokenizer.pad_token_id]
        combined[:new_query.size(0)] = new_query
        combined[new_query.size(0):new_query.size(0)+response.size(0)] = response
        return combined
    
    def _get_reward_score(self, query, response):
        combined = self._combine_query_response(query, response)
        combined_batch = combined.unsqueeze(0)  # Add batch dimension
        attention_mask = (combined_batch != self.tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            reward_score = self.reward_model(
                combined_batch,
                attention_mask=attention_mask
            )
        return reward_score.item()
    
    def save(self, save_path: str):
        """Save the trainer state including policy model and configuration
        
        Args:
            save_path: str, path to save the model checkpoint
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'base_model': self.base_model_name,
            'config': self.policy.config,
            'ppo_config': self.ppo_config,
            'datasets_path': self.datasets_path,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'dry_run': self.dry_run
        }, save_path)
        
        log.info(f'Saved policy model state to {save_path}')
    
    @classmethod
    def load(cls, load_path: str, reward_model_path: str = None) -> 'LMTrainer':
        """Load a saved trainer state
        
        Args:
            load_path: str, path to the saved checkpoint
            reward_model_path: str, optional path to reward model checkpoint.
                             If not provided, will initialize a new reward model.
            
        Returns:
            LMTrainer: Loaded trainer instance
        """
        checkpoint = torch.load(load_path)
        
        # Initialize trainer with saved config
        trainer = cls(
            base_model=checkpoint['base_model'],
            datasets_path=checkpoint['datasets_path'],
            batch_size=checkpoint['batch_size'],
            epochs=checkpoint['epochs'],
            dry_run=checkpoint['dry_run']
        )
        
        # Load policy state
        trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Load reward model if provided
        if reward_model_path:
            from train_reward import RewardTrainer
            reward_trainer = RewardTrainer.load(reward_model_path)
            trainer.reward_model = reward_trainer.model
        
        log.info(f'Loaded policy model state from {load_path}')
        return trainer

    @staticmethod
    def train(base_model: str, datasets_path: list[str], batch_size: int, epochs: int, reward_model_path: str, dry_run: bool):
        """Train language model using PPO with reward model feedback"""
        # TODO: Continue working here, start running
        trainer = LMTrainer(base_model, datasets_path, batch_size, epochs, reward_model_path, dry_run)
        
        log.info("Starting PPO training loop...")
        for epoch in tqdm(range(trainer.epochs), desc="Training epochs"):
            for batch in trainer.ppo_trainer.dataloader:
                query_tensors = batch["input_ids"]
                response_length = trainer.response_length_sampler()
                
                response = trainer.policy.generate(
                    query_tensors,
                    max_new_tokens=response_length,
                    do_sample=True,
                    temperature=1.0
                )
                
                # Get rewards using reward model by directly combining tensors
                rewards = []
                for query, resp in zip(query_tensors, response):
                    reward = trainer._get_reward_score(query, resp)
                    rewards.append(reward)
                
                stats = trainer.ppo_trainer.step(query_tensors, response, rewards)
                
                wandb.log({
                    "reward": torch.mean(torch.tensor(rewards)).item(),
                    "policy_loss": stats["policy/loss"],
                    "value_loss": stats["value/loss"],
                    "kl": stats["objective/kl"],
                    "epoch": epoch
                })
        
        return trainer.policy