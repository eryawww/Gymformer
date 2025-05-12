from gymformer.rl.ppo import PPO, PPOConfig
from typing import List, Optional, Callable, Tuple
import gymnasium as gym
import torch.nn as nn
from scipy.special import logsumexp
import os

def train_agent(env_name: str, actor: nn.Module, critic: nn.Module, ppo_config: PPOConfig, 
        max_episodes: int = 1000, max_timesteps: int = 1000, 
        update_interval: int = 100, save_interval: int = 100, save_path: str = "./models",
        callback: Optional[Callable] = None, log_interval: int = 10, device='cuda') -> List[float]:
    """
    Train a PPO agent in a given environment.
    
    Args:
        env_name: Name of the gym environment
        actor: Actor network
        critic: Critic network
        ppo_config: PPO configuration
        max_episodes: Maximum number of episodes for training
        max_timesteps: Maximum timesteps in one episode
        update_interval: Update policy every n episodes
        save_interval: Save model in the interval
        save_path: Directory path to save model checkpoints
        callback: Optional callback function for logging
        log_interval: Interval for logging progress
        device: Device to run training on
    
    Returns:
        List of average rewards per episode
    """
    # Create environment
    env = gym.make(env_name)
    
    # Initialize PPO agent with device manager
    ppo_agent = PPO(
        actor,
        critic,
        ppo_config,
        device,
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Logging variables
    running_reward = 0
    time_step = 0
    
    # Training loop
    episode_rewards = []
    
    for episode in range(1, max_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(1, max_timesteps+1):
            # Select action
            action = ppo_agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience in buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            time_step += 1
            episode_reward += reward

            state = next_state
            
            if done or truncated:
                break
        
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        episode_rewards.append(running_reward)

        if episode % log_interval == 0:
            if callback is not None:
                callback(state, action, reward, done, truncated, episode, running_reward)
            print(f"Episode {episode}, Running Reward: {running_reward:.2f}")

        # Update if its time
        if episode % update_interval == 0:
            # Make sure we have enough data in the buffer before updating
            if len(ppo_agent.buffer.states) > 0:  # Only update if buffer contains experiences
                ppo_agent.update()
            else:
                print(f"Warning: Episode {episode} - Buffer is empty, skipping update")
        
        # Save model
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(save_path, f"ppo_{env_name}_episode_{episode}.pth")
            ppo_agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    env.close()
    return episode_rewards

def train_reward(
    model_name: str,
    datasets_path: str,
    reward_model_path: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    epochs: int = 100,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    learning_rate: float = 1e-5,
    device: str = 'cuda',
    seed: int = 42,
    dry_run: bool = False,
    resume_checkpoint: Optional[str] = None
) -> nn.Module:
    """Train a reward model using preference data.
    
    Args:
        model_name: Name or path of the base model to use (e.g. 'openai-community/gpt2')
        datasets_path: Path to the preference dataset
        reward_model_path: Path to save the trained reward model
        test_size: Proportion of data to use for testing
        batch_size: Batch size for training
        epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        learning_rate: Learning rate for training
        device: Device to train on ('cuda', 'cpu', or 'mps')
        seed: Random seed for reproducibility
        dry_run: If True, use a small subset of data for quick testing
        resume_checkpoint: Optional path to resume training from checkpoint
        
    Returns:
        Trained RewardModel
    """
    from gymformer.data.base import RewardData
    from gymformer.lm.reward import RewardModelWrapper
    from transformers import TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback
    from transformers import AdamW, get_cosine_schedule_with_warmup
    import os
    import numpy as np


    # Initialize model and tokenizer
    model = RewardModelWrapper(model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and prepare dataset
    datasets = RewardData.from_openai_format(tokenizer, datasets_path).to_dataset()
    if dry_run:
        datasets = datasets.select(range(100)) 
    
    splits = datasets.train_test_split(test_size=test_size, seed=seed)
    train_dataset = RewardData.from_dataset(tokenizer, splits['train'])
    test_dataset = RewardData.from_dataset(tokenizer, splits['test'])

    os.makedirs(reward_model_path, exist_ok=True)

    def compute_metrics(eval_pred):
        # Unpack the raw arrays
        raw_preds = eval_pred.predictions
        labels   = eval_pred.label_ids

        # If your model returned shape (batch*sample_size, 1), squeeze to (batch*sample_size,)
        if raw_preds.ndim == 2 and raw_preds.shape[1] == 1:
            raw_preds = raw_preds.squeeze(-1)

        # Now raw_preds should be 1-D of length batch*sample_size, or 2-D of (batch, sample_size)
        if raw_preds.ndim == 1:
            # Figure out how many samples per example
            batch_size  = labels.shape[0]
            sample_size = raw_preds.shape[0] // batch_size
            logits = raw_preds.reshape(batch_size, sample_size)
        else:
            # Already (batch, sample_size)
            logits = raw_preds

        # Compute metrics
        preds = np.argmax(logits, axis=1)
        accuracy = (preds == labels).mean()

        # 2) MRR
        sorted_idx = np.argsort(-logits, axis=1)
        ranks = np.where(sorted_idx == labels.reshape(-1, 1))[1] + 1
        mrr = (1.0 / ranks).mean()

        # 3) The ranking loss exactly as forward
        selected = logits[np.arange(len(labels)), labels] # r(x, y_best)
        lse      = logsumexp(logits, axis=1) # logsumexp over candidates
        loss     = -np.mean(selected - lse)

        return {
            "accuracy": accuracy,
            "mrr": mrr,
            "loss": loss
        }

    training_args = TrainingArguments(
        output_dir=reward_model_path,
        num_train_epochs=1 if dry_run else epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir='logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        remove_unused_columns=False,
        optim='adamw_torch',
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        label_names=["best"],
        greater_is_better=False
    )
    
    optimizer = AdamW(model.model.score.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * len(train_dataset) // batch_size
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=train_dataset.collate_fn,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )
    
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # Save the best model
    print(f"Saving best reward model to {reward_model_path}")
    model.save(reward_model_path)
    print(f"Reward model saved successfully to {reward_model_path}")
    
    # remove RewardModel wrapper, return AutoModelForSequenceClassification
    return RewardModelWrapper.from_pretrained(reward_model_path, device=device) 

def load_reward_model(model_path: str, device: str = 'cuda') -> nn.Module:
    """Load a reward model from a given path.
    
    Args:
        model_path: Path to the reward model
        device: Device to load the model on ('cuda', 'cpu', or 'mps')
    """
    from gymformer.lm.reward import RewardModelWrapper
    model = RewardModelWrapper.from_pretrained(model_path, device=device)
    return model

def load_ppo_agent(model_path: str, actor: nn.Module, critic: nn.Module, device: str = 'cuda') -> PPO:
    """Load a PPO agent from a given path.
    
    Args:
        model_path: Path to the PPO agent
        actor: Actor network for architecture reference, weight will be loaded
        critic: Critic network for architecture reference, weight will be loaded
        device: Device to load the model on ('cuda', 'cpu', or 'mps')
    """
    from gymformer.rl.ppo import PPO
    ppo_agent = PPO.load(model_path, actor, critic, device=device)
    return ppo_agent