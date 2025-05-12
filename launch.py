import logging
from transformers import EvalPrediction
import wandb
import argparse
import yaml
from types import SimpleNamespace
from scipy.special import logsumexp

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train LM from human preferences')
    # Task
    parser.add_argument('--train-reward', action='store_true', help='Train reward model')
    parser.add_argument('--reward-model', type=str, default='models/reward_model', help='Saving (--train-reward)/Loading (--train-lm) reward model path')
    
    parser.add_argument('--train-lm', action='store_true', help='Train language model')
    parser.add_argument('--lm-model', type=str, default='models/lm_model', help='Saving (--train-lm)/Loading (--inference) language model path')

    parser.add_argument('--train-sft', action='store_true', help='Train SFT model')
    parser.add_argument('--sft-model', type=str, default='models/sft_model', help='Saving (--train-sft)/Loading (--inference) SFT model path')

    parser.add_argument('--inference', action='store_true', default=False, help='Run inference')
    # Run config
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'], help='Device to run on (cuda, cpu, or mps)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialize wandb
    with open("hyperparameters/hyperparameters.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = SimpleNamespace(**config)
    wandb.init(project="lm-human-preferences", config=config)
    
    log.info(f'Running with config: {config}')
    if args.dry_run:
        log.info('Running in DRY-Mode')

    if args.train_reward:
        train_reward(args, config)
    elif args.train_lm:
        train_lm(args, config)
    elif args.train_sft:
        train_sft(args, config)
    elif args.inference:
        inference(args, config)
    else:
        raise ValueError('Either --train-reward, --train-lm, --train-sft, or --inference must be set')

def train_reward(args, config):

    from lm_human_preferences.data.base import RewardData
    from lm_human_preferences.lm.reward import RewardModel
    from transformers import TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback
    from transformers import AdamW, get_cosine_schedule_with_warmup
    import os
    import numpy as np

    log.info(f'Training Reward Model with config {config}')

    model = RewardModel(config.reward['model']).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(config.reward['model'])

    datasets = RewardData.from_openai_format(tokenizer, config.reward['datasets_path']).to_dataset()
    if args.dry_run:
        datasets = datasets.select(range(100)) 
    
    splits = datasets.train_test_split(test_size=config.reward['test_size'], seed=args.seed)
    train_dataset = RewardData.from_dataset(tokenizer, splits['train'])
    test_dataset = RewardData.from_dataset(tokenizer, splits['test'])

    os.makedirs(args.reward_model, exist_ok=True)
    # TODO: Fix compute metrics error after evaluate phase
    def compute_metrics(eval_pred: EvalPrediction):
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

        # --- compute your metrics ---

        # 1) Accuracy
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
            "mrr":      mrr,
            "loss":     loss   # will become eval_loss if you set metric_for_best_model="loss"
        }

    training_args = TrainingArguments(
        output_dir=args.reward_model,
        num_train_epochs=100 if not args.dry_run else 1,
        per_device_train_batch_size=config.reward['batch_size'],
        per_device_eval_batch_size=config.reward['batch_size'],
        warmup_steps=config.reward['warmup_steps'],
        weight_decay=config.reward['weight_decay'],
        logging_dir='logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        # eval_steps=5,
        save_strategy="epoch",
        # save_steps=500,
        remove_unused_columns=False,
        optim='adamw_torch',
        save_total_limit=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        label_names=["best"],
        greater_is_better=False
    )
    
    optimizer = AdamW(model.model.score.parameters(), lr=config.reward['lr'])

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.reward['warmup_steps'],
        num_training_steps=100 * len(train_dataset) // config.reward['batch_size']
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop if no improvement for 3 evaluation calls
        early_stopping_threshold=0.001  # Minimum change to qualify as improvement
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
    
    trainer.train(resume_from_checkpoint="models/reward_model/checkpoint-5652")

    # Save the best model
    log.info(f"Saving best reward model to {args.reward_model}")
    model.save(args.reward_model)
    log.info(f"Reward model saved successfully to {args.reward_model}")
    
    return model

def train_lm(args, config):
    log.info(f'Training LM Model with config {config}')

    from lm_human_preferences.env.rlhf_env import RLHFEnv
    from lm_human_preferences.data.base import QueryData
    from lm_human_preferences.lm.reward import RewardModel
    from lm_human_preferences.rl.policy import LanguageAgent
    from lm_human_preferences.rl.ppo_trainer import PPOTrainer
    from transformers import AutoTokenizer
    import gymnasium as gym

    reward_model = RewardModel.from_pretrained('models/reward_model').to(args.device)

    def make_env(idx):
        base_seed = 0
        def wrapper():
            # Create a unique seed for each environment
            env_seed = base_seed + idx
            env = RLHFEnv(
                ref_model_name=config.lm['model'],
                reward_model=reward_model,
                dataset=QueryData.from_openai_format(
                    AutoTokenizer.from_pretrained(config.lm['model']),
                    config.lm['datasets_path']
                ),
                kl_coef=config.lm['ppo']['kl_coef'],
                max_generation=config.lm['response_max_len'],
                device=args.device,
                seed=env_seed
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return wrapper
    
    n_env = config.lm['num_envs']
    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_env)])

    agent = LanguageAgent(config.lm['model'], device=args.device)
    trainer = PPOTrainer(
        envs,
        agent,
        gamma=config.lm['gamma'],
        clip_coef=config.lm['ppo']['kl_coef'],
    )
    trainer.train(num_iter=config.lm['epochs'], num_steps=config.lm['response_max_len'])
    agent.save(args.lm_model)

def train_sft(args, config):
    # TODO: Implement
    model_path = args.lm_model
    
    raise NotImplementedError()

def inference(args, config):
    # TODO: Implement
    model_path = args.lm_model
    
    raise NotImplementedError()

def validate_running_params(config):
    assert config.train_reward != config.train_lm, 'Either train_reward or train_lm should be true and not both'

if __name__ == "__main__":
    main()