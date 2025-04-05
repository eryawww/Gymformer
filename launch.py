import logging
import wandb
import argparse
import lm_human_preferences
import yaml
from types import SimpleNamespace

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train LM from human preferences')
    # Task
    parser.add_argument('--train-reward', action='store_true', help='Train reward model')
    
    parser.add_argument('--train-lm', action='store_true', help='Train language model')
    parser.add_argument('--reward-model', type=str, default='models/reward_model.pt', help='Saving (--train-reward)/Loading (--train-lm) reward model path')

    # Run config
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
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
    else:
        raise ValueError('Either --train-reward or --train-lm must be set')

def train_reward(args, config):
    log.info(f'Training Reward Model with config {config}')

    trainer = lm_human_preferences.train_reward.RewardTrainer.train(
        base_model=config.reward['model'],
        datasets_path=config.reward['datasets_path'],
        batch_size=config.reward['batch_size'],
        epochs=config.reward['epochs'],
        dry_run=args.dry_run
    )

    trainer.save(args.reward_model)

def train_lm(args, config):
    log.info(f'Training LM Model with config {config}')
    lm_human_preferences.train_lm.LMTrainer.train(
        base_model=config.lm['model'],
        datasets_path=config.lm['datasets_path'],
        batch_size=config.lm['batch_size'],
        epochs=config.lm['epochs'],
        reward_model_path=args.reward_model,
        dry_run=args.dry_run
    )

def validate_running_params(config):
    assert config.train_reward != config.train_lm, 'Either train_reward or train_lm should be true and not both'

if __name__ == "__main__":
    main()