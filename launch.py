import logging
import hydra
from omegaconf import DictConfig
import lm_human_preferences

import sys
import argparse
from typing import Optional

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

@hydra.main(version_base=None, config_path="config", config_name="config", )
def main(cfg: DictConfig):
    run_config = cfg.run
    log.info(f'Running with config : {run_config}')
    if run_config.dry_run:
        log.info('Running in DRY-Mode')

    if run_config.train_reward:
        train_reward(cfg)
    elif run_config.train_lm:
        pass
    else:
        raise ValueError('Either --train-reward or --train-lm must be set')

def train_reward(cfg: DictConfig):
    log.info(f'Training Reward Model with config {cfg.reward}')

    reward_config = cfg.reward
    lm_human_preferences.train_reward.RewardTrainer.train(
        base_model=reward_config.model, 
        datasets_path=reward_config.datasets_path,
        batch_size=reward_config.batch_size,
        dry_run=cfg.run.dry_run
    )

@hydra.main(version_base=None, config_path="config", config_name="config")
def train_lm(cfg: DictConfig):
    # TODO:
    log.info('Training LM Model')

def validate_running_params(run_config: DictConfig):
    assert run_config.train_reward != run_config.train_lm, 'Either train_reward or train_lm should be true and not both'

if __name__ == "__main__":
    main()