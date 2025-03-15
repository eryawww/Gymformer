import logging
import hydra
from omegaconf import DictConfig
import lm_human_preferences

import sys
import argparse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

@hydra.main(version_base=None, config_path="config", config_name="config", )
def train(cfg: DictConfig):
    log.debug('Training')
    print(cfg)
    print(sys.argv)

@hydra.main(version_base=None, config_path="config", config_name="config")
def inference(cfg: DictConfig):
    log.debug('Inference')
    print(cfg)
    print(sys.argv)

def parse_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="train the model")
    parser.add_argument("--inference", action='store_true', help="inference the model")
    return parser.parse_args()

if __name__ == "__main__":
    argv = parse_argv()
    train()