import gymnasium as gym
from gymformer.env.rlhf_env import RLHFEnv
from gymformer.data.base import QueryData
from transformers import AutoTokenizer
from gymformer.lm.reward import RewardModelWrapper
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

gym.envs.register(
    id='RLHFEnv-v0',
    entry_point='gymformer.env.rlhf_env:RLHFEnv',
    kwargs={
        'ref_model_name': 'openai-community/gpt2',
        'reward_model': RewardModelWrapper.from_pretrained('../models/reward_model'),
        'dataset': QueryData.from_openai_format(
            AutoTokenizer.from_pretrained('openai-community/gpt2'),
            '../data/descriptiveness_offline_5k'
        ),
        'kl_coef': 0.01,
        'max_generation': 64,
        'device': device,
        'seed': 42
    }
)

gym.envs.register(
    id='CoTEnv-v0',
    entry_point='gymformer.env.cot_env:CoTEnv',
    kwargs={
        'model_name': 'Qwen/Qwen2-0.5B-Instruct',
        'dataset_path': '../data/math_eval_rlvr.json',
        'max_generation': 128,
        'seed': 42,
        'test_split': 0.1
    }
)

ENV_LIST = [
    'RLHFEnv-v0',
    'CoTEnv-v0'
]
