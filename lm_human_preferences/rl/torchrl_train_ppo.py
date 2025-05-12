from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from lm_human_preferences.env.rlhf_env import RLHFEnv
from lm_human_preferences.lm.reward import RewardModel
from lm_human_preferences.data.base import QueryData
from transformers import AutoTokenizer
import gymnasium as gym
import os

# 1 Hyperparameter
device = torch.device("cuda")
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

# 1.1 Data Collection parameters
frames_per_batch = 1000
total_frames = 10_000

# 1.2 PPO Parameter
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# 2. Configuring Environment
reward_model_path = os.path.abspath("models/reward_model")
reward_model = RewardModel.from_pretrained(reward_model_path)
gym.envs.register(
    id='RLHFEnv-v0',
    entry_point='lm_human_preferences.env.rlhf_env:RLHFEnv',
    kwargs={
        'ref_model_name': 'openai-community/gpt2',
        'reward_model': reward_model,
        'dataset': QueryData.from_openai_format(
            AutoTokenizer.from_pretrained('openai-community/gpt2'),
            'data/descriptiveness_offline_5k'
        ),
        'kl_coef': 0.01,
        'max_generation': 64,
        'device': device,
        'seed': 42
    }
)
env = GymEnv("RLHFEnv-v0", categorical_action_encoding=True, device=device)

# 3. Define policy
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_space.n, device=device),
    NormalParamExtractor()
)
policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

# 4. Define value
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)
value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))