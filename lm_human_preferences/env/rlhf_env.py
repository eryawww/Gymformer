from dataclasses import dataclass
from typing import Union, Optional, Any, Dict, Tuple, List
import gymnasium as gym
from gymnasium import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
import torch
from torch import tensor, Tensor
import torch.nn.functional as F
from lm_human_preferences.data.base import QueryData
import numpy as np
import random
from gymnasium.vector import VectorEnv

StateTypes = Union[List[int], str]  # Tokenized sequence or string prompt

def state_str_to_int(tokenizer, state: StateTypes) -> List[int]:
    """Convert string state to token IDs if needed.
    
    Args:
        tokenizer: Tokenizer to convert strings to token IDs
        state: Either a string to tokenize or already tokenized IDs
        
    Returns:
        List of token IDs
    """
    if isinstance(state, str):
        return tokenizer(state).input_ids
    assert isinstance(state, list) and isinstance(state[0], int)
    return state


class RLHFEnv(gym.Env):
    """Reinforcement Learning from Human Feedback (RLHF) Environment for language models.
    
    This environment implements the Gymnasium interface for RLHF training of language models.
    It uses a reference model to compute KL divergence penalties and a reward model
    to provide feedback signals. The environment supports sequential token generation
    with reward shaping based on human preferences.

    Assumption:
        reward_model and ref_model_name does not share the same backbone architecture
    
    Attributes:
        tokenizer: Tokenizer for the language model
        reference_model: Base language model to compare against for KL penalty
        reward_model: Model that provides reward signals based on human preferences
        vocab_size: Size of the vocabulary of the language model
        dataset: Optional dataset for sampling initial states
        max_generation: Maximum number of tokens to generate
        kl_coef: Coefficient for KL divergence penalty
        device: Device to run models on ('cuda' or 'cpu')
        seed: Random seed for reproducibility
    """
    metadata = {
        "render_modes": ["human"]
    }
    
    def __init__(
        self, 
        ref_model_name: str, 
        reward_model: torch.nn.Module, 
        dataset: Optional[QueryData] = None, 
        kl_coef: float = 0.01, 
        max_generation: int = 64, 
        device: str = 'cuda', 
        seed: int = 42
    ):
        """Initialize the RLHF Environment.
        
        Args:
            ref_model_name: Name or path of the reference language model to use
            reward_model: Reward model to use for feedback
            dataset: Optional dataset for sampling initial states
            kl_coef: Coefficient for KL divergence penalty
            max_generation: Maximum number of tokens to generate
            device: Device to run models on ('cuda' or 'cpu')
            seed: Random seed for reproducibility
        """
        super(RLHFEnv, self).__init__()

        # Set seed for reproducibility
        self.seed_value = seed
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.reference_model = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device).requires_grad_(False)
        self.reward_model = reward_model.to(device).requires_grad_(False)

        self.vocab_size = self.reference_model.config.vocab_size
        
        # Create instance-specific random generators
        self.np_random = np.random.RandomState(seed)
        self.rng = random.Random(seed)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.vocab_size)
        self.observation_space = spaces.Sequence(
            spaces.Discrete(self.vocab_size),
            stack=True
        )
        
        # Environment configuration
        self.dataset = dataset
        self.max_generation = max_generation
        self.kl_coef = kl_coef
        self.stop_token_id = (
            self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 
            self.tokenizer.pad_token_id
        )
        self.pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 
            self.tokenizer.eos_token_id
        )
        self.device = device

        # State variables
        self.state = []
        self.step_counter = 0
        self.approx_reward_mean = 0
        self.approx_reward_var = 0

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Optional seed to use for the random number generators
            options: Optional configuration dictionary with keys:
                - initial_state: Optional initial state to use instead of sampling from dataset
        
        Returns:
            observation: The initial observation (token IDs)
            info: Additional information
        """
        super().reset(seed=seed)
        options = options or {}
        
        # Update seed if provided
        if seed is not None:
            self.seed_value = seed
            self.rng = random.Random(seed)
            self.np_random = np.random.RandomState(seed)
            self.action_space.seed(seed)
        
        self.step_counter = 0
        
        # Get initial state from options or sample from dataset
        initial_state = options.get('initial_state', None)
        if initial_state is None and self.dataset is not None:
            # Random sample from dataset
            idx = self.rng.randint(0, len(self.dataset) - 1)
            self.state = self.dataset[idx]
        else:
            self.state = state_str_to_int(self.tokenizer, initial_state)
        
        return self.state, {}

    def step(
        self, 
        action: int
    ) -> Tuple[List[int], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment with the given action.
        
        Args:
            action: Token ID to append to the current sequence
            logprobs: Optional log probabilities of all tokens in vocabulary from the policy
                     (used for KL divergence calculation)
        
        Returns:
            observation: The new state after taking the action
            reward: The reward for taking the action
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Additional information
        """
        self.step_counter += 1
        
        # Check if this is a termination condition
        terminated = action == self.stop_token_id
        truncated = self.step_counter >= self.max_generation
        
        # Update state with the new action
        next_state = self.state + [action]
        tensor_next_state = torch.tensor([next_state]).to(self.device)  # [1, seq_len]
        
        # Compute reward components
        raw_reward = self._get_raw_reward(tensor_next_state)
        assert isinstance(raw_reward, float), f"raw_reward must be float, got {type(raw_reward)}"

        # compute reference_model logprobs 
        with torch.no_grad():
            outputs = self.reference_model(tensor_next_state[:, :-1])
            logits = outputs.logits[:, -1, :]
            logprobs = F.log_softmax(logits, dim=-1)[0]        
        kl_penalty = self._get_kl_penalty(tensor_next_state, action, logprobs)
        assert isinstance(kl_penalty, float), f"kl_penalty must be float, got {type(kl_penalty)}"
        
        reward = raw_reward + kl_penalty

        # Whiten rewards for more stable training
        whiten_reward = self._get_whiten_reward(reward)

        # Prepare info dictionary
        info = {
            "reward": reward,
            "raw_reward": raw_reward,
            "kl": kl_penalty,
            'whiten_reward': whiten_reward,
            "decoded_curr_state": self.tokenizer.decode(self.state),
            "decoded_next_state": self.tokenizer.decode(next_state),
            "logprobs": logprobs.detach().cpu()
        }

        self.state = next_state
        return self.state, reward, terminated, truncated, info

    def _get_raw_reward(self, state: Tensor) -> float:
        """Calculate raw reward from the reward model.
        
        Args:
            state: State tensor
            
        Returns:
            Float reward value
        """
        # last_hidden_states [1, seq_len, hidden_size]
        seq_reward = self.reward_model.model(state).logits  # [1, seq_len, 1]
        return seq_reward.item()
    
    def _get_kl_penalty(self, state: Tensor, action: int, policy_logprobs: Tensor) -> float:
        """Calculate the KL divergence penalty.
        
        This penalizes the policy for diverging too much from the reference model.
        
        Args:
            state: Current state tensor
            action: Action token ID
            policy_logprobs: Log probabilities from policy
            
        Returns:
            KL penalty value (negative for actual penalty)
        """
        # Get reference model logits
        ref_full_logits = self.reference_model(state).logits  # [1, seq_len, vocab_size]
        ref_logprobs = F.log_softmax(ref_full_logits[0, -1, :], dim=-1)  # [vocab_size]
        
        # Extract relevant log probabilities
        action_policy_logprobs = policy_logprobs[action].item()  # [1]
        action_ref_logprobs = ref_logprobs[action].item()  # [1]

        # Calculate KL penalty (negative because we want to minimize KL divergence)
        return -self.kl_coef * (action_policy_logprobs - action_ref_logprobs)
    
    def _get_whiten_reward(self, reward: float) -> float:
        """Normalize rewards using running statistics.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Whitened (normalized) reward
        """
        assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"
        old_approx_reward_mean = self.approx_reward_mean

        # Update running statistics for reward normalization
        self.approx_reward_mean = self.approx_reward_mean + \
                                (reward - self.approx_reward_mean) / max(1, self.step_counter)
        self.approx_reward_var += (reward - old_approx_reward_mean) * (reward - self.approx_reward_mean)
        
        # Normalize reward
        norm_factor = max(1e-8, np.sqrt(self.approx_reward_var / max(1, self.step_counter)))
        return (reward - self.approx_reward_mean) / norm_factor

    def render(self, mode='human'):
        """Render the current state of the environment.
        
        Args:
            mode: Rendering mode (only 'human' supported)
        """
        if mode == 'human':
            decoded_state = self.tokenizer.decode(self.state)
            print(f"Current sequence: {decoded_state}")
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def close(self):
        """Clean up environment resources."""
        pass
    
    @classmethod
    def create_vectorized_envs(
        cls, 
        model_name: str, 
        reward_model: torch.nn.Module,
        dataset: Optional[QueryData] = None,
        num_envs: int = 4,
        kl_coef: float = 0.01,
        max_generation: int = 64,
        device: str = 'cuda',
        start_seed: int = 42
    ) -> VectorEnv:
        """Create a vectorized environment with multiple RLHF environments.
        
        Args:
            model_name: Name or path of the language model to use
            reward_model: Reward model to use for feedback
            dataset: Optional dataset for sampling initial states
            num_envs: Number of environments to create
            kl_coef: Coefficient for KL divergence penalty
            max_generation: Maximum number of tokens to generate
            device: Device to run models on
            start_seed: Starting seed for the environments
            
        Returns:
            VectorEnv containing multiple RLHF environments
        """
        from gymnasium.vector.sync_vector_env import SyncVectorEnv
        
        def make_env(idx: int):
            def _init():
                env = cls(
                    model_name=model_name,
                    reward_model=reward_model,
                    dataset=dataset,
                    kl_coef=kl_coef,
                    max_generation=max_generation,
                    device=device,
                    seed=start_seed + idx
                )
                return env
            return _init
            
        return SyncVectorEnv([make_env(i) for i in range(num_envs)])


if __name__ == '__main__':
    # Example usage
    from transformers import AutoModelForCausalLM
    from lm_human_preferences.lm.reward import RewardModel
    
    MODEL_NAME = 'openai-community/gpt2'
    reward_model = RewardModel.from_pretrained('models/reward_model').to('cuda')
    
    env = RLHFEnv(
        MODEL_NAME, 
        reward_model,
        dataset=QueryData.from_openai_format(
            AutoTokenizer.from_pretrained(MODEL_NAME),
            'data/descriptiveness_offline_5k'
        )
    )
    
    state, info = env.reset()
    print(f"Initial state: {env.tokenizer.decode(state)}")
    
    for _ in range(5):
        action_id = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action_id)
        
        print(f"Token: {env.tokenizer.decode([action_id])}")
        print(f"Reward: {reward:.4f} (raw: {info['raw_reward']:.4f}, kl: {info['kl']:.4f})")
        
        if terminated or truncated:
            print("Episode finished")
            break
    
    print(f"Final text: {env.tokenizer.decode(state)}")