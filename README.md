# Gymformer: PyTorch framework for training Transformer agents in Gymnasium environments

<div align="center">

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

A PyTorch-based framework for training and fine-tuning language models with reinforcement learning. Gymformer bridges the gap between transformer models and Gymnasium environments, allowing seamless integration of language tasks into the RL pipeline.

⚠️ **A word of caution on performance critical task:** Gymformer prioritizes research and education over raw speed. While most RL language model libraries bypass Gymnasium to unleash GPU parallelism, we embrace Gymnasium's sequential MDP framework - trading GPU acceleration for pedagogical clarity and standardization. If you need production-level performance, you'll want to look elsewhere. But if you're here to learn and experiment, you're in the right place!

## 🌟 Key Features: Let the Code Speak

### 💻 Multiple Environments, One Framework

```python
# RLHF: Train LMs to follow human preferences
train_agent(env_name='RLHFEnv-v0', actor=lm_actor, critic=lm_critic, ...)

# Chain-of-Thought: Train LMs to solve math problems step-by-step
train_agent(env_name='CoTEnv-v0', actor=lm_actor, critic=lm_critic, ...)

# Classic RL: Train agents for traditional control tasks
train_agent(env_name='CartPole-v1', actor=mlp_actor, critic=mlp_critic, ...)
```

With minimal code changes, train your models on completely different tasks - from language generation to step-by-step reasoning to classic control problems!

### 🧩 Extensible Modular Architecture

Gymformer is designed for easy extension across its core modules: `env`, `lm`, and `rl`. Creating a new environment is as simple as implementing the Gymnasium interface:

```python
# Create a text summarization environment
class SummarizationEnv(gym.Env):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.tokenizer))
        self.observation_space = spaces.Sequence(spaces.Discrete(len(self.tokenizer)))
        
        # Sample articles to summarize
        self.articles = [
            "Scientists have discovered a new species of deep-sea fish...",
            "The economic outlook suggests inflation will continue to decrease..."
        ]
    
    def reset(self, *, seed=None, options=None):
        # Choose a random article
        article_idx = random.randint(0, len(self.articles) - 1)
        self.current_article = self.articles[article_idx]
        
        # Initial prompt includes the article and a summarization instruction
        prompt = f"Article: {self.current_article}\n\nWrite a concise summary:"
        self.state = self.tokenizer.encode(prompt)
        self.prompt_len = len(self.state)
        
        # Also create reference summary for reward calculation
        self.reference_summary = self._get_reference_summary(article_idx)
        
        return self.state, {"article": self.current_article}
    
    def step(self, action):
        # Update state with the new token
        self.state.append(action)
        generated_text = self.tokenizer.decode(self.state[self.prompt_len:])
        
        # End episode if stop token generated or max length reached
        done = (action == self.tokenizer.eos_token_id) or (len(generated_text) > 100)
        
        # Simple reward: Increasing for brevity, plus bonus for using key terms
        reward = 0
        if done:
            # Length reward: shorter is better (after minimum useful length)
            summary_len = len(generated_text.split())
            if summary_len >= 10:
                reward += max(0, 1.0 - (summary_len / 50))
            
            # Key terms reward
            key_terms = self._extract_key_terms(self.current_article)
            for term in key_terms:
                if term.lower() in generated_text.lower():
                    reward += 0.2
        
        return self.state, reward, done, False, {"summary": generated_text}
    
    def _get_reference_summary(self, article_idx):
        # Would contain reference summaries in a real implementation
        references = [
            "New deep-sea fish discovered with unique adaptations.",
            "Economic reports predict continued decline in inflation rates."
        ]
        return references[article_idx]
    
    def _extract_key_terms(self, article):
        # Simplified extraction of important terms
        # In a real implementation, this would use NLP techniques
        if "species" in article:
            return ["species", "fish", "deep-sea", "discovered"]
        else:
            return ["economic", "inflation", "decrease"]

# Register your environment
gym.envs.register(
    id='Summarization-v0',
    entry_point='your_module.custom_envs:SummarizationEnv',
    kwargs={'model_name': 'openai-community/gpt2'}
)
```

Then use it with the same training interface:

```python
# Train a summarization agent
train_agent(env_name='Summarization-v0', actor=lm_actor, critic=lm_critic, ppo_config=config)
```

Similarly, you can extend the `lm` module with custom reward models or the `rl` module with new algorithms beyond PPO.

**TODO: `rl` and `lm` abstraction for easy customizable is on the work.**

### 🔄 Framework Components

- **Language Model Policy Gradient**: Train language models as RL agents using policy gradient methods
- **Custom Environments**: Ready-to-use environments for RLHF and Chain-of-Thought reasoning
- **Modular Architecture**: Easily extend with new environments, reward functions, and model architectures
- **Reproducible Research**: Solid foundation for transformer-based RL experiments
- **Gymnasium Integration**: Fully compatible with the Gymnasium interface for RL training
- **Reward Modeling**: Train reward models from human preference data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gymformer.git
cd gymformer

# Install dependencies
poetry install
```

### Basic Usage Examples

Gymformer supports both traditional RL environments and language model environments through a unified interface.

#### 1. Training a Reward Model

```python
from gymformer.trainer import train_reward

# Train a reward model on human preference data
model = train_reward(
    model_name="openai-community/gpt2",
    datasets_path="data/descriptiveness_offline_5k",
    reward_model_path="models/reward_model",
    batch_size=32,
    epochs=3
)
```

#### 2. Training an LM with PPO in the RLHF Environment

```python
from gymformer.rl.transformers_nn import LMActorNetwork, LMCriticNetwork
from gymformer.rl.ppo import PPOConfig
from gymformer.trainer.trainer import train_agent
from gymformer.env.rlhf_env import RLHFEnv
from transformers import AutoTokenizer

# Initialize actor and critic networks
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
actor = LMActorNetwork(
    model_name="openai-community/gpt2",
    pad_token_id=tokenizer.eos_token_id,
)
critic = LMCriticNetwork(
    model_name="openai-community/gpt2",
    pad_token_id=tokenizer.eos_token_id,
)

# Configure PPO
ppo_config = PPOConfig(
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    eps_clip=0.2,
    update_epochs=10
)

# Train the agent
rewards = train_agent(
    env_name='RLHFEnv-v0',
    actor=actor,
    critic=critic,
    ppo_config=ppo_config,
    max_episodes=10,
    max_timesteps=100,
    update_interval=5,
    save_interval=5
)
```

#### 3. Training a Chain-of-Thought Math Solver

```python
from gymformer.rl.transformers_nn import LMActorNetwork, LMCriticNetwork
from gymformer.rl.ppo import PPOConfig
from gymformer.trainer.trainer import train_agent
from transformers import AutoTokenizer

# Initialize with a suitable model for CoT reasoning
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

actor = LMActorNetwork(
    model_name=MODEL_NAME,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
)
critic = LMCriticNetwork(
    model_name=MODEL_NAME,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
)

# Train on the CoT environment
rewards = train_agent(
    env_name='CoTEnv-v0',
    actor=actor,
    critic=critic,
    ppo_config=PPOConfig(),
    max_episodes=3,
    max_timesteps=400
)
```

#### 4. Classic RL Environments

```python
from gymformer.rl.common_nn import MLPActorNetwork, MLPCriticNetwork
from gymformer.rl.ppo import PPOConfig
from gymformer.trainer.trainer import train_agent
import gymnasium as gym

# Setup for classic CartPole environment
env_name = 'CartPole-v1'
env = gym.make(env_name)

actor = MLPActorNetwork(
    env.observation_space.shape[0],
    64,
    env.action_space.n,
)
critic = MLPCriticNetwork(
    env.observation_space.shape[0],
    64,
)

# Train the agent
rewards = train_agent(
    env_name=env_name,
    actor=actor,
    critic=critic,
    ppo_config=PPOConfig(),
    max_episodes=1000
)
```

## 📊 Experiments

The `examples` directory contains Jupyter notebooks demonstrating various use cases:

- Language model policy optimization with RLHF
- Chain-of-Thought reasoning for solving math problems
- Classic RL environments with MLP policies
- Reward model training from human preferences

## 🔍 Under the Hood

Gymformer builds on several key technologies:

- **PyTorch**: For deep learning model implementation
- **Gymnasium**: For the RL environment interface
- **Transformers**: For language model architecture
- **Weights & Biases**: For experiment tracking

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Hugging Face for their Transformers library
- OpenAI for their work on GPT models and RLHF
- The Gymnasium team for their RL environment interface