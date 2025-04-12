# LM Human Preferences

<div align="center">

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

A PyTorch implementation of language model fine-tuning from human preferences, inspired by the [InstructGPT paper](https://arxiv.org/abs/2203.02155) and [Anthropic's RLHF work](https://arxiv.org/abs/2204.05862). This project provides a modular, reproducible framework for training language models using reinforcement learning from human feedback (RLHF).

## 🌟 Features

- **Reward Model Training**: Train a reward model from human preference data
- **PPO Fine-tuning**: Fine-tune language models using PPO with the trained reward model
- **Reproducibility**: Consistent seeds and device management for reproducible results
- **Experiment Tracking**: Integration with Weights & Biases for experiment tracking
- **Modular Design**: Clean, modular codebase for easy extension and modification

## 📋 Implementation Status

### ✅ Implemented

- Reward model training from human preference data
- PPO-based language model fine-tuning
- Reproducibility controls (seed setting, device management)
- Data processing pipeline for OpenAI format datasets
- Experiment tracking with Weights & Biases

### 🚧 TODO

- Supervised fine-tuning (SFT) implementation
- Inference API for fine-tuned models
- Distributed training support
- More comprehensive evaluation metrics
- Support for additional model architectures

## 🔍 Algorithm

The implementation follows the RLHF approach:

1. Train a reward model from human preference data
2. Fine-tune a language model using PPO with the reward model

Mathematically:

1. We initialize a causal language model $p$:
   $$p(x_1, \dots, x_t) = \prod_{k=0}^t p(x_{k}\mid x_0, \dots, x_{k-1})$$

2. We initialize policy $\pi=p$ and reward function $r:\mathcal{X}\times \mathcal{Y}\to \mathbb{R}$ with the objective:
   $$\mathbb{E}_\pi[r] = \mathbb{E}_{x\sim D, y\sim \pi(\cdot\mid x)}[r(x, y)]$$

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lm-human-preferences.git
cd lm-human-preferences

# Install dependencies
poetry install
```

### 🔄 Reproducibility

This project prioritizes reproducibility through several mechanisms:

1. **Consistent Seed Setting**: All random operations use the same seed across frameworks
   - Python's `random` module
   - NumPy
   - PyTorch (CPU and CUDA)
   - Dataset splitting and shuffling

2. **Device Management**: Flexible device selection with proper handling
   - Run on CPU: `--device cpu`
   - Run on CUDA: `--device cuda`
   - Run on MPS (Apple Silicon): `--device mps`

3. **Checkpoint Metadata**: Training metadata is saved with checkpoints
   - Seed values
   - Device information
   - Configuration parameters

To run the complete pipeline with reproducible results:

```bash
# Run the entire pipeline with default settings
bash run_experiment.sh

# Or run individual steps with specific device and seed
poetry run python launch.py --train-reward --device cuda --seed 42
poetry run python launch.py --train-lm --device cuda --seed 42
```

## 📊 Training Pipeline

1. **Data Preparation**:
   ```bash
   # Download the OpenAI preference datasets
   source scripts/download_openai_data.sh
   
   # Process the data into a usable format
   python scripts/decode_openai_data.py
   ```

2. **Reward Model Training**:
   ```bash
   poetry run python launch.py --train-reward --device cuda
   ```

3. **Language Model Fine-tuning**:
   ```bash
   poetry run python launch.py --train-lm --device cuda
   ```

## 🧪 Experiment Configuration

Configuration is managed through YAML files in the `hyperparameters` directory:

```yaml
# Example configuration in hyperparameters/hyperparameters.yaml
reward:
  model: openai-community/gpt2
  datasets_path:
    - data/descriptiveness_offline_5k
  batch_size: 32
  epochs: 3

lm:
  model: openai-community/gpt2
  datasets_path:
    - data/descriptiveness_offline_5k
  batch_size: 2
  epochs: 5
```

## 📝 Code of Conduct

### Development Practices

1. **Experiment Tracking**: Integration with Weights & Biases
2. **Dependency Management**: Using Poetry
3. **Logging and Debugging**: Comprehensive logging with Python's logging module
4. **Code Organization**: Modular design with clear inheritance patterns
5. **Hardware Optimization**: Memory management and OOM handling

### Commit Format
- `[FEAT]`: New features
- `[EXPERIMENTS]`: Experimental changes
- `[REFACTOR]`: Code refactoring

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenAI for their work on InstructGPT and releasing preference datasets
- Anthropic for their research on RLHF
- Hugging Face for their Transformers library and TRL implementation