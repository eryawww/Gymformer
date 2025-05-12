# Changelog

## [v1.0.0] - 2023-05-13
### Major Framework Transformation: LM Human Preferences → Gymformer
- Complete rebranding from "LM Human Preferences" to "Gymformer" to better reflect the framework's capabilities
- Fully embraced Gymnasium as the unified interface for all environments
- Created modular architecture with extendable components:
  - `env`: Added CoTEnv for Chain-of-Thought math reasoning alongside RLHF
  - `lm`: Reorganized reward model implementations for better interoperability
  - `rl`: Enhanced PPO implementation with direct transformer model support

### New Features
- Custom language model environments now fully compatible with Gymnasium API
- Unified training interface through `train_agent()` for all environment types
- Support for both language tasks (RLHF, CoT) and classic RL problems with the same API
- Simplified extension points for creating custom environments and models
- Enhanced documentation with code examples for multiple use cases

### Technical Changes
- Completely refactored directory structure for better modularity
- Improved environment registration system for custom environments
- Better reward model implementation with clearer integration points
- Standardized observation and action space handling for language models

## [v0.last] - 2025-04-23
### Legacy architecture final state
- This commit marks the last state of the original architecture before a major migration.
- All further development will move towards a new architecture, which may introduce breaking changes.
- The `v0.last` tag and branch serve as a reference point for the legacy codebase and for any hotfixes or maintenance required on the old architecture.

#### Architecture Overview
- The core architecture uses a custom reward model implemented as a subclass of `torch.nn.Module`, trained on human preference data.
- This reward model is integrated with the TRL (Transformer Reinforcement Learning) PPO (Proximal Policy Optimization) trainer to fine-tune language models.
- The system is modular and reproducible, with consistent seed and device management, and comprehensive experiment tracking using Weights & Biases.
- Configuration is managed through YAML files for both reward model and language model training.
- The codebase is organized for extensibility and maintainability, following clear inheritance and modular design patterns.

## Planned Upcoming Milestone
### 1.X Performance focus implementation