# Changelog

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

## [Upcoming1]
### Migration to stable-baselines3, gymnasium, and transformer-interaction with focus on implementation correctness
- The upcoming architecture will replace the previous RL training approach, which used a custom reward model with the `trl` library, with a new stack based on `stable-baselines3`, `gymnasium`, and the `transformer-interaction` library.
- This change is intended to provide full flexibility, extensibility, and compatibility with modern RL and simulation environments.
- Users and contributors should refer to the `v0.last` tag for the last stable version of the legacy system based on `trl`.

## [Upcoming2]
### Performance focus implementation