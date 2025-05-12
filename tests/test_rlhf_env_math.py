# import pytest
# import torch
# import numpy as np
# import sys
# import os
# from unittest.mock import patch, MagicMock

# # Add the parent directory to the path so we can import the modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from gymformer.env.rlhf_env import RLHFEnv


# @pytest.fixture
# def env_instance():
#     """Create a minimal RLHFEnv instance for testing."""
#     env = RLHFEnv.__new__(RLHFEnv)  # Create instance without calling __init__
#     env.kl_coef = 0.1
#     env.step_counter = 1
#     env.approx_reward_mean = 0.0
#     env.approx_reward_var = 0.0
#     return env


# def test_kl_penalty_calculation(env_instance):
#     """Test the KL penalty calculation using the actual implementation."""
#     env = env_instance
    
#     # Create mock policy and reference model
#     env.policy = MagicMock()
#     env.reference_model = MagicMock()
    
#     # Set up mock logits
#     policy_logits = torch.zeros((1, 3, 5))
#     policy_logits[0, 2, 3] = 2.0  # log probability for action 3 is 2.0
    
#     ref_logits = torch.zeros((1, 3, 5))
#     ref_logits[0, 2, 3] = 1.0  # log probability for action 3 is 1.0
    
#     # Configure mocks to return the logits
#     env.policy.return_value.logits = policy_logits
#     env.reference_model.return_value.logits = ref_logits
    
#     # Patch the actual _get_kl_penalty method
#     with patch.object(RLHFEnv, '_get_kl_penalty', wraps=RLHFEnv._get_kl_penalty):
#         # Call the method with a mock state and action
#         state = torch.tensor([[1, 2, 3]])
#         action = 3
        
#         # Call the actual implementation
#         kl_penalty = RLHFEnv._get_kl_penalty(env, state, action)
        
#         # Expected: -kl_coef * (policy_logprob - ref_logprob)
#         # = -0.1 * (2.0 - 1.0) = -0.1
#         expected_kl_penalty = -0.1
        
#         assert abs(kl_penalty - expected_kl_penalty) < 1e-5


# def test_whiten_reward_calculation(env_instance):
#     """Test the reward whitening calculation using the actual implementation."""
#     env = env_instance
    
#     # First reward: 2.0
#     reward1 = 2.0
#     whitened1 = RLHFEnv._get_whiten_reward(env, reward1)
    
#     # Expected calculations:
#     # step 1:
#     # old_mean = 0.0
#     # new_mean = 0.0 + (2.0 - 0.0) / 1 = 2.0
#     # var += (2.0 - 0.0) * (2.0 - 2.0) = 0.0
#     # whitened = (2.0 - 2.0) / (0.0 + 1e-8) = 0.0
    
#     assert abs(env.approx_reward_mean - 2.0) < 1e-5
#     assert abs(env.approx_reward_var - 0.0) < 1e-5
#     assert abs(whitened1 - 0.0) < 1e-5
    
#     # Second reward: 4.0
#     env.step_counter = 2
#     reward2 = 4.0
#     whitened2 = RLHFEnv._get_whiten_reward(env, reward2)
    
#     # Expected calculations:
#     # step 2:
#     # old_mean = 2.0
#     # new_mean = 2.0 + (4.0 - 2.0) / 2 = 3.0
#     # var += (4.0 - 2.0) * (4.0 - 3.0) = 2.0
#     # whitened = (4.0 - 3.0) / (2.0 + 1e-8) = 0.5
    
#     assert abs(env.approx_reward_mean - 3.0) < 1e-5
#     assert abs(env.approx_reward_var - 2.0) < 1e-5
#     assert abs(whitened2 - 0.5) < 1e-5
    
#     # Third reward: 0.0
#     env.step_counter = 3
#     reward3 = 0.0
#     whitened3 = RLHFEnv._get_whiten_reward(env, reward3)
    
#     # Expected calculations:
#     # step 3:
#     # old_mean = 3.0
#     # new_mean = 3.0 + (0.0 - 3.0) / 3 = 2.0
#     # var += (0.0 - 3.0) * (0.0 - 2.0) = 6.0
#     # whitened = (0.0 - 2.0) / (8.0 + 1e-8) = -0.25
    
#     assert abs(env.approx_reward_mean - 2.0) < 1e-5
#     assert abs(env.approx_reward_var - 8.0) < 1e-5
#     assert abs(whitened3 - (-0.25)) < 1e-5


# def test_whiten_reward_edge_cases():
#     """Test edge cases for reward whitening using the actual implementation."""
#     # Create a fresh environment instance for this test
#     env = RLHFEnv.__new__(RLHFEnv)
#     env.step_counter = 1
#     env.approx_reward_mean = 0.0
#     env.approx_reward_var = 0.0
    
#     # Test with zero reward
#     reward = 0.0
#     whitened = RLHFEnv._get_whiten_reward(env, reward)
#     assert abs(whitened - 0.0) < 1e-5
    
#     # Reset again
#     env = RLHFEnv.__new__(RLHFEnv)
#     env.step_counter = 1
#     env.approx_reward_mean = 0.0
#     env.approx_reward_var = 0.0
    
#     # Test with negative reward
#     reward = -1.0
#     whitened = RLHFEnv._get_whiten_reward(env, reward)
#     assert abs(whitened - 0.0) < 1e-5  # First step, mean = reward, so whitened = 0
    
#     # Add another step with same reward to test variance
#     env.step_counter = 2
#     whitened = RLHFEnv._get_whiten_reward(env, -1.0)
#     assert abs(whitened - 0.0) < 1e-5  # No variance yet
    
#     # Add a different reward to create variance
#     env.step_counter = 3
#     whitened = RLHFEnv._get_whiten_reward(env, 2.0)
#     assert abs(whitened) > 0.0  # Now we should have non-zero whitened reward
