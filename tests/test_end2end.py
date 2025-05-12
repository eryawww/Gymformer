# import subprocess
# import pytest
# import os
# import shutil

# def cleanup_generated_files():
#     # Cleanup
#     folder_to_remove = ['models/test_reward_model.pt', 'models/test_lm']
#     for file in folder_to_remove:
#         if os.path.exists(file):
#             if os.path.isdir(file):
#                 shutil.rmtree(file)
#             else:
#                 os.remove(file)

# @pytest.mark.order(1)
# def test_train_reward():
#     result = subprocess.run(
#         ['poetry', 'run', 'python', 'launch.py', '--train-reward', '--dry-run', '--reward-model', 'models/test_reward_model.pt'],
#         capture_output=True, text=True
#     )
#     assert result.returncode == 0, f"Failed with error: {result.stderr}"

# @pytest.mark.order(2)
# def test_train_lm():
#     result = subprocess.run(
#         ['poetry', 'run', 'python', 'launch.py', '--train-lm', '--dry-run', '--reward-model', 'models/test_reward_model.pt', '--lm-model', 'models/test_lm'],
#         capture_output=True, text=True
#     )
#     assert result.returncode == 0, f"Failed with error: {result.stderr}"

#     cleanup_generated_files()