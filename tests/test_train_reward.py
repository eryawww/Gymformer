# import torch

# from gymformer.train_reward import RewardData, RewardModel
# import os

# PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# def test_combine_query_sample():
#     # Mock RewardModel object
#     class DummyTokenizer:
#         pad_token_id = 0
#     dummy_instance = RewardModel.__new__(RewardModel)
#     dummy_instance.tokenizer = DummyTokenizer()

#     TEST_CASE = [
#         (torch.tensor([1, 2, 0, 0]), torch.tensor([3, 4, 0, 0]), torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])),
#         (torch.tensor([1, 2]), torch.tensor([3, 4, 5, 0]), torch.tensor([1, 2, 3, 4, 5, 0])),
#         (torch.tensor([1, 2, 0]), torch.tensor([3, 4, 5]), torch.tensor([1, 2, 3, 4, 5, 0])),
#         (torch.tensor([1, 2, 3, 4, 0]), torch.tensor([5, 0, 0]), torch.tensor([1, 2, 3, 4, 5, 0, 0, 0])),
#     ]

#     for query, sample, expected in TEST_CASE:
#         combined = dummy_instance._combine_query_sample(query, sample)
#         assert torch.all(combined == expected)


