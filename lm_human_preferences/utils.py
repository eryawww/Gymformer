import torch

def gpt2_replace_unk_to_pad(query_tokens: torch.Tensor, pad_token: int) -> torch.Tensor:
    """
        Special handle function because GPT default doesn't have any pad tokens
        in this dataset, they pad with 50259 ('') which we need to remove for query before concatenating
    """
    query_tokens[query_tokens == 50259] = pad_token
    return query_tokens