import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm
import os

# Set default seed for reproducibility
DEFAULT_SEED = 42

def set_seed(seed=DEFAULT_SEED):
    """Set seed for reproducibility across all frameworks"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

FILES = [
    'data/cnndm_offline_60k.json',
    'data/cnndm_online_45k.json',
    'data/descriptiveness_offline_5k.json',
    'data/sentimen_offline_5k.json',
    'data/tldr_offline_60k.json',
    'data/tldr_online_45k.json'
]
MODEL_NAME = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def parse_openai_json_to_text(json_file_path, seed=DEFAULT_SEED):
    # Set seed for reproducibility
    set_seed(seed)
    
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Function to decode text using GPT-2 in batches
    def decode_batch(ids):
        decoded_texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
        return decoded_texts
    batch_size = 256  # Define a batch size

    # Assuming the JSON contains a list of items with relevant fields
    batched_data = []
    for i in tqdm(range(0, len(data), batch_size), total=len(data)//batch_size+1):
        batch = data[i:i + batch_size]
        queries = [item.get('query', '') for item in batch]
        samples0 = [item.get('sample0', '') for item in batch]
        samples1 = [item.get('sample1', '') for item in batch]
        samples2 = [item.get('sample2', '') for item in batch]
        samples3 = [item.get('sample3', '') for item in batch]

        # Decode each field separately
        decoded_queries = decode_batch(queries)
        decoded_samples0 = decode_batch(samples0)
        decoded_samples1 = decode_batch(samples1)
        decoded_samples2 = decode_batch(samples2)
        decoded_samples3 = decode_batch(samples3)

        for j, item in enumerate(batch):
            batched_data.append({
                'query': decoded_queries[j],
                'sample0': decoded_samples0[j],
                'sample1': decoded_samples1[j],
                'sample2': decoded_samples2[j],
                'sample3': decoded_samples3[j],
                'best': item.get('best', 0)  # Keep 'best' as is
            })

    # Convert to datasets.Dataset
    dataset = Dataset.from_list(batched_data)

    # Save the dataset to a file
    output_path = json_file_path.removesuffix('.json')
    dataset.save_to_disk(output_path)
    
    # Save seed information
    with open(f"{output_path}/seed_info.json", 'w') as f:
        json.dump({"seed": seed}, f)

# Set seed before processing any files
set_seed(DEFAULT_SEED)

for file in FILES:
    parse_openai_json_to_text(file, DEFAULT_SEED)