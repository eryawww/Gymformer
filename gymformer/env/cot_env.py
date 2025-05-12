import json
from datasets import Dataset, DatasetDict
from typing import Optional, List, Dict, Tuple, Any, Union
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import random
import re
from transformers import AutoTokenizer

def load_math_dataset(json_path: str, test_split: float = 0.1, seed: int = 42) -> DatasetDict:
    """Loads math problems from JSON and splits into train/test datasets."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for item in data:
        item['answer'] = str(item['answer'])
        
    dataset = Dataset.from_list(data)
    
    if test_split > 0:
        dataset_dict = dataset.train_test_split(test_size=test_split, seed=seed)
        return dataset_dict
    else:
        return DatasetDict({'train': dataset})


def _apply_cot_few_shot_template(question):
    """Applies the CoT few-shot template from the prototype notebook."""
    # This template structure should match your cot_ablation.ipynb
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful reasoning assistant.\n"
                "ALWAYS follow the order of operations (BODMAS/PEMDAS):\n"
                "- First, evaluate Parentheses\n"
                "- Then Multiplication and Division (left to right)\n"
                "- Then Addition and Subtraction (left to right)\n"
                "Rewrite the expression at each step by substituting the evaluated part.\n"
                "Use multiple <think> ... </think> blocks.\n"
                "Conclude with <final_answer> ... </final_answer>."
            )
        },
        # Add the few-shot examples from your prototype here...
        {
            "role": "user",
            "content": "What is (3 + 2) * 4?"
        },
        {
            "role": "assistant",
            "content": "<think>Original expression: (3 + 2) * 4</think>\n<think>Evaluate (3 + 2): 5 → becomes 5 * 4</think>\n<think>Now multiply: 5 * 4 = 20</think>\n<final_answer>20</final_answer>"
        },
         {
            "role": "user",
            "content": "What is 2 + 3 * 4?"
        },
        {
            "role": "assistant",
            "content": "<think>Original expression: 2 + 3 * 4</think>\n<think>Multiplication before addition: 3 * 4 = 12 → becomes 2 + 12</think>\n<think>Now add: 2 + 12 = 14</think>\n<final_answer>14</final_answer>"
        },
        # Add other examples as needed...
        
        # The actual question for this episode
        {
            "role": "user",
            "content": f"What is {question}?"
        }
    ]
    return messages

class CoTEnv(gym.Env):
    """
    Chain-of-Thought (CoT) Environment based on math problems.
    Uses a few-shot template and rewards based on <final_answer> accuracy.
    Compatible with Gymnasium and PPO agent.
    """
    metadata = {
        "render_modes": ["human"]
    }

    # Define penalties
    MISSING_ANSWER_PENALTY = -10.0
    INVALID_ANSWER_PENALTY = -5.0

    def __init__(
        self,
        model_name: str, # For tokenizer
        dataset_path: str, # Path to math_eval_rlvr.json
        max_generation: int = 128,
        seed: int = 42,
        test_split: float = 0.1 # Use 0 to use all data for training
    ):
        super().__init__()
        self.seed_value = seed
        self.rng = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        self.max_generation = max_generation

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Define and potentially add special tokens
        self.special_tokens = ["<think>", "</think>", "<final_answer>", "</final_answer>"]
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens})
        if num_added > 0:
            print(f"Added {num_added} special tokens to tokenizer.")
        # Store IDs for checking termination
        self.stop_sequence_ids = self.tokenizer.encode("</final_answer>", add_special_tokens=False)
        self.final_answer_start_id = self.tokenizer.encode("<final_answer>", add_special_tokens=False)[0]
        self.final_answer_end_id = self.tokenizer.encode("</final_answer>", add_special_tokens=False)[-1]


        self.dataset_dict = load_math_dataset(dataset_path, test_split=test_split, seed=seed)
        self.problems = self.dataset_dict['train'] # Use training split

        self.action_space = spaces.Discrete(len(self.tokenizer))
        self.observation_space = spaces.Sequence(spaces.Discrete(len(self.tokenizer)))

        self.state_ids = []
        self.current_problem = None
        self.prompt_len = 0
        self.step_counter = 0
        self.done = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        if seed is not None:
            self.seed_value = seed
            self.rng = random.Random(seed)
            self.np_random = np.random.RandomState(seed)
            self.action_space.seed(seed) # Seed the action space if needed
        
        self.step_counter = 0
        self.done = False
        
        # Sample a new problem
        idx = self.rng.randint(0, len(self.problems))
        self.current_problem = self.problems[idx]
        question = self.current_problem["question"]
        
        # Build prompt using the template
        chat_template = _apply_cot_few_shot_template(question)
        prompt_text = self.tokenizer.apply_chat_template(
            chat_template, 
            tokenize=False, 
            add_generation_prompt=True # Adds assistant prompt turn
        )
        
        self.state_ids = self.tokenizer.encode(prompt_text)
        self.prompt_len = len(self.state_ids)
        
        info = {
            "question": question,
            "correct_answer_str": self.current_problem["answer"]
        }
        return self.state_ids[:], info # Return a copy

    def _check_termination(self) -> bool:
        """Checks if the stop sequence </final_answer> is generated."""
        if len(self.state_ids) < self.prompt_len + len(self.stop_sequence_ids):
            return False
        return self.state_ids[-len(self.stop_sequence_ids):] == self.stop_sequence_ids

    def _parse_final_answer(self) -> Optional[float]:
        """Extracts the numeric answer from <final_answer> tag."""
        full_text = self.tokenizer.decode(self.state_ids[self.prompt_len:])
        match = re.search(r"<final_answer>(.*?)</final_answer>", full_text, re.DOTALL)
        if match:
            answer_str = match.group(1).strip()
            try:
                # Try converting to float, handle potential errors
                return float(answer_str)
            except ValueError:
                return None # Invalid number inside tag
        return None # Tag not found

    def step(self, action: int) -> Tuple[List[int], float, bool, bool, Dict[str, Any]]:
        if self.done:
            # Return dummy values if called after done
            # Alternatively, raise an error as before
            print("Warning: step() called after environment is done.")
            return self.state_ids[:], 0.0, True, True, {}
            # raise RuntimeError("Episode is done. Call reset().")
        
        self.step_counter += 1
        self.state_ids.append(action)
        
        # Check termination conditions
        terminated_by_sequence = self._check_termination()
        truncated_by_length = self.step_counter >= self.max_generation
        self.done = terminated_by_sequence or truncated_by_length
        
        reward = 0.0
        info = {}
        
        if self.done:
            generated_answer_val = self._parse_final_answer()
            correct_answer_str = self.current_problem["answer"]
            
            try:
                correct_answer_val = float(correct_answer_str)
                info["correct_answer_float"] = correct_answer_val
            except ValueError:
                 # Should not happen if dataset loading worked
                print(f"Error: Could not convert correct answer '{correct_answer_str}' to float.")
                correct_answer_val = None
                # Assign max penalty if the ground truth is invalid
                reward = self.MISSING_ANSWER_PENALTY 

            if generated_answer_val is None:
                # Penalty for missing or invalid <final_answer>
                reward = self.MISSING_ANSWER_PENALTY if not terminated_by_sequence else self.INVALID_ANSWER_PENALTY
                info["parse_error"] = "Missing or invalid final_answer tag/content"
            elif correct_answer_val is not None:
                 # Reward based on distance if both answers are valid floats
                difference = abs(generated_answer_val - correct_answer_val)
                reward = -difference # Closer is better (less negative reward)
                info["generated_answer_float"] = generated_answer_val
                info["difference"] = difference
            else:
                # This case handles when correct_answer_val is None but generated_answer_val is not.
                # We already assigned a penalty above when correct_answer_str failed conversion.
                pass # Keep the penalty from the correct_answer conversion failure
                
            info["final_generated_text"] = self.tokenizer.decode(self.state_ids[self.prompt_len:])
        
        # Return state copy, reward, terminated, truncated, info
        # Note: `terminated` signifies reaching a goal state (like finding the answer tag)
        # `truncated` signifies reaching a limit (like max_generation)
        return self.state_ids[:], reward, terminated_by_sequence, truncated_by_length, info

    def render(self, mode='human'):
        if mode == 'human':
            print("-"*20 + f" Step {self.step_counter} " + "-"*20)
            prompt_text = self.tokenizer.decode(self.state_ids[:self.prompt_len])
            generated_text = self.tokenizer.decode(self.state_ids[self.prompt_len:])
            print(f"Prompt:\n{prompt_text}")
            print(f"Generated:\n{generated_text}")
            if self.current_problem:
                print(f"Correct Answer: {self.current_problem['answer']}")
            print("-"*50)
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def close(self):
        pass

# Example Usage (Optional)
if __name__ == '__main__':
    # You need a model name for the tokenizer, e.g., 'gpt2' or a specific chat model
    env = CoTEnv(model_name='openai-community/gpt2', dataset_path='data/math_eval_rlvr.json')
    state, info = env.reset()
    print(f"Initial State Length: {len(state)}")
    print(f"Question: {info['question']}")
    print(f"Correct Answer: {info['correct_answer_str']}")
    env.render()

    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample() # Replace with agent policy
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # env.render() # Uncomment to see each step
        if terminated or truncated:
            done = True
    
    print("\nEpisode Finished.")
    env.render()
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward}")
