# Installation

1. Run `scripts/download_openai_data.sh` to download original openai data.
2. Decode openai tensor format into text first to generalize the code
3. Run `poetry run python launch.py` to run the script

# Tech Stack

1. Experiment Tracking
* Experiment Tracking W&B
* Log metadata with ID

2. Dependency Management
* pipenv
* Use linux (or WSL)

3. Reproducibility
* Set random seed
* Version control, run experiments on committed code
* Document and Automate Reproduction

4. Logging and Debugging
* Use logging module
* Transparency mindset, for every written input and output, consider writing the log
* Dump csv (optional format) as the output
* Profile Performance
* Handle Errors Gracefully using try/except on long running code

5. Code organization & Design Patterns
* Project Structure: Organize your project directory with clarity
* Modular Design: Apply OOP
* Separate Configuration and Hyperparameters in one place: Using hydra
* Unit testing on some crucial and well defined part (like mathematics)
* Use dryrun pattern so that we could test the code locally, utilize proxy pattern

6. Automation & CI/CD
* Commit format [FEAT], [EXPERIMENTS], [REFACTOR]
* Pre-commit Hooks for [EXPERIMENTS]
    * Making sure that onle file in 

7. Hadware Check
* Plan Model Size for Memory, use torch.cuda.get_device_properties(0).total_memory and make sure model parameter fit enough
* Test Allocation or Forward Pass to probe memory usage and computational time required. Do a single model(x) then check torch.cuda.memory_allocated() or use torch.cuda.max_memory_allocated() to see peak usage.
* Catch OOM and Adjust 
```python
try:
    train(model, batch_size=current_bs)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        current_bs = current_bs // 2
        torch.cuda.empty_cache()
        print(f"OOM caught, lowering batch size to {current_bs} and retrying...")
        train(model, batch_size=current_bs)
```
* Monitoring, `nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1` will print the GPU core and memory utilization each second

8. Remote Resource Optimization
* Use a Job Scheduler or Queue System, just use python script for simple clusters
* Using screen for multitask between monitoring and running


# Algorithm
1. We initialize causal language model $p$
$$
p(x_1, \dots, x_t) = \prod_{k=0}^t p(x_{k}\mid x_0, \dots, x_{k-1})
$$
2. We initialize policy $\pi=p$ and reward function $r:\mathcal X\times \mathcal Y\to \mathbb R$ with the objective of
$$
\mathbb E_\pi[r] = \mathbb E_{x\sim D, y\sim \pi(\cdot\mid x)}[r(x, y)]
$$

# Components
1. WnB & Hydra (Logging and config management) integration
2. Commitizen
3. 

# TODO
1. Implement reward model
2. Implemeng hydra for hyperparameter space
3. Understand WnB and track metrics
4. Implement Hydra run and hydra sweep