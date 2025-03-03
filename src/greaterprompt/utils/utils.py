import argparse
import string
import time
from typing import List, Tuple


def clean_string(prompts: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    cleaned_prompts = []

    for prompt, score in prompts:
        prompt = prompt.strip("'")
        prompt = prompt.rstrip('.')
        prompt = prompt.translate(str.maketrans('', '', string.punctuation))
        cleaned_prompts.append((prompt, score))
    
    cleaned_prompts = list(set(cleaned_prompts))
    cleaned_prompts.sort(key=lambda x: x[1])

    return cleaned_prompts


def apo_pe2_args(trainer: str) -> argparse.Namespace:
    args = argparse.Namespace()
    args.do_train = True
    args.do_validate = True
    args.do_test = True
    args.trainer = trainer
    args.backtrack = True
    args.resume = True
    args.meta_prompts_dir = f"./src/greaterprompt/core/PE2/meta_prompts/{trainer}"
    args.model = "zeroshotcot"
    args.task = "bbh"
    args.subtask = "boolean_expressions"
    args.data_dir = "./src/greaterprompt/core/PE2/data/bbh/boolean_expressions"
    args.output_dir = f"./src/greaterprompt/core/PE2/output/boolean_expressions/{trainer}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.task_model = "openai_gpt35_turbo_instruct"
    args.optim_model = "openai_gpt4_turbo"
    args.n_beam = 2
    args.n_expand = 2
    args.train_steps = 2
    args.init_temperature = 0.7
    args.init_method = "file"
    args.init_prompt_file = "prompt.md"
    args.init_n_demo = None
    args.init_n_prompts = 1
    args.prompt_max_tokens = 50
    args.batch_size = 2
    args.step_size = 10
    args.optim_use_gradient = False
    args.optim_use_momentum = False
    args.optim_use_step_size = False
    args.optim_use_instruction = False
    args.optim_use_demonstrations = False
    args.optim_use_optim_tutorial = False
    args.batching = "hard"
    args.bandit = "all"
    args.seed = 42
    args.debug = False

    return args

