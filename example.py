from greaterprompt import (
    ApeOptimizer, ApoOptimizer, GreaterOptimizer, GreaterDataloader, Pe2Optimizer, TextGradOptimizer
)

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------ Part 1: Build your dataloader ------ #

# method1: use jsonl file to build the dataloader
dataloader1 = GreaterDataloader(data_path="./data/boolean_expressions.jsonl")

# method2: use custom inputs to build the dataloader
dataloader2 = GreaterDataloader(custom_inputs=[
    {
        "question": "((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =", 
        "prompt": "Use logical reasoning and think step by step.", 
        "answer": "24"
    },
    {
        "question": "((-9 * -5 - 6 + -2) - (-8 - -6 * -3 * 1)) =",
        "prompt": "Use logical reasoning and think step by step.",
        "answer": "63"
     },
    {
        "question": "((3 * -3 * 6 + -5) - (-2 + -7 - 7 - -7)) =",
        "prompt": "Use logical reasoning and think step by step.",
        "answer": "-50"
    }
])

# ------ Part 2: Initialize the model and tokenizer for GreaterOptimizer only------ #

# init model and tokenzier if you want to use GreaterOptimizer
MODEL_PATH = "/scratch1/wmz5132/models/huggingface/gemma-2-9b-it"
DEVICE = "cuda"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ------ Part 3: Set the optimize configs (Optional) ------ # 

# for GreaterOptimizer
greater_optimize_config = {
    "intersect_q": 5,
    "candidates_topk": 10,
    "loss_function": F.cross_entropy,
    "perplexity_loss": True,
    "perplexity_lambda": 0.2,
    "filter": True,
    "generate_config": {
        "do_sample": True,
        "temperature": 0.2,
        "max_new_tokens": 512,
    }
}

# for ApeOptimizer, ApoOptimizer, Pe2Optimizer
optimize_config = {
    "task_model": "openai_gpt35_turbo_instruct",
    "optim_model": "openai_gpt4_turbo",
}

# for TextGradOptimizer
textgrad_optimize_config = {
    "evaluation_engine": "meta-llama/Meta-Llama-3-8B-Instruct",
    "test_engine": "meta-llama/Meta-Llama-3-8B-Instruct",
    "device": "cuda"
}

# ------ Part 4: Initialize the Optimizers ------ #

ape_optimizer = ApeOptimizer(
    optimize_config=optimize_config # Optional
)

apo_optimizer = ApoOptimizer(
    optimize_config=optimize_config # Optional
)

greater_optimizer = GreaterOptimizer(
    model=model,
    tokenizer=tokenizer,
    optimize_config=greater_optimize_config # Optional
)

pe2_optimizer = Pe2Optimizer(
    optimize_config=optimize_config # Optional
)

textgrad_optimizer = TextGradOptimizer(
    textgrad_optimize_config # Optional
)

# ------ Part 5: Pass the Dataloader to Optimizer and Optimize ------ #

ape_result = ape_optimizer.optimize(dataloader1, p_init="think step by step")
apo_result = apo_optimizer.optimize(dataloader2, p_init="think step by step")
greater_result = greater_optimizer.optimize(
    dataloader1,
    p_extractor="\nNext, only give the exact answer, no extract words or any punctuation:",
    rounds=80
)
pe2_result = pe2_optimizer.optimize(dataloader2, p_init="think step by step")
textgrad_result = textgrad_optimizer.optimize(dataloader1, p_init="think step by step")

# print results
print(f'ape_result: {ape_result}')
print(f'-' * 30)
print(f'apo_result: {apo_result}')
print(f'-' * 30)
print(f'greater_result: {greater_result}')
print(f'-' * 30)
print(f'pe2_result: {pe2_result}')
print(f'-' * 30)
print(f'textgrad_result: {textgrad_result}')
