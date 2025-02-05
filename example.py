from optimizer import GreaterOptimizer
from utils.dataloader import GreaterDataSet

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example1, use jsonl file to build dataset
dataset1 = GreaterDataSet(data_path="./data/boolean_expressions.jsonl")

# init model and tokenzier
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
)

# optimizer config
optimize_config = {
    "intersection": False,
    "candidates_topk": 3,
    "generate_config": {
        "do_sample": False,
    }
}
T = 105

# optimize
optimizer = GreaterOptimizer(
    model=model,
    tokenizer=tokenizer,
    optimize_config=optimize_config
)

p_stars, meta_info = optimizer.optimize(
    inputs=dataset1, 
    # this extractor will be applied to all prompts inside the dataset
    extractor="Only return the exact answer. \
               Therefore, the final answer (use exact format: '$ True' or '$ False') is $ ",
    rounds=T
)
# print results
for p_star, info in zip(p_stars, meta_info):
    print(f'question: {info["question"]}')
    print(f'p_init: {info["prompt"]}')
    print(f'p_stars: ')
    for i, p in enumerate(p_star):
        print(f'{i + 1}: {p}')
    print('--------------------------------')


# Example2, use custom inputs to build dataset
dataset2 = GreaterDataSet(custom_inputs=[
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
p_stars, meta_info = optimizer.optimize(
    inputs=dataset2, 
    # this extractor will be applied to all prompts inside the dataset
    extractor="Therefore, the final answer (use exactly this format: **NUMBER**, \
               where NUMBER is a positive or negative integer) is **",
    rounds=T
)
for p_star, info in zip(p_stars, meta_info):
    print(f'question: {info["question"]}')
    print(f'p_init: {info["prompt"]}')
    print(f'p_stars: ')
    for i, p in enumerate(p_star):
        print(f'{i + 1}: {p}')
    print('--------------------------------')
