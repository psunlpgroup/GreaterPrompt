from core.optimizer import GreaterOptimizer
from utils.dataloader import GreaterDataSet

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# build dataset
dataset = GreaterDataSet(data_path="./data/BBH/boolean_expressions_refactored.jsonl")

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

p_stars, meta_info = optimizer.optimize(inputs=dataset, rounds=T)

# print results
for p_star, info in zip(p_stars, meta_info):
    print(f'question: {info["question"]}')
    print(f'p_init: {info["p_init"]}')
    print(f'p_star: {p_star}')
    print('--------------------------------')
