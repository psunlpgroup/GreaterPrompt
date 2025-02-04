from core.optimizer import GreaterOptimizer

import torch
from torch.utils.data import DataLoader

from utils.dataloader import GreaterDataSet

# build dataset
dataset = GreaterDataSet(data_path="./data/BBH/boolean_expressions_refactored.jsonl")

# optimizer config
optimize_config = {
    "intersection": False,
    "candidates_topk": 3,
    "generate_config": {
        # "temperature": 1e-5
    }
}
T = 105

# optimize
optimizer = GreaterOptimizer(
    model="/scratch1/wmz5132/models/huggingface/Llama-3.1-8B-Instruct",
    model_params={"torch_dtype": torch.bfloat16, "device_map": "cuda:6", "low_cpu_mem_usage": True, "trust_remote_code": True},
    tokenizer="/scratch1/wmz5132/models/huggingface/Llama-3.1-8B-Instruct",
    tokenizer_params={"use_fast": True, "padding_side": "left"},
    optimize_config=optimize_config
)

p_stars, meta_info = optimizer.optimize(inputs=dataset, rounds=T)

for p_star, info in zip(p_stars, meta_info):
    print(f'question: {info["question"]}')
    print(f'p_init: {info["p_init"]}')
    print(f'p_star: {p_star}')
    print('--------------------------------')
