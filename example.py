from core.optimizer import GreaterOptimizer

import torch
from torch.utils.data import DataLoader

from utils.dataloader import GreaterDataSet


dataset = GreaterDataSet(data_path="./data/BBH/boolean_expressions_refactored.jsonl")
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)


optimize_config = {
    "intersection": False,
    "candidate_topk": 3,
    "generate_config": {
        "temperature": 1e-5,
        "max_new_tokens": 100
    }
}

optimizer = GreaterOptimizer(
    model="/scratch2/wmz5132/models/huggingface/gemma-2-2b",
    model_params={},
    tokenizer="/scratch2/wmz5132/models/huggingface/gemma-2-2b",
    tokenizer_params={},
    optimize_config=optimize_config
)

p_stars = optimizer.optimize(inputs=dataloader, rounds=10)
