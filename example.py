from core.optimizer import GreaterOptimizer

import torch
from torch.utils.data import DataLoader

from utils.dataloader import GreaterDataSet


dataset = GreaterDataSet(data_path="./data/BBH/boolean_expressions_refactored.jsonl")
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)


optimizer = GreaterOptimizer(
    model=None,
    model_params=None,
    intersection=False,
    candidate_topk=3
)
output = optimizer.optimize()

