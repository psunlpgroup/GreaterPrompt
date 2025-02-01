import json
import os
from typing import Any, Dict, List

from torch.utils.data import Dataset

'''
an example of custom inputs:

inputs = [
    {"question": "True and not not ( not False ) is", "answer": "True", "init_prompt": "Use logic thinking step by step"},
    {"question": "What's (5 - 2) * 3 + 10 / 2?", "answer": "17", "init_prompt": "Use math thinking step by step"},
    {"question": "What's the capital of France?", "answer": "Paris", "init_prompt": "Use geography thinking step by step"},
]
'''

class GreaterDataSet(Dataset):
    def __init__(
        self,
        custom_inputs: List[dict] | dict | None = None,
        data_path: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.custom_inputs = custom_inputs
        self.data_path = data_path
        self.items = []

        self._load_data()


    def _load_data(self):
        if self.data_path:
            with open(self.data_path, "r") as f:
                self.items = [json.loads(line) for line in f]
        else:
            self.items = [self.custom_inputs] if isinstance(self.custom_inputs, dict) else self.custom_inputs


    def __len__(self) -> int:
        return len(self.items)
    

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
