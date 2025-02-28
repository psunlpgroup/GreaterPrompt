import json
from typing import Any, Dict, List

from torch.utils.data import Dataset


class GreaterDataloader(Dataset):
    def __init__(
        self, custom_inputs: List[dict] | None = None, data_path: str | None = None, **kwargs
    ):
        super().__init__()

        self.custom_inputs = custom_inputs
        self.data_path = data_path
        self.items = []

        self._load_data() if data_path else self._build_data()


    def _load_data(self):
        with open(self.data_path, "r") as f:
            self.items = [json.loads(line) for line in f]


    def _build_data(self):
        for i, line in enumerate(self.custom_inputs):
            self.items.append({
                "id": line.get("id", str(i)),
                "question": line["question"],
                "prompt": line["prompt"],
                "answer": line["answer"]
            })


    def __len__(self) -> int:
        return len(self.items)
    

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]
