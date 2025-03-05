import os
import re
import json

import numpy as np
import pandas as pd

import random
import string

from .task import AbstractTask
from .ii_utils import get_em_score

class ZeroshotCoTTask(AbstractTask):
    def _load_split(self, split):
        filename = os.path.join(self.data_dir, "{}.csv".format(split))
        data, size = None, 0
        
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col=0)
            size = len(data)

        if self.args.debug:
            data = data.sample(n=20, random_state=self.args.seed)
        
        return data, size
    
    def load_data(self):
        self.train_data, self.train_size = self._load_split("train")
        self.dev_data, self.dev_size = self._load_split("dev")
        self.test_data, self.test_size = self._load_split("test")
        
        self.data["train"] = self.train_data
        self.data["dev"] = self.dev_data
        self.data["test"] = self.test_data

        self.logger.info("Loading data... # Train: {}; # Dev: {}; # Test: {}".format(self.train_size, self.dev_size, self.test_size))
        
    def evaluate(self, result_df):
        result_df = self._postprocess(result_df)
        labels = result_df["label"].tolist()
        outputs = result_df["output"].tolist()

        scores = []
        for prediction, ans_ in zip(outputs, labels):
            if self.args.task == "math":
                score = prediction == float(ans_)
            elif self.args.task == "bbh":
                score = get_em_score(prediction, str(ans_))
            scores.append(score)

        result_df["score"] = scores
        return result_df, result_df["score"].mean()

    def _postprocess(self, data):
        all_reasoning = []
        all_output = []
        for pred in data["raw_output"].tolist():
            final_pred = pred.strip(" ").strip(".").strip('"').strip("'").strip(":").strip(" ")
            all_reasoning.append(pred)
            all_output.append(final_pred)

        data["output"] = all_output

        return data