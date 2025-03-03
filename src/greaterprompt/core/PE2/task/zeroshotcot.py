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
            '''
            try:
                if self.args.task == "math":
                    final_pred = pred.replace(",", "")
                    final_pred = [s for s in re.findall(r'-?\d+\.?\d*', final_pred)]
                    final_pred = float(final_pred[0].rstrip("."))
                elif self.args.task == "bbh":
                    if self.args.subtask in ["disambiguation_qa", "date_understanding", "disambiguation_qa",
                                              "geometric_shapes", "hyperbaton", "logical_deduction_five_objects", 
                                              "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation",
                                              "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names", 
                                              "salient_translation_error_detection", "snarks", "temporal_sequences", 
                                              "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_three_objects", 
                                              "tracking_shuffled_objects_seven_objects"]:
                        # multiple-choice, answer is (A), (B), (C), ...
                        final_pred = re.findall(r'\([A-Z]\)', pred)[0]
                    elif self.args.subtask in ["object_counting", "multistep_arithmetic_two"]:
                        # answer is an interger
                        final_pred = pred.replace(",", "")
                        final_pred = [s for s in re.findall(r'-?\d+\.?\d*', final_pred)]
                        final_pred = final_pred[0].rstrip(".")
                    elif self.args.subtask in ["formal_fallacies", "navigate", "sports_understanding", "web_of_lies"]:
                        pred = pred.lstrip(":")
                        tokens = pred.split()
                        tokens_without_punct = [''.join(char for char in token if char not in string.punctuation) for token in tokens]
                        final_pred = tokens_without_punct[0]
                    else: # dyck_languages, word_sorting
                        final_pred = pred.strip(" ").strip(".").strip('"').strip("'").strip(":").strip(" ")

            except Exception as e:
                final_pred = -1e10 if self.args.task == "math" else pred
                self.logger.info("[Postprocess Exception] Raw Output: {}; Exception: {}".format(pred, e))
            '''
            all_reasoning.append(pred)
            all_output.append(final_pred)

        data["output"] = all_output

        return data