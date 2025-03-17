import os
import json
import ast

import numpy as np
import pandas as pd

import random

from .task import AbstractTask
from .ii_utils import TASK_TO_METRIC, score_fn

class DirectTask(AbstractTask):
    
    def __init__(self, args, logger, data_dir):
        AbstractTask.__init__(self, args, logger, data_dir)
        self.sample_id = str(data_dir.split("/")[-1])

    def _load_split(self, split):
        filename = os.path.join(self.data_dir, "{}.csv".format(split))
        data, size = None, 0
            
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col=0)
            data.fillna('', inplace=True)
            
            indices, inputs, outputs = [], [], []
            for i, row in data.iterrows():
                index = row.name
                # breakpoint()
                # adapted from https://github.com/keirp/automatic_prompt_engineer/blob/main/experiments/data/instruction_induction/load_data.py
                if self.subtask == 'cause_and_effect':
                    cause, effect = row['cause'], row['effect']
                    # Pick an order randomly
                    if random.random() < 0.5:
                        input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
                    else:
                        input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
                    output_ = [cause]
                elif self.subtask == 'common_concept':
                    items = ast.literal_eval(row['items'])
                    # Make comma separated list of items
                    input_ = ', '.join(items) # ? why does the original code has items[:-1]?
                    output_ = ast.literal_eval(row['all_common_concepts'])
                elif self.subtask == 'rhymes':
                    input_, output_ = row['input'], ast.literal_eval(row['other_rhymes'])
                elif 'translation' in self.subtask:
                    input_, output_ = row['input'], ast.literal_eval(row['possible_translations'])
                elif self.subtask in ["sum", "diff"]:
                    input_, output_ = row['input'], [str(row['output'])] # do exact match for strings
                elif self.subtask == "num_to_verbal":
                    input_, output_ = str(row['input']), [row['output']] # do exact match for strings
                elif "arithmetic_base" in self.subtask:
                    input_, output_ = row['input'], [str(row['output'])] # do exact match for strings
                else:
                    input_, output_ = row['input'], [row['output']]
                
                indices.append(index)
                inputs.append(input_)
                outputs.append(output_) # output_ is a list of acceptable answers
            
            data_dict = {'input': inputs, 'label': outputs, 'index': indices}
            data = pd.DataFrame(data_dict).set_index('index')

            if self.args.debug and split in ["train", "dev"]:
                data = data.sample(n=5, random_state=self.args.seed)
                self.logger.info("Loading 5 {} examples in debug mode".format(split))
                
            size = len(data)
        
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
        labels = result_df["label"].tolist()
        outputs = result_df["raw_output"].tolist()
        result_df["output"] = result_df["raw_output"] # to ensure there is a column named output, will be used in packing batches
        metric = TASK_TO_METRIC.get(self.subtask, "em")

        scores = []
        for prediction, ans_ in zip(outputs, labels):
            score = score_fn(prediction, ans_, metric=metric)
            scores.append(score)
            
        result_df["score"] = scores
        return result_df, result_df["score"].mean()
    
    def __str__(self):
        s = "Task: {}; ".format(self.task)
        s += "Subtask: {}; ".format(self.subtask)
        s += "Sample ID: {}; ".format(self.sample_id)
        s += "Data Dir: {}".format(self.data_dir)
        return s

