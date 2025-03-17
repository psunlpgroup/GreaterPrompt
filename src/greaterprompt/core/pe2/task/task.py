import os
import pandas as pd

class AbstractTask:
    def __init__(self, args, logger, data_dir):
        self.args = args
        self.logger = logger
        self.task = args.task
        self.subtask = args.subtask
        self.data_dir = data_dir
        
        self.data = {} # split_name: data

    def load_data(self):
        # ideally train.csv, dev.csv, test.csv in `self.data_dir`
        raise NotImplementedError

    def save_predictions(self, split="test", method=None):
        # rename the file as sth like test_pred_<method_name>.csv
        raise NotImplementedError

    def evaluate(self, labels, predictions):
        # return a list of per-example scores and an overall score
        raise NotImplementedError
    
    def get_data_split(self, split):
        if split not in self.data:
            raise Exception("data split {} does not exist.".format(split))
        else:
            return self.data[split]
    
    def __str__(self):
        s = "Task: {}; ".format(self.task)
        s += "Subtask: {}; ".format(self.subtask)
        s += "Data Dir: {}".format(self.data_dir)
        return s