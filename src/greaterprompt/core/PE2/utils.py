import os
import logging

import random
import numpy as np

from .task.collection import Task2Class
from .model.collection import Model2Class
from .trainer.collection import Trainer2Class

def get_logger(args):
    # Create a logger
    logger = logging.getLogger("LABO")
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to output.log
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "output.log"))
    file_handler.setLevel(logging.INFO)

    # Create a stream handler that prints to the screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # Avoid interference with guidance logger
    logger.propagate = False

    return logger

def get_task(args, logger):
    _cls = Task2Class(args.task) 
    task = _cls(args, logger, args.data_dir)
    task.load_data()
    return task
    
def get_model(args, logger):
    _cls = Model2Class(args.model)
    model = _cls(args, logger)
    model.load_prompt()
    return model

def get_trainer(args, logger):
    _cls = Trainer2Class(args.trainer)
    trainer = _cls(args, logger)
    return trainer

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)