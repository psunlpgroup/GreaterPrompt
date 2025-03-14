import os
import argparse
import logging

from .run import run
from .utils import get_logger, seed_everything

def ape_apo_pe2_optimizer(args):
    # create output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # create logger
    logger = get_logger(args)
    logger.info(args)
    
    # seed everything
    seed_everything(args.seed)

    p_start = run(args, logger)

    return p_start
