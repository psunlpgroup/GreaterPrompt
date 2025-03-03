import os
import argparse
import logging

from run import run
from utils import get_logger, seed_everything

def main():
    parser = argparse.ArgumentParser(description="language-based optimizer")

    # basic
    parser.add_argument("--output_dir", default="output/test", type=str)
    parser.add_argument('--resume', action='store_true', help="resume training/searching from the checkpoint if there is one in output_dir")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_validate', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    
    # task
    parser.add_argument("--task", default=None, required=True)
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--data_dir", default=None, required=True)
    parser.add_argument("--model", choices=["direct", "zeroshotcot"], required=True)

    # llms
    parser.add_argument("--task_model", default="openai_td003")
    parser.add_argument("--optim_model", default="openai_gpt4")
    
    # initialize
    parser.add_argument("--init_method", default="file", choices=["file", "induction"])
    ## used when init method is file
    parser.add_argument("--init_prompt_file", default="prompt.md", type=str, help="the file should be in `data_dir`, supports regex matching.")
    ## used when init method is induction
    parser.add_argument("--init_n_demo", default=None, type=int, help="number of examples used in instruction induction to generate initial prompts.")
    parser.add_argument("--init_n_prompts", default=1, type=int, help="number of initial prompt candidates.")
    parser.add_argument("--init_temperature", default=0.0, type=float, help="number of initial prompt candidates.")
    parser.add_argument("--prompt_max_tokens", default=50, type=int, help="max number of tokens in a prompt")

    # optim
    parser.add_argument("--train_steps", default=10, type=int, help="# number of optimization steps")
    parser.add_argument("--batch_size", default=2, type=int, help="# failure examples included in one optimizaiton step")
    parser.add_argument("--step_size", default=10, type=int, help="# tokens that can be changed during on optimization step")
    parser.add_argument("--n_beam", default=2, type=int, help="beam size in beam search")
    parser.add_argument("--n_expand", default=2, type=int, help="# prompts to expand at each time stamp")
    
    # optim - guidance
    parser.add_argument("--meta_prompts_dir", default="meta_prompts/default", type=str)
    parser.add_argument("--optim_use_gradient", action='store_true')
    parser.add_argument("--optim_use_step_size", action='store_true')
    parser.add_argument("--optim_use_momentum", action='store_true')
    parser.add_argument("--optim_use_instruction", action='store_true')
    parser.add_argument("--optim_use_demonstrations", action='store_true')
    parser.add_argument("--optim_use_optim_tutorial", action='store_true')
    
    # search (TODO)
    parser.add_argument("--trainer", default="default", choices=["default", "ape", "apo", "pe2"])
    parser.add_argument("--batching", default="hard", choices=["random", "hard", "hard_weighted"])
    parser.add_argument("--bandit", default="all", choices=["all", "ucb"])
    parser.add_argument("--backtrack", action='store_true')
    
    # others
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")

    # parse args
    args = parser.parse_args()

    # create output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # create logger
    logger = get_logger(args)
    logger.info(args)
    
    # seed everything
    seed_everything(args.seed)

    run(args, logger)

if __name__ == "__main__":
    main()