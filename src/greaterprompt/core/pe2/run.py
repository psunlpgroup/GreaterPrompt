import os

from .trainer.default_trainer import DefaultTrainer
from .trainer.node import Node

from .utils import get_task, get_model, get_trainer

def run(args, logger):
    
    # load data
    task = get_task(args, logger)
    logger.info("Initializing task: {}".format(task))

    # load model
    model = get_model(args, logger)

    # get trainner
    trainer = get_trainer(args, logger)
    initial_nodes = trainer.initialize(task)
    
    if args.do_train:
        best_score, best_node, states = trainer.train(model, initial_nodes, task)
    
    if args.do_test:
        if args.do_train:
            test_node = best_node
        else:
            assert len(initial_nodes)==1
            test_node = initial_nodes[0]
        
        data = task.get_data_split("test")
        result, final_score = trainer.evaluate(model, test_node, task, data, "test")
        
        logger.info("[Test] Node ID: {}; Score: {}".format(test_node.id, final_score))
        
        # result.to_csv(os.path.join(args.output_dir, "test_result.csv"))

    return best_node.prompt
