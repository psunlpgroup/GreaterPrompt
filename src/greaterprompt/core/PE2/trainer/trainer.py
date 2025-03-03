class AbstractTrainer:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def initialize(self, task):
        # initialize from (1) files or (2) induction from few examples
        raise NotImplementedError

    def train(self, model, initial_node, task):
        # returns a best prompt and the final metric on the train set
        raise NotImplementedError
    
    def evaluate(self, model, node, task, split="test"):
        # returns a result_df and a final performance score
        raise NotImplementedError
        
    def load_session(self, path):
        # load checkpoints for resuming from crashes
        raise NotImplementedError
        
    def save_session(self, path):
        # save session at each timestamp
        raise NotImplementedError