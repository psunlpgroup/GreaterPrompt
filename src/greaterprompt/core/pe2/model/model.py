from .utils import get_llm

class AbstractModel:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    def load_prompt(self):
        raise NotImplementedError
    
    def run(self, data):
        raise NotImplementedError