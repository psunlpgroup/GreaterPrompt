import os
import json

class Node:
    def __init__(self, timestamp, id, prompt, history=None, reasoning=None,
                 parent=None, optim_batch=None, 
                 train_results=None, train_score=None, 
                 dev_results=None, dev_score=None):
        
        self.timestamp = timestamp
        self.id = id # a string
        self.prompt = prompt
        self.history = history

        self.parent = parent # id of the parent node
        self.optim_batch = optim_batch # indices of training examples that are used to update parent -> this node
        
        self.scores = {} # split_name: score on the split
        self.results = {} # split_name: dataframe of the split

        self.n_child = 0
        self.reasoning = reasoning

    def __str__(self):
        s = "Timestamp: {}".format(self.timestamp)
        s += "\nID: {}".format(self.id)
        s += "\nPrompt: {}".format(self.prompt[:200])
        if len(self.prompt) > 200:
            s += "\n[Displaying the first 200 characters of the prompt ...]"
        return s
    
    def to_dict(self):
        d = {
            "timestamp": self.timestamp,
            "id": self.id,
            "parent": self.parent,
            "scores": self.scores,
            "prompt": self.prompt,
            "history": self.history,
            "optim_batch": self.optim_batch,
            "reasoning": self.reasoning,
        }
            
        return d

    def save(self, out_dir):
        node_dir = os.path.join(out_dir, self.id)
        os.makedirs(node_dir, exist_ok=True)
        
        # full node
        node_file = os.path.join(node_dir, "node.json")
        d = self.to_dict()
        with open(node_file, "w") as f:
            json.dump(d, f, indent=4)
            
        # prompt
        prompt_file = os.path.join(node_dir, "prompt.md")
        with open(prompt_file, "w") as f:
            f.write(self.prompt)
            
        # results (if any)
        for key, value in self.results.items():
            result_file = os.path.join(node_dir, "{}_result.csv".format(key))
            value.to_csv(result_file)
            
    def register_results(self, results, score, split_name):
        self.scores[split_name] = score
        self.results[split_name] = results