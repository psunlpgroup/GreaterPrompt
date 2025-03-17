import os
import pickle
import guidance
import glob
import random
import itertools
import openai

from .trainer import AbstractTrainer
from .utils import deduplicate, clean_string, get_llm, pack_demo_string, _load_prompt
from .node import Node

class DefaultTrainer(AbstractTrainer):
    
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.llm = get_llm(self.args.optim_model)
        self.prepare_meta_prompts()

    def initialize(self, task):
        train_data = task.get_data_split("train") # training data
        
        # get a list of initial prompts
        if self.args.init_method == "induction":
            prompts = self.initialize_induction(train_data)
            self.logger.info("Initialization prompts: {}".format(prompts))
        elif self.args.init_method == "file":
            prompts = self.initialize_file()
            
        # pack them into nodes for optimization
        initial_nodes = self._pack_initial_nodes(prompts)
        return initial_nodes
    
    def initialize_file(self):
        prompt_files = glob.glob(os.path.join(self.args.data_dir, self.args.init_prompt_file))
        all_prompts = []
        for filename in prompt_files:
            prompt = _load_prompt(filename)
            all_prompts.append(prompt)
        return all_prompts
    
    def initialize_induction(self, train_data):

        init_batch, _ = self._sample_batch(train_data, k=self.args.init_n_demo, method="random")
        demo_string = pack_demo_string(init_batch)
        
        f = self.initializer(demos=demo_string, n_demo=self.args.init_n_demo, max_tokens=self.args.prompt_max_tokens)
        init_instructions = f["instruction"]
        
        # when n=1 guidance returns a string; otherwise it returns a list of strings
        if self.args.init_n_prompts == 1:
            return [clean_string(init_instructions)]
        else:
            return [clean_string(item) for item in init_instructions]
        
    def train(self, model, initial_nodes, task):
        self.model = model
        
        train_data = task.get_data_split("train") # training data
        dev_data = task.get_data_split("dev") # dev data
        
        T = self.args.train_steps # total optimization steps
        
        # initialization/loading states
        checkpoint_path = os.path.join(self.args.output_dir, "checkpoint.pkl")
        if os.path.exists(checkpoint_path) and self.args.resume:
            states, t = self.load_session(checkpoint_path)
            self.logger.info("Resuming from checkpoint {}; t={}".format(checkpoint_path, t))
        else:
            states, t = [initial_nodes], 0
            self.logger.info("Start training from scratch".format(checkpoint_path))
            
        stop_training = False

        # training
        while t < T:
            
            # evaluate all prompt candidates
            for i, node in enumerate(states[-1]):
                
                if self.args.do_validate:
                    dev_results, dev_score = self.evaluate(model, node, task, dev_data, "dev")
                    self.logger.info("[Node {}] Dev Score: {}".format(node.id, dev_score))
                    
                node.save(self.args.output_dir)
                
                if self.args.do_validate and dev_score == 1.0:
                    stop_training = True
                    break
                
                # save checkpoint (save evaluation progress)
                self.save_session(checkpoint_path, states, t)
                
            if stop_training:
                break
                
            # select top k prompts
            selection_criterion = "dev"
            # `backtracking`: selects best states globally
            candidates = list(itertools.chain(*states)) if self.args.backtrack else states[-1]
            selected_nodes = sorted(candidates, key=lambda x: x.scores.get(selection_criterion, -1.0), reverse=True)[:self.args.n_beam] # select best
            self.logger.info("[t={}] Selected nodes: {}".format(t, str([node.id for node in selected_nodes])))
            
            # at the last timestamp do not expand new prompt candidates
            if t == T-1:
                break
            
            # expand new prompt candidates
            new_nodes = []
            for node in selected_nodes:
                _, _ = self.evaluate(model, node, task, train_data, "train")
                new_nodes += self._propose_new_nodes(node)
                    
            # deduplicate
            new_nodes = deduplicate(new_nodes, prev_nodes=list(itertools.chain(*states)))
            states.append(new_nodes)

            # increment the timestamp
            t += 1
            
            # save checkpoint
            self.save_session(checkpoint_path, states, t)
        
        # find the best node
        selection_criterion = "dev" if self.args.do_validate else "train"
        candidates = list(itertools.chain(*states))
        best_node = sorted(candidates, key=lambda x: x.scores.get(selection_criterion, -1.0), reverse=True)[0] # select best
        best_score = best_node.scores[selection_criterion]  

        return best_score, best_node, states
    
    def evaluate(self, model, node, task, data, split_name):
        # data is a dataframe
        
        # the node may be evaluated before the code crashes;
        # in this case we don't need to run evaluation again.
        if split_name not in node.scores:

            if split_name == "train":
                n_target_errors = self.args.batch_size * max(self.args.n_expand, 2) / 2 # ensuring enough error cases for generating feedback
                data = model.run_till_enough_errors(node.prompt, data, task, n_target_errors)
            else:
                data = model.run(node.prompt, data)
                
            data, final_score = task.evaluate(data)
            node.register_results(data, final_score, split_name)
            if self.args.optim_use_momentum and split_name == "train":
                node.history += "\n * At timestamp {}, the accuracy on the training set is {:.4f}".format(
                    node.timestamp, final_score
                )

        return node.results[split_name], node.scores[split_name]
    
    def load_session(self, checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            states, t = pickle.load(f)
        return states, t
            
    def save_session(self, checkpoint_file, states, t):
        with open(checkpoint_file, "wb") as f:
            pickle.dump((states, t), f)

    def _propose_new_nodes(self, node):
        new_nodes = []
        for id in range(self.args.n_expand):
            # sample batch
            batch, indices = self._sample_batch(node.results["train"], k=self.args.batch_size, method=self.args.batching)
            new_node = self._update(node, batch, indices)
            new_node.save(self.args.output_dir)
            new_nodes.append(new_node)
        return new_nodes
    
    def _update(self, node, batch, indices):
        raise NotImplementedError

    def _pack_batch(self, batch):
        s = ""
        if self.args.task in ["math", "bbh"]:
            for i, row in batch.iterrows():
                s += "### Example {}\n".format(i)
                s += "Input: {}\n".format(row["input"])
                s += "Reasoning: {}\n".format(row["reasoning"])
                s += "Output: {}\n".format(row["output"])
                s += "Label: {}\n\n".format(row["label"])
        elif self.args.task == "ii":
            for i, row in batch.iterrows():
                s += "### Example {}\n".format(i)
                s += "Input: {}\n".format(row["input"])
                s += "Output: {}\n".format(row["output"])
                # II tasks may have multiple correct answers
                s += "All Correct Labels: {}\n\n".format(row["label"])
        elif self.args.task == "cf":
            for i, row in batch.iterrows():
                s += "### Example {}\n".format(i)
                s += "Input: {}\n".format(row["input"])
                s += "Output: {}\n".format(row["output"])
                s += "Label: {}\n\n".format(row["label"][0])
        return s
    
    def _pack_initial_nodes(self, prompts):
        nodes = []
        for i, prompt in enumerate(prompts):
            node = Node(
                timestamp=0,
                id=str(i), 
                prompt=prompt
            )
            nodes.append(node)
        return nodes
    
    def _sample_batch(self, results, k, method="hard"):
        if method == "random":

            indices = results.index.tolist()
            random_indices = random.sample(indices, k=k)

            selected_examples = results.loc[random_indices]
            return selected_examples, random_indices

        elif method == "hard":

            failure_indices = results[results['score'] == 0.0].index.tolist()
            true_k = min(k, len(failure_indices)) # in the rare case that k < n_failures

            random_indices = random.sample(failure_indices, k=true_k)
            selected_examples = results.loc[random_indices]
            return selected_examples, random_indices
        
        elif method == "hard_weighted":
            
            failure_indices = results[results['score'] == 0.0].index.tolist()
            failure_scores = 1.0 - results.loc[failure_indices]['score'] # lower score -> higher prob of being selected
            true_k = min(k, len(failure_indices))
        
            # seems that there is no function for weighted unique sampling...
            random_indices = random.choices(failure_indices, weights=failure_scores, k=true_k) 
            selected_examples = results.loc[random_indices]
            return selected_examples, random_indices
            
            