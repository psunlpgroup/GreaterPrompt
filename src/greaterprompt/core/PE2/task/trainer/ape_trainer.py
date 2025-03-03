import os
import pickle
import guidance
import glob
import itertools

from .default_trainer import DefaultTrainer
from .utils import clean_string, _load_prompt, deduplicate
from .node import Node

class APETrainer(DefaultTrainer):
    ## simplify/customize the default training loop for APE and iterative APE
    
    def prepare_meta_prompts(self):
        # there is only initializer and proposer in APE
        
        # initializer
        init_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "initializer.md"))
        generation_config = "n={} temperature={} max_tokens={}".format(self.args.init_n_prompts, self.args.init_temperature, self.args.prompt_max_tokens)
        prompt = init_prompt.replace("[[GENERATION_CONFIG]]", generation_config)
        self.initializer = guidance(prompt, llm=self.llm) # guidance program for initialization

        # proposer: ~ optimizer.step()
        proposer_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "proposer.md"))
        generation_config = "n={} temperature={} max_tokens={}".format(self.args.n_expand, self.args.init_temperature, self.args.prompt_max_tokens) # re-using `init_temperature`
        prompt = proposer_prompt.replace("[[GENERATION_CONFIG]]", generation_config)
        self.proposer = guidance(prompt, llm=self.llm)

    def train(self, model, initial_nodes, task):
        # no gradient, just expand
        
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
            
        # initialization only methods (original/non-iterative APE)
        if T == 0:
            selection_criterion = "dev"
            for i, node in enumerate(states[-1]):
                dev_results, dev_score = self.evaluate(model, node, task, dev_data, "dev")
                self.logger.info("[Node {}] Dev Score: {}".format(node.id, dev_score))
                
                node.save(self.args.output_dir)
                self.save_session(checkpoint_path, states, t)
                
        # iterative APE
        while t < T:
            
            # evaluate all prompt candidates
            for i, node in enumerate(states[-1]):
                # train_results, train_score = self.evaluate(model, node, task, train_data, "train")
                # self.logger.info("[Node {}] Train Score: {}".format(node.id, train_score))
                if self.args.do_validate:
                    dev_results, dev_score = self.evaluate(model, node, task, dev_data, "dev")
                    self.logger.info("[Node {}] Dev Score: {}".format(node.id, dev_score))
                    
                node.save(self.args.output_dir)

                # save checkpoint (save evaluation progress)
                self.save_session(checkpoint_path, states, t)
                
            # select top k prompts
            selection_criterion = "dev" if self.args.do_validate else "train"
            candidates = list(itertools.chain(*states)) if self.args.backtrack else states[-1]
            selected_nodes = sorted(candidates, key=lambda x: x.scores[selection_criterion], reverse=True)[:self.args.n_beam] # select best
            # self.logger.info("[t={}] Selected nodes: {}".format(t, str([node.id for node in selected_nodes])))
            for node in selected_nodes:
                self.logger.info("[t={}] [Node: {}] Prompt: {}".format(t, node.id, node.prompt))

            # at the last timestamp do not expand new prompt candidates
            if t == T-1:
                break
            
            # expand new prompt candidates
            new_nodes = []
            for node in selected_nodes:
                # update the prompt
                new_nodes_from_one_node = self._propose_new_nodes(node)
                new_nodes += new_nodes_from_one_node
                    
            # deduplicate
            new_nodes = deduplicate(new_nodes, prev_nodes=list(itertools.chain(*states)))
            states.append(new_nodes)

            # increment the timestamp
            t += 1
            
            # save checkpoint
            self.save_session(checkpoint_path, states, t)

        # find the best node
        best_score, best_node = -1e10, None
        for t in range(len(states)):
            for node in states[t]:
                if node.scores[selection_criterion] > best_score:
                    best_score = node.scores[selection_criterion]
                    best_node = node 

        return best_score, best_node, states
    
    def _propose_new_nodes(self, node):
        guidance.llms.OpenAI.cache.clear()
        f = self.proposer(prompt=node.prompt, max_tokens=self.args.prompt_max_tokens)
        new_nodes = []

        # directly sample n different new prompts
        for idx, item in enumerate(f["new_prompt"]):
            new_prompt = clean_string(item)
            new_node = Node(
                timestamp=node.timestamp+1,
                id=node.id+"-{}".format(node.n_child),
                parent=node.id,
                prompt=new_prompt
            )
            new_node.save(self.args.output_dir)
            new_nodes.append(new_node)
            node.n_child += 1

        return new_nodes
