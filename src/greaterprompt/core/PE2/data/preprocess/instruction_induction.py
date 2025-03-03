# preprocess data of instruction induction into the format used in this codebase
import os
import json

import pandas as pd

INPUT_DIR = "/home/azureuser/automatic_prompt_engineer/experiments/data/instruction_induction/raw"
OUTPUT_DIR = "../data/instruction_induction"

SUBTASKS = ['antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']

N_SAMPLES = 5
SEEDS = [0, 16, 32, 42, 99]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for subtask in SUBTASKS:
        # create subtask dir
        subtask_path = os.path.join(OUTPUT_DIR, subtask)
        os.makedirs(subtask_path, exist_ok=True)
        
        # load subtask data
        induce_data_path = os.path.join(INPUT_DIR, "induce", "{}.json".format(subtask))
        with open(induce_data_path, "r") as fin:
            induce_data = json.load(fin)
        n_induce = induce_data["metadata"]["num_examples"]
        induce_df = pd.DataFrame.from_dict(induce_data["examples"], orient="index")
        # print(induce_df.head())
        
        # following configurations here:
        # https://github.com/keirp/automatic_prompt_engineer/blob/main/experiments/run_instruction_induction.py
        n_train = min(int(n_induce * 0.5), 100)
        n_dev = min(n_induce - n_train, 20)
            
        execute_data = os.path.join(INPUT_DIR, "execute", "{}.json".format(subtask))
        with open(execute_data, "r") as fin:
            execute_data = json.load(fin)
        n_test = execute_data["metadata"]["num_examples"]
        execute_df = pd.DataFrame.from_dict(execute_data["examples"], orient="index")
        # print(execute_df.head())
        
        # looks like all takss has <= 100 examples
        if n_test > 100:
            print("Subtask {} has {} test examples".format(subtask, n_test))
            
        print("Subtask {}: # Train: {}; # Dev: {}; # Test: {}".format(subtask, n_train, n_dev, n_test))
            
        for i in range(N_SAMPLES):
            sample_dir = os.path.join(subtask_path, str(i))
            os.makedirs(sample_dir, exist_ok=True)
            
            shuffled_induce_df = induce_df.sample(frac=1, random_state=SEEDS[i]) # random shuffle
            train_df = shuffled_induce_df[:n_train]
            dev_df = shuffled_induce_df[n_train:n_train + n_dev]
            test_df = execute_df
            
            train_df.to_csv(os.path.join(sample_dir, "train.csv"))
            dev_df.to_csv(os.path.join(sample_dir, "dev.csv"))
            test_df.to_csv(os.path.join(sample_dir, "test.csv"))
    
if __name__ == "__main__":
    main()