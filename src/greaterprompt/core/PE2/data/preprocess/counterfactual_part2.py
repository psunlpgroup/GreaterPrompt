import os
import json

import pandas as pd

DATA_DIR = "../data/counterfactual-evaluation"

SUBTASKS = [
    "arithmetic_base10", "arithmetic_base9", "arithmetic_base8", "arithmetic_base11", "arithmetic_base16", 
    "chess_original", "chess_cf",
    "syntax_osv", "syntax_ovs", "syntax_sov", "syntax_svo", "syntax_vos", "syntax_vso"
]

N_SAMPLES = 5
SEEDS = [0, 16, 32, 42, 99]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for subtask in SUBTASKS:
        # create subtask dir
        subtask_path = os.path.join(DATA_DIR, subtask)

        # load subtask data
        data_path = os.path.join(subtask_path, "data.csv")
        df = pd.read_csv(data_path, index_col=0)
        
        n_train = 100
        n_dev = 20
        n_test = 100
            
        for i in range(N_SAMPLES):
            sample_dir = os.path.join(subtask_path, str(i))
            os.makedirs(sample_dir, exist_ok=True)
            
            shuffled_df = df.sample(frac=1, random_state=SEEDS[i]) # random shuffle
            train_df = shuffled_df[:n_train]
            dev_df = shuffled_df[n_train:n_train + n_dev]
            test_df = shuffled_df[n_train + n_dev: n_train + n_dev + n_test]
            
            train_df.to_csv(os.path.join(sample_dir, "train.csv"))
            dev_df.to_csv(os.path.join(sample_dir, "dev.csv"))
            test_df.to_csv(os.path.join(sample_dir, "test.csv"))
    
if __name__ == "__main__":
    main()