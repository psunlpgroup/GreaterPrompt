import os
import glob

import pandas as pd
import numpy as np

DATA_DIR = "../data/counterfactual-evaluation-original"
OUTPUT_DIR = "../data/counterfactual-evaluation"


def get_label(expr, base):
    lhs, rhs = expr.split("+")
    lhs_base10 = int(lhs, base)
    rhs_base10 = int(rhs, base)
    sum_base10 = lhs_base10 + rhs_base10
    return np.base_repr(sum_base10, base)

def load_file(data_path):
    with open(data_path, "r") as fin:
        lines = fin.readlines()
    return lines

def main():
    print("hello")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # arithmetic
    task = "arithmetic"
    print(task)
    for base in [8,9,10,11,16]:
        data_path = os.path.join(DATA_DIR, task, "data", "0shot", "base{}.txt".format(base))
        lines = load_file(data_path)
        inputs = [line.strip() for line in lines]
        outputs = [get_label(_input, base) for _input in inputs]
        
        df = pd.DataFrame({"input": inputs, "output": outputs})
        
        path = os.path.join(OUTPUT_DIR, "{}_base{}".format(task, base))
        os.makedirs(path, exist_ok=True)
        
        save_path = os.path.join(path, "data.csv")
        df.to_csv(save_path)
        
    # chess
    task = "chess"
    print(task)
    data_path1 = os.path.join(DATA_DIR, task, "data", "chess_4_move", "counter_factual_F_T.txt")
    data_path2 = os.path.join(DATA_DIR, task, "data", "chess_4_move", "counter_factual_T_F.txt")
    lines1 = load_file(data_path1)
    lines2 = load_file(data_path2)
    
    ## original
    inputs = lines1 + lines2
    outputs = ["illegal"] * len(lines1) + ["legal"] * len(lines2)
    
    df = pd.DataFrame({"input": inputs, "output": outputs})
    path = os.path.join(OUTPUT_DIR, "{}_original".format(task))
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, "data.csv")
    df.to_csv(save_path)
    
    ## counterfactual
    inputs = lines1 + lines2
    outputs = ["legal"] * len(lines1) + ["illegal"] * len(lines2)
    
    df = pd.DataFrame({"input": inputs, "output": outputs})
    path = os.path.join(OUTPUT_DIR, "{}_cf".format(task))
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, "data.csv")
    df.to_csv(save_path)
    
    # syntax
    task = "syntax"
    for order in ["osv", "ovs", "sov", "svo", "vos", "vso"]:
        data_path = os.path.join(DATA_DIR, task, "data", "ptb_filtered_data", order, "deps_train.csv")
        df = pd.read_csv(data_path)
        
        inputs = df["reordered_sent"]
        outputs = df["main_subj"] + " " + df["main_verb"]
        
        new_df = pd.DataFrame({"input": inputs, "output": outputs})
        
        path = os.path.join(OUTPUT_DIR, "{}_{}".format(task, order))
        os.makedirs(path, exist_ok=True)
        
        save_path = os.path.join(path, "data.csv")
        new_df.to_csv(save_path)
    

if __name__ == "__main__":
    main()