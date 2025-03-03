import os
import pandas as pd
import json

data_dir = "../data/math/gsm8k/"

def file_to_df(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    return df


### train and dev

# train_file = os.path.join(data_dir, "train.jsonl")
# df = file_to_df(train_file)

# df["label"] = df["answer"].apply(lambda s: s.split("\n####")[-1].strip().replace(",", ""))

# # checking if all solutions are integer
# is_int = df["label"].apply(lambda x: x.strip("-").replace(",", "").isdigit())
# print(all(is_int))

# df.rename(columns={'question': 'input', 'answer': 'reasoning'}, inplace=True)

# shuffled_df = df.sample(frac=1, random_state=42) # random shuffle
# train_df = shuffled_df[:100]
# dev_df = shuffled_df[100:200]

# train_df.to_csv(os.path.join(data_dir, "train.csv"))
# dev_df.to_csv(os.path.join(data_dir, "dev.csv"))

## test

test_file = os.path.join(data_dir, "test.jsonl")
test_df = file_to_df(test_file)

test_df["label"] = test_df["answer"].apply(lambda s: s.split("\n####")[-1].strip().replace(",", ""))
test_df.rename(columns={'question': 'input', 'answer': 'reasoning'}, inplace=True)

test_df.to_csv(os.path.join(data_dir, "test0.csv"))