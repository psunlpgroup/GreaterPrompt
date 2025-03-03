import os
import pandas as pd
import json

data_dir = "../data/math/multiarith/"

data_file = os.path.join(data_dir, "MultiArith.json")

df = pd.read_json(data_file)
# df.rename(columns={'sQuestion': 'input'}, inplace=True)

# checking if all solutions are integer
is_int = df["lSolutions"].apply(lambda x: x[0].is_integer())
print(all(is_int))

df["input"] = df["sQuestion"].apply(lambda x: x.strip())
df["label"] = df["lSolutions"].apply(lambda x: int(x[0]))
df["reasoning"] = df["lEquations"].apply(lambda x: x[0])


df.drop(columns=['lEquations', 'lSolutions', 'lAlignments', 'iIndex', 'sQuestion'], inplace=True)

print(df)

shuffled_df = df.sample(frac=1, random_state=42) # random shuffle

n_train = 100
n_dev = 100

train_df = shuffled_df[:n_train]
dev_df = shuffled_df[n_train:n_train + n_dev]
test_df = shuffled_df[n_train + n_dev: ]

train_df.to_csv(os.path.join(data_dir, "train.csv"))
dev_df.to_csv(os.path.join(data_dir, "dev.csv"))
test_df.to_csv(os.path.join(data_dir, "test.csv"))