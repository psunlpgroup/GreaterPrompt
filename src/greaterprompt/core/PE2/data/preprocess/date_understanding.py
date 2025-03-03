import json
import pandas as pd
import os

input_file = "../data/date_understanding/task.json"

with open(input_file, "r") as fin:
    d = json.load(fin)

print(len(d["examples"]))

_inputs, _outputs = [], []

for example in d["examples"]:
    keys_with_value_1 = [key for key, value in example["target_scores"].items() if value == 1]
    if len(keys_with_value_1) != 1:
        continue
    _inputs.append(example["input"])
    _outputs.append(keys_with_value_1[0])

df = pd.DataFrame({'input': _inputs, 'label': _outputs})
df_shuffled = df.sample(frac=1.0, random_state=42)

df_train = df_shuffled[:100]
df_dev = df_shuffled[100:200]
df_test = df_shuffled[200:]

sample_dir = "../data/date_understanding/"
df_train.to_csv(os.path.join(sample_dir, "train.csv"))
df_dev.to_csv(os.path.join(sample_dir, "dev.csv"))
df_test.to_csv(os.path.join(sample_dir, "test.csv"))

