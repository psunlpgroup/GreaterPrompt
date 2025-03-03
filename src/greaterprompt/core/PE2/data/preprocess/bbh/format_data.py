import json
import pandas as pd
import os

task_names = ["causal_judgement", "disambiguation_qa", "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
              "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects", "multistep_arithmetic_two",
              "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
              "snarks", "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
              "web_of_lies", "word_sorting"]

for task_name in ["date_understanding", "movie_recommendation"]:
    input_file = "../../data/bbh/{}/{}.json".format(task_name, task_name)

    with open(input_file, "r") as fin:
        d = json.load(fin)

    print(len(d["examples"]))
    if len(d["examples"]) < 250:
        print("Task {} has less than 250 examples. ".format(task_name))
        continue

    _inputs, _outputs = [], []

    if "input" not in d["examples"][0] or "target" not in d["examples"][0]:
        print("Task {} needs special processing. ".format(task_name))
        continue

    for example in d["examples"]:
        _inputs.append(example["input"])
        _outputs.append(example["target"])

    df = pd.DataFrame({'input': _inputs, 'label': _outputs})
    df_shuffled = df.sample(frac=1.0, random_state=42)

    df_train = df_shuffled[:100]
    df_dev = df_shuffled[100:200]
    df_test = df_shuffled[200:]

    sample_dir = "../../data/bbh/{}".format(task_name)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    df_train.to_csv(os.path.join(sample_dir, "train.csv"))
    df_dev.to_csv(os.path.join(sample_dir, "dev.csv"))
    df_test.to_csv(os.path.join(sample_dir, "test.csv"))

# for task_name in ["causal_judgement", "penguins_in_a_table", "snarks"]:
#     input_file = "../../data/bbh/{}/{}.json".format(task_name, task_name)

#     with open(input_file, "r") as fin:
#         d = json.load(fin)

#     print(len(d["examples"]))
#     _inputs, _outputs = [], []

#     for example in d["examples"]:
#         _inputs.append(example["input"])
#         _outputs.append(example["target"])

#     df = pd.DataFrame({'input': _inputs, 'label': _outputs})
#     df_shuffled = df.sample(frac=1.0, random_state=42)

#     df_train = df_shuffled[:50]
#     df_dev = df_shuffled[50:100]
#     df_test = df_shuffled[100:]

#     sample_dir = "../../data/bbh/{}".format(task_name)
#     if not os.path.exists(sample_dir):
#         os.makedirs(sample_dir)

#     df_train.to_csv(os.path.join(sample_dir, "train.csv"))
#     df_dev.to_csv(os.path.join(sample_dir, "dev.csv"))
#     df_test.to_csv(os.path.join(sample_dir, "test.csv"))