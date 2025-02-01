import json

with open("./BBH/boolean_expressions_refactored.jsonl", "r") as f:
    for line in f:
        x = json.loads(line)
        print(x)
        break

import sys
sys.exit()

data = []
with open("./BBH/boolean_expressions.json", "r") as f:
    for i, line in enumerate(f):
        x = line.strip().split(",")
        data.append(
            {
                "id": str(i),
                "question": x[0],
                "answer": x[1],
            }
        )

with open("./BBH/boolean_expressions_refactored.jsonl", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")

# print(data)
