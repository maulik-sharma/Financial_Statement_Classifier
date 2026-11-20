import json

with open("vocabulary.json") as f:
    data = json.load(f)

print(len(data))