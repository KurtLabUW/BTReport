import json

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
