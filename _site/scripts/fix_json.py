# %%
import json
import yaml

JSON_FILE = "../_data/_channels.json"
YAML_FILE = "../_data/channels.yaml"

with open(JSON_FILE, "r") as f:
    channels = json.load(f)

channels

with open(YAML_FILE, "w") as f:
    yaml.dump({"channels": list(channels.values())}, f)

# %%
