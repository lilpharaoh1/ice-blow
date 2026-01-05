import os
import json


class Logger:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.data = []

    def log_episode(self, info):
        self.data.append(info)

    def flush(self):
        with open(f"{self.path}/metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)
