import os
import json


class Logger:
    def __init__(self, path):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.data = []
        self.training_metrics = []

    def log_episode(self, info):
        self.data.append(info)

    def log_training_step(self, step, metrics):
        """Log training metrics like loss, Q-values, epsilon."""
        metrics["step"] = step
        self.training_metrics.append(metrics)

    def flush(self):
        with open(f"{self.path}/metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)

        if self.training_metrics:
            with open(f"{self.path}/training_metrics.json", "w") as f:
                json.dump(self.training_metrics, f, indent=2)
