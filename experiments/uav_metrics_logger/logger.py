import csv
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_dir="logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.reward_log = os.path.join(self.log_dir, "episode_rewards.csv")
        self.pid_log = os.path.join(self.log_dir, "pid_params.csv")
        self.metrics_log = os.path.join(self.log_dir, "metrics.csv")

        with open(self.reward_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length"])

        with open(self.pid_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "agent_id", "Kp", "Ki", "Kd", "mae"])

        with open(self.metrics_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "distance_to_target", "mae", "collisions"])

    def log_episode(self, episode, reward, length):
        with open(self.reward_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, length])

    def log_pid(self, step, agent_id, kp, ki, kd, mae=0.0):
        with open(self.pid_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, agent_id, kp, ki, kd, mae])

    def log_metrics(self, step, distance, mae, collisions):
        with open(self.metrics_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, distance, mae, collisions])
