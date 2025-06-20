# Добавьте этот класс в experiments/run_experiment.py (или отдельный файл)
from stable_baselines3.common.callbacks import BaseCallback


class SimpleProgress(BaseCallback):
    def __init__(self, total_timesteps, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.total = total_timesteps
        self.freq = print_freq

    def _on_step(self) -> bool:
        # self.num_timesteps счётчик уже выполненных шагов
        if self.num_timesteps % self.freq == 0:
            pct = (self.num_timesteps / self.total) * 100
            print(f"[Progress] {self.num_timesteps}/{self.total} steps ({pct:.1f}%)")
        return True
