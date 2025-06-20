from stable_baselines3.common.callbacks import BaseCallback
import numpy as np, csv

class EpisodeRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        # rewards и dones — списки из VecEnv
        rewards = self.locals['rewards']
        dones   = self.locals['dones']
        truncs  = self.locals.get('truncates', [False]*len(dones))

        # накапливаем
        for r in rewards:
            self.current_rewards.append(r)

        # для каждого env проверяем end-of-episode
        for done, trunc in zip(dones, truncs):
            if done or trunc:
                ep_r = float(np.sum(self.current_rewards))
                self.episode_rewards.append(ep_r)
                mean = float(np.mean(self.episode_rewards))
                # лог в TensorBoard
                self.logger.record("rollout/ep_rew_mean", mean)
                # сбросим
                self.current_rewards = []
        return True

    def _on_training_end(self) -> None:
        # сохраняем CSV
        with open("episode_rewards.csv","w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode","reward"])
            for i,r in enumerate(self.episode_rewards,1):
                w.writerow([i,r])
