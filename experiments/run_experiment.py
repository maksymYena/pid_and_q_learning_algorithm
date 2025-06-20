import os
import subprocess
import sys

proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import gymnasium as gym
import hydra
import numpy as np
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from experiments.uav_metrics_logger.logger import MetricsLogger

# Регистрация должна отработать в core/env.py, но на всякий:
from gymnasium.envs.registration import register

register("AHTMEnv-v0", entry_point="core.env:AHTMEnv")


class EpisodeRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episodes = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        if "episode" in info:
            ep = info["episode"]
            self.episodes.append({
                "r": ep["r"],
                "l": ep["l"],
                "step": self.num_timesteps
            })
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.episodes)
        df.to_csv("episode_rewards.csv", index=False)
        print(f"Wrote {len(self.episodes)} episodes to episode_rewards.csv")


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Фабрика одной среды
    def make_env():
        env = gym.make("AHTMEnv-v0", config=cfg)
        return TimeLimit(env, max_episode_steps=cfg["simulation"].get("max_episode_steps", 200))

    # Вектор из 4 параллельных сред
    vec_env = DummyVecEnv([make_env for _ in range(4)])

    # Создаём PPO
    agent = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=cfg["adaptive"].get("tb_log", None),
        ent_coef=0.01
    )

    # Логгер эпизодов
    cb = EpisodeRewardLogger()
    agent.learn(
        total_timesteps=cfg["training"].timesteps,
        callback=cb
    )
    print("TensorBoard logs in:", cfg["adaptive"].get("tb_log"))

    try:
        subprocess.run([sys.executable, "uav_metrics_logger/plot_metrics.py"], check=True)
    except Exception as e:
        print("Не вдалося побудувати графіки:", e)

    test_env = TimeLimit(
        gym.make("AHTMEnv-v0", config=cfg),
        max_episode_steps=cfg["simulation"].get("max_episode_steps", 200)
    )
    metrics_logger = MetricsLogger(log_dir="logs")
    num_episodes = 50
    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        trajectories = []
        done = False
        step = 0
        cum_reward = 0.0

        while not done and step < cfg["simulation"].get("max_episode_steps", 200):
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = test_env.step(action)

            # Логування метрик
            errors = info.get("errors", np.array([np.nan]))
            mae = float(np.mean(errors))
            pid = info.get("pid", [0.0, 0.0, 0.0])
            kp, ki, kd = pid
            collisions = info.get("collisions", 0)

            metrics_logger.log_metrics(step, distance=mae, mae=mae, collisions=collisions)
            metrics_logger.log_pid(step, agent_id=0, kp=kp, ki=ki, kd=kd, mae=mae)
            positions = obs[:2]  # або obs[:num_dim], якщо agent >1
            trajectories.append(positions.copy())

            cum_reward += reward
            done = terminated or truncated
            step += 1

        # Для першого епізоду можна зберегти траєкторію (або для всіх)
        if episode == 0:
            traj_path = os.path.join(metrics_logger.log_dir, "trajectories.csv")
            pd.DataFrame(trajectories).to_csv(traj_path, index=False)

        metrics_logger.log_episode(episode, reward=cum_reward, length=step)
        print(f"Test episode {episode + 1}/{num_episodes} finished in {step} steps with reward {cum_reward:.2f}")


if __name__ == "__main__":
    main()
