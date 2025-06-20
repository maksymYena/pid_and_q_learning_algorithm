import gymnasium
import hydra
import numpy as np
# Регистрация должна отработать в core/env.py, но на всякий:
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

register("AHTMEnv-v0", entry_point="core.env:AHTMEnv")


def static_pid_policy(obs, target_coord, Kp=1.0):
    pos = obs[:2]
    error = target_coord - pos
    action = Kp * error
    action = np.clip(action, -1.0, 1.0)
    action3 = np.zeros(3)
    action3[:2] = action
    return action3


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    def make_env():
        env = gymnasium.make("AHTMEnv-v0", config=cfg)
        return TimeLimit(env, max_episode_steps=cfg["simulation"]["max_episode_steps"])

    # === Навчання RL-агента ===
    vec_env = DummyVecEnv([make_env])
    agent = PPO("MlpPolicy", vec_env, verbose=1)
    agent.learn(total_timesteps=cfg["training"]["timesteps"])

    # === Оцінка RL ===
    test_env = make_env()
    rl_rewards, pid_rewards = [], []
    num_episodes = 30
    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        done, cum_reward = False, 0
        while not done:
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = test_env.step(action)
            cum_reward += reward
            done = terminated or truncated
        rl_rewards.append(cum_reward)

    # === Оцінка Baseline ===
    target_coord = test_env.unwrapped.target if hasattr(test_env.unwrapped, "target") else np.array([0.0, 0.0])
    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        done, cum_reward = False, 0
        while not done:
            action = static_pid_policy(obs, target_coord)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            cum_reward += reward
            done = terminated or truncated
        pid_rewards.append(cum_reward)

    print("Mean RL reward:", np.mean(rl_rewards))
    print("Mean PID reward:", np.mean(pid_rewards))

    # === Графік для статті ===
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(rl_rewards, label="RL+Swarm", color="red", marker='o')
    plt.plot(pid_rewards, label="PID baseline", color="green", marker='x')
    plt.xlabel("Епізод")
    plt.ylabel("Сумарна винагорода")
    plt.title("Порівняння RL+Swarm та Baseline PID")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("compare_rewards.png")
    plt.show()
    print("Графік збережено: compare_rewards.png")


if __name__ == "__main__":
    main()
