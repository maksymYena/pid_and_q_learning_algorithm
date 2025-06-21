import gymnasium
import hydra
import numpy as np
import pandas as pd
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
    ## До изменений
    # === Навчання RL-агента ===
    vec_env = DummyVecEnv([make_env])
    agent = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        ent_coef=0.05,
        learning_rate=0.0003,
        clip_range=0.25,
        policy_kwargs=dict(net_arch=[128, 128])
    )
    agent.learn(total_timesteps=cfg["training"]["timesteps"])

    # === Оцінка RL ===
    test_env = make_env()
    num_episodes = 30

    rl_rewards, pid_rewards = [], []
    episode_rewards = []
    all_metrics = []
    all_pid_params = []

    rl_mae_all, pid_mae_all = [], []
    rl_successes, pid_successes = 0, 0
    rl_steps, pid_steps = [], []

    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        done, cum_reward = False, 0
        episode_mae = []
        step = 0
        while not done:
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = test_env.step(action)
            cum_reward += reward
            done = terminated or truncated

            # Для MAE
            errors = info.get("errors", np.array([np.nan]))
            mae = float(np.mean(errors))
            episode_mae.append(mae)
            step += 1

        rl_rewards.append(cum_reward)
        rl_mae_all.append(np.mean(episode_mae))
        rl_steps.append(step)
        if mae < test_env.unwrapped.done_thresh:
            rl_successes += 1

    pd.DataFrame(episode_rewards).to_csv("episode_rewards.csv", index=False)
    pd.DataFrame(all_metrics).to_csv("metrics.csv", index=False)
    pd.DataFrame(all_pid_params).to_csv("pid_params.csv", index=False)

    # === Оцінка Baseline ===
    target_coord = test_env.unwrapped.target if hasattr(test_env.unwrapped, "target") else np.array([0.0, 0.0])
    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        done, cum_reward = False, 0
        episode_mae = []
        while not done:
            action = static_pid_policy(obs, target_coord)
            obs, reward, terminated, truncated, info = test_env.step(action)
            cum_reward += reward
            done = terminated or truncated

            errors = info.get("errors", np.array([np.nan]))
            mae = float(np.mean(errors))
            episode_mae.append(mae)
            step += 1

        pid_rewards.append(cum_reward)
        pid_mae_all.append(np.mean(episode_mae))
        pid_steps.append(step)
        if mae < test_env.unwrapped.done_thresh:
            pid_successes += 1

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

    print("Mean RL reward:", np.mean(rl_rewards))
    print("Mean PID reward:", np.mean(pid_rewards))

    # MAE
    print("Mean RL MAE:", np.mean(rl_mae_all))
    print("Mean PID MAE:", np.mean(pid_mae_all))

    # % Success
    print("RL success rate:", rl_successes / num_episodes * 100, "%")
    print("PID success rate:", pid_successes / num_episodes * 100, "%")

    # Steps to goal
    print("Mean RL steps:", np.mean(rl_steps))
    print("Mean PID steps:", np.mean(pid_steps))

    # Побудова графіків
    plt.figure(figsize=(8, 5))
    plt.plot(rl_rewards, label="RL+Swarm reward", color="red", marker='o')
    plt.plot(pid_rewards, label="PID baseline reward", color="green", marker='x')
    plt.xlabel("Епізод")
    plt.ylabel("Сумарна винагорода")
    plt.title("Порівняння RL+Swarm та Baseline PID (Reward)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("compare_rewards.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(rl_mae_all, label="RL+Swarm MAE", color="red", marker='o')
    plt.plot(pid_mae_all, label="PID baseline MAE", color="green", marker='x')
    plt.xlabel("Епізод")
    plt.ylabel("MAE")
    plt.title("MAE на епізод")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("compare_mae.png")
    plt.show()

    # % Success
    labels = ['RL+Swarm', 'PID baseline']
    success_rates = [rl_successes / num_episodes * 100, pid_successes / num_episodes * 100]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, success_rates, color=['red', 'green'])
    plt.ylabel('% Успішних епізодів')
    plt.title('Відсоток досягнення цілі')
    plt.tight_layout()
    plt.savefig("compare_success.png")
    plt.show()

    # Steps to goal
    plt.figure(figsize=(8, 5))
    plt.plot(rl_steps, label="RL+Swarm steps", color="red", marker='o')
    plt.plot(pid_steps, label="PID baseline steps", color="green", marker='x')
    plt.xlabel("Епізод")
    plt.ylabel("Кількість кроків")
    plt.title("Середня кількість кроків до цілі")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("compare_steps.png")
    plt.show()


if __name__ == "__main__":
    main()

# Mean RL reward: 450.1154849761437
# Mean PID reward: 420.48454045507566
