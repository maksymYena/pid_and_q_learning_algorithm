import os

import matplotlib.pyplot as plt
import pandas as pd


def get_latest_run(log_dir="experiments/logs"):
    all_runs = sorted([d for d in os.listdir(log_dir) if d.startswith("run_")])
    if not all_runs:
        raise FileNotFoundError(f"У {log_dir} не знайдено жодної папки run_*")
    return os.path.join(log_dir, all_runs[-1])


def plot_rewards(df, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(df["episode"], df["reward"], marker='o')
    plt.xlabel("Епізод")
    plt.ylabel("Сумарна винагорода")
    plt.title("Зміна сумарної винагороди за епізод")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_per_episode.png"))
    plt.close()


def plot_pid_params(df, out_dir):
    if df.shape[1] == 6:
        # з mae
        step_col, agent_col, kp_col, ki_col, kd_col, mae_col = df.columns
    else:
        step_col, agent_col, kp_col, ki_col, kd_col = df.columns

    agent_ids = df[agent_col].unique()
    for aid in agent_ids:
        adf = df[df[agent_col] == aid]
        plt.figure(figsize=(8, 5))
        plt.plot(adf[step_col], adf[kp_col], label="Kp")
        plt.plot(adf[step_col], adf[ki_col], label="Ki")
        plt.plot(adf[step_col], adf[kd_col], label="Kd")
        plt.xlabel("Крок")
        plt.ylabel("Значення параметра")
        plt.title(f"Динаміка PID-параметрів (Агент {int(aid) + 1})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pid_params_agent_{int(aid) + 1}.png"))
        plt.close()


def plot_metrics(df, out_dir):
    plt.figure(figsize=(9, 6))
    if "mae" in df.columns:
        plt.plot(df["step"], df["mae"], label="MAE")
    if "distance_to_target" in df.columns:
        plt.plot(df["step"], df["distance_to_target"], label="Відстань до цілі")
    if "collisions" in df.columns:
        plt.plot(df["step"], df["collisions"], label="Кількість колізій")
    plt.xlabel("Крок")
    plt.ylabel("Величина метрики")
    plt.title("Динаміка MAE, відстані до цілі та кількості колізій")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance_metrics.png"))
    plt.close()


def plot_trajectories(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 7))
    if df.shape[1] == 2:
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o', label="Агент 1")
    elif df.shape[1] % 2 == 0:
        num_agents = df.shape[1] // 2
        for i in range(num_agents):
            plt.plot(df.iloc[:, 2 * i], df.iloc[:, 2 * i + 1], marker='o', label=f"Агент {i + 1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Траєкторії руху агентів у просторі")
    plt.legend()
    plt.axis("equal")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectories_agents.png"))
    plt.close()


def main():
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
    run_dir = get_latest_run(log_dir)
    print(f"Обробка результатів з: {run_dir}")

    # Епізодичні винагороди
    reward_path = os.path.join(run_dir, "episode_rewards.csv")
    if os.path.exists(reward_path):
        rewards_df = pd.read_csv(reward_path)
        plot_rewards(rewards_df, run_dir)
        print("Збережено: reward_per_episode.png")
    else:
        print("Відсутній episode_rewards.csv")

    # PID параметри
    pid_path = os.path.join(run_dir, "pid_params.csv")
    if os.path.exists(pid_path):
        pid_df = pd.read_csv(pid_path)
        plot_pid_params(pid_df, run_dir)
        print("Збережено: pid_params_agent_X.png")
    else:
        print("Відсутній pid_params.csv")

    # Метрики
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        plot_metrics(metrics_df, run_dir)
        print("Збережено: performance_metrics.png")
    else:
        print("Відсутній metrics.csv")

    # Траєкторії
    traj_path = os.path.join(run_dir, "trajectories.csv")
    if os.path.exists(traj_path):
        plot_trajectories(traj_path, run_dir)
        print("Збережено: trajectories_agents.png")
    else:
        print("Відсутній trajectories.csv")


if __name__ == "__main__":
    main()
