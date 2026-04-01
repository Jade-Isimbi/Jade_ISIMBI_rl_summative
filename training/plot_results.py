from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_if_exists(path: str):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


def main():
    dqn = load_if_exists("models/analysis/dqn_results.csv")
    ppo = load_if_exists("models/analysis/ppo_results.csv")
    reinforce = load_if_exists("models/analysis/reinforce_results.csv")

    Path("models/analysis/plots").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    mapping = [("DQN", dqn), ("PPO", ppo), ("REINFORCE", reinforce)]
    for ax, (name, df) in zip(axes.flatten(), mapping):
        if df is None or df.empty:
            ax.set_title(f"{name} (no data)")
            continue
        ax.plot(df["run"], df["mean_reward"], marker="o", label="Mean Reward")
        ax.fill_between(df["run"], df["mean_reward"] - df["std_reward"], df["mean_reward"] + df["std_reward"], alpha=0.2)
        ax.set_title(name)
        ax.set_xlabel("Run")
        ax.set_ylabel("Evaluation Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("models/analysis/plots/cumulative_reward_comparison.png", dpi=200)

    # Adherence comparison
    plt.figure(figsize=(10, 5))
    for name, df in mapping:
        if df is not None and not df.empty:
            plt.plot(df["run"], df["mean_adherence"], marker="o", label=name)
    plt.title("Generalization / Adherence Comparison")
    plt.xlabel("Run")
    plt.ylabel("Mean adherence ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/analysis/plots/adherence_comparison.png", dpi=200)
    print("Saved plots to models/analysis/plots/")


if __name__ == "__main__":
    main()
