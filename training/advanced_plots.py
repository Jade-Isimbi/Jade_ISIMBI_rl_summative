from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN, PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from environment.custom_env import ChronicCareAdherenceEnv
from training.common import evaluate_policy
from training.pg_training import PolicyNet, ReinforceAgent
import torch


CSV_FILES = {
    "DQN": "models/analysis/dqn_results.csv",
    "PPO": "models/analysis/ppo_results.csv",
    "REINFORCE": "models/analysis/reinforce_results.csv",
}


def _load_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def _plot_convergence(all_dfs):
    plt.figure(figsize=(10, 5))
    for name, df in all_dfs.items():
        if df is None or df.empty:
            continue
        curve = df["mean_reward"].cummax()
        plt.plot(df["run"], curve, marker="o", label=name)
    plt.title("Convergence Trend by Algorithm (Best-so-far Reward)")
    plt.xlabel("Hyperparameter run index")
    plt.ylabel("Best-so-far mean reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/analysis/plots/convergence_plot.png", dpi=200)


def _plot_tb_scalar(log_dir: str, tag: str, out_path: str, title: str, ylabel: str):
    p = Path(log_dir)
    if not p.exists():
        return False
    event_files = list(p.rglob("events.out.tfevents.*"))
    if not event_files:
        return False

    steps = []
    vals = []
    for ef in event_files:
        try:
            acc = EventAccumulator(str(ef))
            acc.Reload()
            tags = acc.Tags().get("scalars", [])
            if tag in tags:
                for s in acc.Scalars(tag):
                    steps.append(s.step)
                    vals.append(s.value)
        except Exception:
            continue
    if not vals:
        return False

    pairs = sorted(zip(steps, vals), key=lambda x: x[0])
    steps = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, vals, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    return True


def _load_model(algo: str, model_path: str):
    if algo == "DQN":
        return DQN.load(model_path if model_path.endswith(".zip") else model_path + ".zip")
    if algo == "PPO":
        return PPO.load(model_path if model_path.endswith(".zip") else model_path + ".zip")
    if algo == "REINFORCE":
        env = ChronicCareAdherenceEnv()
        net = PolicyNet(env.observation_space.shape[0], env.action_space.n)
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        return ReinforceAgent(net)
    raise ValueError(algo)


def _generalization_tests(all_dfs):
    scenarios = {
        "digital_low_high_severity": {
            "digital_preference": 0.15,
            "condition_severity": 0.9,
            "baseline_adherence": 0.45,
            "fatigue_sensitivity": 0.75,
            "support_network": 0.3,
        },
        "digital_high_strong_support": {
            "digital_preference": 0.9,
            "condition_severity": 0.5,
            "baseline_adherence": 0.55,
            "fatigue_sensitivity": 0.45,
            "support_network": 0.85,
        },
    }

    rows = []
    for algo, df in all_dfs.items():
        if df is None or df.empty:
            continue
        best = df.sort_values("mean_reward", ascending=False).iloc[0]
        model = _load_model(algo, best["model_path"])
        for scenario_name, profile in scenarios.items():
            env = ChronicCareAdherenceEnv(max_days=60)
            rewards = []
            adherences = []
            for ep in range(10):
                obs, _ = env.reset(seed=5000 + ep, options={"profile_override": profile})
                done = False
                total = 0.0
                adheres = 0
                steps = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, terminated, truncated, info = env.step(int(action))
                    total += r
                    adheres += int(info["adhered"])
                    steps += 1
                    done = terminated or truncated
                rewards.append(total)
                adherences.append(adheres / max(1, steps))
            env.close()
            rows.append(
                {
                    "algorithm": algo,
                    "scenario": scenario_name,
                    "mean_reward": float(sum(rewards) / len(rewards)),
                    "mean_adherence": float(sum(adherences) / len(adherences)),
                }
            )

    if not rows:
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv("models/analysis/generalization_results.csv", index=False)

    plt.figure(figsize=(11, 5))
    for scenario_name in out_df["scenario"].unique():
        sdf = out_df[out_df["scenario"] == scenario_name]
        plt.plot(sdf["algorithm"], sdf["mean_reward"], marker="o", label=scenario_name)
    plt.title("Generalization Test: Mean Reward by Scenario")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/analysis/plots/generalization_tests.png", dpi=200)


def main():
    Path("models/analysis/plots").mkdir(parents=True, exist_ok=True)
    all_dfs = {name: _load_csv(path) for name, path in CSV_FILES.items()}

    _plot_convergence(all_dfs)
    _plot_tb_scalar(
        log_dir="models/analysis/tb/dqn",
        tag="train/loss",
        out_path="models/analysis/plots/dqn_objective_curve.png",
        title="DQN Objective Curve (Loss)",
        ylabel="Loss",
    )
    _plot_tb_scalar(
        log_dir="models/analysis/tb/ppo",
        tag="train/entropy_loss",
        out_path="models/analysis/plots/pg_entropy_curve.png",
        title="Policy Gradient Entropy Curve",
        ylabel="Entropy loss",
    )
    _generalization_tests(all_dfs)
    print("Saved advanced analysis artifacts in models/analysis/plots/ and models/analysis/generalization_results.csv")


if __name__ == "__main__":
    main()
