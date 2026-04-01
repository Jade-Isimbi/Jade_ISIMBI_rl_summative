import argparse
from pathlib import Path

from stable_baselines3 import DQN

from training.common import append_result_csv, evaluate_policy, make_env


DQN_RUNS = [
    {"learning_rate": 3e-4, "gamma": 0.98, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.2},
    {"learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 100_000, "batch_size": 64, "exploration_fraction": 0.3},
    {"learning_rate": 5e-4, "gamma": 0.97, "buffer_size": 80_000, "batch_size": 128, "exploration_fraction": 0.25},
    {"learning_rate": 2e-4, "gamma": 0.995, "buffer_size": 120_000, "batch_size": 64, "exploration_fraction": 0.15},
    {"learning_rate": 7e-4, "gamma": 0.96, "buffer_size": 60_000, "batch_size": 32, "exploration_fraction": 0.4},
    {"learning_rate": 4e-4, "gamma": 0.985, "buffer_size": 90_000, "batch_size": 128, "exploration_fraction": 0.2},
    {"learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 200_000, "batch_size": 256, "exploration_fraction": 0.1},
    {"learning_rate": 6e-4, "gamma": 0.975, "buffer_size": 70_000, "batch_size": 64, "exploration_fraction": 0.3},
    {"learning_rate": 8e-4, "gamma": 0.95, "buffer_size": 40_000, "batch_size": 32, "exploration_fraction": 0.45},
    {"learning_rate": 3e-4, "gamma": 0.992, "buffer_size": 150_000, "batch_size": 128, "exploration_fraction": 0.12},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path("models/dqn")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(DQN_RUNS, start=1):
        env = make_env(seed=args.seed + i)
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed + i,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            buffer_size=cfg["buffer_size"],
            batch_size=cfg["batch_size"],
            exploration_fraction=cfg["exploration_fraction"],
            tensorboard_log="models/analysis/tb/dqn",
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        run_name = f"dqn_run_{i}"
        save_path = out_dir / run_name
        model.save(str(save_path))

        metrics = evaluate_policy(model, episodes=10, seed=50 + i)
        row = {"algorithm": "DQN", "run": i, **cfg, **metrics, "model_path": str(save_path)}
        append_result_csv("models/analysis/dqn_results.csv", row)
        env.close()
        print(f"[DQN] Completed run {i}: {metrics}")


if __name__ == "__main__":
    main()
