import argparse
from pathlib import Path

from stable_baselines3 import PPO

from training.common import append_result_csv, evaluate_policy, make_env


PPO_FOCUSED_RUNS = [
    {"learning_rate": 7e-4, "gamma": 0.98, "n_steps": 512, "ent_coef": 0.005, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 6e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.003, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.985, "n_steps": 768, "ent_coef": 0.005, "clip_range": 0.2, "gae_lambda": 0.96},
    {"learning_rate": 4e-4, "gamma": 0.99, "n_steps": 1024, "ent_coef": 0.002, "clip_range": 0.2, "gae_lambda": 0.97},
    {"learning_rate": 3e-4, "gamma": 0.992, "n_steps": 1024, "ent_coef": 0.001, "clip_range": 0.2, "gae_lambda": 0.97},
    {"learning_rate": 8e-4, "gamma": 0.975, "n_steps": 512, "ent_coef": 0.008, "clip_range": 0.2, "gae_lambda": 0.95},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 1536, "ent_coef": 0.002, "clip_range": 0.15, "gae_lambda": 0.97},
    {"learning_rate": 6e-4, "gamma": 0.985, "n_steps": 768, "ent_coef": 0.004, "clip_range": 0.15, "gae_lambda": 0.96},
    {"learning_rate": 4e-4, "gamma": 0.995, "n_steps": 1024, "ent_coef": 0.001, "clip_range": 0.2, "gae_lambda": 0.98},
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.0, "clip_range": 0.2, "gae_lambda": 0.95},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=700)
    args = parser.parse_args()

    out_dir = Path("models/pg")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, cfg in enumerate(PPO_FOCUSED_RUNS, start=1):
        env = make_env(seed=args.seed + i)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed + i,
            tensorboard_log="models/analysis/tb/ppo_focused",
            **cfg,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        save_path = out_dir / f"ppo_focused_v2_run_{i}"
        model.save(str(save_path))
        metrics = evaluate_policy(model, episodes=12, seed=900 + i)
        append_result_csv(
            "models/analysis/ppo_focused_v2_results.csv",
            {"algorithm": "PPO_FOCUSED_V2", "run": i, **cfg, **metrics, "model_path": str(save_path)},
        )
        env.close()
        print(f"[PPO_FOCUSED] run={i} metrics={metrics}")


if __name__ == "__main__":
    main()
