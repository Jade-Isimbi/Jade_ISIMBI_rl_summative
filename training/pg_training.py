import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from torch.distributions import Categorical

from environment.custom_env import ChronicCareAdherenceEnv
from training.common import append_result_csv, evaluate_policy, make_env


PPO_RUNS = [
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "ent_coef": 0.01},
    {"learning_rate": 1e-4, "gamma": 0.995, "n_steps": 2048, "ent_coef": 0.005},
    {"learning_rate": 5e-4, "gamma": 0.98, "n_steps": 512, "ent_coef": 0.02},
    {"learning_rate": 2e-4, "gamma": 0.97, "n_steps": 1024, "ent_coef": 0.03},
    {"learning_rate": 7e-4, "gamma": 0.96, "n_steps": 512, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "gamma": 0.985, "n_steps": 1536, "ent_coef": 0.015},
    {"learning_rate": 2e-4, "gamma": 0.992, "n_steps": 2048, "ent_coef": 0.008},
    {"learning_rate": 4e-4, "gamma": 0.975, "n_steps": 768, "ent_coef": 0.02},
    {"learning_rate": 6e-4, "gamma": 0.95, "n_steps": 512, "ent_coef": 0.04},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 1024, "ent_coef": 0.0},
]

REINFORCE_RUNS = [
    {"learning_rate": 1e-3, "gamma": 0.99},
    {"learning_rate": 7e-4, "gamma": 0.98},
    {"learning_rate": 5e-4, "gamma": 0.995},
    {"learning_rate": 2e-3, "gamma": 0.97},
    {"learning_rate": 1e-4, "gamma": 0.99},
    {"learning_rate": 8e-4, "gamma": 0.96},
    {"learning_rate": 3e-4, "gamma": 0.985},
    {"learning_rate": 1.5e-3, "gamma": 0.975},
    {"learning_rate": 6e-4, "gamma": 0.992},
    {"learning_rate": 9e-4, "gamma": 0.95},
]


@dataclass
class ReinforceAgent:
    model: nn.Module

    def predict(self, obs, deterministic=True):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.model(obs_t)
        action = int(torch.argmax(logits, dim=-1).item())
        return action, None


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_reinforce(run_idx: int, cfg: dict, episodes: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = ChronicCareAdherenceEnv(max_days=60)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = PolicyNet(obs_dim, n_actions)
    optimizer = optim.Adam(net.parameters(), lr=cfg["learning_rate"])

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        log_probs = []
        rewards = []
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = net(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(float(reward))
            done = terminated or truncated

        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + cfg["gamma"] * g
            returns.insert(0, g)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.stack([-lp * ret for lp, ret in zip(log_probs, returns_t)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--reinforce_episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    Path("models/pg").mkdir(parents=True, exist_ok=True)
    Path("models/pg/reinforce").mkdir(parents=True, exist_ok=True)

    # PPO runs
    for i, cfg in enumerate(PPO_RUNS, start=1):
        env = make_env(seed=args.seed + i)
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed + i, tensorboard_log="models/analysis/tb/ppo", **cfg)
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        save_path = Path(f"models/pg/ppo_run_{i}")
        model.save(str(save_path))
        metrics = evaluate_policy(model, episodes=10, seed=100 + i)
        append_result_csv("models/analysis/ppo_results.csv", {"algorithm": "PPO", "run": i, **cfg, **metrics, "model_path": str(save_path)})
        env.close()
        print(f"[PPO] Completed run {i}: {metrics}")

    # REINFORCE runs
    for i, cfg in enumerate(REINFORCE_RUNS, start=1):
        net = train_reinforce(i, cfg, episodes=args.reinforce_episodes, seed=args.seed + 200 + i)
        save_path = Path(f"models/pg/reinforce/reinforce_run_{i}.pt")
        torch.save(net.state_dict(), save_path)
        agent = ReinforceAgent(net)
        metrics = evaluate_policy(agent, episodes=10, seed=300 + i)
        append_result_csv(
            "models/analysis/reinforce_results.csv",
            {"algorithm": "REINFORCE", "run": i, **cfg, **metrics, "model_path": str(save_path)},
        )
        print(f"[REINFORCE] Completed run {i}: {metrics}")


if __name__ == "__main__":
    main()
