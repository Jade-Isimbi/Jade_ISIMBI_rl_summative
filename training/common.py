from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import ChronicCareAdherenceEnv


def make_env(seed: int = 0, max_days: int = 60):
    env = ChronicCareAdherenceEnv(max_days=max_days)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_policy(model, episodes: int = 8, seed: int = 123, max_days: int = 60) -> Dict[str, float]:
    env = ChronicCareAdherenceEnv(max_days=max_days)
    rewards = []
    adherences = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        adherence_hits = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            steps += 1
            total_reward += r
            adherence_hits += int(info["adhered"])
            done = terminated or truncated
        rewards.append(total_reward)
        adherences.append(adherence_hits / max(1, steps))
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_adherence": float(np.mean(adherences)),
    }


class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            self.rewards.extend([float(r) for r in self.locals["rewards"]])
        return True


def append_result_csv(csv_path: str, row: Dict):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
