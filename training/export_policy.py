import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import DQN, PPO

from environment.custom_env import ChronicCareAdherenceEnv
from training.pg_training import PolicyNet, ReinforceAgent


def load_model(agent: str, model_path: str):
    if agent == "dqn":
        return DQN.load(model_path)
    if agent == "ppo":
        return PPO.load(model_path)
    if agent == "reinforce":
        env = ChronicCareAdherenceEnv()
        net = PolicyNet(env.observation_space.shape[0], env.action_space.n)
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        return ReinforceAgent(net)
    raise ValueError(f"Unsupported agent: {agent}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["dqn", "ppo", "reinforce"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="deployment/policy_export.json")
    args = parser.parse_args()

    env = ChronicCareAdherenceEnv()
    model = load_model(args.agent, args.model_path)

    samples = []
    for seed in [7, 21, 42]:
        obs, _ = env.reset(seed=seed)
        action, _ = model.predict(obs, deterministic=True)
        samples.append(
            {
                "seed": seed,
                "observation": [float(v) for v in obs.tolist()],
                "predicted_action_id": int(action),
                "predicted_action_name": env.ACTIONS[int(action)],
            }
        )

    payload = {
        "mission": "Chronic disease adherence optimization",
        "model_type": args.agent,
        "model_path": args.model_path,
        "observation_schema": [
            "day_progress",
            "adherence_probability",
            "adherence_streak",
            "missed_streak",
            "fatigue",
            "clinical_risk",
            "engagement",
            "condition_severity",
            "digital_preference",
            "support_network",
        ],
        "action_map": {str(k): v for k, v in env.ACTIONS.items()},
        "sample_predictions": samples,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved policy export JSON to {out_path}")


if __name__ == "__main__":
    main()
