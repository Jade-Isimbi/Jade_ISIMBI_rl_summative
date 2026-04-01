import argparse
import time
from pathlib import Path

import torch
from stable_baselines3 import DQN, PPO

from environment.custom_env import ChronicCareAdherenceEnv
from training.pg_training import PolicyNet, ReinforceAgent


def load_agent(agent_type: str, model_path: str):
    if agent_type == "dqn":
        return DQN.load(model_path)
    if agent_type == "ppo":
        return PPO.load(model_path)
    if agent_type == "reinforce":
        env = ChronicCareAdherenceEnv()
        net = PolicyNet(env.observation_space.shape[0], env.action_space.n)
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        return ReinforceAgent(net)
    raise ValueError(f"Unsupported agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="ppo", choices=["dqn", "ppo", "reinforce"])
    parser.add_argument("--model_path", type=str, default="models/pg/ppo_run_1.zip")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    env = ChronicCareAdherenceEnv(render_mode="human")
    model = load_agent(args.agent, args.model_path)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            env.render()
            time.sleep(0.12)
            done = terminated or truncated
        print(f"Episode {ep} | reward={total:.2f} | final risk={info['clinical_risk']:.2f} | fatigue={info['fatigue']:.2f}")
    env.close()


if __name__ == "__main__":
    main()
