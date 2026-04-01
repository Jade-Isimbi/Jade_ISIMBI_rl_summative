import argparse

import torch
from fastapi import FastAPI
from pydantic import BaseModel
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


class PredictRequest(BaseModel):
    observation: list[float]


def create_app(agent: str, model_path: str):
    env = ChronicCareAdherenceEnv()
    model = load_model(agent, model_path)
    app = FastAPI(title="Smart Care RL Policy API")

    @app.get("/health")
    def health():
        return {"status": "ok", "agent": agent}

    @app.post("/predict")
    def predict(req: PredictRequest):
        if len(req.observation) != 10:
            return {"error": "observation must have exactly 10 values"}
        action, _ = model.predict(req.observation, deterministic=True)
        a = int(action)
        return {"action_id": a, "action_name": env.ACTIONS[a]}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["dqn", "ppo", "reinforce"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.agent, args.model_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
