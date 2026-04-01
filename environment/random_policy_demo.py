import argparse
from pathlib import Path

import imageio.v2 as imageio

from environment.custom_env import ChronicCareAdherenceEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=str, default="models/pg/random_policy.gif")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = ChronicCareAdherenceEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    frames = []
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    imageio.mimsave(out_path, frames, duration=1 / 6)
    print(f"Saved random-policy demo to: {out_path}")


if __name__ == "__main__":
    main()
