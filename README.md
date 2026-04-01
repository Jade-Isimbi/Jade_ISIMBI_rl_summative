# smart-care-rl

Reinforcement learning project for CKD renal-diet adherence support.

## Project structure

```text
project_root/
├── environment/
│   ├── custom_env.py
│   ├── rendering.py
├── training/
│   ├── dqn_training.py
│   ├── pg_training.py
├── models/
│   ├── dqn/
│   └── pg/
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run commands

Random demo (no training):
```bash
python3 -m environment.random_policy_demo --steps 300 --output models/pg/random_policy.gif
```

Train DQN:
```bash
python3 -m training.dqn_training --timesteps 30000
```

Train PPO + REINFORCE:
```bash
python3 -m training.pg_training --timesteps 30000 --reinforce_episodes 300
```

Focused PPO tuning:
```bash
python3 -m training.ppo_focused_training --timesteps 100000
```

Generate plots:
```bash
python3 -m training.plot_results
python3 -m training.advanced_plots
```

Run best model:
```bash
python3 main.py --agent ppo --model_path models/pg/ppo_focused_v2_run_3.zip --episodes 3
```
