# Reinforcement learning

## Introduction

This project is a **reinforcement learning system** for helping people with **chronic kidney disease (CKD)** follow a **renal diet**. Each simulated day, an agent chooses an intervention—such as doing nothing, sending a diet reminder, suggesting a meal, sharing education, or escalating to a nurse or caregiver—to improve dietary adherence and keep nutrient-related risks (potassium, sodium, protein) under control, while limiting reminder fatigue .

The environment is implemented in **Gymnasium**; policies are trained with **Stable-Baselines3** (DQN, PPO) and a custom **REINFORCE** implementation. Results, plots, and saved models live under `models/`.

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
