# Agent-in-Environment Diagram

```mermaid
flowchart LR
    S[Patient State\nadherence, fatigue, risk,\nengagement, profile] --> O[Observation Vector]
    O --> A[RL Agent\nDQN / REINFORCE / PPO]
    A --> X[Action\nreminder channel + intensity]
    X --> E[Environment Dynamics\nfatigue update + adherence probability\nrisk progression]
    E --> R[Reward\nadherence gain + health stability\n- fatigue - intervention cost]
    R --> A
    E --> S
```

## Description

- The agent receives a compact patient-state observation.
- It chooses one intervention action each day.
- Environment updates adherence, fatigue, and clinical risk with stochastic dynamics.
- Reward encourages long-term adherence and health outcomes while discouraging over-contact fatigue.
