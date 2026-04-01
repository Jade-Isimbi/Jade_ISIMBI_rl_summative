from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class PatientProfile:
    ckd_stage: int
    baseline_adherence: float
    fatigue_sensitivity: float
    support_network: float


class ChronicCareAdherenceEnv(gym.Env):
    """
    CKD dietary-adherence environment.
    Each step is one day where the agent picks an intervention strategy.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    ACTIONS = {
        0: "Do nothing",
        1: "Diet reminder (SMS/app)",
        2: "Personalized meal suggestion",
        3: "Educational CKD message",
        4: "Nurse call",
        5: "Caregiver escalation",
    }

    def __init__(self, max_days: int = 30, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.max_days = max_days
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.profile: Optional[PatientProfile] = None
        self.day = 0
        self.adherence_level = 0.5
        self.fatigue = 0.1
        self.potassium_risk = 0.4
        self.sodium_risk = 0.4
        self.protein_risk = 0.4
        self.engagement = 0.5
        self.last_action = 0
        self.recent_intensity = np.zeros(5, dtype=np.float32)

        self._renderer = None

    def _sample_profile(self) -> PatientProfile:
        return PatientProfile(
            ckd_stage=int(self.rng.integers(3, 6)),
            baseline_adherence=float(self.rng.uniform(0.35, 0.75)),
            fatigue_sensitivity=float(self.rng.uniform(0.2, 0.9)),
            support_network=float(self.rng.uniform(0.1, 0.9)),
        )

    def _obs(self) -> np.ndarray:
        assert self.profile is not None
        return np.array(
            [
                (self.profile.ckd_stage - 3) / 2.0,
                self.adherence_level,
                self.fatigue,
                self.potassium_risk,
                self.sodium_risk,
                self.protein_risk,
                self.engagement,
            ],
            dtype=np.float32,
        )

    def _action_effect(self, action: int) -> Tuple[float, float, float]:
        assert self.profile is not None
        adherence_boost = {0: 0.00, 1: 0.06, 2: 0.09, 3: 0.07, 4: 0.11, 5: 0.10 + 0.03 * self.profile.support_network}[action]
        fatigue_increase = {0: 0.00, 1: 0.02, 2: 0.03, 3: 0.025, 4: 0.06, 5: 0.05}[action]
        nutrient_protection = {0: 0.00, 1: 0.02, 2: 0.035, 3: 0.03, 4: 0.045, 5: 0.04}[action]
        return adherence_boost, fatigue_increase, nutrient_protection

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        profile_override = options.get("profile_override")
        if profile_override:
            sampled = self._sample_profile()
            self.profile = PatientProfile(
                ckd_stage=int(profile_override.get("ckd_stage", sampled.ckd_stage)),
                baseline_adherence=float(profile_override.get("baseline_adherence", sampled.baseline_adherence)),
                fatigue_sensitivity=float(profile_override.get("fatigue_sensitivity", sampled.fatigue_sensitivity)),
                support_network=float(profile_override.get("support_network", sampled.support_network)),
            )
        else:
            self.profile = self._sample_profile()
        self.day = 0
        self.adherence_level = self.profile.baseline_adherence
        self.fatigue = float(self.rng.uniform(0.05, 0.2))
        base_risk = 0.30 + 0.07 * (self.profile.ckd_stage - 3)
        self.potassium_risk = float(np.clip(base_risk + self.rng.uniform(-0.08, 0.08), 0.05, 0.95))
        self.sodium_risk = float(np.clip(base_risk + self.rng.uniform(-0.08, 0.08), 0.05, 0.95))
        self.protein_risk = float(np.clip(base_risk + self.rng.uniform(-0.08, 0.08), 0.05, 0.95))
        self.engagement = float(self.rng.uniform(0.3, 0.6))
        self.last_action = 0
        self.recent_intensity = np.zeros(5, dtype=np.float32)
        return self._obs(), {}

    def step(self, action: int):
        assert self.profile is not None
        self.day += 1
        self.last_action = int(action)

        boost, fatigue_cost, nutrient_protection = self._action_effect(action)
        action_intensity = action / (len(self.ACTIONS) - 1)
        self.recent_intensity = np.roll(self.recent_intensity, 1)
        self.recent_intensity[0] = action_intensity
        pressure = float(np.mean(self.recent_intensity))

        fatigue_growth = fatigue_cost + pressure * self.profile.fatigue_sensitivity * 0.05
        fatigue_recovery = 0.035 if action == 0 else 0.01
        self.fatigue = float(np.clip(self.fatigue + fatigue_growth - fatigue_recovery, 0.0, 1.0))

        fatigue_penalty = self.fatigue * (0.08 + 0.35 * self.profile.fatigue_sensitivity)
        nutrient_risk = (self.potassium_risk + self.sodium_risk + self.protein_risk) / 3.0
        stage_pressure = 0.10 + 0.04 * (self.profile.ckd_stage - 3)
        adherence_noise = float(self.rng.normal(0.0, 0.03))

        self.adherence_level = float(
            np.clip(
                self.adherence_level * 0.84
                + self.profile.baseline_adherence * 0.14
                + boost
                + 0.05 * self.engagement
                - fatigue_penalty
                - nutrient_risk * stage_pressure
                + adherence_noise,
                0.01,
                0.99,
            )
        )

        adhered_today = self.rng.random() < self.adherence_level
        diet_quality = self.adherence_level - 0.4 * self.fatigue
        drift = 0.035 * (1.0 - max(diet_quality, 0.0))
        self.potassium_risk = float(np.clip(self.potassium_risk + drift - nutrient_protection, 0.0, 1.0))
        self.sodium_risk = float(np.clip(self.sodium_risk + drift * 0.95 - nutrient_protection * 0.9, 0.0, 1.0))
        self.protein_risk = float(np.clip(self.protein_risk + drift * 0.85 - nutrient_protection * 0.8, 0.0, 1.0))

        engagement_shift = 0.03 if action in (1, 2, 3) and adhered_today else -0.02 * (action in (4, 5))
        self.engagement = float(np.clip(self.engagement + engagement_shift - 0.01 * self.fatigue, 0.0, 1.0))

        nutrient_safety = 1.0 - ((self.potassium_risk + self.sodium_risk + self.protein_risk) / 3.0)
        adherence_reward = 2.2 if adhered_today else -1.4
        nutrient_reward = 1.8 * nutrient_safety
        fatigue_reward = -0.9 * self.fatigue
        intervention_cost = -0.12 * action_intensity
        risk_penalty = -1.2 * max(self.potassium_risk, self.sodium_risk, self.protein_risk)
        reward = adherence_reward + nutrient_reward + fatigue_reward + intervention_cost + risk_penalty

        severe_risk = max(self.potassium_risk, self.sodium_risk, self.protein_risk)
        terminated = severe_risk >= 0.98
        truncated = self.day >= self.max_days
        info = {
            "adhered": adhered_today,
            "action_name": self.ACTIONS[action],
            "adherence_prob": self.adherence_level,
            "fatigue": self.fatigue,
            "clinical_risk": severe_risk,
            "potassium_risk": self.potassium_risk,
            "sodium_risk": self.sodium_risk,
            "protein_risk": self.protein_risk,
            "ckd_stage": self.profile.ckd_stage,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from environment.rendering import PygameRenderer

            self._renderer = PygameRenderer(width=960, height=540)
        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
