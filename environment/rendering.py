from __future__ import annotations

import numpy as np
import pygame


class PygameRenderer:
    def __init__(self, width: int = 960, height: int = 540):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CKD Renal Diet RL Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 22)
        self.small = pygame.font.SysFont("Arial", 18)

    def _bar(self, x: int, y: int, w: int, h: int, value: float, color):
        pygame.draw.rect(self.screen, (60, 60, 70), (x, y, w, h), border_radius=8)
        fill = int(max(0, min(1, value)) * w)
        pygame.draw.rect(self.screen, color, (x, y, fill, h), border_radius=8)
        pygame.draw.rect(self.screen, (180, 180, 190), (x, y, w, h), 2, border_radius=8)

    def render(self, env):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

        self.screen.fill((25, 27, 34))
        title = self.font.render("CKD Diet Adherence", True, (240, 240, 240))
        self.screen.blit(title, (24, 16))

        self.screen.blit(self.small.render(f"Day: {env.day}/{env.max_days}", True, (220, 220, 220)), (24, 60))
        self.screen.blit(self.small.render(f"Action: {env.ACTIONS[env.last_action]}", True, (220, 220, 220)), (24, 88))
        self.screen.blit(self.small.render(f"CKD Stage: {env.profile.ckd_stage}", True, (220, 220, 220)), (560, 60))

        labels = [
            ("Diet Adherence", env.adherence_level, (90, 200, 120)),
            ("Patient Fatigue", env.fatigue, (220, 130, 90)),
            ("Potassium Risk", env.potassium_risk, (225, 70, 70)),
            ("Sodium Risk", env.sodium_risk, (245, 120, 70)),
            ("Protein Risk", env.protein_risk, (210, 90, 170)),
            ("Engagement", env.engagement, (90, 170, 240)),
        ]
        y = 130
        for label, val, color in labels:
            self.screen.blit(self.small.render(f"{label}: {val:.2f}", True, (225, 225, 230)), (24, y))
            self._bar(290, y + 2, 420, 24, val, color)
            y += 48

        fatigue_color = (255, 165, 0) if env.fatigue < 0.7 else (255, 70, 70)
        severe = max(env.potassium_risk, env.sodium_risk, env.protein_risk)
        risk_color = (95, 210, 105) if severe < 0.45 else (255, 95, 95)
        pygame.draw.circle(self.screen, fatigue_color, (790, 210), 38)
        pygame.draw.circle(self.screen, risk_color, (900, 210), 38)
        self.screen.blit(self.small.render("Fatigue", True, (230, 230, 230)), (758, 258))
        self.screen.blit(self.small.render("Severe Risk", True, (230, 230, 230)), (850, 258))

        pygame.draw.rect(self.screen, (36, 40, 50), (24, 390, 912, 120), border_radius=10)
        desc = [
            "Reward = diet adherence + nutrient safety - fatigue - intervention cost.",
            "Nutrient risks track potassium, sodium, and protein exposure in CKD diet management.",
            "Best policy balances timely intervention with patient fatigue and intervention cost.",
        ]
        for i, line in enumerate(desc):
            self.screen.blit(self.small.render(line, True, (208, 212, 220)), (36, 406 + i * 30))

        pygame.display.flip()
        self.clock.tick(6)

        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))

    def close(self):
        pygame.quit()
