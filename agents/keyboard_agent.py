import pygame
import numpy as np
from agents.base import BaseAgent


class KeyboardAgent(BaseAgent):
    """
    Arrow keys:
      UP    = forward
      LEFT  = left
      RIGHT = right
      DOWN  = backward (if applicable)
    """

    def __init__(self, action_space, env_type="discrete", speed=1.0, **kwargs):
        super().__init__(action_space)
        self.env_type = env_type
        self.speed = speed

    def act(self, obs, **kwargs):
        keys = pygame.key.get_pressed()

        if self.env_type == "gridworld":
            if keys[pygame.K_LEFT]:
                return 1
            if keys[pygame.K_RIGHT]:
                return 2
            if keys[pygame.K_UP]:
                return 3
            if keys[pygame.K_DOWN]:
                return 4
            return 0 # no-op
        
        elif self.env_type == "discrete":
            if keys[pygame.K_LEFT]:
                return 1
            if keys[pygame.K_RIGHT]:
                return 2
            if keys[pygame.K_UP]:
                return 3
            if keys[pygame.K_DOWN]:
                return 4
            return 0 # no-op

        else:  # continuous
            action = np.zeros(self.action_space.shape[0], dtype=np.float32)

            if keys[pygame.K_LEFT]:
                action[0] -= self.speed
            if keys[pygame.K_RIGHT]:
                action[0] += self.speed
            if keys[pygame.K_UP]:
                action[1] -= self.speed
            if keys[pygame.K_DOWN]:
                action[1] += self.speed

            return action
