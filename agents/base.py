from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, action_space, obs_space=None, **kwargs):
        self.action_space = action_space
        self.obs_space = obs_space

    @abstractmethod
    def act(self, obs):
        pass

    def reset(self):
        pass
