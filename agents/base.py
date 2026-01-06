from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, action_space, obs_space=None, **kwargs):
        self.action_space = action_space
        self.obs_space = obs_space

    def act(self, obs, explore=True):
        raise NotImplementedError

    def store(self, obs, action, reward, next_obs, done):
        pass

    def update(self):
        pass

    def reset(self):
        pass
