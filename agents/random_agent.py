from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def act(self, obs, **kwargs):
        return self.action_space.sample()
