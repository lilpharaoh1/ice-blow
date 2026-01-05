from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def act(self, obs):
        return self.action_space.sample()
