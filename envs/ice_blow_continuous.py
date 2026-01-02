# envs/ice_blow_continuous.py
import numpy as np
from gymnasium import spaces
from .ice_blow_base import IceBlowBaseEnv

class IceBlowContinuousEnv(IceBlowBaseEnv):
    def __init__(self, dt=0.1, friction=0.98, **kwargs):
        super().__init__(world_size=1.0, **kwargs)

        self.dt = dt
        self.friction = friction

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,))

        self.vel = np.zeros(2)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.vel = np.zeros(2)
        return obs, info

    def sample_agent_pos(self):
        return np.random.rand(2)

    def sample_goal_pos(self):
        return np.random.rand(2)

    def _apply_action(self, action):
        self.vel += action * self.dt
        self.vel *= self.friction
        self.agent_pos += self.vel * self.dt
        self.agent_pos = np.clip(self.agent_pos, 0.0, 1.0)

    def _get_obs(self):
        phase_map = {"idle": 0.0, "warning": 1.0, "active": 2.0}

        return np.concatenate([
            self.agent_pos,
            self.vel,
            self.goal_pos,
            np.array([
                phase_map[self.blow_phase],
                -1.0 if self.blow_axis is None else float(self.blow_axis),
                -1.0 if self.blow_coord is None else self.blow_coord,
            ])
        ])


