# envs/ice_blow_continuous.py

import numpy as np
from gymnasium import spaces
from .ice_blow_base import IceBlowBaseEnv


class IceBlowContinuousEnv(IceBlowBaseEnv):
    """
    Continuous analogue of IceBlowDiscreteEnv.

    Differences:
      - agent_pos is continuous in [0, grid_size)
      - actions are continuous accelerations
      - dynamics include velocity + friction
    """

    def __init__(self, grid_size=10, dt=0.1, friction=0.98, **kwargs):
        super().__init__(world_size=grid_size, **kwargs)

        self.grid_size = grid_size
        self.dt = dt
        self.friction = friction

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(
                low=0.0,
                high=grid_size,
                shape=(2,),
                dtype=np.float32,
            ),
            "velocity": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(2,),
                dtype=np.float32,
            ),
            "goal": spaces.Box(
                low=0.0,
                high=grid_size,
                shape=(2,),
                dtype=np.float32,
            ),
            "blow_phase": spaces.Discrete(3),     # idle, warning, active
            "blow_axis": spaces.Discrete(2),      # -1=None, 0=x, +1=y
            "blow_centers": spaces.Box(
                low=0,
                high=grid_size - 1,
                shape=(self.num_blow_lines,),
                dtype=np.int32,
            ),
            "blow_width": spaces.Discrete(grid_size),
        })

        self.vel = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.vel[:] = 0.0
        return obs, info

    def sample_agent_pos(self):
        # Continuous position in grid coordinates
        return np.random.uniform(0.0, self.grid_size - 1.0, size=2)

    def sample_goal_pos(self):
        return np.random.uniform(0.0, self.grid_size - 1.0, size=2)

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)

        self.vel += action * self.dt
        self.vel *= self.friction

        self.agent_pos += self.vel * self.dt
        self.agent_pos = np.clip(
            self.agent_pos,
            0.0,
            self.grid_size - 1.0,
        )

    def _get_obs(self):
        phase_map = {"idle": 0, "warning": 1, "active": 2}

        blow_centers = (
            np.array(self.blow_centers, dtype=np.int32)
            if self.blow_centers
            else np.full(self.num_blow_lines, -1, dtype=np.int32)
        )
        blow_axis = (
            self.blow_axis
            if self.blow_axis
            else -1
        )

        return {
            "agent": self.agent_pos.astype(np.float32),
            "velocity": self.vel.astype(np.float32),
            "goal": self.goal_pos.astype(np.float32),
            "blow_phase": phase_map[self.blow_phase],
            "blow_axis": blow_axis,
            "blow_centers": blow_centers,
            "blow_width": self.blow_width,
        }
