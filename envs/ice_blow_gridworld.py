# envs/ice_blow_gridworld.py
import numpy as np
from gymnasium import spaces
from .ice_blow_base import IceBlowBaseEnv

class IceBlowGridworldEnv(IceBlowBaseEnv):
    def __init__(self, grid_size=10, **kwargs):
        self.grid_size = grid_size

        super().__init__(
            world_size=grid_size,
            **kwargs
        )

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(
                low=0, 
                high=grid_size - 1, 
                shape=(2,), 
                dtype=np.int32
            ),
            "goal": spaces.Box(
                low=0, 
                high=grid_size - 1, 
                shape=(2,), 
                dtype=np.int32
            ),
            # "blow_phase": espaces.Discrete(3),  # idle, warning, active
            "blow_axis": spaces.Discrete(3),
            "blow_centers": spaces.Box(
                low=0,
                high=grid_size - 1,
                shape=(self.num_blow_lines,),
                dtype=np.int32,
            ),
            # "blow_width": spaces.Discrete(grid_size),
        })


    def sample_agent_pos(self):
        return np.random.randint(0, self.grid_size, size=2)

    def sample_goal_pos(self):
        return np.random.randint(0, self.grid_size, size=2)

    def _apply_action(self, action):
        move = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 1]),
        }[action]

        self.agent_pos = np.clip(
            self.agent_pos + move,
            0,
            self.grid_size - 1
        )
        
    def _get_obs(self):
        agent_normalized = self.agent_pos / self.grid_size
        goal_normalized = self.goal_pos / self.grid_size

        # Handle blow information consistently (no -1 values)
        if self.blow_centers is not None:
            # Normalize blow centers to [0, 1]
            blow_centers = np.array(self.blow_centers, dtype=np.float32) / self.grid_size
        else:
            # Use 0.0 when inactive (valid value in [0, 1])
            blow_centers = np.zeros(self.num_blow_lines, dtype=np.float32)

        blow_axis = self.blow_axis if self.blow_axis is not None else 2

        phase_map = {"idle": 0, "warning": 1, "active": 2}
        return {
            "agent": agent_normalized,
            "goal": goal_normalized,
            # "blow_phase": phase_map[self.blow_phase],
            "blow_axis": blow_axis,
            "blow_centers": blow_centers
            # "blow_width": self.blow_width,
        }

