# envs/ice_blow_discrete.py
import numpy as np
from gymnasium import spaces
from .ice_blow_base import IceBlowBaseEnv

class IceBlowDiscreteEnv(IceBlowBaseEnv):
    """
    Non-grid world with discrete actions that adjust velocity.
    The world is continuous on position and velocity.
    """

    def __init__(
        self,
        grid_size=10,
        dt=0.1,
        friction=0.98,
        vel_scale=1.0,
        **kwargs
    ):
        super().__init__(world_size=grid_size, **kwargs)

        self.grid_size = grid_size
        self.dt = dt
        self.friction = friction
        self.vel_scale = vel_scale

        # Discrete action mapping:
        # 0 = no change
        # 1 = +x accel
        # 2 = -x accel
        # 3 = +y accel
        # 4 = -y accel
        self.action_space = spaces.Discrete(5)

        # Observation: matching discrete layout
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(
                low=0.0,
                high=float(grid_size),
                shape=(2,),
                dtype=np.float32
            ),
            "velocity": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(2,),
                dtype=np.float32
            ),
            "goal": spaces.Box(
                low=0.0,
                high=float(grid_size),
                shape=(2,),
                dtype=np.float32
            ),
            "blow_phase": spaces.Discrete(3),
            "blow_axis": spaces.Discrete(2),
            "blow_centers": spaces.Box(
                low=0,
                high=grid_size - 1,
                shape=(self.num_blow_lines,),
                dtype=np.int32
            ),
            "blow_width": spaces.Discrete(grid_size),
        })

        self.vel = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.vel[:] = 0.0
        return obs, info

    def sample_agent_pos(self):
        # top-left corner of agent box
        return np.random.uniform(
            low=0.0,
            high=self.world_size - 1.0,
            size=(2,)
        )

    def sample_goal_pos(self):
        return np.random.uniform(
            low=0.0,
            high=self.world_size - 1.0,
            size=(2,)
        )

    def _apply_action(self, action):
        """
        Instead of grid movement or continuous action,
        we interpret discrete actions as velocity increments.
        """
        # velocity increment mapping
        dvel = np.zeros_like(self.vel)

        if action == 1:
            dvel[0] += self.vel_scale
        elif action == 2:
            dvel[0] -= self.vel_scale
        elif action == 3:
            dvel[1] += self.vel_scale
        elif action == 4:
            dvel[1] -= self.vel_scale
        # action 0 = no change

        # apply discrete velocity adjustment
        self.vel += dvel * self.dt

        # friction
        self.vel *= self.friction

        # integrate position
        self.agent_pos += self.vel * self.dt

        # constrain world
        self.agent_pos = np.clip(self.agent_pos, 0.0, self.grid_size - 1.0)

    def _get_obs(self):
        phase_map = {"idle": 0, "warning": 1, "active": 2}
        axis_map = {"x": 0, "y": 1}

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
