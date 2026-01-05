# envs/ice_blow_base.py
import gymnasium as gym
import numpy as np

class IceBlowBaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        world_size,
        blow_interval=50,
        warning_duration=10,
        active_duration=5,
        blow_width=0,
        num_blow_lines=1,
        goal_radius=0.05,
        render_mode=None,
    ):


        self.world_size = world_size
        self.goal_radius = goal_radius
        self.render_mode = render_mode

        self.step_count = 0
        self.blow_timer = 0
        self.blow_interval = blow_interval
        self.blow_active = False
        self.blow_phase = "idle"   # "idle" | "warning" | "active"
        self.blow_axis = None     # 0 (x) or 1 (y)
        self.blow_centers = None  # list of coords
        self.blow_width = blow_width
        self.num_blow_lines = num_blow_lines


        self.warning_duration = warning_duration
        self.active_duration = active_duration
        self.phase_timer = 0


        self.agent_pos = None
        self.goal_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.blow_timer = 0
        self.blow_active = False

        self.agent_pos = self.sample_agent_pos()
        self.goal_pos = self.sample_goal_pos()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        self._update_blow_state()
        self._apply_action(action)

        terminated = False
        reward = 0.0

        if self._agent_exposed():
            reward = -1.0
            terminated = True


        elif self._reached_goal():
            reward = +1.0
            self.goal_pos = self.sample_goal_pos()

        return self._get_obs(), reward, terminated, False, {}

    def _update_blow_state(self):
        # Start a new blow cycle
        if self.blow_phase == "idle":
            if self.step_count > 0 and self.step_count % self.blow_interval == 0:
                self.blow_phase = "warning"
                self.phase_timer = self.warning_duration
                self._sample_blow_axis()

        # Warning phase
        elif self.blow_phase == "warning":
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.blow_phase = "active"
                self.phase_timer = self.active_duration

        # Active phase
        elif self.blow_phase == "active":
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.blow_phase = "idle"
                self.blow_axis = None
                self.blow_centers = None

    def _sample_blow_axis(self):
        self.blow_axis = self.np_random.integers(0, 2)

        if self.world_size > 1:
            max_coord = self.world_size - 1
            self.blow_centers = self.np_random.choice(
                max_coord + 1,
                size=self.num_blow_lines,
                replace=False,
            ).tolist()
        else:
            self.blow_centers = self.np_random.random(
                size=self.num_blow_lines
            ).tolist()

    def _reached_goal(self):
        return np.linalg.norm(self.agent_pos - self.goal_pos) < self.goal_radius

    def _agent_exposed(self):
        if self.blow_phase != "active":
            return False

        agent_val = self.agent_pos[self.blow_axis]

        for c in self.blow_centers:
            if self.world_size > 1:
                # Discrete
                if abs(agent_val - c) <= self.blow_width:
                    return True
            else:
                # Continuous
                if abs(agent_val - c) <= self.blow_width:
                    return True

        return False



    # Methods subclasses MUST implement
    def _apply_action(self, action):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def sample_agent_pos(self):
        raise NotImplementedError

    def sample_goal_pos(self):
        raise NotImplementedError

