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
        time_limit=25000,
        step_penalty=0.01,
        render_mode=None,
    ):


        self.world_size = world_size
        self.goal_radius = goal_radius
        self.time_limit = time_limit
        self.step_penalty = step_penalty
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
        self.blow_phase = "idle"   # "idle" | "warning" | "active"
        self.blow_axis = None     # 0 (x) or 1 (y)
        self.blow_centers = None  # list of coords
        self.phase_timer = 0 


        self.agent_pos = self.sample_agent_pos()
        self.goal_pos = self.sample_goal_pos()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        self._update_blow_state()
        self._apply_action(action)

        terminated = False
        reward = -self.step_penalty  # Small penalty for each step

        if self._agent_exposed():
            reward = -1.0
            terminated = True
        elif self._reached_goal():
            reward = +1.0
            self.goal_pos = self.sample_goal_pos()

        if self.step_count >= self.time_limit:
            terminated = True

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

    def _intervals_overlap(self, a_min, a_max, b_min, b_max):
        return (a_min <= b_max) and (b_min <= a_max)

    def _reached_goal(self):
        # Agent box
        agent_min = self.agent_pos
        agent_max = self.agent_pos + 1.0

        # Goal box
        goal_min = self.goal_pos
        goal_max = self.goal_pos + 1.0

        return (
            self._intervals_overlap(agent_min[0], agent_max[0], goal_min[0], goal_max[0])
            and
            self._intervals_overlap(agent_min[1], agent_max[1], goal_min[1], goal_max[1])
        )

    def _agent_exposed(self):
        if self.blow_phase != "active":
            return False

        axis = self.blow_axis  # 0 = x, 1 = y

        # Agent interval (top-left corner, width = 1)
        agent_min = self.agent_pos[axis]
        agent_max = agent_min + 1.0

        for c in self.blow_centers:
            # Blow slab interval
            blow_min = c - self.blow_width
            blow_max = c + self.blow_width + 1.0

            if self._intervals_overlap(agent_min, agent_max, blow_min, blow_max):
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

