# agents/assisted_pathfinding_discrete_agent.py
"""
Human-in-the-loop pathfinding agent for the discrete environment.
Uses hybrid A* + controller approach for safety overrides.
"""
import pygame
import numpy as np
from agents.base import BaseAgent
from agents.pathfinding_discrete_agent import PathfindingDiscreteAgent


class AssistedPathfindingDiscreteAgent(BaseAgent):
    """
    Human-in-the-loop agent for continuous environments with velocity-based movement.

    Takes human input via arrow keys, but overrides dangerous actions
    using the hybrid pathfinding approach.

    Modes:
      risk_averse=True:  Override any action leading toward danger
      risk_averse=False: Allow risky actions if there's time to recover
    """

    def __init__(self, action_space, obs_space=None, env_type="discrete",
                 world_size=10, blow_width=0, dt=0.033, friction=0.90,
                 vel_scale=0.5, max_vel=5.0, grid_resolution=20,
                 warning_duration=100, risk_averse=True, **kwargs):
        super().__init__(action_space, obs_space)

        self.world_size = world_size
        self.blow_width = blow_width
        self.dt = dt
        self.friction = friction
        self.vel_scale = vel_scale
        self.max_vel = max_vel
        self.warning_duration = warning_duration
        self.risk_averse = risk_averse

        # Create internal pathfinding agent
        self.pathfinder = PathfindingDiscreteAgent(
            action_space=action_space,
            obs_space=obs_space,
            env_type=env_type,
            world_size=world_size,
            blow_width=blow_width,
            dt=dt,
            friction=friction,
            vel_scale=vel_scale,
            max_vel=max_vel,
            grid_resolution=grid_resolution,
        )

        # Track warning phase timing
        self._last_blow_axis = 2
        self._warning_steps_elapsed = 0

        # Action effects for simulation
        self.action_to_dvel = {
            0: np.array([0.0, 0.0]),
            1: np.array([-vel_scale, 0.0]),
            2: np.array([vel_scale, 0.0]),
            3: np.array([0.0, -vel_scale]),
            4: np.array([0.0, vel_scale]),
        }

        self.action_names = {
            0: "coast",
            1: "left",
            2: "right",
            3: "up",
            4: "down",
        }

    def _get_keyboard_action(self):
        """Get action from keyboard input."""
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            return 1
        if keys[pygame.K_RIGHT]:
            return 2
        if keys[pygame.K_UP]:
            return 3
        if keys[pygame.K_DOWN]:
            return 4
        return 0

    def _update_warning_tracker(self, obs):
        """Track warning phase timing."""
        blow_axis = obs["blow_axis"]

        if blow_axis == 2:
            self._warning_steps_elapsed = 0
            self._last_blow_axis = 2
        elif self._last_blow_axis == 2 and blow_axis != 2:
            self._warning_steps_elapsed = 1
            self._last_blow_axis = blow_axis
        elif blow_axis != 2:
            self._warning_steps_elapsed += 1
            self._last_blow_axis = blow_axis

    def _get_warning_steps_remaining(self):
        """Estimate steps remaining before blow becomes active."""
        return max(0, self.warning_duration - self._warning_steps_elapsed)

    def _simulate_step(self, pos, vel, action):
        """Simulate one physics step."""
        vel = vel + self.action_to_dvel[action]
        vel = vel * self.friction
        vel = np.clip(vel, -self.max_vel, self.max_vel)
        pos = pos + vel * self.dt
        pos = np.clip(pos, 0.0, self.world_size - 1.0)
        return pos, vel

    def _simulate_trajectory(self, pos, vel, action, steps):
        """Simulate trajectory for given number of steps."""
        pos = np.array(pos, dtype=np.float64)
        vel = np.array(vel, dtype=np.float64)

        trajectory = []
        # Apply action once, then coast
        pos, vel = self._simulate_step(pos, vel, action)
        trajectory.append((pos.copy(), vel.copy()))

        for _ in range(steps - 1):
            pos, vel = self._simulate_step(pos, vel, 0)  # Coast
            trajectory.append((pos.copy(), vel.copy()))

        return trajectory

    def _trajectory_enters_danger(self, trajectory, blow_axis, blow_centers):
        """Check if trajectory enters danger zone."""
        for pos, _ in trajectory:
            if self.pathfinder._is_pos_in_danger(pos, blow_axis, blow_centers):
                return True
        return False

    def _estimate_escape_steps(self, pos, vel, blow_axis, blow_centers):
        """Estimate steps needed to escape from current position."""
        if not self.pathfinder._is_pos_in_danger(pos, blow_axis, blow_centers):
            return 0

        # Try all escape actions and find the fastest
        best_steps = float('inf')

        for escape_action in [1, 2, 3, 4]:  # Try all movement actions
            sim_pos = np.array(pos, dtype=np.float64)
            sim_vel = np.array(vel, dtype=np.float64)

            for step in range(200):
                sim_pos, sim_vel = self._simulate_step(sim_pos, sim_vel, escape_action)
                if not self.pathfinder._is_pos_in_danger(sim_pos, blow_axis, blow_centers):
                    best_steps = min(best_steps, step + 1)
                    break

        return best_steps

    def _is_in_danger(self, obs):
        """Check if currently in danger zone."""
        blow_axis = obs["blow_axis"]
        if blow_axis == 2:
            return False

        pos = self.pathfinder._denormalize_pos(obs["agent"])
        return self.pathfinder._is_pos_in_danger(pos, blow_axis, obs["blow_centers"])

    def _would_enter_danger(self, obs, action, lookahead=30):
        """Check if action would lead into danger."""
        blow_axis = obs["blow_axis"]
        if blow_axis == 2:
            return False

        pos = self.pathfinder._denormalize_pos(obs["agent"])
        vel = np.array(obs["velocity"])

        trajectory = self._simulate_trajectory(pos, vel, action, lookahead)
        return self._trajectory_enters_danger(trajectory, blow_axis, obs["blow_centers"])

    def _must_escape_now(self, obs):
        """Check if escape is time-critical."""
        if not self._is_in_danger(obs):
            return False

        pos = self.pathfinder._denormalize_pos(obs["agent"])
        vel = np.array(obs["velocity"])
        blow_axis = obs["blow_axis"]

        escape_steps = self._estimate_escape_steps(pos, vel, blow_axis, obs["blow_centers"])
        steps_remaining = self._get_warning_steps_remaining()

        # Add safety buffer
        return escape_steps >= steps_remaining - 5

    def _can_recover_after_action(self, obs, action):
        """Check if we can escape after taking this action."""
        pos = self.pathfinder._denormalize_pos(obs["agent"])
        vel = np.array(obs["velocity"])

        # Simulate one step
        new_pos, new_vel = self._simulate_step(pos, vel, action)

        # Check if we can escape from new position
        blow_axis = obs["blow_axis"]
        escape_steps = self._estimate_escape_steps(new_pos, new_vel, blow_axis, obs["blow_centers"])
        steps_remaining = self._get_warning_steps_remaining() - 1

        return escape_steps < steps_remaining - 5

    def act(self, obs, explore=True):
        """Get action - uses keyboard if safe, otherwise overrides."""
        self._update_warning_tracker(obs)

        human_action = self._get_keyboard_action()
        in_danger = self._is_in_danger(obs)
        blow_axis = obs["blow_axis"]

        # Non-risk_averse mode: allow more freedom
        if not self.risk_averse and in_danger:
            if self._must_escape_now(obs):
                safe_action = self.pathfinder.act(obs, explore=False)
                steps_left = self._get_warning_steps_remaining()
                if safe_action != human_action:
                    print(f"\033[93m[OVERRIDE] Must escape NOW! ({steps_left} steps left) "
                          f"Ignoring '{self.action_names[human_action]}', "
                          f"using '{self.action_names[safe_action]}'\033[0m")
                return safe_action
            else:
                steps_left = self._get_warning_steps_remaining()
                if human_action == 0:
                    print(f"\033[94m[INFO] In danger zone, ~{steps_left} steps to escape\033[0m")
                return human_action

        # Risk averse mode or not in danger
        if in_danger:
            safe_action = self.pathfinder.act(obs, explore=False)
            if safe_action != human_action:
                print(f"\033[93m[OVERRIDE] In danger! "
                      f"Ignoring '{self.action_names[human_action]}', "
                      f"escaping with '{self.action_names[safe_action]}'\033[0m")
            return safe_action

        # Check if human action would lead to danger
        if blow_axis != 2 and self._would_enter_danger(obs, human_action):
            # In non-risk_averse mode, allow if recoverable
            if not self.risk_averse and self._can_recover_after_action(obs, human_action):
                steps_left = self._get_warning_steps_remaining()
                print(f"\033[94m[INFO] Entering danger zone ({steps_left} steps to escape)\033[0m")
                return human_action

            safe_action = self.pathfinder.act(obs, explore=False)

            if safe_action != human_action and not self._would_enter_danger(obs, safe_action):
                if self.risk_averse:
                    print(f"\033[93m[OVERRIDE] '{self.action_names[human_action]}' leads to danger! "
                          f"Using '{self.action_names[safe_action]}'\033[0m")
                else:
                    steps_left = self._get_warning_steps_remaining()
                    print(f"\033[93m[OVERRIDE] '{self.action_names[human_action]}' - can't recover! "
                          f"({steps_left} steps) Using '{self.action_names[safe_action]}'\033[0m")
                return safe_action
            elif safe_action == human_action:
                print(f"\033[91m[WARNING] No safe option, proceeding with '{self.action_names[human_action]}'\033[0m")

        return human_action

    def store(self, obs, action, reward, next_obs, done):
        pass

    def update(self):
        pass

    def reset(self):
        """Reset internal state."""
        self._last_blow_axis = 2
        self._warning_steps_elapsed = 0
        self.pathfinder.reset()
