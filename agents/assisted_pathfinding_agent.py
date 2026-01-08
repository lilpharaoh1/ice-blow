# agents/assisted_pathfinding_agent.py
import pygame
import numpy as np
from agents.base import BaseAgent
from agents.pathfinding_agent import PathfindingAgent


class AssistedPathfindingAgent(BaseAgent):
    """
    Human-in-the-loop pathfinding agent.

    Takes human input via arrow keys, but overrides dangerous actions
    with the pathfinding algorithm's safer choice.

    Modes:
      risk_averse=True:  Never allow entering danger zone during warning
      risk_averse=False: Allow entering if there's enough time to escape

    Arrow keys:
      LEFT  = move left
      RIGHT = move right
      UP    = move up
      DOWN  = move down
    """

    def __init__(self, action_space, obs_space=None, env_type="gridworld",
                 grid_size=10, blow_width=0, warning_duration=5,
                 safety_margin=0, risk_averse=True, **kwargs):
        super().__init__(action_space, obs_space)
        self.env_type = env_type
        self.grid_size = grid_size
        self.blow_width = blow_width
        self.warning_duration = warning_duration
        self.risk_averse = risk_averse

        # Create internal pathfinding agent for safety checks and overrides
        self.pathfinder = PathfindingAgent(
            action_space=action_space,
            obs_space=obs_space,
            env_type=env_type,
            grid_size=grid_size,
            blow_width=blow_width,
            safety_margin=safety_margin,
        )

        # Track warning phase timing
        self._last_blow_axis = 2  # 2 = inactive
        self._warning_steps_elapsed = 0

        # Action mapping
        self.action_to_delta = {
            0: (0, 0),   # no-op
            1: (-1, 0),  # left
            2: (1, 0),   # right
            3: (0, -1),  # up
            4: (0, 1),   # down
        }

        self.action_names = {
            0: "stay",
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
        return 0  # no-op

    def _update_warning_tracker(self, obs):
        """Track warning phase timing to estimate remaining steps."""
        blow_axis = obs["blow_axis"]

        if blow_axis == 2:
            # Blow inactive - reset tracker
            self._warning_steps_elapsed = 0
            self._last_blow_axis = 2
        elif self._last_blow_axis == 2 and blow_axis != 2:
            # New warning just started
            self._warning_steps_elapsed = 1
            self._last_blow_axis = blow_axis
        elif blow_axis != 2:
            # Warning ongoing
            self._warning_steps_elapsed += 1
            self._last_blow_axis = blow_axis

    def _get_warning_steps_remaining(self):
        """Estimate how many steps remain before blow becomes active."""
        return max(0, self.warning_duration - self._warning_steps_elapsed)

    def _get_escape_distance(self, pos, danger_cells, blow_axis):
        """
        Calculate minimum steps needed to escape from a position.
        Returns the distance to the nearest safe cell.
        """
        pos = tuple(pos)
        if pos not in danger_cells:
            return 0

        # BFS to find nearest safe cell
        from collections import deque
        visited = {pos}
        queue = deque([(pos, 0)])

        while queue:
            current, dist = queue.popleft()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = current[0] + dx
                ny = current[1] + dy

                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue

                neighbor = (nx, ny)
                if neighbor in visited:
                    continue

                visited.add(neighbor)

                if neighbor not in danger_cells:
                    return dist + 1

                queue.append((neighbor, dist + 1))

        return float('inf')  # No escape possible

    def _can_escape_in_time(self, pos, obs):
        """Check if agent can escape from position before blow activates."""
        blow_axis = obs["blow_axis"]
        if blow_axis == 2:
            return True

        danger_cells = self.pathfinder._get_danger_cells(blow_axis, obs["blow_centers"])
        escape_distance = self._get_escape_distance(pos, danger_cells, blow_axis)
        steps_remaining = self._get_warning_steps_remaining()

        # Need at least escape_distance steps to get out
        # Add 1 step safety buffer
        return escape_distance < steps_remaining

    def _would_die(self, obs, action):
        """
        Check if taking this action would put the agent in danger.

        In risk_averse mode: Returns True if action enters danger zone at all
        In non-risk_averse mode: Returns True only if escape is impossible
        """
        blow_axis = obs["blow_axis"]

        # No danger if blow is inactive
        if blow_axis == 2:
            return False

        # Get current position and calculate new position after action
        agent_pos = self.pathfinder._denormalize_pos(obs["agent"])
        dx, dy = self.action_to_delta[action]

        new_x = np.clip(agent_pos[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(agent_pos[1] + dy, 0, self.grid_size - 1)
        new_pos = (new_x, new_y)

        # Get danger cells
        danger_cells = self.pathfinder._get_danger_cells(blow_axis, obs["blow_centers"])

        if new_pos not in danger_cells:
            return False

        # Position is in danger zone
        if self.risk_averse:
            # Risk averse: any entry into danger is considered deadly
            return True
        else:
            # Non-risk averse: only deadly if we can't escape in time
            return not self._can_escape_in_time(new_pos, obs)

    def _is_in_danger(self, obs):
        """Check if agent is currently in a danger zone."""
        blow_axis = obs["blow_axis"]

        if blow_axis == 2:
            return False

        agent_pos = self.pathfinder._denormalize_pos(obs["agent"])
        danger_cells = self.pathfinder._get_danger_cells(blow_axis, obs["blow_centers"])

        return tuple(agent_pos) in danger_cells

    def _must_escape_now(self, obs):
        """
        Check if agent must escape immediately (no time to spare).
        Used in non-risk_averse mode to force escape when time is critical.
        """
        if not self._is_in_danger(obs):
            return False

        agent_pos = self.pathfinder._denormalize_pos(obs["agent"])
        blow_axis = obs["blow_axis"]
        danger_cells = self.pathfinder._get_danger_cells(blow_axis, obs["blow_centers"])

        escape_distance = self._get_escape_distance(agent_pos, danger_cells, blow_axis)
        steps_remaining = self._get_warning_steps_remaining()

        # Must escape if we have just enough time (or less)
        # Use <= to ensure we start escaping with exactly enough time
        return escape_distance >= steps_remaining

    def act(self, obs, explore=True):
        """
        Get action - uses keyboard input if safe, otherwise overrides with pathfinder.
        """
        # Update warning phase tracker
        self._update_warning_tracker(obs)

        human_action = self._get_keyboard_action()
        blow_axis = obs["blow_axis"]
        in_danger = self._is_in_danger(obs)

        # In non-risk_averse mode, allow more freedom but force escape when necessary
        if not self.risk_averse and in_danger:
            if self._must_escape_now(obs):
                safe_action = self.pathfinder.act(obs, explore=False)
                steps_left = self._get_warning_steps_remaining()
                if safe_action != human_action:
                    print(f"\033[93m[OVERRIDE] Must escape NOW! ({steps_left} steps left) "
                          f"Ignoring '{self.action_names[human_action]}', "
                          f"escaping with '{self.action_names[safe_action]}'\033[0m")
                return safe_action
            else:
                # In danger but have time - allow human control
                steps_left = self._get_warning_steps_remaining()
                # Optionally warn the user
                if human_action == 0:  # Staying still in danger
                    print(f"\033[94m[INFO] In danger zone, {steps_left} steps to escape\033[0m")
                return human_action

        # Risk averse mode OR not in danger: original behavior
        if in_danger:
            safe_action = self.pathfinder.act(obs, explore=False)
            if safe_action != human_action:
                print(f"\033[93m[OVERRIDE] You're in danger! "
                      f"Ignoring '{self.action_names[human_action]}', "
                      f"escaping with '{self.action_names[safe_action]}'\033[0m")
            return safe_action

        # Check if human action would lead to danger
        if self._would_die(obs, human_action):
            safe_action = self.pathfinder.act(obs, explore=False)

            # Only override if safe action is different and actually safer
            if safe_action != human_action and not self._would_die(obs, safe_action):
                if self.risk_averse:
                    print(f"\033[93m[OVERRIDE] '{self.action_names[human_action]}' would enter danger! "
                          f"Using '{self.action_names[safe_action]}' instead\033[0m")
                else:
                    steps_left = self._get_warning_steps_remaining()
                    print(f"\033[93m[OVERRIDE] '{self.action_names[human_action]}' - not enough time to escape! "
                          f"({steps_left} steps left) Using '{self.action_names[safe_action]}'\033[0m")
                return safe_action
            elif safe_action == human_action:
                # Pathfinder also chose this - might be unavoidable
                print(f"\033[91m[WARNING] No safe option available, proceeding with '{self.action_names[human_action]}'\033[0m")

        return human_action

    def store(self, obs, action, reward, next_obs, done):
        """No learning for this agent."""
        pass

    def update(self):
        """No learning for this agent."""
        pass

    def reset(self):
        """Reset internal state for new episode."""
        self._last_blow_axis = 2
        self._warning_steps_elapsed = 0
