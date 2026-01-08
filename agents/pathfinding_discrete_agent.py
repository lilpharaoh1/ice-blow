# agents/pathfinding_discrete_agent.py
"""
Hybrid pathfinding agent for the discrete (continuous position, velocity-based) environment.

Approach:
1. Discretize continuous space into a grid for pathfinding
2. Use A* to find safe waypoints avoiding danger zones
3. Use simple proportional controller to navigate to waypoints
4. Find truly safe positions (not between danger zones, with buffer from edges)
"""
import numpy as np
import heapq
from agents.base import BaseAgent


class PathfindingDiscreteAgent(BaseAgent):
    """
    Hybrid pathfinding + control agent for continuous environments.

    Uses A* on a discretized grid to find safe paths, then a simple
    controller to navigate between waypoints.
    """

    def __init__(self, action_space, obs_space=None, env_type="discrete",
                 world_size=10, blow_width=0, dt=0.033, friction=0.90,
                 vel_scale=0.5, max_vel=5.0, grid_resolution=20,
                 waypoint_threshold=0.5, brake_threshold=1.0,
                 safety_buffer=1.5, **kwargs):
        super().__init__(action_space, obs_space)

        self.world_size = world_size
        self.blow_width = blow_width
        self.dt = dt
        self.friction = friction
        self.vel_scale = vel_scale
        self.max_vel = max_vel
        self.safety_buffer = safety_buffer  # Extra distance to stay from danger edges

        # Discretization for pathfinding
        self.grid_resolution = grid_resolution
        self.cell_size = world_size / grid_resolution

        # Controller parameters
        self.waypoint_threshold = waypoint_threshold
        self.brake_threshold = brake_threshold

        # Current path and waypoint
        self.current_path = []
        self.current_waypoint_idx = 0
        self.last_goal = None
        self.last_danger_state = None
        self.safe_haven = None  # Position to retreat to when goal is blocked

        self.action_names = {
            0: "coast",
            1: "left",
            2: "right",
            3: "up",
            4: "down",
        }

    def _world_to_grid(self, pos):
        """Convert world position to grid cell."""
        gx = int(np.clip(pos[0] / self.cell_size, 0, self.grid_resolution - 1))
        gy = int(np.clip(pos[1] / self.cell_size, 0, self.grid_resolution - 1))
        return (gx, gy)

    def _grid_to_world(self, cell):
        """Convert grid cell to world position (center of cell)."""
        wx = (cell[0] + 0.5) * self.cell_size
        wy = (cell[1] + 0.5) * self.cell_size
        return np.array([wx, wy])

    def _get_danger_zones(self, blow_axis, blow_centers_norm):
        """Get list of danger zone intervals (min, max) in world coordinates."""
        if blow_axis == 2:
            return []

        zones = []
        for center_norm in blow_centers_norm:
            center = center_norm * self.world_size
            # Match environment's interval overlap logic
            blow_min = center - self.blow_width
            blow_max = center + self.blow_width + 0.999
            zones.append((blow_min, blow_max))

        return sorted(zones, key=lambda z: z[0])

    def _get_danger_cells(self, blow_axis, blow_centers_norm):
        """Get set of grid cells that are in danger zones (with safety buffer)."""
        danger_cells = set()

        if blow_axis == 2:
            return danger_cells

        zones = self._get_danger_zones(blow_axis, blow_centers_norm)

        for blow_min, blow_max in zones:
            # Add safety buffer to avoid sitting at edges
            buffered_min = blow_min - self.safety_buffer
            buffered_max = blow_max + self.safety_buffer

            if blow_axis == 0:  # X-axis danger
                for gx in range(self.grid_resolution):
                    cell_min = gx * self.cell_size
                    cell_max = (gx + 1) * self.cell_size
                    # Check if cell overlaps buffered danger zone
                    if buffered_min <= cell_max + 0.999 and cell_min <= buffered_max + 0.999:
                        for gy in range(self.grid_resolution):
                            danger_cells.add((gx, gy))
            else:  # Y-axis danger
                for gy in range(self.grid_resolution):
                    cell_min = gy * self.cell_size
                    cell_max = (gy + 1) * self.cell_size
                    if buffered_min <= cell_max + 0.999 and cell_min <= buffered_max + 0.999:
                        for gx in range(self.grid_resolution):
                            danger_cells.add((gx, gy))

        return danger_cells

    def _is_pos_in_danger(self, pos, blow_axis, blow_centers_norm, use_buffer=False):
        """Check if a world position is in danger using interval overlap."""
        if blow_axis == 2:
            return False

        buffer = self.safety_buffer if use_buffer else 0
        agent_min = pos[blow_axis]
        agent_max = agent_min + 0.999

        zones = self._get_danger_zones(blow_axis, blow_centers_norm)

        for blow_min, blow_max in zones:
            check_min = blow_min - buffer
            check_max = blow_max + buffer
            if agent_min <= check_max and check_min <= agent_max:
                return True

        return False

    def _is_between_dangers(self, pos, blow_axis, blow_centers_norm):
        """Check if position is between two danger zones (dangerous spot)."""
        if blow_axis == 2:
            return False

        zones = self._get_danger_zones(blow_axis, blow_centers_norm)
        if len(zones) < 2:
            return False

        coord = pos[blow_axis]

        # Check if we're between any two consecutive zones
        for i in range(len(zones) - 1):
            zone1_max = zones[i][1]
            zone2_min = zones[i + 1][0]

            # Gap between zones
            gap = zone2_min - zone1_max

            # If gap is small and we're in it, we're in danger
            if gap < self.safety_buffer * 3:  # Small gap
                if zone1_max < coord < zone2_min:
                    return True

        return False

    def _find_safe_haven(self, pos, blow_axis, blow_centers_norm):
        """
        Find a truly safe position to retreat to.
        - Not in danger zone
        - Not between danger zones
        - Has buffer from edges
        - Preferably close to current position
        """
        if blow_axis == 2:
            return pos

        zones = self._get_danger_zones(blow_axis, blow_centers_norm)
        if not zones:
            return pos

        # Find safe regions (gaps between danger zones and world edges)
        safe_regions = []

        # Region before first danger zone
        first_zone_min = zones[0][0] - self.safety_buffer
        if first_zone_min > self.safety_buffer:
            safe_regions.append((self.safety_buffer, first_zone_min - 0.5))

        # Region after last danger zone
        last_zone_max = zones[-1][1] + self.safety_buffer
        if last_zone_max < self.world_size - self.safety_buffer - 1:
            safe_regions.append((last_zone_max + 0.5, self.world_size - self.safety_buffer - 1))

        # Gaps between zones (only if large enough)
        for i in range(len(zones) - 1):
            zone1_max = zones[i][1] + self.safety_buffer
            zone2_min = zones[i + 1][0] - self.safety_buffer

            gap = zone2_min - zone1_max
            if gap > self.safety_buffer * 2:  # Gap is large enough to be safe
                safe_regions.append((zone1_max + 0.5, zone2_min - 0.5))

        if not safe_regions:
            # No safe region found - try to find the largest gap
            # This is a fallback for difficult situations
            best_gap = None
            best_gap_size = 0

            # Check edge regions
            first_zone_min = zones[0][0]
            if first_zone_min > best_gap_size:
                best_gap_size = first_zone_min
                best_gap = (0, first_zone_min - 1)

            last_zone_max = zones[-1][1]
            remaining = self.world_size - last_zone_max - 1
            if remaining > best_gap_size:
                best_gap_size = remaining
                best_gap = (last_zone_max + 1, self.world_size - 1)

            # Check gaps between zones
            for i in range(len(zones) - 1):
                gap_start = zones[i][1]
                gap_end = zones[i + 1][0]
                gap_size = gap_end - gap_start
                if gap_size > best_gap_size:
                    best_gap_size = gap_size
                    best_gap = (gap_start + 0.5, gap_end - 0.5)

            if best_gap:
                safe_regions = [best_gap]

        if not safe_regions:
            return pos  # No safe region at all

        # Find closest safe region to current position
        coord = pos[blow_axis]
        best_target = None
        best_dist = float('inf')

        for region_min, region_max in safe_regions:
            # Target center of safe region
            target = (region_min + region_max) / 2

            dist = abs(coord - target)
            if dist < best_dist:
                best_dist = dist
                best_target = target

        if best_target is None:
            return pos

        # Create safe haven position
        safe_pos = pos.copy()
        safe_pos[blow_axis] = best_target
        return safe_pos

    def _heuristic(self, a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _astar(self, start_cell, goal_cell, danger_cells):
        """A* pathfinding on the discretized grid."""
        if start_cell == goal_cell:
            return [start_cell]

        # If start is in danger, allow escaping through danger
        start_in_danger = start_cell in danger_cells

        open_set = [(self._heuristic(start_cell, goal_cell), 0, start_cell, [start_cell])]
        closed_set = set()
        g_scores = {start_cell: 0}
        counter = 0

        while open_set:
            _, _, current, path = heapq.heappop(open_set)

            if current == goal_cell:
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy

                if not (0 <= nx < self.grid_resolution and 0 <= ny < self.grid_resolution):
                    continue

                neighbor = (nx, ny)

                # Skip danger cells unless we're escaping from danger
                if neighbor in danger_cells and not start_in_danger:
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current] + 1

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_cell)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor, path + [neighbor]))

        return None

    def _compute_control_action(self, pos, vel, target_pos, should_stop=False):
        """
        Simple proportional controller to navigate to target.
        If should_stop=True, prioritize braking at target.
        """
        error = target_pos - pos
        dist = np.linalg.norm(error)
        speed = np.linalg.norm(vel)

        # If we should stop and we're close, brake
        if should_stop and dist < self.waypoint_threshold * 2:
            if speed > 0.2:
                # Brake - apply opposite thrust
                if abs(vel[0]) > abs(vel[1]):
                    return 1 if vel[0] > 0 else 2
                else:
                    return 3 if vel[1] > 0 else 4
            return 0  # Coast when stopped

        # If very close, coast or brake
        if dist < self.waypoint_threshold:
            if speed > 0.3:
                if abs(vel[0]) > abs(vel[1]):
                    return 1 if vel[0] > 0 else 2
                else:
                    return 3 if vel[1] > 0 else 4
            return 0

        # Determine desired direction
        direction = error / (dist + 1e-6)

        # Check if we need to brake (approaching target too fast)
        approach_vel = np.dot(vel, direction)
        if dist < self.brake_threshold and approach_vel > 1.0:
            if abs(vel[0]) > abs(vel[1]):
                return 1 if vel[0] > 0 else 2
            else:
                return 3 if vel[1] > 0 else 4

        # Accelerate toward target
        if abs(error[0]) > abs(error[1]):
            if error[0] > 0 and vel[0] < self.max_vel * 0.7:
                return 2
            elif error[0] < 0 and vel[0] > -self.max_vel * 0.7:
                return 1
            elif error[1] > 0 and vel[1] < self.max_vel * 0.7:
                return 4
            elif error[1] < 0 and vel[1] > -self.max_vel * 0.7:
                return 3
        else:
            if error[1] > 0 and vel[1] < self.max_vel * 0.7:
                return 4
            elif error[1] < 0 and vel[1] > -self.max_vel * 0.7:
                return 3
            elif error[0] > 0 and vel[0] < self.max_vel * 0.7:
                return 2
            elif error[0] < 0 and vel[0] > -self.max_vel * 0.7:
                return 1

        return 0

    def _denormalize_pos(self, normalized_pos):
        """Convert normalized [0,1] position to world coordinates."""
        return np.array(normalized_pos) * self.world_size

    def act(self, obs, explore=True):
        """Choose action using hybrid pathfinding + control."""
        pos = self._denormalize_pos(obs["agent"])
        vel = np.array(obs["velocity"])
        goal_pos = self._denormalize_pos(obs["goal"])
        blow_axis = obs["blow_axis"]
        blow_centers = obs["blow_centers"]

        # Check current safety status
        in_danger = self._is_pos_in_danger(pos, blow_axis, blow_centers, use_buffer=False)
        in_buffered_danger = self._is_pos_in_danger(pos, blow_axis, blow_centers, use_buffer=True)
        between_dangers = self._is_between_dangers(pos, blow_axis, blow_centers)

        # Get danger cells for pathfinding
        danger_cells = self._get_danger_cells(blow_axis, blow_centers)

        current_cell = self._world_to_grid(pos)
        goal_cell = self._world_to_grid(goal_pos)

        # Replan conditions
        danger_state = (blow_axis, tuple(blow_centers) if blow_axis != 2 else None)
        goal_changed = self.last_goal is None or np.linalg.norm(goal_pos - self.last_goal) > 1.0
        danger_changed = danger_state != self.last_danger_state

        if goal_changed or danger_changed:
            self.last_goal = goal_pos.copy()
            self.last_danger_state = danger_state
            self.current_path = []
            self.current_waypoint_idx = 0
            self.safe_haven = None

        # PRIORITY 1: If in immediate danger, escape
        if in_danger:
            safe_haven = self._find_safe_haven(pos, blow_axis, blow_centers)
            return self._compute_control_action(pos, vel, safe_haven, should_stop=False)

        # PRIORITY 2: If between dangers or in buffered danger, move to safe haven
        if between_dangers or in_buffered_danger:
            if self.safe_haven is None:
                self.safe_haven = self._find_safe_haven(pos, blow_axis, blow_centers)
            return self._compute_control_action(pos, vel, self.safe_haven, should_stop=True)

        # PRIORITY 3: If no danger, navigate to goal
        if blow_axis == 2:
            self.safe_haven = None
            return self._compute_control_action(pos, vel, goal_pos, should_stop=False)

        # PRIORITY 4: Danger active but we're safe - check if we can reach goal
        if not self.current_path:
            # Try to find path to goal
            self.current_path = self._astar(current_cell, goal_cell, danger_cells)

            if self.current_path is None:
                # Can't reach goal - find safe haven and wait
                self.safe_haven = self._find_safe_haven(pos, blow_axis, blow_centers)
                self.current_path = []

        # If we have a safe haven and no path to goal, go to safe haven and wait
        if self.safe_haven is not None and not self.current_path:
            safe_haven_dist = np.linalg.norm(pos - self.safe_haven)
            speed = np.linalg.norm(vel)

            # If we're at safe haven, stop
            if safe_haven_dist < self.waypoint_threshold and speed < 0.3:
                return 0  # Stay still

            return self._compute_control_action(pos, vel, self.safe_haven, should_stop=True)

        # Navigate along path to goal
        if self.current_path:
            while self.current_waypoint_idx < len(self.current_path):
                waypoint_cell = self.current_path[self.current_waypoint_idx]
                waypoint_pos = self._grid_to_world(waypoint_cell)

                dist_to_waypoint = np.linalg.norm(pos - waypoint_pos)

                if dist_to_waypoint < self.waypoint_threshold:
                    self.current_waypoint_idx += 1
                else:
                    break

            if self.current_waypoint_idx >= len(self.current_path):
                target = goal_pos
            else:
                target = self._grid_to_world(self.current_path[self.current_waypoint_idx])

            return self._compute_control_action(pos, vel, target, should_stop=False)

        # Fallback: go to goal directly
        return self._compute_control_action(pos, vel, goal_pos, should_stop=False)

    def store(self, obs, action, reward, next_obs, done):
        pass

    def update(self):
        pass

    def reset(self):
        """Reset path state for new episode."""
        self.current_path = []
        self.current_waypoint_idx = 0
        self.last_goal = None
        self.last_danger_state = None
        self.safe_haven = None
