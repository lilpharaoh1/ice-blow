# agents/pathfinding_agent.py
import numpy as np
import heapq
from agents.base import BaseAgent


class PathfindingAgent(BaseAgent):
    """
    A pathfinding agent that uses A* to navigate to the goal while avoiding dangers.

    Behavior:
    - Uses A* algorithm to find optimal path to goal
    - During warning/active blow phases, avoids dangerous cells
    - If in danger zone, prioritizes escaping to safety
    - If path to goal is blocked by danger, gets as close as safely possible
    """

    def __init__(self, action_space, obs_space=None, env_type="gridworld",
                 grid_size=10, blow_width=0, safety_margin=1, **kwargs):
        super().__init__(action_space, obs_space)
        self.grid_size = grid_size
        self.blow_width = blow_width
        self.safety_margin = safety_margin  # Extra cells to avoid around danger
        self.current_path = []
        self.last_goal = None

        # Action mapping: action -> (dx, dy)
        self.action_to_delta = {
            0: (0, 0),   # no-op
            1: (-1, 0),  # left
            2: (1, 0),   # right
            3: (0, -1),  # up
            4: (0, 1),   # down
        }

        # Reverse mapping: (dx, dy) -> action
        self.delta_to_action = {v: k for k, v in self.action_to_delta.items()}

    def act(self, obs, explore=True):
        """
        Choose action based on current observation using A* pathfinding.
        """
        # Denormalize positions to grid coordinates
        agent_pos = self._denormalize_pos(obs["agent"])
        goal_pos = self._denormalize_pos(obs["goal"])

        # Get danger information
        blow_axis = obs["blow_axis"]  # 0=X, 1=Y, 2=inactive
        blow_centers = obs["blow_centers"]

        # Determine if we're in a danger phase (warning or active)
        is_danger_phase = blow_axis != 2

        # Get dangerous cells
        danger_cells = set()
        if is_danger_phase:
            danger_cells = self._get_danger_cells(blow_axis, blow_centers)

        # Check if agent is currently in danger
        agent_tuple = tuple(agent_pos)
        agent_in_danger = agent_tuple in danger_cells

        # Priority 1: If in danger, escape immediately
        if agent_in_danger:
            escape_action = self._find_escape_action(agent_pos, danger_cells)
            if escape_action is not None:
                return escape_action

        # Priority 2: Find path to goal avoiding danger
        path = self._astar(agent_pos, goal_pos, danger_cells)

        if path and len(path) > 1:
            # We have a safe path to goal
            next_pos = path[1]  # path[0] is current position
            return self._get_action_to_move(agent_pos, next_pos)

        # Priority 3: No safe path to goal - get as close as possible safely
        if danger_cells:
            closest_safe = self._find_closest_safe_to_goal(agent_pos, goal_pos, danger_cells)
            if closest_safe is not None and closest_safe != agent_tuple:
                path = self._astar(agent_pos, np.array(closest_safe), danger_cells)
                if path and len(path) > 1:
                    next_pos = path[1]
                    return self._get_action_to_move(agent_pos, next_pos)

        # Priority 4: If we're safe and no path available, wait
        if not agent_in_danger:
            return 0  # no-op, stay safe

        # Fallback: random safe action
        return self._get_random_safe_action(agent_pos, danger_cells)

    def _denormalize_pos(self, normalized_pos):
        """Convert normalized [0,1] position back to grid coordinates."""
        return np.clip(
            np.round(np.array(normalized_pos) * self.grid_size).astype(int),
            0,
            self.grid_size - 1
        )

    def _get_danger_cells(self, blow_axis, blow_centers):
        """
        Get set of all cells that are dangerous.
        blow_axis: 0 = danger along X axis, 1 = danger along Y axis
        blow_centers: normalized positions of blow lines
        """
        danger_cells = set()

        # Denormalize blow centers
        centers = [int(round(c * self.grid_size)) for c in blow_centers]

        for center in centers:
            # Calculate the range of dangerous cells with width and safety margin
            total_width = self.blow_width + self.safety_margin

            for offset in range(-total_width, total_width + 1):
                coord = center + offset
                if 0 <= coord < self.grid_size:
                    if blow_axis == 0:  # X-axis blow - danger spans all Y for specific X
                        for y in range(self.grid_size):
                            danger_cells.add((coord, y))
                    else:  # Y-axis blow - danger spans all X for specific Y
                        for x in range(self.grid_size):
                            danger_cells.add((x, coord))

        return danger_cells

    def _find_escape_action(self, agent_pos, danger_cells):
        """
        Find the best action to escape from danger.
        Uses A* to find path to nearest safe cell if single-step escape isn't possible.
        """
        agent_tuple = tuple(agent_pos)

        # First, try single-step escape (fastest)
        best_action = None
        best_score = float('-inf')

        for action, (dx, dy) in self.action_to_delta.items():
            if action == 0:  # Skip no-op when escaping
                continue

            new_x = np.clip(agent_pos[0] + dx, 0, self.grid_size - 1)
            new_y = np.clip(agent_pos[1] + dy, 0, self.grid_size - 1)
            new_pos = (new_x, new_y)

            if new_pos not in danger_cells:
                # Score based on distance from danger
                min_dist = self._min_distance_to_danger(new_pos, danger_cells)
                if min_dist > best_score:
                    best_score = min_dist
                    best_action = action

        if best_action is not None:
            return best_action

        # No single-step escape possible - need multi-step escape
        # Find nearest safe cell and path to it (allowing travel through danger)
        nearest_safe = self._find_nearest_safe_cell(agent_pos, danger_cells)

        if nearest_safe is not None:
            # Use BFS to find shortest path (even through danger, since we're already in it)
            path = self._bfs_escape(agent_pos, nearest_safe)
            if path and len(path) > 1:
                next_pos = path[1]
                return self._get_action_to_move(agent_pos, next_pos)

        # Fallback: move toward closest safe cell directly
        if nearest_safe is not None:
            return self._get_action_toward(agent_pos, nearest_safe)

        return None

    def _find_nearest_safe_cell(self, agent_pos, danger_cells):
        """Find the nearest cell not in danger using BFS."""
        from collections import deque

        start = tuple(agent_pos)
        if start not in danger_cells:
            return start

        visited = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy

                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue

                neighbor = (nx, ny)
                if neighbor in visited:
                    continue

                visited.add(neighbor)

                if neighbor not in danger_cells:
                    return neighbor

                queue.append(neighbor)

        return None  # No safe cell exists

    def _bfs_escape(self, start, goal):
        """BFS to find shortest path from start to goal (ignoring obstacles)."""
        from collections import deque

        start = tuple(start)
        goal = tuple(goal)

        if start == goal:
            return [start]

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy

                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue

                neighbor = (nx, ny)
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                new_path = path + [neighbor]

                if neighbor == goal:
                    return new_path

                queue.append((neighbor, new_path))

        return None

    def _get_action_toward(self, current_pos, target_pos):
        """Get action that moves toward target position."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        # Prefer larger delta
        if abs(dx) >= abs(dy) and dx != 0:
            return self.delta_to_action.get((1 if dx > 0 else -1, 0), 0)
        elif dy != 0:
            return self.delta_to_action.get((0, 1 if dy > 0 else -1), 0)
        return 0

    def _min_distance_to_danger(self, pos, danger_cells):
        """Calculate minimum Manhattan distance to any danger cell."""
        if not danger_cells:
            return float('inf')

        min_dist = float('inf')
        for dc in danger_cells:
            dist = abs(pos[0] - dc[0]) + abs(pos[1] - dc[1])
            min_dist = min(min_dist, dist)
        return min_dist

    def _astar(self, start, goal, obstacles):
        """
        A* pathfinding algorithm.
        Returns path as list of (x, y) tuples, or None if no path exists.
        """
        start = tuple(start)
        goal = tuple(goal)

        if start == goal:
            return [start]

        # Priority queue: (f_score, counter, position, path)
        counter = 0
        open_set = [(self._heuristic(start, goal), counter, start, [start])]
        closed_set = set()
        g_scores = {start: 0}

        while open_set:
            _, _, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in closed_set:
                continue
            closed_set.add(current)

            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = current[0] + dx
                ny = current[1] + dy

                # Check bounds
                if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    continue

                neighbor = (nx, ny)

                # Skip obstacles (danger cells)
                if neighbor in obstacles:
                    continue

                # Skip if already processed
                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current] + 1

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor, path + [neighbor]))

        return None  # No path found

    def _heuristic(self, a, b):
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _find_closest_safe_to_goal(self, agent_pos, goal_pos, danger_cells):
        """
        Find the closest safe cell to the goal that we can reach.
        Uses BFS from goal, stopping at first safe cell reachable from agent.
        """
        goal = tuple(goal_pos)
        agent = tuple(agent_pos)

        # Get all safe cells
        all_cells = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        safe_cells = all_cells - danger_cells

        if not safe_cells:
            return None

        # BFS from goal to find closest safe cell
        # Sort safe cells by distance to goal
        safe_by_dist = sorted(safe_cells, key=lambda c: self._heuristic(c, goal))

        # Find the closest safe cell that's reachable from agent
        for safe_cell in safe_by_dist:
            if safe_cell == agent:
                return safe_cell
            # Check if we can reach this safe cell
            path = self._astar(agent_pos, np.array(safe_cell), danger_cells)
            if path:
                return safe_cell

        return None

    def _get_action_to_move(self, current_pos, target_pos):
        """Get the action needed to move from current to target position."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        # Clamp to single step
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))

        # Only move in one direction at a time
        if dx != 0:
            return self.delta_to_action.get((dx, 0), 0)
        elif dy != 0:
            return self.delta_to_action.get((0, dy), 0)
        return 0

    def _get_random_safe_action(self, agent_pos, danger_cells):
        """Get a random action that doesn't move into danger."""
        safe_actions = []

        for action, (dx, dy) in self.action_to_delta.items():
            new_x = np.clip(agent_pos[0] + dx, 0, self.grid_size - 1)
            new_y = np.clip(agent_pos[1] + dy, 0, self.grid_size - 1)

            if (new_x, new_y) not in danger_cells:
                safe_actions.append(action)

        if safe_actions:
            return np.random.choice(safe_actions)
        return 0  # No safe action, stay put

    def store(self, obs, action, reward, next_obs, done):
        """No learning for this agent."""
        pass

    def update(self):
        """No learning for this agent."""
        pass
