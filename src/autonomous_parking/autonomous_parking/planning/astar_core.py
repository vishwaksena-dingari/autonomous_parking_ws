"""
astar_core.py

Core A* planner for hierarchical RL parking agent.

This module ONLY handles:
- Grid-based A* search
- Converting between world and grid coordinates
- Reconstructing waypoints from a given obstacle grid

Path smoothing and obstacle grid construction live in:
- smoothing.py
- obstacle_grid.py
"""

import heapq
from typing import List, Tuple, Optional, Dict

import numpy as np

from .smoothing import smooth_path


class AStarPlanner:
    """
    Grid-based A* path planner for parking lot navigation.

    Usage:
        planner = AStarPlanner(world_bounds=(-25, 25, -25, 25), resolution=0.5)
        path = planner.plan(start, goal, obstacle_grid)
    """

    def __init__(self, world_bounds=(-25, 25, -25, 25), resolution: float = 0.5) -> None:
        """
        Args:
            world_bounds: (x_min, x_max, y_min, y_max) in meters.
            resolution: Grid cell size in meters.
        """
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        self.resolution = float(resolution)

        # Initial grid size (may be overridden to match obstacles in plan()).
        self.grid_width = int((self.x_max - self.x_min) / self.resolution)
        self.grid_height = int((self.y_max - self.y_min) / self.resolution)

    # -------------------------------------------------------------------------
    # Coordinate transforms
    # -------------------------------------------------------------------------
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid indices (gx, gy)."""
        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices (gx, gy) to world coordinates (center of cell)."""
        x = gx * self.resolution + self.x_min + self.resolution / 2.0
        y = gy * self.resolution + self.y_min + self.resolution / 2.0
        return x, y

    # -------------------------------------------------------------------------
    # Grid validity & heuristic
    # -------------------------------------------------------------------------
    def is_valid(self, gx: int, gy: int, obstacles: np.ndarray) -> bool:
        """
        Check if grid cell is valid.

        Valid means:
        - inside bounds
        - not marked as obstacle (cell value == 0)
        """
        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False
        return obstacles[gy, gx] == 0

    @staticmethod
    def heuristic(gx1: int, gy1: int, gx2: int, gy2: int) -> float:
        """Euclidean distance heuristic in grid space."""
        return float(np.sqrt((gx1 - gx2) ** 2 + (gy1 - gy2) ** 2))

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        obstacles: np.ndarray,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan a path from start to goal using A* on a pre-built obstacle grid.

        IMPORTANT:
        The obstacle grid is assumed to be built using the same
        world_bounds and resolution as this planner. We still
        sync grid_width / grid_height to obstacles.shape for safety.

        Args:
            start: (x, y, theta) starting pose in world coordinates.
            goal: (x, y, theta) goal pose in world coordinates.
            obstacles: 2D numpy array where 1 = obstacle, 0 = free.

        Returns:
            List of waypoints [(x, y, theta), ...] in world coordinates,
            or None if no path is found.
        """
        # 🔴 NEW: sync internal grid size with given obstacles array
        self.grid_height, self.grid_width = obstacles.shape

        start_x, start_y, start_theta = start
        goal_x, goal_y, goal_theta = goal

        # Convert world coords to grid indices
        start_gx, start_gy = self.world_to_grid(start_x, start_y)
        goal_gx, goal_gy = self.world_to_grid(goal_x, goal_y)

        # Basic sanity: if start or goal is invalid, bail early
        if not self.is_valid(start_gx, start_gy, obstacles):
            print("[AStarPlanner] Start cell is invalid (in obstacle or out of bounds).")
            return None
        if not self.is_valid(goal_gx, goal_gy, obstacles):
            print("[AStarPlanner] Goal cell is invalid (in obstacle or out of bounds).")
            return None

        # A* open set: (f_score, gx, gy)
        open_set: List[Tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, start_gx, start_gy))

        # For path reconstruction & cost tracking
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(start_gx, start_gy): 0.0}
        f_score: Dict[Tuple[int, int], float] = {
            (start_gx, start_gy): self.heuristic(start_gx, start_gy, goal_gx, goal_gy)
        }

        # 8-connected neighbor moves
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        while open_set:
            _, current_gx, current_gy = heapq.heappop(open_set)

            # Goal check
            if current_gx == goal_gx and current_gy == goal_gy:
                return self._reconstruct_path(
                    came_from,
                    (current_gx, current_gy),
                    start_theta,
                    goal_theta,
                    obstacles,
                )

            current_key = (current_gx, current_gy)
            current_g = g_score[current_key]

            for dx, dy in neighbors:
                neighbor_gx = current_gx + dx
                neighbor_gy = current_gy + dy
                neighbor_key = (neighbor_gx, neighbor_gy)

                if not self.is_valid(neighbor_gx, neighbor_gy, obstacles):
                    continue

                # Movement cost (diagonal = sqrt(2), straight = 1)
                move_cost = float(np.sqrt(dx * dx + dy * dy))
                tentative_g = current_g + move_cost

                if neighbor_key not in g_score or tentative_g < g_score[neighbor_key]:
                    came_from[neighbor_key] = current_key
                    g_score[neighbor_key] = tentative_g
                    f = tentative_g + self.heuristic(neighbor_gx, neighbor_gy, goal_gx, goal_gy)
                    f_score[neighbor_key] = f
                    heapq.heappush(open_set, (f, neighbor_gx, neighbor_gy))

        # No path found
        print("[AStarPlanner] No path found.")
        return None

    # -------------------------------------------------------------------------
    # Path reconstruction & smoothing
    # -------------------------------------------------------------------------
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
        start_theta: float,
        goal_theta: float,
        obstacles: np.ndarray,
    ) -> List[Tuple[float, float, float]]:
        """
        Reconstruct and smooth path from A* search results.

        Steps:
        1. Rebuild path in grid coordinates.
        2. Convert to world coordinates.
        3. Smooth using smoothing.smooth_path (with collision checks).
        4. Add orientations along the smoothed path.
        """
        # 1) Backtrack from goal to start in grid space
        path_grid: List[Tuple[int, int]] = [current]
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)
        path_grid.reverse()

        # 2) Convert to world coordinates
        world_path: List[Tuple[float, float]] = [
            self.grid_to_world(gx, gy) for gx, gy in path_grid
        ]

        # 3) Smooth path using helper module
        smoothed_xy = smooth_path(
            path=world_path,
            obstacles=obstacles,
            world_to_grid_fn=self.world_to_grid,
            is_valid_fn=lambda gx, gy: self.is_valid(gx, gy, obstacles),
            resolution=self.resolution,
        )

        # 4) Attach headings
        waypoints: List[Tuple[float, float, float]] = []
        n = len(smoothed_xy)
        for i, (x, y) in enumerate(smoothed_xy):
            if i == 0:
                theta = start_theta
            elif i == n - 1:
                theta = goal_theta
            else:
                dx = smoothed_xy[i + 1][0] - x
                dy = smoothed_xy[i + 1][1] - y
                theta = float(np.arctan2(dy, dx))
            waypoints.append((x, y, theta))

        return waypoints
