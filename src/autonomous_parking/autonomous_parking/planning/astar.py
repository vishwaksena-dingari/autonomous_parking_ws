"""
astar.py (compatibility shim)

Legacy entry point kept for backward-compatibility.

Old code used:
    from autonomous_parking.planning.astar import AStarPlanner, create_obstacle_grid

After refactor, the real implementations live in:
    - astar_core.py    → AStarPlanner
    - obstacle_grid.py → create_obstacle_grid

This file simply re-exports them so existing imports keep working.
"""

from .astar_core import AStarPlanner
from .obstacle_grid import create_obstacle_grid

__all__ = [
    "AStarPlanner",
    "create_obstacle_grid",
]


# """
# A* Path Planner for Hierarchical RL approach.

# Generates collision-free waypoints from start to goal in parking lot.
# """

# import numpy as np
# import heapq
# from typing import List, Tuple, Optional
# # from ..utils.create_obstacle_grid import create_obstacle_grid

# class AStarPlanner:
#     """Grid-based A* path planner for parking lot navigation."""
    
#     def __init__(self, world_bounds=(-25, 25, -25, 25), resolution=0.5):
#         """
#         Args:
#             world_bounds: (x_min, x_max, y_min, y_max) in meters
#             resolution: Grid cell size in meters
#         """
#         self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
#         self.resolution = resolution
        
#         # Create grid
#         self.grid_width = int((self.x_max - self.x_min) / resolution)
#         self.grid_height = int((self.y_max - self.y_min) / resolution)
        
#     def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
#         """Convert world coordinates to grid indices."""
#         gx = int((x - self.x_min) / self.resolution)
#         gy = int((y - self.y_min) / self.resolution)
#         return gx, gy
    
#     def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
#         """Convert grid indices to world coordinates."""
#         x = gx * self.resolution + self.x_min + self.resolution / 2
#         y = gy * self.resolution + self.y_min + self.resolution / 2
#         return x, y
    
#     def is_valid(self, gx: int, gy: int, obstacles: np.ndarray) -> bool:
#         """Check if grid cell is valid (in bounds and not obstacle)."""
#         if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
#             return False
#         return obstacles[gy, gx] == 0
    
#     def heuristic(self, gx1: int, gy1: int, gx2: int, gy2: int) -> float:
#         """Euclidean distance heuristic."""
#         return np.sqrt((gx1 - gx2)**2 + (gy1 - gy2)**2)
    
#     def plan(
#         self,
#         start: Tuple[float, float, float],
#         goal: Tuple[float, float, float],
#         obstacles: np.ndarray
#     ) -> Optional[List[Tuple[float, float, float]]]:
#         """
#         Plan path from start to goal using A*.
        
#         Args:
#             start: (x, y, theta) starting pose
#             goal: (x, y, theta) goal pose
#             obstacles: 2D grid of obstacles (1 = obstacle, 0 = free)
            
#         Returns:
#             List of waypoints [(x, y, theta), ...] or None if no path
#         """
#         start_x, start_y, start_theta = start
#         goal_x, goal_y, goal_theta = goal
        
#         # Convert to grid coordinates
#         start_gx, start_gy = self.world_to_grid(start_x, start_y)
#         goal_gx, goal_gy = self.world_to_grid(goal_x, goal_y)
        
#         # A* search
#         open_set = []
#         heapq.heappush(open_set, (0, start_gx, start_gy))
        
#         came_from = {}
#         g_score = {(start_gx, start_gy): 0}
#         f_score = {(start_gx, start_gy): self.heuristic(start_gx, start_gy, goal_gx, goal_gy)}
        
#         while open_set:
#             _, current_gx, current_gy = heapq.heappop(open_set)
            
#             # Goal reached
#             # Goal reached
#             if current_gx == goal_gx and current_gy == goal_gy:
#                 return self._reconstruct_path(came_from, (current_gx, current_gy), start_theta, goal_theta, obstacles)
            
#             # Explore neighbors (8-connected)
#             for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
#                 neighbor_gx = current_gx + dx
#                 neighbor_gy = current_gy + dy
                
#                 if not self.is_valid(neighbor_gx, neighbor_gy, obstacles):
#                     continue
                
#                 # Cost to reach neighbor
#                 move_cost = np.sqrt(dx**2 + dy**2)
#                 tentative_g = g_score[(current_gx, current_gy)] + move_cost
                
#                 if (neighbor_gx, neighbor_gy) not in g_score or tentative_g < g_score[(neighbor_gx, neighbor_gy)]:
#                     came_from[(neighbor_gx, neighbor_gy)] = (current_gx, current_gy)
#                     g_score[(neighbor_gx, neighbor_gy)] = tentative_g
#                     f = tentative_g + self.heuristic(neighbor_gx, neighbor_gy, goal_gx, goal_gy)
#                     f_score[(neighbor_gx, neighbor_gy)] = f
#                     heapq.heappush(open_set, (f, neighbor_gx, neighbor_gy))
        
#         # No path found
#         return None
    
#     def _reconstruct_path(
#         self,
#         came_from: dict,
#         current: Tuple[int, int],
#         start_theta: float,
#         goal_theta: float,
#         obstacles: np.ndarray
#     ) -> List[Tuple[float, float, float]]:
#         """Reconstruct path from A* search and smooth it."""
#         # Get grid path
#         path = [current]
#         while current in came_from:
#             current = came_from[current]
#             path.append(current)
#         path.reverse()
        
#         # Convert to world coordinates
#         world_path = []
#         for gx, gy in path:
#             x, y = self.grid_to_world(gx, gy)
#             world_path.append((x, y))
        
#         # Smooth path (Douglas-Peucker with collision check)
#         smoothed = self._smooth_path(world_path, obstacles)
        
#         # Add orientations
#         waypoints = []
#         for i, (x, y) in enumerate(smoothed):
#             if i == 0:
#                 theta = start_theta
#             elif i == len(smoothed) - 1:
#                 theta = goal_theta
#             else:
#                 # Point toward next waypoint
#                 dx = smoothed[i+1][0] - x
#                 dy = smoothed[i+1][1] - y
#                 theta = np.arctan2(dy, dx)
#             waypoints.append((x, y, theta))
        
#         return waypoints
    
#     def _smooth_path(self, path: List[Tuple[float, float]], obstacles: np.ndarray) -> List[Tuple[float, float]]:
#         """Smooth path using Douglas-Peucker algorithm with collision checking."""
#         if len(path) <= 2:
#             return path
        
#         # Keep first and last
#         result = [path[0]]
#         i = 0
#         while i < len(path) - 1:
#             # Try to connect to furthest visible point
#             for j in range(len(path) - 1, i, -1):
#                 if self._is_visible(path[i], path[j], obstacles):
#                     result.append(path[j])
#                     i = j
#                     break
#             else:
#                 # No direct connection, take next point
#                 result.append(path[i + 1])
#                 i += 1
        
#         return result
    
#     def _is_visible(self, p1: Tuple[float, float], p2: Tuple[float, float], obstacles: np.ndarray) -> bool:
#         """Check if two points have line-of-sight (collision-free)."""
#         x1, y1 = p1
#         x2, y2 = p2
        
#         dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         if dist < 0.1:
#             return True
            
#         # Limit connection distance to prevent skipping too much context
#         if dist > 10.0:
#             return False
            
#         # Ray casting / Sampling
#         steps = int(dist / (self.resolution / 2)) + 1
#         for i in range(steps + 1):
#             t = i / steps
#             x = x1 + t * (x2 - x1)
#             y = y1 + t * (y2 - y1)
            
#             gx, gy = self.world_to_grid(x, y)
#             if not self.is_valid(gx, gy, obstacles):
#                 return False
                
#         return True


# def create_obstacle_grid(
#     world_bounds=(-25, 25, -25, 25),
#     resolution=0.5,
#     bays=None,
#     roads=None,
#     margin=0.5,
#     goal_bay=None
# ) -> np.ndarray:
#     """
#     Create road-aware obstacle grid for A* planning.
    
#     CRITICAL: Mark everything as obstacle EXCEPT roads and goal bay.
#     This ensures paths stay on roads and only enter parking bays at the goal.
    
#     Args:
#         world_bounds: (x_min, x_max, y_min, y_max)
#         resolution: Grid resolution in meters
#         bays: List of parking bay dicts with 'x', 'y', 'theta' keys (non-goal bays)
#         roads: List of road dicts with 'x', 'y', 'length', 'width', 'yaw'
#         margin: Safety margin around obstacles in meters
#         goal_bay: Optional dict for the target bay to explicitly mark as FREE
        
#     Returns:
#         2D numpy array where 1 = obstacle, 0 = free (road)
#     """
#     x_min, x_max, y_min, y_max = world_bounds
#     grid_width = int((x_max - x_min) / resolution)
#     grid_height = int((y_max - y_min) / resolution)
    
#     # START WITH EVERYTHING AS OBSTACLE (1)
#     grid = np.ones((grid_height, grid_width), dtype=np.uint8)
    
#     # MARK ROADS AS FREE SPACE (0)
#     if roads:
#         for road in roads:
#             rx, ry = road["x"], road["y"]
#             r_len = road["length"]
#             r_width = road["width"]
#             r_yaw = road["yaw"]
            
#             # Determine bounding box based on orientation
#             # We assume roads are roughly axis-aligned (0 or 90 degrees) for grid simplicity
#             # If needed, we can implement full rotated rectangle rasterization later
            
#             norm_yaw = abs(r_yaw) % np.pi
#             is_horizontal = (norm_yaw < 0.78) or (norm_yaw > 2.35) # < 45 deg or > 135 deg
            
#             if is_horizontal:
#                 # Horizontal road: length along X, width along Y
#                 half_len = r_len / 2.0
#                 half_wid = r_width / 2.0
                
#                 x_start = rx - half_len
#                 x_end = rx + half_len
#                 y_start = ry - half_wid
#                 y_end = ry + half_wid
#             else:
#                 # Vertical road: width along X, length along Y
#                 half_len = r_len / 2.0
#                 half_wid = r_width / 2.0
                
#                 x_start = rx - half_wid
#                 x_end = rx + half_wid
#                 y_start = ry - half_len
#                 y_end = ry + half_len
            
#             # Convert to grid indices
#             gx_min = int((x_start - x_min) / resolution)
#             gx_max = int((x_end - x_min) / resolution)
#             gy_min = int((y_start - y_min) / resolution)
#             gy_max = int((y_end - y_min) / resolution)
            
#             # Clip to grid bounds
#             gx_min = max(0, gx_min)
#             gx_max = min(grid_width, gx_max)
#             gy_min = max(0, gy_min)
#             gy_max = min(grid_height, gy_max)
            
#             # Mark as free space
#             grid[gy_min:gy_max, gx_min:gx_max] = 0
    
#     # MARK PARKING BAYS AS OBSTACLES (except goal bay which is not in this list)
#     if bays:
#         bay_width = 2.7  # meters (actual bay width)
#         bay_length = 5.5  # meters (actual bay length)
        
#         for bay in bays:
#             bx, by = bay["x"], bay["y"]
#             # Convert to grid
#             gx = int((bx - x_min) / resolution)
#             gy = int((by - y_min) / resolution)
            
#             # Mark bay area as obstacle
#             half_w = int((bay_width / 2 + margin) / resolution)
#             half_l = int((bay_length / 2 + margin) / resolution)
            
#             gx_min = max(0, gx - half_w)
#             gx_max = min(grid_width, gx + half_w)
#             gy_min = max(0, gy - half_l)
#             gy_max = min(grid_height, gy + half_l)
            
#             grid[gy_min:gy_max, gx_min:gx_max] = 1
    
#     # Mark world boundaries as obstacles
#     border_cells = int(2.0 / resolution)  # 2m border
#     grid[:border_cells, :] = 1  # Bottom
#     grid[-border_cells:, :] = 1  # Top
#     grid[:, :border_cells] = 1  # Left
#     grid[:, -border_cells:] = 1  # Right
    
#     # CRITICAL: EXPLICITLY MARK GOAL BAY AS FREE (0) BUT KEEP WALLS
#     if goal_bay:
#         bay_width = 2.7
#         bay_length = 5.5
#         bx, by = goal_bay["x"], goal_bay["y"]
#         byaw = goal_bay["yaw"]
        
#         gx = int((bx - x_min) / resolution)
#         gy = int((by - y_min) / resolution)
        
#         # Calculate dimensions in grid cells
#         half_w = int((bay_width / 2) / resolution)
#         half_l = int((bay_length / 2) / resolution)
        
#         # 1. Clear the interior (slightly smaller than full bay to keep walls)
#         # We clear a central corridor
#         inner_w = max(1, half_w - 1)
#         inner_l = max(1, half_l - 1)
        
#         gx_min = max(0, gx - inner_w)
#         gx_max = min(grid_width, gx + inner_w)
#         gy_min = max(0, gy - inner_l)
#         gy_max = min(grid_height, gy + inner_l)
        
#         grid[gy_min:gy_max, gx_min:gx_max] = 0
        
#         # 2. Explicitly OPEN the entrance side based on orientation
#         # Normalize yaw
#         norm_yaw = (byaw + np.pi) % (2 * np.pi) - np.pi
        
#         # Determine entrance direction
#         if abs(norm_yaw) < np.pi / 4:  # Facing South (down)
#             # Entrance is at bottom (-y), Back is top (+y)
#             # Clear bottom edge
#             grid[gy_min-2:gy_min, gx_min:gx_max] = 0
#         elif abs(norm_yaw - np.pi) < np.pi / 4 or abs(norm_yaw + np.pi) < np.pi / 4:  # Facing North (up)
#             # Entrance is at top (+y), Back is bottom (-y)
#             # Clear top edge
#             grid[gy_max:gy_max+2, gx_min:gx_max] = 0
#         elif abs(norm_yaw - np.pi/2) < np.pi / 4:  # Facing East (right)
#             # Entrance is at right (+x), Back is left (-x)
#             # Clear right edge
#             grid[gy_min:gy_max, gx_max:gx_max+2] = 0
#         elif abs(norm_yaw + np.pi/2) < np.pi / 4:  # Facing West (left)
#             # Entrance is at left (-x), Back is right (+x)
#             # Clear left edge
#             grid[gy_min:gy_max, gx_min-2:gx_min] = 0
            
#     return grid
