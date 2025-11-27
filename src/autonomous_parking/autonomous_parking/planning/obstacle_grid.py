"""
obstacle_grid.py

Road-aware obstacle grid construction for A* planning.

Key design:
- Start with EVERYTHING as obstacle (1).
- Explicitly carve out:
    - Roads as free space (0).
    - Only the target parking bay interior + entrance as free space.
- All other bays remain obstacles so A* stays on the road
  and only enters the goal bay.
"""

from typing import List, Optional, Dict, Tuple

import numpy as np


WorldBounds = Tuple[float, float, float, float]


def create_obstacle_grid(
    world_bounds: WorldBounds = (-25, 25, -25, 25),
    resolution: float = 0.5,
    bays: Optional[List[Dict]] = None,
    roads: Optional[List[Dict]] = None,
    margin: float = 0.5,
    goal_bay: Optional[Dict] = None,
) -> np.ndarray:
    """
    Create road-aware obstacle grid for A* planning.

    CRITICAL: Mark everything as obstacle EXCEPT roads and goal bay.
    This ensures paths stay on roads and only enter parking bays at the goal.

    Args:
        world_bounds: (x_min, x_max, y_min, y_max) in meters.
        resolution: Grid resolution in meters.
        bays: List of NON-GOAL bay dicts, each with at least 'x', 'y' keys.
        roads: List of road dicts with:
            - 'x', 'y': center position of road segment (world coords)
            - 'length': length along main axis (meters)
            - 'width': width across minor axis (meters)
            - 'yaw': orientation (radians)
        margin: Safety margin around bays (meters).
        goal_bay: Optional dict with at least:
            - 'x', 'y', 'yaw': center position and orientation of the target bay.

    Returns:
        2D numpy array of shape (H, W) where:
            - 1 = obstacle
            - 0 = free space (roads + target bay interior/entrance)
    """
    x_min, x_max, y_min, y_max = world_bounds
    resolution = float(resolution)

    grid_width = int((x_max - x_min) / resolution)
    grid_height = int((y_max - y_min) / resolution)

    # Start with everything as obstacle.
    grid = np.ones((grid_height, grid_width), dtype=np.uint8)

    # -------------------------------------------------------------------------
    # Mark roads as free space
    # -------------------------------------------------------------------------
    if roads:
        for road in roads:
            rx = float(road["x"])
            ry = float(road["y"])
            r_len = float(road["length"])
            r_width = float(road["width"])
            r_yaw = float(road["yaw"])

            # Very simple orientation check: assume roads are ~axis-aligned
            norm_yaw = abs(r_yaw) % np.pi
            is_horizontal = (norm_yaw < 0.78) or (norm_yaw > 2.35)  # < 45° or > 135°

            if is_horizontal:
                # Horizontal: length along X, width along Y
                half_len = r_len / 2.0
                half_wid = r_width / 2.0

                x_start = rx - half_len
                x_end = rx + half_len
                y_start = ry - half_wid
                y_end = ry + half_wid
            else:
                # Vertical: width along X, length along Y
                half_len = r_len / 2.0
                half_wid = r_width / 2.0

                x_start = rx - half_wid
                x_end = rx + half_wid
                y_start = ry - half_len
                y_end = ry + half_len

            # Convert world bounds to grid indices
            gx_min = int((x_start - x_min) / resolution)
            gx_max = int((x_end - x_min) / resolution)
            gy_min = int((y_start - y_min) / resolution)
            gy_max = int((y_end - y_min) / resolution)

            # Clip to grid
            gx_min = max(0, gx_min)
            gx_max = min(grid_width, gx_max)
            gy_min = max(0, gy_min)
            gy_max = min(grid_height, gy_max)

            # Mark as free space
            grid[gy_min:gy_max, gx_min:gx_max] = 0

    # -------------------------------------------------------------------------
    # Mark NON-goal parking bays as obstacles
    # -------------------------------------------------------------------------
    if bays:
        bay_width = 2.7  # meters
        bay_length = 5.5  # meters

        for bay in bays:
            bx = float(bay["x"])
            by = float(bay["y"])

            gx = int((bx - x_min) / resolution)
            gy = int((by - y_min) / resolution)

            half_w = int((bay_width / 2.0 + margin) / resolution)
            half_l = int((bay_length / 2.0 + margin) / resolution)

            gx_min = max(0, gx - half_w)
            gx_max = min(grid_width, gx + half_w)
            gy_min = max(0, gy - half_l)
            gy_max = min(grid_height, gy + half_l)

            grid[gy_min:gy_max, gx_min:gx_max] = 1

    # -------------------------------------------------------------------------
    # Mark world boundaries as obstacles (2 m border)
    # -------------------------------------------------------------------------
    border_cells = int(2.0 / resolution)
    if border_cells > 0:
        grid[:border_cells, :] = 1      # bottom
        grid[-border_cells:, :] = 1     # top
        grid[:, :border_cells] = 1      # left
        grid[:, -border_cells:] = 1     # right

    # -------------------------------------------------------------------------
    # Explicitly mark GOAL bay interior + entrance as free (0), but keep walls
    # -------------------------------------------------------------------------
    if goal_bay is not None:
        bay_width = 2.7
        bay_length = 5.5

        bx = float(goal_bay["x"])
        by = float(goal_bay["y"])
        byaw = float(goal_bay["yaw"])

        gx_center = int((bx - x_min) / resolution)
        gy_center = int((by - y_min) / resolution)

        half_w = int((bay_width / 2.0) / resolution)
        half_l = int((bay_length / 2.0) / resolution)

        # 1) Clear interior corridor (keep "wall" cells around it)
        inner_w = max(1, half_w - 1)
        inner_l = max(1, half_l - 1)

        gx_min = max(0, gx_center - inner_w)
        gx_max = min(grid_width, gx_center + inner_w)
        gy_min = max(0, gy_center - inner_l)
        gy_max = min(grid_height, gy_center + inner_l)

        grid[gy_min:gy_max, gx_min:gx_max] = 0

        # 2) Open the entrance edge based on yaw
        norm_yaw = (byaw + np.pi) % (2.0 * np.pi) - np.pi

        try:
            if abs(norm_yaw) < np.pi / 4.0:
                # Facing South (down): entrance at bottom (-y)
                grid[max(0, gy_min - 2):gy_min, gx_min:gx_max] = 0
            elif abs(norm_yaw - np.pi) < np.pi / 4.0 or abs(norm_yaw + np.pi) < np.pi / 4.0:
                # Facing North (up): entrance at top (+y)
                grid[gy_max:min(grid_height, gy_max + 2), gx_min:gx_max] = 0
            elif abs(norm_yaw - np.pi / 2.0) < np.pi / 4.0:
                # Facing East (right): entrance at right (+x)
                grid[gy_min:gy_max, gx_max:min(grid_width, gx_max + 2)] = 0
            elif abs(norm_yaw + np.pi / 2.0) < np.pi / 4.0:
                # Facing West (left): entrance at left (-x)
                grid[gy_min:gy_max, max(0, gx_min - 2):gx_min] = 0
        except IndexError:
            # In case bay is very near boundary; safer to ignore than crash
            pass

    return grid
