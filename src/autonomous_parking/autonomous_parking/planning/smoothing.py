"""
smoothing.py

Path smoothing utilities for A*.

Implements a Douglas-Peucker-style simplification with collision checking:
- Tries to connect farther points directly if line-of-sight is free.
- Uses the same obstacle grid and world_to_grid mapping as the A* planner.
"""

from typing import Callable, List, Tuple

import numpy as np


WorldPoint = Tuple[float, float]
GridIndexFn = Callable[[float, float], Tuple[int, int]]
IsValidFn = Callable[[int, int], bool]


def smooth_path(
    path: List[WorldPoint],
    obstacles: np.ndarray,
    world_to_grid_fn: GridIndexFn,
    is_valid_fn: IsValidFn,
    resolution: float,
) -> List[WorldPoint]:
    """
    Smooth path using a visibility-based simplification with collision checks.

    Args:
        path: List of (x, y) in world coordinates.
        obstacles: Grid (2D numpy array) where 1 = obstacle, 0 = free.
        world_to_grid_fn: Function mapping (x, y) -> (gx, gy).
        is_valid_fn: Function checking if (gx, gy) is valid (free & in bounds).
        resolution: Grid cell size (meters).

    Returns:
        Smoothed path as list of (x, y) points.
    """
    if len(path) <= 2:
        return path

    result: List[WorldPoint] = [path[0]]
    i = 0
    last_index = len(path) - 1

    while i < last_index:
        # Try to directly connect point i to a farther point j
        connected = False
        for j in range(last_index, i, -1):
            if _is_visible(
                path[i],
                path[j],
                obstacles=obstacles,
                world_to_grid_fn=world_to_grid_fn,
                is_valid_fn=is_valid_fn,
                resolution=resolution,
            ):
                result.append(path[j])
                i = j
                connected = True
                break

        if not connected:
            # No direct connection found; step to next point
            next_idx = i + 1
            if next_idx <= last_index:
                result.append(path[next_idx])
            i = next_idx

    return result


def _is_visible(
    p1: WorldPoint,
    p2: WorldPoint,
    obstacles: np.ndarray,
    world_to_grid_fn: GridIndexFn,
    is_valid_fn: IsValidFn,
    resolution: float,
) -> bool:
    """
    Check if two world points have line-of-sight (collision-free).

    Uses simple ray-sampling between p1 and p2 and queries the obstacle grid.
    """
    x1, y1 = p1
    x2, y2 = p2

    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 0.1:
        # Too close to matter; treat as visible.
        return True

    # Limit max smoothing jump to avoid skipping too much context.
    if dist > 10.0:
        return False

    # Sample points along the line between p1 and p2
    step_length = resolution / 2.0
    num_steps = int(dist / step_length) + 1

    for i in range(num_steps + 1):
        t = i / max(num_steps, 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        gx, gy = world_to_grid_fn(x, y)
        if not is_valid_fn(gx, gy):
            return False

    return True
