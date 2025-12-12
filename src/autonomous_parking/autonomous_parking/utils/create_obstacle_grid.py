"""
Backward-compatible wrapper for the road-aware obstacle grid.

New code should import from:
    autonomous_parking.planning.astar import create_obstacle_grid

Some older scripts might use:
    from autonomous_parking.utils.create_obstacle_grid import create_obstacle_grid

This module simply forwards the call to the planning.astar implementation.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ..planning.astar import create_obstacle_grid as _create_obstacle_grid


def create_obstacle_grid(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Thin wrapper around autonomous_parking.planning.astar.create_obstacle_grid.

    Accepts the same arguments as the original function, e.g.:

        create_obstacle_grid(
            world_bounds=(-25, 25, -25, 25),
            resolution=0.25,
            bays=bays,
            roads=roads,
            margin=0.5,
            goal_bay=goal_bay,
        )
    """
    return _create_obstacle_grid(*args, **kwargs)
