"""
Geometry utilities for the autonomous_parking project.

Centralizes small helpers like angle wrapping, angle difference,
and distance computations so they can be reused across envs.
"""

from __future__ import annotations

import math
from typing import Tuple


def wrap_to_pi(angle: float) -> float:
    """
    Wrap an angle to the range [-pi, pi].

    Args:
        angle: Angle in radians (unbounded).

    Returns:
        Wrapped angle in [-pi, pi].
    """
    # Use atan2(sin, cos) for robust wrapping
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(target: float, source: float) -> float:
    """
    Smallest signed angular difference from source -> target.

    Args:
        target: Target angle (radians).
        source: Source angle (radians).

    Returns:
        Angle in [-pi, pi] that, when added to `source`,
        takes you closest to `target`.
    """
    return wrap_to_pi(target - source)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Euclidean distance between two 2D points.

    Args:
        p1: (x1, y1)
        p2: (x2, y2)

    Returns:
        Distance between p1 and p2.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.hypot(dx, dy)


def squared_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Squared Euclidean distance (avoids sqrt if you only compare distances).

    Args:
        p1: (x1, y1)
        p2: (x2, y2)

    Returns:
        Squared distance between p1 and p2.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return dx * dx + dy * dy
