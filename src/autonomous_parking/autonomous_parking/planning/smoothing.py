"""
Path Smoothing Module for Autonomous Parking

Centralized location for all path smoothing techniques:
- Douglas-Peucker visibility-based smoothing
- B-spline parametric curve smoothing
- Reeds-Shepp curve smoothing (kinematically feasible)

Usage:
    from autonomous_parking.planning.smoothing import (
        smooth_path_douglas_peucker,
        smooth_path_bspline,
        smooth_path_reeds_shepp,
        SmoothingMethod
    )
"""

from typing import Callable, List, Tuple, Optional
from enum import Enum

import numpy as np
from scipy.interpolate import splprep, splev


# Type aliases
WorldPoint = Tuple[float, float]
WorldPose = Tuple[float, float, float]  # (x, y, theta)
GridIndexFn = Callable[[float, float], Tuple[int, int]]
IsValidFn = Callable[[int, int], bool]  # Fixed: removed extra bracket


class SmoothingMethod(Enum):
    """Available path smoothing methods."""
    DOUGLAS_PEUCKER = "douglas_peucker"
    BSPLINE = "bspline"
    REEDS_SHEPP = "reeds_shepp"
    NONE = "none"


# ============================================================================
# Douglas-Peucker Visibility-Based Smoothing (Original)
# ============================================================================

def smooth_path_douglas_peucker(
    path: List[WorldPoint],
    obstacles: np.ndarray,
    world_to_grid_fn: GridIndexFn,
    is_valid_fn: IsValidFn,
    resolution: float,
) -> List[WorldPoint]:
    """
    Smooth path by removing waypoints if a straight line is collision-free.

    This is a visibility-based smoother: it tries to skip intermediate
    waypoints if the direct line-of-sight is obstacle-free.

    Args:
        path: List of (x, y) waypoints
        obstacles: 2D grid (0=free, 1=occupied)
        world_to_grid_fn: Function to convert (x, y) -> (gx, gy)
        is_valid_fn: Function to check if (gx, gy) is valid
        resolution: Grid cell size (meters)

    Returns:
        Smoothed path as list of (x, y) waypoints
    """
    if len(path) < 3:
        return path

    def is_line_clear(p1: WorldPoint, p2: WorldPoint) -> bool:
        """Check if straight line from p1 to p2 is collision-free."""
        dist = np.linalg.norm([p2[0] - p1[0], p2[1] - p1[1]])
        num_samples = max(int(dist / (resolution * 0.5)), 2)

        for i in range(num_samples + 1):
            alpha = i / num_samples
            x = p1[0] + alpha * (p2[0] - p1[0])
            y = p1[1] + alpha * (p2[1] - p1[1])

            gx, gy = world_to_grid_fn(x, y)
            if not is_valid_fn(gx, gy):
                return False

        return True

    smooth = [path[0]]
    i = 0

    while i < len(path) - 1:
        # Try to skip as many waypoints as possible
        j = len(path) - 1
        while j > i + 1:
            if is_line_clear(path[i], path[j]):
                smooth.append(path[j])
                i = j
                break
            j -= 1
        else:
            # Couldn't skip any, move to next waypoint
            smooth.append(path[i + 1])
            i += 1

    return smooth


# ============================================================================
# B-spline Parametric Curve Smoothing
# ============================================================================

def smooth_path_bspline(
    waypoints: List[WorldPose],
    smoothness: float = 0.1,
    densification_factor: float = 3.0,
    validate_collision_free: bool = False,
    obstacles: Optional[np.ndarray] = None,
    world_to_grid_fn: Optional[GridIndexFn] = None,
) -> List[WorldPose]:
    """
    Smooth (x, y) path with a B-spline and reconstruct yaw from tangent.

    IMPROVEMENTS APPLIED:
    - Reduced smoothing parameter (s=0.1 instead of 2.0)
    - No phantom end-point (spline goes exactly from start to goal)
    - Dense sampling for visually smoother curves
    - Optional collision validation

    Args:
        waypoints: List of (x, y, theta) poses
        smoothness: B-spline smoothing parameter (0.0 = interpolate exactly, higher = smoother)
        densification_factor: How many samples per input waypoint (3.0 = 3x denser)
        validate_collision_free: If True, check smoothed path doesn't cut corners
        obstacles: Obstacle grid (required if validate_collision_free=True)
        world_to_grid_fn: Grid conversion function (required if validate_collision_free=True)

    Returns:
        Smoothed path as list of (x, y, theta) poses
        Falls back to original if spline fails or collision detected
    """
    if len(waypoints) < 3:
        return waypoints

    try:
        pts = np.array([[w[0], w[1]] for w in waypoints])
        goal_x, goal_y, goal_theta = waypoints[-1]

        # Fit spline directly through the original points (no phantom)
        tck, _ = splprep(
            [pts[:, 0], pts[:, 1]],
            s=smoothness,  # Tight fit
            k=min(3, len(pts) - 1),  # Cubic spline if enough points
        )

        # Dense sampling for smooth visualization
        num_samples = max(int(len(waypoints) * densification_factor), len(waypoints) + 2)
        u_new = np.linspace(0.0, 1.0, num_samples)
        sx, sy = splev(u_new, tck)

        # Reconstruct poses with tangent-based yaw
        smooth: List[WorldPose] = []
        for i in range(len(sx)):
            if i < len(sx) - 1:
                dx = sx[i + 1] - sx[i]
                dy = sy[i + 1] - sy[i]
                theta = float(np.arctan2(dy, dx))
            else:
                # Force final orientation to match bay yaw
                theta = goal_theta
            smooth.append((float(sx[i]), float(sy[i]), theta))

        # Optional: Validate collision-free
        if validate_collision_free and obstacles is not None and world_to_grid_fn is not None:
            if not _validate_bspline_collision_free(smooth, obstacles, world_to_grid_fn):
                print("[WARN] B-spline cut corners through obstacles, using original path")
                return waypoints

        return smooth

    except Exception as e:
        print(f"[WaypointEnv] WARNING: B-spline failed ({e}), using raw waypoints")
        return waypoints


def _validate_bspline_collision_free(
    smooth_path: List[WorldPose],
    obstacles: np.ndarray,
    world_to_grid_fn: GridIndexFn,
) -> bool:
    """
    Check if B-spline path is collision-free.

    Args:
        smooth_path: Smoothed path to validate
        obstacles: 2D grid (0=free, 1=occupied)
        world_to_grid_fn: Function to convert (x, y) -> (gx, gy)

    Returns:
        True if path is collision-free, False otherwise
    """
    for x, y, _ in smooth_path:
        gx, gy = world_to_grid_fn(x, y)
        grid_height, grid_width = obstacles.shape

        if gx < 0 or gx >= grid_width or gy < 0 or gy >= grid_height:
            return False

        if obstacles[gy, gx] != 0:
            return False

    return True


# ============================================================================
# Reeds-Shepp Curve Smoothing
# ============================================================================

def smooth_path_reeds_shepp(
    waypoints: List[WorldPose],
    turning_radius: float = 5.0,
    step_size: float = 0.5,
) -> List[WorldPose]:
    """
    Smooth path using Reeds-Shepp curves (forward/backward arcs).
    
    This connects each consecutive pair of waypoints with a Reeds-Shepp
    path that respects minimum turning radius. The last waypoint's pose
    (x, y, theta) is preserved exactly.
    
    Args:
        waypoints: List of (x, y, theta) poses
        turning_radius: Minimum turning radius (meters)
        step_size: Discretization step along the curve (meters)
        
    Returns:
        Smoothed path as list of (x, y, theta) poses.
        Falls back to original segment if RS fails.
    """
    from .reeds_shepp import reeds_shepp_path_planning
    
    if len(waypoints) < 2:
        return waypoints
    
    maxc = 1.0 / max(turning_radius, 1e-6)
    full_path: List[WorldPose] = []
    
    for i in range(len(waypoints) - 1):
        sx, sy, syaw = waypoints[i]
        gx, gy, gyaw = waypoints[i + 1]
        
        xs, ys, yaws, _, _ = reeds_shepp_path_planning(
            sx, sy, syaw,
            gx, gy, gyaw,
            maxc=maxc,
            step_size=step_size,
        )
        
        if xs is None:
            # Fallback: just keep the start waypoint
            full_path.append((sx, sy, syaw))
            continue
        
        # Add all RS points except the very last one (to avoid duplicates)
        for x, y, yaw in zip(xs[:-1], ys[:-1], yaws[:-1]):
            full_path.append((float(x), float(y), float(yaw)))
    
    # Force final waypoint exactly as specified (x, y, goal theta)
    gx, gy, gyaw = waypoints[-1]
    full_path.append((float(gx), float(gy), float(gyaw)))
    
    return full_path


# ============================================================================
# Theta Reconstruction Helper
# ============================================================================

def reconstruct_theta_from_xy(
    smoothed_points: List[WorldPoint],
    original_waypoints: List[WorldPose],
) -> List[WorldPose]:
    """
    Reconstruct (x, y, theta) from smoothed (x, y) points.

    Uses tangent direction between consecutive points for theta.
    """
    if len(smoothed_points) == 0:
        return []

    result: List[WorldPose] = []
    for i in range(len(smoothed_points)):
        x, y = smoothed_points[i]

        if i < len(smoothed_points) - 1:
            # Tangent direction to next point
            dx = smoothed_points[i + 1][0] - x
            dy = smoothed_points[i + 1][1] - y
            theta = float(np.arctan2(dy, dx))
        else:
            # Use last original waypoint's theta
            theta = original_waypoints[-1][2]

        result.append((x, y, theta))

    return result
