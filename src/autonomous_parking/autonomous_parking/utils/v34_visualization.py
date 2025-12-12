#!/usr/bin/env python3
"""
v34 visualization utilities.

Shared helper functions for:
- ParkingEnv.render_v34_overlays (training-time overlays)
- visualize_paths_v34_final.py (offline path visualization)

Conventions (v42):
- Each waypoint is a tuple (x, y, theta).
- goal_bay is a dict with at least: {"x", "y", "yaw"}.
- goal_bay["yaw"] points INTO the bay (same as parked car heading).
- Bay depth axis is aligned with yaw; bay width is perpendicular.
"""

from typing import Sequence, Tuple, List
import math

import numpy as np
from scipy.interpolate import splprep, splev


Waypoint = Tuple[float, float, float]
Point2D = Tuple[float, float]


def compute_path_tangents(waypoints: Sequence[Waypoint]) -> List[Waypoint]:
    """
    Compute orientation at each waypoint from local path tangents.

    This fixes the issue where waypoint theta is sometimes overwritten with the
    final goal yaw, which breaks corridor generation and curvature reasoning.

    Args:
        waypoints: list of (x, y, theta) tuples. theta is ignored and recomputed.

    Returns:
        List of (x, y, tangent_theta) with the same length as input.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    xs = np.array([w[0] for w in waypoints], dtype=float)
    ys = np.array([w[1] for w in waypoints], dtype=float)

    dxs = np.empty_like(xs)
    dys = np.empty_like(ys)

    # Endpoints: one-sided
    dxs[0] = xs[1] - xs[0]
    dys[0] = ys[1] - ys[0]
    dxs[-1] = xs[-1] - xs[-2]
    dys[-1] = ys[-1] - ys[-2]

    if len(waypoints) > 2:
        # Interior: central differences
        dxs[1:-1] = 0.5 * (xs[2:] - xs[:-2])
        dys[1:-1] = 0.5 * (ys[2:] - ys[:-2])

    thetas = np.arctan2(dys, dxs)

    corrected: List[Waypoint] = []
    for (x, y, _), th in zip(waypoints, thetas):
        corrected.append((float(x), float(y), float(th)))

    return corrected


def _dense_spline_path(
    waypoints: Sequence[Waypoint],
    num_points: int = 100,
    smoothing: float = 0.5,
) -> List[Waypoint]:
    """
    Fit a B-spline to the waypoint XY positions and sample a dense, smooth
    path with consistent orientation.

    This is used only for visualization / corridor estimation, not for the
    actual control path inside the environment.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    pts = np.asarray(waypoints, dtype=float)
    xs = pts[:, 0]
    ys = pts[:, 1]

    try:
        # k must be <= len(waypoints) - 1
        k = min(3, len(waypoints) - 1)
        tck, _ = splprep([xs, ys], s=smoothing, k=k)
        u_dense = np.linspace(0.0, 1.0, num_points)
        x_dense, y_dense = splev(u_dense, tck)

        x_dense = np.asarray(x_dense)
        y_dense = np.asarray(y_dense)

        dx = np.gradient(x_dense)
        dy = np.gradient(y_dense)
        theta_dense = np.arctan2(dy, dx)

        dense = list(zip(x_dense, y_dense, theta_dense))
    except Exception:
        # Fallback: original waypoints with tangent-based headings
        dense = compute_path_tangents(waypoints)

    return [(float(x), float(y), float(th)) for x, y, th in dense]


def _bay_geometry_from_dict(
    bay: dict,
    default_depth: float = 5.5,
    default_width: float = 2.7,
) -> Tuple[float, float, float, float, float]:
    """
    Extract bay center, yaw, depth, and width from a bay dict with sensible defaults.
    """
    bx = float(bay["x"])
    by = float(bay["y"])
    yaw = float(bay["yaw"])

    depth = float(bay.get("depth", default_depth))
    width = float(bay.get("width", default_width))

    return bx, by, yaw, depth, width


def calculate_corridor_boundaries(
    waypoints: Sequence[Waypoint],
    goal_bay: dict,
    corridor_width: float = 3.0,
    default_depth: float = 5.5,
    default_width: float = 2.7,
) -> Tuple[List[Point2D], List[Point2D]]:
    """
    Hybrid corridor boundaries using a dense smoothed path:

    1. From start to bay entrance: boundaries follow the smoothed path
       (no sharp corners / loops).
    2. Inside the bay: boundaries are extended using the bay's physical
       rectangle (straight lines).

    Args:
        waypoints: list of (x, y, theta) waypoints (sparse).
        goal_bay: dict with at least {"x", "y", "yaw"}; optional "depth", "width".
        corridor_width: full corridor width in meters (default 3.0).

    Returns:
        (left_boundary, right_boundary) as lists of (x, y) points.
    """
    if len(waypoints) < 2:
        return [], []

    dense_path = _dense_spline_path(waypoints, num_points=100, smoothing=0.5)

    bx, by, b_yaw, b_depth, b_width = _bay_geometry_from_dict(
        goal_bay, default_depth=default_depth, default_width=default_width
    )

    half_depth = 0.5 * b_depth
    half_width = 0.5 * b_width

    # Entrance = "open" side of bay: half-depth BACK along negative yaw direction.
    entrance_x = bx - half_depth * math.cos(b_yaw)
    entrance_y = by - half_depth * math.sin(b_yaw)

    # Find dense-path point closest to entrance (search latter half only).
    split_idx = len(dense_path) - 1
    min_dist = float("inf")
    search_start = len(dense_path) // 2

    for i in range(search_start, len(dense_path)):
        px, py, _ = dense_path[i]
        d = math.hypot(px - entrance_x, py - entrance_y)
        if d < min_dist:
            min_dist = d
            split_idx = i

    approach_path = dense_path[: split_idx + 1]

    offset = 0.5 * corridor_width
    left_boundary: List[Point2D] = []
    right_boundary: List[Point2D] = []

    # Corridor following the path
    for x, y, th in approach_path:
        # Perpendicular to path direction
        perp_x = -math.sin(th)
        perp_y = math.cos(th)

        left_boundary.append((x + offset * perp_x, y + offset * perp_y))
        right_boundary.append((x - offset * perp_x, y - offset * perp_y))

    # Extend inside bay with straight bay edges.
    # Width axis is perpendicular to yaw.
    bay_perp_yaw = b_yaw + math.pi / 2.0

    # Front edge (entrance side)
    fl_x = entrance_x + half_width * math.cos(bay_perp_yaw)
    fl_y = entrance_y + half_width * math.sin(bay_perp_yaw)
    fr_x = entrance_x - half_width * math.cos(bay_perp_yaw)
    fr_y = entrance_y - half_width * math.sin(bay_perp_yaw)

    # Back edge (deep inside bay)
    back_x = bx + half_depth * math.cos(b_yaw)
    back_y = by + half_depth * math.sin(b_yaw)
    bl_x = back_x + half_width * math.cos(bay_perp_yaw)
    bl_y = back_y + half_width * math.sin(bay_perp_yaw)
    br_x = back_x - half_width * math.cos(bay_perp_yaw)
    br_y = back_y - half_width * math.sin(bay_perp_yaw)

    left_boundary.extend([(fl_x, fl_y), (bl_x, bl_y)])
    right_boundary.extend([(fr_x, fr_y), (br_x, br_y)])

    return left_boundary, right_boundary


def calculate_8_point_bay_reference(
    goal_bay: dict,
    default_depth: float = 5.5,
    default_width: float = 2.7,
) -> List[Point2D]:
    """
    Compute an 8-point reference set anchored to the bay rectangle:

    - 4 corners (depth × width)
    - 4 midpoints (two along depth, two along width)

    Useful as a visual reference when debugging parking precision.
    """
    bx, by, b_yaw, b_depth, b_width = _bay_geometry_from_dict(
        goal_bay, default_depth=default_depth, default_width=default_width
    )

    half_depth = 0.5 * b_depth
    half_width = 0.5 * b_width

    depth_axis = np.array([math.cos(b_yaw), math.sin(b_yaw)])
    width_axis = np.array(
        [math.cos(b_yaw + math.pi / 2.0), math.sin(b_yaw + math.pi / 2.0)]
    )

    center = np.array([bx, by], dtype=float)

    points: List[Point2D] = []

    # 4 corners in local (depth, width) frame
    local_corners = [
        (-half_depth, +half_width),
        (-half_depth, -half_width),
        (+half_depth, -half_width),
        (+half_depth, +half_width),
    ]

    for d, w in local_corners:
        world = center + d * depth_axis + w * width_axis
        points.append((float(world[0]), float(world[1])))

    # 4 midpoints: entrance center, back center, right mid, left mid
    local_mids = [
        (-half_depth, 0.0),        # entrance center
        (+half_depth, 0.0),        # back center
        (0.0, -half_width),        # right mid
        (0.0, +half_width),        # left mid
    ]

    for d, w in local_mids:
        world = center + d * depth_axis + w * width_axis
        points.append((float(world[0]), float(world[1])))

    return points
