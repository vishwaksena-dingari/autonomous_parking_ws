#!/usr/bin/env python3
"""
Corridor Path Planning Module

Provides corridor boundary calculation for path following and constraint-based rewards.
This is CORE PLANNING LOGIC for the autonomous parking system.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Dict, Optional


def compute_path_tangents(waypoints: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """
    Compute orientation at each waypoint from path tangents.
    
    This fixes the issue where waypoint theta is set to goal_yaw prematurely
    in waypoint_env.py, causing incorrect orientations for approach waypoints.
    
    Args:
        waypoints: List of (x, y, theta) tuples
        
    Returns:
        List of (x, y, corrected_theta) tuples with tangent-based orientations
    """
    if len(waypoints) < 2:
        return waypoints
        
    corrected = []
    for i, (x, y, _) in enumerate(waypoints):
        if i == 0:
            # First point: use direction to next point
            dx = waypoints[i+1][0] - x
            dy = waypoints[i+1][1] - y
        elif i == len(waypoints) - 1:
            # Last point: use direction from previous point
            dx = x - waypoints[i-1][0]
            dy = y - waypoints[i-1][1]
        else:
            # Middle points: average of incoming and outgoing directions
            dx1 = x - waypoints[i-1][0]
            dy1 = y - waypoints[i-1][1]
            dx2 = waypoints[i+1][0] - x
            dy2 = waypoints[i+1][1] - y
            dx = (dx1 + dx2) / 2.0
            dy = (dy1 + dy2) / 2.0
            
        theta = np.arctan2(dy, dx)
        corrected.append((x, y, theta))
        
    return corrected


def calculate_corridor_boundaries(
    waypoints: List[Tuple[float, float, float]], 
    goal_bay: Dict,
    corridor_width: float = 2.2  # Car (1.9m) + 0.3m margin (was 3.0m)
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Calculate hybrid corridor boundaries for path following.
    
    This is the CORE PLANNING LOGIC for corridor-based path following:
    1. Approach phase: Smooth corridor from dense B-spline path
    2. Bay entry phase: Straight boundaries aligned with bay geometry
    
    Args:
        waypoints: List of (x, y, theta) - sparse waypoints
        goal_bay: Dictionary with bay geometry (x, y, yaw, width, depth)
        corridor_width: Width of the approach corridor (meters)
        
    Returns:
        Tuple of (left_boundary, right_boundary) as lists of (x, y) points
    """
    if len(waypoints) < 2:
        return [], []
    
    # # --- STEP 1: Create DENSE smoothed path ---
    # # This eliminates loops and sharp corners by working with an already-smooth curve
    # waypoints_array = np.array(waypoints)
    # x_sparse = waypoints_array[:, 0]
    # y_sparse = waypoints_array[:, 1]
    
    # # Generate dense smooth path (100 points)
    # try:
    #     tck, u = splprep([x_sparse, y_sparse], s=0.5, k=min(3, len(waypoints)-1))
    #     u_dense = np.linspace(0, 1, 100)
    #     x_dense, y_dense = splev(u_dense, tck)
        
    #     # Compute tangents from dense path
    #     dx = np.gradient(x_dense)
    #     dy = np.gradient(y_dense)
    #     theta_dense = np.arctan2(dy, dx)
        
    #     dense_path = list(zip(x_dense, y_dense, theta_dense))
    # except:
    #     # Fallback to sparse waypoints if smoothing fails
    #     dense_path = waypoints
        
    # --- STEP 1: Use waypoints with light densification ---
    # 🔧 v46 FIX: Waypoints are already smoothed, just densify slightly for corridor
    waypoints_array = np.array(waypoints)
    x_sparse = waypoints_array[:, 0]
    y_sparse = waypoints_array[:, 1]
    
    # 🔧 FIX: Adaptive densification based on waypoint count
    # Use 2x waypoints (not fixed 100) to preserve waypoint structure
    target_points = min(len(waypoints) * 2, 60)  # Cap at 60 for performance
    
    try:
        # 🔧 FIX: Reduce smoothing parameter (s=0.1 instead of 0.5)
        # This makes corridor follow waypoints more closely
        # Lower s = tighter fit to waypoints = corridor matches actual path better
        tck, u = splprep([x_sparse, y_sparse], s=0.1, k=min(3, len(waypoints)-1))
        u_dense = np.linspace(0, 1, target_points)
        x_dense, y_dense = splev(u_dense, tck)
        
        # Compute tangents from dense path
        dx = np.gradient(x_dense)
        dy = np.gradient(y_dense)
        theta_dense = np.arctan2(dy, dx)
        
        dense_path = list(zip(x_dense, y_dense, theta_dense))
    except:
        # Fallback to sparse waypoints if smoothing fails
        dense_path = waypoints

    # --- STEP 2: Identify Bay Entrance ---
    bx, by = goal_bay['x'], goal_bay['y']
    b_yaw = goal_bay['yaw']  # Direction INTO the bay (car heading when parked)
    b_depth = goal_bay.get('depth', 5.5)
    b_width = goal_bay.get('width', 2.7)
    
    # # CRITICAL: Bay's physical rectangle is PERPENDICULAR to b_yaw
    # bay_orientation = b_yaw + np.pi / 2.0
    
    # # Entrance is at the "front" of the bay
    # entrance_x = bx - (b_depth / 2.0) * np.cos(bay_orientation)
    # entrance_y = by - (b_depth / 2.0) * np.sin(bay_orientation)
    bay_orientation = b_yaw  # REMOVED: + np.pi / 2.0
    entrance_x = bx + (-b_depth / 2.0) * np.cos(bay_orientation)
    entrance_y = by + (-b_depth / 2.0) * np.sin(bay_orientation)
    
    # Find split index in dense path
    split_idx = len(dense_path) - 1
    min_dist = float('inf')
    
    search_start = max(0, len(dense_path) // 2)
    
    for i in range(search_start, len(dense_path)):
        px, py, _ = dense_path[i]
        dist = np.hypot(px - entrance_x, py - entrance_y)
        if dist < min_dist:
            min_dist = dist
            split_idx = i
            
    approach_path = dense_path[:split_idx+1]
    
    # --- STEP 3: Generate Approach Corridor from DENSE PATH ---
    # No additional smoothing needed - the path is already smooth!
    offset = corridor_width / 2.0
    left_boundary = []
    right_boundary = []
    
    for x, y, theta in approach_path:
        perp_x = -np.sin(theta)
        perp_y = np.cos(theta)
        
        left_boundary.append((x + offset * perp_x, y + offset * perp_y))
        right_boundary.append((x - offset * perp_x, y - offset * perp_y))
    
    # --- STEP 4: Generate Bay Boundaries (Straight Lines) ---
    # Front-Left (Entrance side)
    fl_x = entrance_x + (b_width/2.0) * np.cos(bay_orientation + np.pi/2)
    fl_y = entrance_y + (b_width/2.0) * np.sin(bay_orientation + np.pi/2)
    
    # Front-Right (Entrance side)
    fr_x = entrance_x + (b_width/2.0) * np.cos(bay_orientation - np.pi/2)
    fr_y = entrance_y + (b_width/2.0) * np.sin(bay_orientation - np.pi/2)
    
    # Back-Left (Deep end of bay)
    back_x = bx + (b_depth/2.0) * np.cos(bay_orientation)
    back_y = by + (b_depth/2.0) * np.sin(bay_orientation)
    
    bl_x = back_x + (b_width/2.0) * np.cos(bay_orientation + np.pi/2)
    bl_y = back_y + (b_width/2.0) * np.sin(bay_orientation + np.pi/2)
    
    # Back-Right (Deep end of bay)
    br_x = back_x + (b_width/2.0) * np.cos(bay_orientation - np.pi/2)
    br_y = back_y + (b_width/2.0) * np.sin(bay_orientation - np.pi/2)
    
    # Append bay lines to boundaries
    left_boundary.append((fl_x, fl_y))
    left_boundary.append((bl_x, bl_y))
    
    right_boundary.append((fr_x, fr_y))
    right_boundary.append((br_x, br_y))
    
    return left_boundary, right_boundary


# def calculate_8_point_bay_reference(goal_bay: Dict) -> List[Tuple[float, float]]:
#     """
#     Calculate 8-point bay reference system (4 corners + 4 edge midpoints).
    
#     This provides spatial awareness of bay boundaries for the agent.
    
#     Args:
#         goal_bay: Dictionary with bay geometry (x, y, yaw, width, depth)
        
#     Returns:
#         List of 8 (x, y) points in world frame
#     """
#     bx, by = goal_bay['x'], goal_bay['y']
#     b_yaw = goal_bay['yaw']
#     b_depth = goal_bay.get('depth', 5.0)
#     b_width = goal_bay.get('width', 2.8)
    
#     # Bay orientation is perpendicular to goal_yaw
#     bay_orientation = b_yaw + np.pi / 2.0
    
#     half_l = b_depth / 2.0
#     half_w = b_width / 2.0
    
#     # 8 points in bay-aligned frame
#     bay_points_local = [
#         (-half_l, half_w),   # Corner 1
#         (-half_l, -half_w),  # Corner 2
#         (half_l, -half_w),   # Corner 3
#         (half_l, half_w),    # Corner 4
#         (-half_l, 0),        # Edge midpoint 1
#         (0, -half_w),        # Edge midpoint 2
#         (half_l, 0),         # Edge midpoint 3
#         (0, half_w),         # Edge midpoint 4
#     ]
    
#     # Transform to world frame
#     bay_points_world = []
#     cos_b = np.cos(bay_orientation)
#     sin_b = np.sin(bay_orientation)
    
#     for lx, ly in bay_points_local:
#         wx = bx + lx * cos_b - ly * sin_b
#         wy = by + lx * sin_b + ly * cos_b
#         bay_points_world.append((wx, wy))
        
#     return bay_points_world



def calculate_8_point_bay_reference(goal_bay: Dict) -> List[Tuple[float, float]]:
    """
    Calculate 8-point bay reference system (4 corners + 4 edge midpoints).
    
    v42 FIX: Use bay yaw directly (no rotation offset needed)
    Bay local frame: +X = depth (into bay), +Y = width (lateral)
    """
    bx, by = goal_bay['x'], goal_bay['y']
    b_yaw = goal_bay['yaw']
    b_depth = goal_bay.get('depth', 5.5)  # Match bay_length
    b_width = goal_bay.get('width', 2.7)  # Match bay_width
    
    # v42 FIX: Use bay yaw directly (it already points INTO the bay)
    bay_orientation = b_yaw  # ✅ FIXED! No extra rotation
    
    half_l = b_depth / 2.0  # 2.75m
    half_w = b_width / 2.0  # 1.35m
    
    # v42 FIX: Bay local frame - SWAPPED to match new convention
    # Local +X = depth (into bay), +Y = width (lateral)
    bay_points_local = [
        (-half_l, -half_w),  # Entrance left (back toward road)
        (-half_l, half_w),   # Entrance right
        (half_l, half_w),    # Deep left (away from road)
        (half_l, -half_w),   # Deep right
        (-half_l, 0),        # Entrance center
        (0, half_w),         # Right edge midpoint
        (half_l, 0),         # Deep center
        (0, -half_w),        # Left edge midpoint
    ]
    
    # Transform to world frame
    bay_points_world = []
    cos_b = np.cos(bay_orientation)
    sin_b = np.sin(bay_orientation)
    
    for lx, ly in bay_points_local:
        wx = bx + lx * cos_b - ly * sin_b
        wy = by + lx * sin_b + ly * cos_b
        bay_points_world.append((wx, wy))
        
    return bay_points_world