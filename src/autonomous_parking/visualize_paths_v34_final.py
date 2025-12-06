#!/usr/bin/env python3
"""
Enhanced path visualization showing v34 multi-point options.

For each parking bay, generates a visualization showing:
- The B-spline path (yellow dots)
- Option 1: 4 corner offset targets (colored dots)
- Option 2: Corridor boundaries (red dashed lines)
- Car position and parking bay

Based on visualize_paths.py but enhanced with v34 overlays.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches

# Import environment
from autonomous_parking.env2d.waypoint_env import WaypointEnv

def calculate_offset_waypoints(waypoint_x, waypoint_y, waypoint_theta):
    """
    Calculate 4 offset target positions for car corners (Option 1).
    
    Offsets are perpendicular to waypoint direction.
    """
    car_length, car_width = 4.5, 2.0
    half_l, half_w = car_length / 2.0, car_width / 2.0
    
    cos_t, sin_t = np.cos(waypoint_theta), np.sin(waypoint_theta)
    
    # Front-Left: +half_length along theta, +half_width perpendicular
    FL = (waypoint_x + half_l * cos_t - half_w * sin_t, 
          waypoint_y + half_l * sin_t + half_w * cos_t)
    
    # Front-Right: +half_length along theta, -half_width perpendicular
    FR = (waypoint_x + half_l * cos_t + half_w * sin_t, 
          waypoint_y + half_l * sin_t - half_w * cos_t)
    
    # Rear-Left: -half_length along theta, +half_width perpendicular
    RL = (waypoint_x - half_l * cos_t - half_w * sin_t, 
          waypoint_y - half_l * sin_t + half_w * cos_t)
    
    # Rear-Right: -half_length along theta, -half_width perpendicular
    RR = (waypoint_x - half_l * cos_t + half_w * sin_t, 
          waypoint_y - half_l * sin_t - half_w * cos_t)
    
    return [FL, FR, RL, RR]

def calculate_corridor_boundaries(waypoints, goal_bay, corridor_width=3.0):
    """
    Hybrid corridor boundaries using DENSE SMOOTHED PATH:
    1. From start to bay entrance: Calculate from smoothed path (no sharp corners!)
    2. Inside bay: Actual straight bay lines (from geometry)
    
    Args:
        waypoints: List of (x, y, theta) - sparse waypoints
        goal_bay: Dictionary with bay geometry (x, y, yaw, width, depth)
        corridor_width: Width of the approach corridor
        
    Returns:
        left_boundary, right_boundary (lists of points)
    """
    from scipy.interpolate import splprep, splev
    
    if len(waypoints) < 2:
        return [], []
    
    # --- CRITICAL FIX: Create a DENSE smoothed path first ---
    # This eliminates loops and sharp corners by working with an already-smooth curve
    waypoints_array = np.array(waypoints)
    x_sparse = waypoints_array[:, 0]
    y_sparse = waypoints_array[:, 1]
    
    # Generate dense smooth path (100 points)
    try:
        tck, u = splprep([x_sparse, y_sparse], s=0.5, k=min(3, len(waypoints)-1))
        u_dense = np.linspace(0, 1, 100)
        x_dense, y_dense = splev(u_dense, tck)
        
        # Compute tangents from dense path
        dx = np.gradient(x_dense)
        dy = np.gradient(y_dense)
        theta_dense = np.arctan2(dy, dx)
        
        dense_path = list(zip(x_dense, y_dense, theta_dense))
    except:
        # Fallback to sparse waypoints if smoothing fails
        dense_path = waypoints
        
    # --- 1. Identify Bay Entrance ---
    bx, by = goal_bay['x'], goal_bay['y']
    b_yaw = goal_bay['yaw']  # Direction INTO the bay (car heading when parked)
    b_depth = goal_bay.get('depth', 5.0)
    b_width = goal_bay.get('width', 2.8)
    
    # CRITICAL: Bay's physical rectangle is PERPENDICULAR to b_yaw
    bay_orientation = b_yaw + np.pi / 2.0
    
    # Entrance is at the "front" of the bay
    entrance_x = bx - (b_depth / 2.0) * np.cos(bay_orientation)
    entrance_y = by - (b_depth / 2.0) * np.sin(bay_orientation)
    
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
    
    # --- 2. Generate Approach Corridor from DENSE PATH ---
    # No additional smoothing needed - the path is already smooth!
    offset = corridor_width / 2.0
    left_boundary = []
    right_boundary = []
    
    for x, y, theta in approach_path:
        perp_x = -np.sin(theta)
        perp_y = np.cos(theta)
        
        left_boundary.append((x + offset * perp_x, y + offset * perp_y))
        right_boundary.append((x - offset * perp_x, y - offset * perp_y))
    
    # --- 3. Generate Bay Boundaries (Straight Lines) ---
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

def compute_path_tangents(waypoints):
    """
    Compute orientation at each waypoint from path tangents.
    This fixes the issue where waypoint theta is set to goal_yaw prematurely.
    
    Returns: List of (x, y, tangent_theta)
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

def visualize_path_with_v34_options(env, lot_name, bay_id, output_dir='path_visualizations_v34'):
    """Generate enhanced visualization with v34 corridor option."""
    
    # Reset environment to get path
    obs, info = env.reset(lot_name=lot_name, goal_bay=bay_id)
    
    # Get waypoints
    waypoints = env.waypoints
    goal_x, goal_y, goal_yaw = env.goal_x, env.goal_y, env.goal_yaw
    
    # CRITICAL FIX: Recompute orientations from path tangents
    # The waypoints from env have theta=goal_yaw for final waypoints, which is wrong
    waypoints_corrected = compute_path_tangents(waypoints)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw parking lot
    env.render_lot(ax)
    
    # Extract path coordinates
    path_x = [w[0] for w in waypoints_corrected]
    path_y = [w[1] for w in waypoints_corrected]
    
    # OPTION 2: Draw corridor boundaries (FIRST, so it's in background)
    # Use dense smoothed path to eliminate sharp corners and loops
    goal_bay = env.unwrapped.goal_bay
    left_boundary, right_boundary = calculate_corridor_boundaries(waypoints_corrected, goal_bay, corridor_width=3.0)
    
    if left_boundary:
        left_x, left_y = zip(*left_boundary)
        ax.plot(left_x, left_y, 'r--', linewidth=2, alpha=0.6, label='Path Corridor', zorder=2)
    
    if right_boundary:
        right_x, right_y = zip(*right_boundary)
        ax.plot(right_x, right_y, 'r--', linewidth=2, alpha=0.6, zorder=2)
    
    # Draw main path (yellow dots)
    ax.plot(path_x, path_y, 'yo', markersize=8, label='Path Waypoints', zorder=3)
    
    # Draw car at start
    car_x, car_y, car_yaw = waypoints_corrected[0]
    draw_car(ax, car_x, car_y, car_yaw, color='blue', alpha=0.5, label='Start Position')
    
    # Draw goal marker
    ax.plot(goal_x, goal_y, 'r*', markersize=20, label='Goal', zorder=5)
    
    # Title and labels
    ax.set_title(f'{lot_name.upper()} - Bay {bay_id}\n'
                f'Path Length: {len(waypoints)} waypoints\n'
                f'Yellow: Waypoints | Red: 3m Corridor | Cyan: Bay Reference Points',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('X [m]', fontsize=10)
    ax.set_ylabel('Y [m]', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{lot_name}_{bay_id}_v34.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'  ✓ Saved to {output_path}')

def draw_car(ax, x, y, yaw, color='blue', alpha=0.5, label=None):
    """Draw a car rectangle."""
    car_length = 4.5
    car_width = 2.0
    
    # Car corners
    corners_local = [
        (0, -car_width/2),
        (car_length, -car_width/2),
        (car_length, car_width/2),
        (0, car_width/2),
    ]
    
    corners_world = []
    for lx, ly in corners_local:
        wx = x + lx * np.cos(yaw) - ly * np.sin(yaw)
        wy = y + lx * np.sin(yaw) + ly * np.cos(yaw)
        corners_world.append((wx, wy))
    
    car_rect = Polygon(corners_world, fill=True, facecolor=color, 
                      alpha=alpha, edgecolor='black', linewidth=2, label=label, zorder=6)
    ax.add_patch(car_rect)

def main():
    """Generate v34 visualizations for all bays."""
    
    output_dir = 'path_visualizations_v34'
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving v34 visualizations to {output_dir}/\n')
    
    # Process both lots
    lots = ['lot_a', 'lot_b']
    
    for lot_name in lots:
        print(f'Processing {lot_name}...')
        
        # Load config to get bays
        from autonomous_parking.config_loader import load_parking_config
        config = load_parking_config(lot_name)
        bays = config['bays']
        
        for bay in bays:
            bay_id = bay['id']
            print(f'  Generating v34 path for Bay {bay_id}...', end=' ', flush=True)
            
            # Create fresh env per bay
            env = WaypointEnv(lot_name=lot_name, render_mode='rgb_array')
            
            try:
                # Reset with specific bay
                obs, info = env.reset(bay_id=bay_id)
                
                # Trigger render
                env.unwrapped.render()
                
                # Access matplotlib axis/figure
                if hasattr(env.unwrapped, 'ax') and env.unwrapped.ax is not None:
                    ax = env.unwrapped.ax
                    fig = env.unwrapped.fig
                    
                    if hasattr(env, 'waypoints') and env.waypoints is not None and len(env.waypoints) > 0:
                        waypoints = env.waypoints
                        
                        # OPTION 2: Draw corridor boundaries (FIRST, background)
                        # Use dense smoothed path to eliminate sharp corners and loops
                        goal_bay = env.unwrapped.goal_bay
                        waypoints_corrected = compute_path_tangents(waypoints)
                        left_boundary, right_boundary = calculate_corridor_boundaries(waypoints_corrected, goal_bay, corridor_width=3.0)
                        
                        if left_boundary:
                            left_x, left_y = zip(*left_boundary)
                            ax.plot(left_x, left_y, 'r--', linewidth=2, alpha=0.6, label='Corridor (Opt 2)', zorder=2)
                        
                        if right_boundary:
                            right_x, right_y = zip(*right_boundary)
                            ax.plot(right_x, right_y, 'r--', linewidth=2, alpha=0.6, zorder=2)
                        
                        # Draw main path (yellow dots like original)
                        wps = np.array(waypoints)
                        ax.scatter(wps[:, 0], wps[:, 1], c='yellow', s=40, zorder=20, 
                                  edgecolors='black', linewidth=1, label='Path Waypoints')
                        
                        # Start/goal markers
                        ax.scatter(wps[0, 0], wps[0, 1], c='lime', s=100, zorder=21, 
                                  label='Start', edgecolors='black')
                        ax.scatter(wps[-1, 0], wps[-1, 1], c='red', s=100, zorder=21, 
                                  label='Goal', edgecolors='black')
                        
                        # v34: Draw 8-point bay reference system (Cyan)
                        # CRITICAL: goal_yaw is the direction INTO the bay (car heading)
                        # The bay's physical rectangle is PERPENDICULAR to this!
                        # So we rotate by 90° to get the bay's actual orientation
                        
                        b_depth = goal_bay.get('depth', 5.0)
                        b_width = goal_bay.get('width', 2.8)
                        bx, by = goal_bay['x'], goal_bay['y']
                        b_yaw = goal_bay['yaw']
                        
                        # Bay orientation is perpendicular to goal_yaw
                        bay_orientation = b_yaw + np.pi / 2.0
                        
                        half_l = b_depth / 2.0
                        half_w = b_width / 2.0
                        
                        # 8 points in bay-aligned frame
                        # (depth along bay_orientation, width perpendicular)
                        bay_points_local = [
                            (-half_l, half_w),   # Corner 1
                            (-half_l, -half_w),  # Corner 2
                            (half_l, -half_w),   # Corner 3
                            (half_l, half_w),    # Corner 4
                            (-half_l, 0),        # Edge midpoint 1
                            (0, -half_w),        # Edge midpoint 2
                            (half_l, 0),         # Edge midpoint 3
                            (0, half_w),         # Edge midpoint 4
                        ]
                        
                        # Transform to world frame
                        cyan_x, cyan_y = [], []
                        cos_b = np.cos(bay_orientation)
                        sin_b = np.sin(bay_orientation)
                        
                        for lx, ly in bay_points_local:
                            wx = bx + lx * cos_b - ly * sin_b
                            wy = by + lx * sin_b + ly * cos_b
                            cyan_x.append(wx)
                            cyan_y.append(wy)
                            
                        ax.scatter(cyan_x, cyan_y, c='cyan', s=30, zorder=22,
                                  edgecolors='black', linewidth=0.5, label='Bay Ref Points')
                        
                        # Title
                        ax.set_title(f'{lot_name} - Bay {bay_id}\\n'
                                    f'Path: {len(waypoints)} waypoints | '
                                    f'Yellow: Waypoints | Red: 3m Corridor | Cyan: Bay Points',
                                    fontsize=10)
                        
                        # Save
                        filename = os.path.join(output_dir, f'{lot_name}_{bay_id}_v34.png')
                        fig.savefig(filename, dpi=100, bbox_inches='tight')
                        print(f'✓ Saved to {filename}')
                    else:
                        print('❌ No waypoints')
                else:
                    print('❌ No axis/figure')
                    
            except Exception as e:
                print(f'❌ ERROR: {e}')
                import traceback
                traceback.print_exc()
            
            finally:
                if hasattr(env.unwrapped, 'fig') and env.unwrapped.fig is not None:
                    plt.close(env.unwrapped.fig)
                env.close()
    
    print(f'\\n✅ All v34 visualizations saved to {output_dir}/')

if __name__ == '__main__':
    main()
