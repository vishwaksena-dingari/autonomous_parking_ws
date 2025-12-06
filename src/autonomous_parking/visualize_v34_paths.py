#!/usr/bin/env python3
"""
Visualize v34 options with PATHS for all parking spots.

Shows example curved paths with both Option 1 (offset waypoints) 
and Option 2 (corridor) overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

def draw_parking_lot(ax, lot_name):
    """Draw simplified parking lot layout."""
    road_color = (0.3, 0.3, 0.3)
    
    if lot_name == "lot_a":
        # Road
        road = Rectangle((-25, -3), 50, 6, facecolor=road_color, edgecolor='none', zorder=0)
        ax.add_patch(road)
        
        # A-row bays (top)
        for i in range(6):
            x = -12.5 + i * 5
            y = 7.5
            draw_bay(ax, x, y, np.pi/2, f'A{i+1}')
        
        # B-row bays (bottom)
        for i in range(6):
            x = -12.5 + i * 5
            y = -7.5
            draw_bay(ax, x, y, -np.pi/2, f'B{i+1}')
    
    else:  # lot_b
        # Horizontal road
        road_h = Rectangle((-25, 7), 50, 6, facecolor=road_color, edgecolor='none', zorder=0)
        ax.add_patch(road_h)
        # Vertical road
        road_v = Rectangle((-3, -25), 6, 36, facecolor=road_color, edgecolor='none', zorder=0)
        ax.add_patch(road_v)
        
        # H-row bays (horizontal)
        for i in range(5):
            x = -10 + i * 5
            y = 15
            draw_bay(ax, x, y, np.pi/2, f'H{i+1}')
        
        # V-row bays (vertical)
        for i in range(5):
            x = 5
            y = -10 + i * 5
            draw_bay(ax, x, y, 0, f'V{i+1}')

def draw_bay(ax, x, y, yaw, label):
    """Draw a single parking bay."""
    width, length = 2.5, 5.0
    
    corners_local = [
        (-length/2, -width/2),
        (-length/2, width/2),
        (length/2, width/2),
        (length/2, -width/2),
    ]
    
    corners_world = []
    for lx, ly in corners_local:
        wx = x + lx * np.cos(yaw) - ly * np.sin(yaw)
        wy = y + lx * np.sin(yaw) + ly * np.cos(yaw)
        corners_world.append((wx, wy))
    
    bay_poly = Polygon(corners_world, fill=False, edgecolor='green', 
                      linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(bay_poly)
    ax.text(x, y, label, ha='center', va='center', 
           fontsize=8, color='green', fontweight='bold', zorder=2)

def generate_synthetic_path(start, goal, goal_yaw, num_points=50):
    """Generate a smooth curved path using simple interpolation."""
    sx, sy = start
    gx, gy = goal
    
    # Create a curved path (simple arc)
    t = np.linspace(0, 1, num_points)
    
    # Add a curve in the middle
    mid_x = (sx + gx) / 2
    mid_y = (sy + gy) / 2
    
    # Control point for curve (offset perpendicular to straight line)
    dx = gx - sx
    dy = gy - sy
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        perp_x = -dy / length
        perp_y = dx / length
    else:
        perp_x, perp_y = 0, 1
    
    # Curve amount based on distance
    curve_amount = min(length * 0.2, 3.0)
    ctrl_x = mid_x + curve_amount * perp_x
    ctrl_y = mid_y + curve_amount * perp_y
    
    # Quadratic Bezier curve
    path_x = (1-t)**2 * sx + 2*(1-t)*t * ctrl_x + t**2 * gx
    path_y = (1-t)**2 * sy + 2*(1-t)*t * ctrl_y + t**2 * gy
    
    # Calculate orientations
    waypoints = []
    for i in range(len(path_x)):
        x, y = path_x[i], path_y[i]
        
        if i < len(path_x) - 1:
            dx = path_x[i+1] - x
            dy = path_y[i+1] - y
            theta = np.arctan2(dy, dx)
        else:
            theta = goal_yaw
        
        waypoints.append((x, y, theta))
    
    return waypoints

def visualize_option1_on_path(ax, waypoints, show_every_n=5):
    """Overlay Option 1 (offset waypoints) on the path."""
    car_length, car_width = 4.5, 2.0
    half_l, half_w = car_length / 2.0, car_width / 2.0
    
    for i, (x, y, theta) in enumerate(waypoints):
        if i % show_every_n != 0:
            continue
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Calculate 4 offset targets
        FL = (x + half_l * cos_t - half_w * sin_t, y + half_l * sin_t + half_w * cos_t)
        FR = (x + half_l * cos_t + half_w * sin_t, y + half_l * sin_t - half_w * cos_t)
        RL = (x - half_l * cos_t - half_w * sin_t, y - half_l * sin_t + half_w * cos_t)
        RR = (x - half_l * cos_t + half_w * sin_t, y - half_l * sin_t - half_w * cos_t)
        
        # Draw small markers for offset targets
        for corner, color in zip([FL, FR, RL, RR], ['red', 'orange', 'purple', 'brown']):
            ax.plot(corner[0], corner[1], 'o', color=color, markersize=3, alpha=0.6, zorder=5)

def visualize_option2_on_path(ax, waypoints, corridor_width=3.0):
    """Overlay Option 2 (corridor) on the path."""
    if len(waypoints) < 2:
        return
    
    # Create corridor boundaries
    left_boundary = []
    right_boundary = []
    
    for i in range(len(waypoints) - 1):
        x1, y1, _ = waypoints[i]
        x2, y2, _ = waypoints[i+1]
        
        # Perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            continue
        
        half_corridor = corridor_width / 2.0
        
        # Left and right points
        left_boundary.append((x1 + half_corridor * perp_x, y1 + half_corridor * perp_y))
        right_boundary.append((x1 - half_corridor * perp_x, y1 - half_corridor * perp_y))
    
    # Draw boundaries
    if left_boundary:
        left_x, left_y = zip(*left_boundary)
        ax.plot(left_x, left_y, 'r--', linewidth=1, alpha=0.5, label='Corridor Boundary', zorder=4)
    
    if right_boundary:
        right_x, right_y = zip(*right_boundary)
        ax.plot(right_x, right_y, 'r--', linewidth=1, alpha=0.5, zorder=4)

def main():
    """Create comprehensive path visualizations."""
    
    # Test cases: (lot, bay_id, start, goal, goal_yaw)
    test_cases = [
        ('lot_a', 'A3', (0, 0), (0, 7.5), np.pi/2),
        ('lot_a', 'B3', (5, 0), (0, -7.5), -np.pi/2),
        ('lot_b', 'H3', (0, 10), (0, 15), np.pi/2),
        ('lot_b', 'V3', (0, 0), (5, 0), 0),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, (lot_name, bay_id, start, goal, goal_yaw) in enumerate(test_cases):
        ax = axes[idx]
        
        # Draw lot
        draw_parking_lot(ax, lot_name)
        
        # Generate path
        waypoints = generate_synthetic_path(start, goal, goal_yaw, num_points=50)
        
        # Draw centerline path
        path_x = [w[0] for w in waypoints]
        path_y = [w[1] for w in waypoints]
        ax.plot(path_x, path_y, 'b-', linewidth=3, label='Path Centerline', zorder=3)
        
        # Overlay Option 1
        visualize_option1_on_path(ax, waypoints, show_every_n=5)
        
        # Overlay Option 2
        visualize_option2_on_path(ax, waypoints, corridor_width=3.0)
        
        # Highlight start and goal
        ax.plot(waypoints[0][0], waypoints[0][1], 'go', markersize=15, 
               label='Start', zorder=6, markeredgecolor='black', markeredgewidth=2)
        ax.plot(waypoints[-1][0], waypoints[-1][1], 'r*', markersize=20, 
               label='Goal', zorder=6)
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-15, 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{lot_name.upper()} → {bay_id}\n'
                    f'Blue: Path | Red dashed: Corridor | Colored dots: Corner Targets',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlabel('X [m]', fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/v34_paths_all_spots.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved path visualization to: results/v34_paths_all_spots.png")

if __name__ == "__main__":
    main()
