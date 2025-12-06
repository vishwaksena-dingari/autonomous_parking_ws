#!/usr/bin/env python3
"""
Smart Corridor Boundary Generation (v34 Enhanced)

Uses side midpoint paths with adaptive B-spline smoothing to create
smooth, physically realistic corridor boundaries that handle tight turns
without creating loops or overshooting.

Key innovations:
1. Calculate left/right side midpoints from waypoint orientations
2. Use lower-degree B-spline (k=2) to prevent overshooting
3. Use tighter smoothing (s=0.2-0.3) to avoid loops
4. Handle tight turns gracefully
"""

import numpy as np
from typing import List, Tuple
from scipy.interpolate import splprep, splev


def calculate_side_midpoints(waypoints: List[Tuple[float, float, float]], 
                             offset_distance: float) -> Tuple[List, List]:
    """
    Calculate left and right side midpoints for each waypoint.
    
    Args:
        waypoints: List of (x, y, theta) tuples
        offset_distance: Distance from centerline to side (e.g., car_width/2 or corridor_width/2)
    
    Returns:
        (left_points, right_points) as lists of (x, y) tuples
    """
    left_points = []
    right_points = []
    
    for x, y, theta in waypoints:
        # Calculate perpendicular direction (90° from heading)
        perp_x = -np.sin(theta)
        perp_y = np.cos(theta)
        
        # Left side midpoint (perpendicular left)
        left_x = x + offset_distance * perp_x
        left_y = y + offset_distance * perp_y
        left_points.append((left_x, left_y))
        
        # Right side midpoint (perpendicular right)
        right_x = x - offset_distance * perp_x
        right_y = y - offset_distance * perp_y
        right_points.append((right_x, right_y))
    
    return left_points, right_points


def smooth_path_adaptive(points: List[Tuple[float, float]], 
                         k: int = 2, 
                         s: float = 0.2, 
                         num_points: int = 50) -> List[Tuple[float, float]]:
    """
    Apply adaptive B-spline smoothing to a path.
    
    Uses lower-degree spline (k=2) and tighter smoothing (s=0.2) to prevent
    loops and overshooting on tight curves.
    
    Args:
        points: List of (x, y) points
        k: Spline degree (2=quadratic, 3=cubic). Lower prevents overshooting.
        s: Smoothing factor (0=interpolate, higher=smoother). Lower prevents loops.
        num_points: Number of points in smoothed output
    
    Returns:
        Smoothed path as list of (x, y) tuples
    """
    if len(points) < k + 1:
        # Not enough points for spline, return original
        return points
    
    # Convert to numpy arrays
    points_array = np.array(points)
    x = points_array[:, 0]
    y = points_array[:, 1]
    
    try:
        # Fit B-spline
        # k=2: quadratic (prevents overshooting)
        # s=0.2: tight fit (prevents loops)
        tck, u = splprep([x, y], s=s, k=k)
        
        # Evaluate spline at uniform intervals
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        
        # Return as list of tuples
        smoothed = list(zip(x_new, y_new))
        return smoothed
        
    except Exception as e:
        print(f"Warning: B-spline smoothing failed ({e}), returning original points")
        return points


def smart_corridor_boundaries(waypoints: List[Tuple[float, float, float]], 
                              corridor_width: float = 3.0,
                              num_smooth_points: int = 50) -> Tuple[List, List]:
    """
    Generate smart corridor boundaries using side midpoint paths.
    
    This approach:
    1. Calculates left/right side midpoints based on waypoint orientations
    2. Smooths both paths independently with adaptive B-spline
    3. Uses k=2 (quadratic) and s=0.2-0.3 to prevent loops on tight turns
    
    Args:
        waypoints: List of (x, y, theta) waypoints
        corridor_width: Total corridor width (meters)
        num_smooth_points: Number of points in smoothed boundaries
    
    Returns:
        (left_boundary, right_boundary) as lists of (x, y) tuples
    """
    # Calculate side midpoints
    offset = corridor_width / 2.0
    left_points, right_points = calculate_side_midpoints(waypoints, offset)
    
    # Apply adaptive smoothing to both sides
    # Inner curve (typically left on right turns) uses tighter smoothing
    left_smooth = smooth_path_adaptive(
        left_points, 
        k=2,           # Quadratic (prevents overshooting)
        s=0.2,         # Tight fit (prevents loops)
        num_points=num_smooth_points
    )
    
    # Outer curve can use slightly more smoothing
    right_smooth = smooth_path_adaptive(
        right_points, 
        k=2,           # Quadratic for consistency
        s=0.3,         # Slightly smoother
        num_points=num_smooth_points
    )
    
    return left_smooth, right_smooth


def visualize_corridor_comparison(waypoints: List[Tuple[float, float, float]], 
                                  corridor_width: float = 3.0):
    """
    Visualize comparison between old (perpendicular offset) and new (side midpoint) methods.
    """
    import matplotlib.pyplot as plt
    
    # Old method: perpendicular offsets (piecewise linear)
    old_left, old_right = calculate_old_corridor(waypoints, corridor_width)
    
    # New method: side midpoint paths with B-spline smoothing
    new_left, new_right = smart_corridor_boundaries(waypoints, corridor_width)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract path coordinates
    path_x = [w[0] for w in waypoints]
    path_y = [w[1] for w in waypoints]
    
    # Plot old method
    ax1.plot(path_x, path_y, 'b-', linewidth=2, label='Path Centerline')
    if old_left:
        old_left_x, old_left_y = zip(*old_left)
        ax1.plot(old_left_x, old_left_y, 'r--', linewidth=2, alpha=0.6, label='Left Boundary (Old)')
    if old_right:
        old_right_x, old_right_y = zip(*old_right)
        ax1.plot(old_right_x, old_right_y, 'r--', linewidth=2, alpha=0.6, label='Right Boundary (Old)')
    ax1.set_title('Old Method: Perpendicular Offsets\n(Sharp inner corners)', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot new method
    ax2.plot(path_x, path_y, 'b-', linewidth=2, label='Path Centerline')
    if new_left:
        new_left_x, new_left_y = zip(*new_left)
        ax2.plot(new_left_x, new_left_y, 'g-', linewidth=2, alpha=0.8, label='Left Boundary (New)')
    if new_right:
        new_right_x, new_right_y = zip(*new_right)
        ax2.plot(new_right_x, new_right_y, 'g-', linewidth=2, alpha=0.8, label='Right Boundary (New)')
    ax2.set_title('New Method: Side Midpoint Paths + B-spline\n(Smooth both sides)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def calculate_old_corridor(waypoints, corridor_width):
    """Old method for comparison: perpendicular offsets (piecewise linear)."""
    if len(waypoints) < 2:
        return [], []
    
    left_boundary = []
    right_boundary = []
    
    for i in range(len(waypoints) - 1):
        x1, y1, _ = waypoints[i]
        x2, y2, _ = waypoints[i+1]
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            continue
        
        half_corridor = corridor_width / 2.0
        
        left_boundary.append((x1 + half_corridor * perp_x, y1 + half_corridor * perp_y))
        right_boundary.append((x1 - half_corridor * perp_x, y1 - half_corridor * perp_y))
    
    return left_boundary, right_boundary


if __name__ == "__main__":
    # Test with a curved path
    print("Testing smart corridor boundary generation...")
    
    # Create a test path with a sharp turn
    test_waypoints = [
        (0, 0, 0),
        (2, 0, 0),
        (4, 0, np.pi/6),
        (6, 1, np.pi/4),
        (7, 3, np.pi/2),
        (7, 5, np.pi/2),
        (7, 7, np.pi/2),
    ]
    
    # Generate smart corridors
    left, right = smart_corridor_boundaries(test_waypoints, corridor_width=3.0)
    
    print(f"✅ Generated {len(left)} left boundary points")
    print(f"✅ Generated {len(right)} right boundary points")
    
    # Visualize comparison
    fig = visualize_corridor_comparison(test_waypoints, corridor_width=3.0)
    fig.savefig('results/corridor_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved comparison to results/corridor_comparison.png")
