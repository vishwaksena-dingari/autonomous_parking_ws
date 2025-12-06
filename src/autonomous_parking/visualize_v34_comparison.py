#!/usr/bin/env python3
"""
Visualize both v34 options across different scenarios.

Shows how Option 1 (offset waypoints) and Option 2 (corridor) work
for various parking scenarios in Lot A and Lot B.
"""

import sys
sys.path.insert(0, '/Users/vishwaksena_dingari/autonomous_parking_ws/src/autonomous_parking')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Import prototypes
from _prototypes.v34_option1_multipoint import MultiPointWaypointReward
from _prototypes.v34_option2_corridor import PathCorridorReward

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
    
    car_rect = plt.Polygon(corners_world, fill=True, facecolor=color, 
                           alpha=alpha, edgecolor='black', linewidth=2, label=label)
    ax.add_patch(car_rect)
    
    # Draw front indicator
    front_x = x + car_length * np.cos(yaw)
    front_y = y + car_length * np.sin(yaw)
    ax.plot([x, front_x], [y, front_y], 'r-', linewidth=3, alpha=0.7)

def visualize_option1_scenario(ax, scenario_name, car_state, waypoint):
    """Visualize Option 1: Offset waypoints."""
    car_x, car_y, car_yaw = car_state
    wp_x, wp_y, wp_theta = waypoint
    
    calculator = MultiPointWaypointReward()
    
    # Draw waypoint
    ax.plot(wp_x, wp_y, 'g*', markersize=20, label='Waypoint Center', zorder=10)
    
    # Draw waypoint direction
    arrow_len = 3.0
    ax.arrow(wp_x, wp_y, arrow_len * np.cos(wp_theta), arrow_len * np.sin(wp_theta),
             head_width=0.5, head_length=0.3, fc='green', ec='green', alpha=0.5)
    
    # Calculate and draw offset targets
    targets = calculator.calculate_offset_waypoints(wp_x, wp_y, wp_theta)
    target_labels = ['FL Target', 'FR Target', 'RL Target', 'RR Target']
    colors = ['red', 'orange', 'purple', 'brown']
    
    for i, (target, label, color) in enumerate(zip(targets, target_labels, colors)):
        ax.plot(target[0], target[1], 'o', color=color, markersize=10, 
                label=label if i < 2 else None, zorder=9)
    
    # Draw car
    draw_car(ax, car_x, car_y, car_yaw, color='blue', alpha=0.3, label='Car')
    
    # Calculate and draw car corners
    car_corners = calculator.calculate_car_corners(car_x, car_y, car_yaw)
    corner_labels = ['FL', 'FR', 'RL', 'RR']
    
    for i, (corner, target, label, color) in enumerate(zip(car_corners, targets, corner_labels, colors)):
        # Draw corner
        ax.plot(corner[0], corner[1], 's', color=color, markersize=8, zorder=11)
        
        # Draw line from corner to target
        ax.plot([corner[0], target[0]], [corner[1], target[1]], 
                '--', color=color, linewidth=1, alpha=0.5)
        
        # Distance text
        dist = np.sqrt((corner[0] - target[0])**2 + (corner[1] - target[1])**2)
        mid_x = (corner[0] + target[0]) / 2
        mid_y = (corner[1] + target[1]) / 2
        ax.text(mid_x, mid_y, f'{dist:.1f}m', fontsize=8, color=color)
    
    # Calculate reward
    reward, debug = calculator.calculate_multi_point_reward(
        car_x, car_y, car_yaw, wp_x, wp_y, wp_theta
    )
    
    ax.set_title(f'{scenario_name}\nOption 1: Offset Waypoints\nReward: {reward:.1f}, Avg Dist: {debug["avg_dist"]:.2f}m',
                 fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7)

def visualize_option2_scenario(ax, scenario_name, car_state, path_segment):
    """Visualize Option 2: Corridor constraint."""
    car_x, car_y, car_yaw = car_state
    seg_start, seg_end = path_segment
    
    calculator = PathCorridorReward(corridor_width=3.0)
    
    # Draw path segment
    ax.plot([seg_start[0], seg_end[0]], [seg_start[1], seg_end[1]], 
            'g-', linewidth=3, label='Path Centerline')
    
    # Draw corridor boundaries
    # Calculate perpendicular direction
    dx = seg_end[0] - seg_start[0]
    dy = seg_end[1] - seg_start[1]
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        perp_x = -dy / length
        perp_y = dx / length
    else:
        perp_x, perp_y = 0, 1
    
    corridor_half = calculator.corridor_width / 2.0
    
    # Left boundary
    left_start = (seg_start[0] + corridor_half * perp_x, seg_start[1] + corridor_half * perp_y)
    left_end = (seg_end[0] + corridor_half * perp_x, seg_end[1] + corridor_half * perp_y)
    ax.plot([left_start[0], left_end[0]], [left_start[1], left_end[1]], 
            'r--', linewidth=2, label='Corridor Boundary', alpha=0.7)
    
    # Right boundary
    right_start = (seg_start[0] - corridor_half * perp_x, seg_start[1] - corridor_half * perp_y)
    right_end = (seg_end[0] - corridor_half * perp_x, seg_end[1] - corridor_half * perp_y)
    ax.plot([right_start[0], right_end[0]], [right_start[1], right_end[1]], 
            'r--', linewidth=2, alpha=0.7)
    
    # Draw car
    draw_car(ax, car_x, car_y, car_yaw, color='blue', alpha=0.3, label='Car')
    
    # Draw check points
    check_points = calculator.calculate_car_check_points(car_x, car_y, car_yaw)
    
    for i, point in enumerate(check_points):
        perp_dist = calculator.point_to_line_distance(point, seg_start, seg_end)
        
        # Color based on violation
        if perp_dist > corridor_half:
            color = 'red'
            marker = 'x'
        else:
            color = 'green'
            marker = 'o'
        
        ax.plot(point[0], point[1], marker, color=color, markersize=6, zorder=11)
    
    # Calculate reward
    reward, debug = calculator.calculate_corridor_reward(
        car_x, car_y, car_yaw, seg_start, seg_end
    )
    
    ax.set_title(f'{scenario_name}\nOption 2: Corridor Constraint\nReward: {reward:.1f}, Violations: {debug["num_violating_points"]}',
                 fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7)

def main():
    """Create comprehensive visualization."""
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Aligned',
            'car': (0, 0, 0),
            'waypoint': (5, 0, 0),
            'path_seg': ((0, 0), (10, 0)),
        },
        {
            'name': 'Scenario 2: Misaligned (Rotated)',
            'car': (0, 0, np.pi/6),  # 30° rotation
            'waypoint': (5, 0, 0),
            'path_seg': ((0, 0), (10, 0)),
        },
        {
            'name': 'Scenario 3: Off-Center',
            'car': (0, 1.5, 0),  # Shifted laterally
            'waypoint': (5, 0, 0),
            'path_seg': ((0, 0), (10, 0)),
        },
        {
            'name': 'Scenario 4: Curved Path',
            'car': (0, 0, np.pi/4),
            'waypoint': (3, 3, np.pi/4),
            'path_seg': ((0, 0), (5, 5)),
        },
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(16, 4*len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        # Option 1 visualization
        visualize_option1_scenario(
            axes[i, 0],
            scenario['name'],
            scenario['car'],
            scenario['waypoint'],
        )
        axes[i, 0].set_xlim(-3, 12)
        axes[i, 0].set_ylim(-4, 8)
        
        # Option 2 visualization
        visualize_option2_scenario(
            axes[i, 1],
            scenario['name'],
            scenario['car'],
            scenario['path_seg'],
        )
        axes[i, 1].set_xlim(-3, 12)
        axes[i, 1].set_ylim(-4, 8)
    
    plt.tight_layout()
    plt.savefig('results/v34_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: results/v34_comparison.png")
    
    # Create summary comparison
    create_summary_comparison()

def create_summary_comparison():
    """Create a summary table comparing both options."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    comparison_text = """
    v34 MULTI-POINT PATH FOLLOWING: COMPARISON
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    OPTION 1: Offset Waypoints                               │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ ✅ PROS:                                                                     │
    │   • Explicit alignment enforcement (all 4 corners must reach targets)       │
    │   • Works with existing waypoint system                                     │
    │   • Natural orientation correction (perpendicular offsets)                  │
    │   • Clear visual feedback (can see target positions)                        │
    │                                                                              │
    │ ⚠️  CONS:                                                                     │
    │   • 4x more distance calculations per step                                  │
    │   • Requires precise waypoint theta values                                  │
    │   • May be too strict for tight maneuvers                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    OPTION 2: Corridor Constraint                            │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │ ✅ PROS:                                                                     │
    │   • More flexible (allows some deviation within corridor)                   │
    │   • Works with path segments (no theta needed)                              │
    │   • Easier to tune (just adjust corridor width)                             │
    │   • Bonus for staying centered (smooth driving)                             │
    │                                                                              │
    │ ⚠️  CONS:                                                                     │
    │   • Less explicit alignment enforcement                                     │
    │   • Requires 7+ point checks per step                                       │
    │   • May allow "snaking" within corridor                                     │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         RECOMMENDATION                                      │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                              │
    │  🎯 START WITH OPTION 1 (Offset Waypoints)                                  │
    │                                                                              │
    │  Reasons:                                                                    │
    │  1. Stronger alignment enforcement (exactly what we need)                   │
    │  2. Simpler to understand and debug                                         │
    │  3. Complements existing waypoint system                                    │
    │  4. Can add Option 2 later if needed                                        │
    │                                                                              │
    │  Implementation:                                                             │
    │  • Add to WaypointRewardCalculator                                          │
    │  • Weight: 0.5 per corner (total 2.0 for all 4)                             │
    │  • Combine with existing path deviation penalty                             │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
    
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('results/v34_summary.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved summary to: results/v34_summary.png")

if __name__ == "__main__":
    main()
