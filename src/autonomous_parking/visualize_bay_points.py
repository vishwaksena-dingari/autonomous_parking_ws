#!/usr/bin/env python3
"""
Visualize the 8-point bay reference system that the agent sees.
Shows what spatial information is in the observation vector.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import math

def visualize_8_point_system():
    """Create a diagram showing the 8 reference points."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Bay parameters
    bay_x, bay_y = 0.0, 7.5  # A3 position
    bay_yaw = np.pi / 2  # Vertical orientation
    bay_width = 2.5
    bay_length = 5.0
    
    # Draw bay
    cos_yaw = np.cos(bay_yaw)
    sin_yaw = np.sin(bay_yaw)
    
    # Bay corners in world frame
    half_w = bay_width / 2.0
    half_l = bay_length / 2.0
    
    corners_local = [
        (-half_l, -half_w),
        (-half_l, half_w),
        (half_l, half_w),
        (half_l, -half_w),
    ]
    
    corners_world = []
    for lx, ly in corners_local:
        wx = bay_x + lx * cos_yaw - ly * sin_yaw
        wy = bay_y + lx * sin_yaw + ly * cos_yaw
        corners_world.append((wx, wy))
    
    # Draw bay rectangle
    bay_rect = plt.Polygon(corners_world, fill=False, edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(bay_rect)
    
    # 8 reference points
    bay_points_local = [
        (-half_l, half_w, "TL (Top-Left Corner)"),
        (-half_l, -half_w, "TR (Top-Right Corner)"),
        (half_l, -half_w, "BR (Bottom-Right Corner)"),
        (half_l, half_w, "BL (Bottom-Left Corner)"),
        (-half_l, 0, "T (Top Edge Midpoint)"),
        (0, -half_w, "R (Right Edge Midpoint)"),
        (half_l, 0, "B (Bottom Edge Midpoint)"),
        (0, half_w, "L (Left Edge Midpoint)"),
    ]
    
    # Transform and plot
    colors = ['red', 'red', 'red', 'red', 'blue', 'blue', 'blue', 'blue']
    for i, (lx, ly, label) in enumerate(bay_points_local):
        wx = bay_x + lx * cos_yaw - ly * sin_yaw
        wy = bay_y + lx * sin_yaw + ly * cos_yaw
        
        ax.plot(wx, wy, 'o', color=colors[i], markersize=12, zorder=10)
        ax.text(wx + 0.3, wy + 0.3, label, fontsize=9, color=colors[i])
    
    # Draw bay center
    ax.plot(bay_x, bay_y, 'x', color='green', markersize=15, markeredgewidth=3, label='Bay Center')
    
    # Draw car at different position
    car_x, car_y, car_yaw = 3.0, 2.0, 0.3
    car_length, car_width = 4.5, 2.0
    
    car_corners_local = [
        (0, -car_width/2),
        (car_length, -car_width/2),
        (car_length, car_width/2),
        (0, car_width/2),
    ]
    
    car_corners_world = []
    for lx, ly in car_corners_local:
        wx = car_x + lx * np.cos(car_yaw) - ly * np.sin(car_yaw)
        wy = car_y + lx * np.sin(car_yaw) + ly * np.cos(car_yaw)
        car_corners_world.append((wx, wy))
    
    car_rect = plt.Polygon(car_corners_world, fill=True, facecolor='blue', alpha=0.5, edgecolor='darkblue', linewidth=2)
    ax.add_patch(car_rect)
    
    # Draw lines from car center to each bay point
    car_center_x = car_x + (car_length / 2) * np.cos(car_yaw)
    car_center_y = car_y + (car_length / 2) * np.sin(car_yaw)
    
    for i, (lx, ly, label) in enumerate(bay_points_local[:4]):  # Just corners for clarity
        wx = bay_x + lx * cos_yaw - ly * sin_yaw
        wy = bay_y + lx * sin_yaw + ly * cos_yaw
        ax.plot([car_center_x, wx], [car_center_y, wy], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlim(-3, 8)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('v32: 8-Point Bay Reference System\n(What the Agent "Sees" in Observation Vector)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Add text box explaining
    textstr = '''Agent receives 16 values in observation:
• 4 Corners (red): TL, TR, BR, BL
• 4 Midpoints (blue): T, R, B, L
• Each point: (x, y) in car frame
• Total: 8 points × 2 coords = 16 dims

This gives precise spatial awareness
of bay boundaries for alignment!'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('results/bay_8_point_reference.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to: results/bay_8_point_reference.png")
    plt.show()

if __name__ == "__main__":
    visualize_8_point_system()
