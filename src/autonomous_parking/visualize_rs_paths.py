#!/usr/bin/env python3
"""
Simplified Reeds-Shepp Path Visualization

Generates comparison images using WaypointEnv directly.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autonomous_parking.env2d.waypoint_env import WaypointEnv


def generate_comparison(lot_name, bay_id, output_dir):
    """Generate B-spline vs Reeds-Shepp comparison for one bay"""
    
    print(f"Generating {lot_name} {bay_id}...")
    
    # Create two environments (one for each path type)
    env_bspline = WaypointEnv(lot_name=lot_name, use_reeds_shepp=False, verbose=False)
    env_rs = WaypointEnv(lot_name=lot_name, use_reeds_shepp=True, 
                         reeds_shepp_turning_radius=2.6, verbose=False)
    
    # Reset both with same bay
    env_bspline.reset(bay_id=bay_id)
    env_rs.reset(bay_id=bay_id)
    
    # Get paths
    bspline_path = env_bspline.full_path
    rs_path = env_rs.full_path
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    for ax, env, path, title in [
        (ax1, env_bspline, bspline_path, "B-Spline (Current)"),
        (ax2, env_rs, rs_path, "Reeds-Shepp (Proposed)")
    ]:
        # Setup
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title}\n{lot_name.upper()} | Bay: {bay_id}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Draw roads (simple gray rectangles)
        if lot_name == "lot_a":
            # Horizontal road at y=0
            ax.add_patch(Rectangle((-25, -3.75), 50, 7.5, 
                                  facecolor='gray', alpha=0.2))
        elif lot_name == "lot_b":
            # Vertical road at x=0
            ax.add_patch(Rectangle((-3.75, -25), 7.5, 36, 
                                  facecolor='gray', alpha=0.2))
            # Horizontal road at y=10
            ax.add_patch(Rectangle((-25, 6.25), 50, 7.5, 
                                  facecolor='gray', alpha=0.2))
        
        # Draw all bays
        for bay in env.bays:
            color = 'limegreen' if bay["id"] == bay_id else 'lightgray'
            alpha = 0.6 if bay["id"] == bay_id else 0.15
            linewidth = 2.5 if bay["id"] == bay_id else 1.0
            
            bx, by, byaw = bay["x"], bay["y"], bay["yaw"]
            cos_b, sin_b = np.cos(byaw), np.sin(byaw)
            
            # Bay corners
            half_l, half_w = 5.5/2, 2.7/2
            corners_local = [
                (-half_l, -half_w), (half_l, -half_w),
                (half_l, half_w), (-half_l, half_w)
            ]
            
            corners_world = []
            for lx, ly in corners_local:
                wx = bx + cos_b * lx - sin_b * ly
                wy = by + sin_b * lx + cos_b * ly
                corners_world.append([wx, wy])
            
            poly = Polygon(corners_world, facecolor=color, 
                          edgecolor='darkgreen', alpha=alpha, linewidth=linewidth)
            ax.add_patch(poly)
            
            # Label
            if bay["id"] == bay_id:
                ax.text(bx, by, bay["id"], ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
        
        # Draw path
        if len(path) > 0:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 0], path_arr[:, 1], 'b-', 
                   linewidth=2.5, alpha=0.8, label='Path', zorder=3)
            
            # Draw direction arrows
            arrow_step = max(1, len(path) // 15)
            for i in range(0, len(path), arrow_step):
                x, y, yaw = path[i]
                dx = 0.6 * np.cos(yaw)
                dy = 0.6 * np.sin(yaw)
                ax.arrow(x, y, dx, dy, head_width=0.4, head_length=0.3,
                        fc='blue', ec='blue', alpha=0.7, zorder=4)
            
            # Highlight start and end
            ax.plot(path[0][0], path[0][1], 'ro', markersize=10, 
                   label='Start', zorder=5)
            ax.plot(path[-1][0], path[-1][1], 'g^', markersize=12,
                   label='Goal', zorder=5)
            
            # Path stats
            path_length = sum(np.linalg.norm([path[i+1][0] - path[i][0],
                                              path[i+1][1] - path[i][1]])
                             for i in range(len(path)-1))
            ax.text(0.02, 0.98, f"Path: {len(path)} pts, {path_length:.1f}m",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.legend(loc='upper right', fontsize=10)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{lot_name}_{bay_id}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename}")
    return filepath


def main():
    output_dir = "results/rs_path_preview"
    
    print("=" * 70)
    print("Reeds-Shepp Path Visualization (Simplified)")
    print("=" * 70)
    
    # All bays to test
    scenarios = [
        ("lot_a", "A1"), ("lot_a", "A2"), ("lot_a", "A3"),
        ("lot_a", "A4"), ("lot_a", "A5"), ("lot_a", "A6"),
        ("lot_a", "B1"), ("lot_a", "B2"), ("lot_a", "B3"),
        ("lot_a", "B4"), ("lot_a", "B5"), ("lot_a", "B6"),
        ("lot_b", "H1"), ("lot_b", "H2"), ("lot_b", "H3"),
        ("lot_b", "V1"), ("lot_b", "V2"), ("lot_b", "V3"),
    ]
    
    generated = []
    for lot, bay in scenarios:
        try:
            filepath = generate_comparison(lot, bay, output_dir)
            generated.append((lot, bay, filepath))
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 70)
    print(f"Generated {len(generated)} images")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("=" * 70)
    
    # Create HTML index
    create_html(output_dir, generated)


def create_html(output_dir, generated):
    """Create HTML gallery"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Reeds-Shepp Path Preview</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f0f0f0; }
        h1 { color: #333; }
        .info { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .gallery { display: grid; grid-template-columns: 1fr; gap: 30px; }
        .item { background: white; padding: 15px; border-radius: 8px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        img { width: 100%; height: auto; border: 1px solid #ddd; }
        .caption { margin-top: 10px; font-size: 16px; font-weight: bold; color: #555; }
        .lot-section { margin-top: 40px; }
        .lot-title { font-size: 24px; color: #2c3e50; border-bottom: 3px solid #3498db; 
                     padding-bottom: 10px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>🚗 Reeds-Shepp Path Comparison</h1>
    <div class="info">
        <strong>Left:</strong> B-Spline (Current - Forward only)<br>
        <strong>Right:</strong> Reeds-Shepp (Proposed - Allows reverse maneuvers)<br>
        <strong>Key Difference:</strong> Reeds-Shepp can use backing up for tighter turns
    </div>
"""
    
    # Group by lot
    lot_a = [(l, b, f) for l, b, f in generated if l == "lot_a"]
    lot_b = [(l, b, f) for l, b, f in generated if l == "lot_b"]
    
    for lot_name, items in [("Lot A", lot_a), ("Lot B", lot_b)]:
        if items:
            html += f'<div class="lot-section"><div class="lot-title">{lot_name}</div><div class="gallery">'
            for lot, bay, filepath in items:
                filename = os.path.basename(filepath)
                html += f"""
                <div class="item">
                    <img src="{filename}" alt="{bay}">
                    <div class="caption">Bay {bay}</div>
                </div>
"""
            html += '</div></div>'
    
    html += '</body></html>'
    
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(html)
    
    abs_path = os.path.abspath(index_path)
    print(f"\n📄 Open in browser: file://{abs_path}")


if __name__ == "__main__":
    main()
