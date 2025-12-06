#!/usr/bin/env python3
"""
Visualize Reeds-Shepp SMOOTHED paths for every bay.

This uses A* waypoints and connects them with Reeds-Shepp curves,
giving us obstacle-aware, kinematically feasible paths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from autonomous_parking.env2d.waypoint_env import WaypointEnv
from autonomous_parking.config_loader import load_parking_config
from autonomous_parking.planning.smoothing import smooth_path_reeds_shepp


def visualize_rs_smoothed_paths(): 
    lots = ["lot_a", "lot_b"]
    output_dir = "rs_smoothed_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving Reeds-Shepp SMOOTHED visualizations to {output_dir}/")

    for lot in lots:
        print(f"\nProcessing {lot}...")
        config = load_parking_config(lot)
        bays = config["bays"]

        for bay in bays:
            bay_id = bay["id"]
            print(f"  Generating RS-smoothed path for Bay {bay_id}...", end=" ", flush=True)

            # Create environment
            env = WaypointEnv(lot_name=lot, render_mode="rgb_array")

            try:
                # Reset to get A* waypoints
                obs, info = env.reset(bay_id=bay_id)
                
                # Render to create figure
                env.unwrapped.render()
                
                if hasattr(env.unwrapped, "ax") and env.unwrapped.ax is not None:
                    ax = env.unwrapped.ax
                    fig = env.unwrapped.fig

                    # Get A* waypoints (these are already obstacle-aware!)
                    if hasattr(env, "waypoints") and env.waypoints is not None and len(env.waypoints) > 0:
                        astar_waypoints = env.waypoints  # List of (x, y, theta)
                        
                        # Pre-align waypoint headings to point toward next waypoint
                        # This prevents RS from creating loops
                        aligned_waypoints = []
                        for i in range(len(astar_waypoints)):
                            x, y, theta = astar_waypoints[i]
                            if i < len(astar_waypoints) - 1:
                                # Point toward next waypoint
                                dx = astar_waypoints[i + 1][0] - x
                                dy = astar_waypoints[i + 1][1] - y
                                aligned_theta = float(np.arctan2(dy, dx))
                            else:
                                # Keep original goal heading
                                aligned_theta = theta
                            aligned_waypoints.append((x, y, aligned_theta))
                        
                        # Apply Reeds-Shepp smoothing with aligned headings
                        rs_smoothed = smooth_path_reeds_shepp(
                            aligned_waypoints,
                            turning_radius=5.0,
                            step_size=0.5
                        )

                        if rs_smoothed and len(rs_smoothed) > 0:
                            # Extract x, y coordinates
                            rs_x = [p[0] for p in rs_smoothed]
                            rs_y = [p[1] for p in rs_smoothed]
                            
                            # Also plot original A* waypoints for comparison
                            astar_x = [p[0] for p in astar_waypoints]
                            astar_y = [p[1] for p in astar_waypoints]

                            # Plot A* waypoints (yellow dots)
                            ax.scatter(astar_x, astar_y, c='yellow', s=50, zorder=20, 
                                      edgecolors='black', linewidth=1, label='A* Waypoints')

                            # Plot Reeds-Shepp smoothed path (Magenta)
                            ax.plot(
                                rs_x,
                                rs_y,
                                "m-",
                                linewidth=2.5,
                                label="Reeds-Shepp Smoothed",
                            )
                            
                            # Plot start and goal
                            ax.scatter(rs_x[0], rs_y[0], c='lime', s=100, zorder=21, 
                                      label='Start', edgecolors='black')
                            ax.scatter(rs_x[-1], rs_y[-1], c='red', s=100, zorder=21, 
                                      label='Goal', edgecolors='black')

                            ax.legend(loc='upper right')
                            ax.set_title(f"Reeds-Shepp Smoothed: {lot} - {bay_id}")

                            # Save
                            save_path = os.path.join(output_dir, f"{lot}_{bay_id}.png")
                            fig.savefig(save_path)
                            print("Saved.")
                        else:
                            print("Failed to smooth path.")
                    else:
                        print("No waypoints available.")

                    plt.close(fig)

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                env.close()

if __name__ == "__main__":
    visualize_rs_smoothed_paths()
