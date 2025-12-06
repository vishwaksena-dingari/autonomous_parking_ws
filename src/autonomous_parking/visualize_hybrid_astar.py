#!/usr/bin/env python3
"""
Visualize Hybrid A* paths for every bay in lot_a and lot_b.

This demonstrates obstacle-aware Reeds-Shepp path planning.
Unlike pure Reeds-Shepp, Hybrid A* respects walls and obstacles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from autonomous_parking.env2d.waypoint_env import WaypointEnv
from autonomous_parking.config_loader import load_parking_config
from autonomous_parking.planning.hybrid_astar import HybridAStarPlanner
from autonomous_parking.planning.astar import create_obstacle_grid


def visualize_hybrid_astar_paths(): 
    lots = ["lot_a", "lot_b"]
    output_dir = "hybrid_astar_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving Hybrid A* visualizations to {output_dir}/")

    for lot in lots:
        print(f"\nProcessing {lot}...")
        config = load_parking_config(lot)
        bays = config["bays"]

        for bay in bays:
            bay_id = bay["id"]
            print(f"  Generating Hybrid A* path for Bay {bay_id}...", end=" ", flush=True)

            # Create environment to get obstacle grid
            env = WaypointEnv(lot_name=lot, render_mode="rgb_array")

            try:
                # Reset to get start pose and obstacle grid
                obs, info = env.reset(bay_id=bay_id)
                
                # Render to create figure
                env.unwrapped.render()
                
                if hasattr(env.unwrapped, "ax") and env.unwrapped.ax is not None:
                    ax = env.unwrapped.ax
                    fig = env.unwrapped.fig

                    # Get start and goal poses
                    state = env.unwrapped.state
                    sx, sy, syaw = state[0], state[1], state[2]
                    gx, gy, gyaw = env.unwrapped.goal_x, env.unwrapped.goal_y, env.unwrapped.goal_yaw

                    # Get obstacle grid
                    obstacle_grid = create_obstacle_grid(
                        world_bounds=(-25, 25, -25, 25),
                        resolution=0.5,  # Match Hybrid A* resolution
                        bays=env.unwrapped.occupied_bays,
                        margin=1.0,
                        roads=env.unwrapped.roads,
                        goal_bay=env.unwrapped.goal_bay,
                    )

                    # Create Hybrid A* planner
                    planner = HybridAStarPlanner(
                        world_bounds=(-25, 25, -25, 25),
                        resolution=0.5,
                        heading_resolution=np.radians(15),  # 15 degree bins
                        max_curvature=0.2,
                        step_size=0.5
                    )

                    # Plan path
                    path = planner.plan(
                        (sx, sy, syaw),
                        (gx, gy, gyaw),
                        obstacle_grid
                    )

                    if path:
                        # Extract x, y coordinates
                        path_x = [p[0] for p in path]
                        path_y = [p[1] for p in path]

                        # Plot Hybrid A* Path (Green)
                        ax.plot(
                            path_x,
                            path_y,
                            "g-",
                            linewidth=2.5,
                            label="Hybrid A* Path",
                        )
                        
                        # Plot start and goal arrows
                        arrow_len = 1.5
                        ax.arrow(sx, sy, arrow_len * np.cos(syaw), arrow_len * np.sin(syaw), 
                                color='lime', width=0.3, head_width=0.8, label='Start')
                        ax.arrow(gx, gy, arrow_len * np.cos(gyaw), arrow_len * np.sin(gyaw), 
                                color='red', width=0.3, head_width=0.8, label='Goal')

                        ax.legend(loc='upper right')
                        ax.set_title(f"Hybrid A*: {lot} - {bay_id}")

                        # Save
                        save_path = os.path.join(output_dir, f"{lot}_{bay_id}.png")
                        fig.savefig(save_path)
                        print("Saved.")
                    else:
                        print("Failed to find path.")

                    plt.close(fig)

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                env.close()

if __name__ == "__main__":
    visualize_hybrid_astar_paths()
