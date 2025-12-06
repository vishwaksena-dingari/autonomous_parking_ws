#!/usr/bin/env python3
"""
Visualize Reeds-Shepp paths for every bay in lot_a and lot_b.

For each bay:
- Reset the WaypointEnv with that bay as the goal.
- Get Start Pose and Goal Pose.
- Generate Reeds-Shepp path connecting them.
- Plot the parking lot + the Reeds-Shepp path.
- Save an image: rs_path_visualizations/<lot>_<bay_id>.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from autonomous_parking.env2d.waypoint_env import WaypointEnv
from autonomous_parking.config_loader import load_parking_config
from autonomous_parking.planning.reeds_shepp import reeds_shepp_path_planning


def visualize_rs_paths(): 
    lots = ["lot_a", "lot_b"]
    output_dir = "rs_path_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving Reeds-Shepp visualizations to {output_dir}/")

    for lot in lots:
        print(f"\nProcessing {lot}...")
        config = load_parking_config(lot)
        bays = config["bays"]

        for bay in bays:
            bay_id = bay["id"]
            print(f"  Generating RS path for Bay {bay_id}...", end=" ", flush=True)

            # Create a fresh env per bay so each figure is clean
            env = WaypointEnv(lot_name=lot, render_mode="rgb_array")

            try:
                # Reset with the specific bay ID so goal_bay=bay_id
                obs, info = env.reset(bay_id=bay_id)

                # Trigger a render so ParkingEnv draws the lot into its figure
                env.unwrapped.render()
                
                # Access matplotlib axis / figure created by ParkingEnv
                if hasattr(env.unwrapped, "ax") and env.unwrapped.ax is not None:
                    ax = env.unwrapped.ax
                    fig = env.unwrapped.fig

                    # Get Start and Goal Poses
                    state = env.unwrapped.state
                    sx, sy, syaw = state[0], state[1], state[2]
                    gx, gy, gyaw = env.unwrapped.goal_x, env.unwrapped.goal_y, env.unwrapped.goal_yaw

                    # Generate Reeds-Shepp Path
                    # maxc = 1.0 / min_radius. Assuming min_radius ~ 4-5m for a car? 
                    # Let's try curvature = 0.2 (radius 5m)
                    rs_x, rs_y, rs_yaw, rs_modes, rs_lengths = reeds_shepp_path_planning(
                        sx, sy, syaw,
                        gx, gy, gyaw,
                        maxc=0.2,  # Curvature (1/R)
                        step_size=0.2
                    )

                    if rs_x:
                        # Plot Reeds-Shepp Path (Magenta)
                        ax.plot(
                            rs_x,
                            rs_y,
                            "m-",
                            linewidth=2.5,
                            label="Reeds-Shepp Path",
                        )
                        
                        # Plot Start and Goal arrows
                        ax.arrow(sx, sy, np.cos(syaw), np.sin(syaw), color='lime', width=0.5, label='Start')
                        ax.arrow(gx, gy, np.cos(gyaw), np.sin(gyaw), color='red', width=0.5, label='Goal')

                        ax.legend(loc='upper right')
                        ax.set_title(f"Reeds-Shepp Path: {lot} - {bay_id}")

                        # Save
                        save_path = os.path.join(output_dir, f"{lot}_{bay_id}.png")
                        fig.savefig(save_path)
                        print("Saved.")
                    else:
                        print("Failed to generate RS path.")

                    plt.close(fig)

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                env.close()

if __name__ == "__main__":
    visualize_rs_paths()
