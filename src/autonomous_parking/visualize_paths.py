#!/usr/bin/env python3
"""
Visualize A* + B-spline waypoint paths for every bay in lot_a and lot_b.

For each bay:
- Reset the WaypointEnv with that bay as the goal.
- Let it generate A* + smoothed waypoints.
- Plot the parking lot + the waypoint path.
- Save an image: path_visualizations/<lot>_<bay_id>.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from autonomous_parking.env2d.waypoint_env import WaypointEnv
from autonomous_parking.config_loader import load_parking_config


def visualize_all_paths(): 
    lots = ["lot_a", "lot_b"]
    output_dir = "path_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving visualizations to {output_dir}/")

    for lot in lots:
        print(f"\nProcessing {lot}...")
        config = load_parking_config(lot)
        bays = config["bays"]

        for bay in bays:
            bay_id = bay["id"]
            print(f"  Generating path for Bay {bay_id}...", end=" ", flush=True)

            # Create a fresh env per bay so each figure is clean
            env = WaypointEnv(lot_name=lot, render_mode="rgb_array")

            try:
                # Reset with the specific bay ID so goal_bay=bay_id
                obs, info = env.reset(bay_id=bay_id)

                # Trigger a render so ParkingEnv draws the lot into its figure
                env.unwrapped.render()
                # plt.pause(0.1)

                # Access matplotlib axis / figure created by ParkingEnv
                if hasattr(env.unwrapped, "ax") and env.unwrapped.ax is not None:
                    ax = env.unwrapped.ax
                    fig = env.unwrapped.fig

                    if (
                        hasattr(env, "waypoints")
                        and env.waypoints is not None
                        and len(env.waypoints) > 0
                    ):
                        wps = np.array(env.waypoints)

                        # Path line (cyan dashed)
                        ax.plot(
                            wps[:, 0],
                            wps[:, 1],
                            "c--",
                            linewidth=2.5,
                            label="A* Path",
                        )

                        # Waypoint dots (yellow with black edges)
                        ax.scatter(
                            wps[:, 0],
                            wps[:, 1],
                            c="yellow",
                            s=40,
                            zorder=20,
                            edgecolors="black",
                            linewidth=1,
                        )

                        # Start, pre-goal, and goal markers
                        ax.scatter(
                            wps[0, 0],
                            wps[0, 1],
                            c="lime",
                            s=100,
                            zorder=21,
                            label="Start",
                            edgecolors="black",
                        )
                        ax.scatter(
                            wps[-1, 0],
                            wps[-1, 1],
                            c="red",
                            s=100,
                            zorder=21,
                            label="Goal",
                            edgecolors="black",
                        )
                        if len(wps) > 1:
                            ax.scatter(
                                wps[-2, 0],
                                wps[-2, 1],
                                c="orange",
                                s=80,
                                zorder=21,
                                label="Pre-Goal",
                                edgecolors="black",
                            )

                        ax.set_title(
                            f"{lot} - Bay {bay_id}\n"
                            f"Path Length: {len(wps)} waypoints"
                        )

                        filename = os.path.join(output_dir, f"{lot}_{bay_id}.png")
                        fig.savefig(filename, dpi=100, bbox_inches="tight")
                        print(f"✓ Saved to {filename}")
                    else:
                        print("❌ No waypoints generated")

                else:
                    print("❌ Could not access matplotlib axis/figure")

            except Exception as e:
                print(f"❌ ERROR: {e}")
                import traceback

                traceback.print_exc()

            finally:
                # Close env and figure to avoid memory leaks and cross-contamination
                if hasattr(env.unwrapped, "fig") and env.unwrapped.fig is not None:
                    plt.close(env.unwrapped.fig)
                env.close()


if __name__ == "__main__":
    visualize_all_paths()
