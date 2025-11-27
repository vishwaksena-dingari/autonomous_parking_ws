#!/usr/bin/env python3
"""
Offline visualization of A* + B-spline waypoint paths for all bays.

- Loads lot_a and lot_b configs
- For each bay, resets WaypointEnv with that bay as goal
- Lets WaypointEnv generate A* + smoothed waypoints
- Overlays the path + waypoints on the existing matplotlib lot figure
- Saves one PNG per bay in a folder `path_visualizations/`
"""

import os
import sys

# ---------------------------------------------------------------------
# Add src/ to PYTHONPATH so we can import the package
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ---------------------------------------------------------------------
# Matplotlib (non-interactive backend so we can save PNGs headless)
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # important: do not open a GUI window

import matplotlib.pyplot as plt
import numpy as np

from autonomous_parking.env2d.waypoint_env import WaypointEnv
from autonomous_parking.config_loader import load_parking_config


def visualize_all_paths(output_dir: str = "path_visualizations_v15"):
    """
    Generate and save path visualizations for all bays in lot_a and lot_b.

    For each (lot, bay):
      1. env.reset(bay_id=...) → A* + smoothing runs inside WaypointEnv.reset()
      2. env.unwrapped.render() draws the lot (roads, bays, car, goal)
      3. We overlay the smooth waypoint path on top of that figure
      4. Save as PNG: {output_dir}/{lot}_{bay_id}.png
    """
    lots = ["lot_a", "lot_b"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving visualizations to {output_dir}/")

    for lot in lots:
        print(f"\nProcessing {lot}...")

        # 1) Load config to get list of bay IDs
        config = load_parking_config(lot)
        bays = config["bays"]

        # 2) Create a single env per lot and reuse it for all bays
        env = WaypointEnv(lot_name=lot, render_mode="rgb_array")

        try:
            for bay in bays:
                bay_id = bay["id"]
                print(f"  Generating path for Bay {bay_id}...", end=" ", flush=True)

                # 3) Reset env with this bay as goal
                _, info = env.reset(bay_id=bay_id)

                # Force a render so ParkingEnv creates/updates its matplotlib figure/axes
                env.unwrapped.render()

                # 4) Get the matplotlib axis + figure from the base ParkingEnv
                if hasattr(env.unwrapped, "ax") and env.unwrapped.ax is not None:
                    ax = env.unwrapped.ax
                    fig = env.unwrapped.fig

                    # Optionally clear old overlays between bays (keep static lot drawing)
                    # If your ParkingEnv has a method to redraw static background, call it.
                    # For now we keep it simple and just overlay new paths each time.

                    # 5) Grab waypoints from WaypointEnv
                    if getattr(env, "waypoints", None):
                        wps = np.array(env.waypoints)  # shape (N, 3): x, y, theta

                        # Path line
                        ax.plot(
                            wps[:, 0],
                            wps[:, 1],
                            "c--",
                            linewidth=2.0,
                            label="A* + B-spline path",
                        )

                        # Waypoint dots
                        ax.scatter(
                            wps[:, 0],
                            wps[:, 1],
                            c="yellow",
                            s=35,
                            zorder=20,
                            edgecolors="black",
                            linewidth=0.8,
                        )

                        # Start
                        ax.scatter(
                            wps[0, 0],
                            wps[0, 1],
                            c="lime",
                            s=80,
                            zorder=21,
                            edgecolors="black",
                            label="Start",
                        )

                        # Goal
                        ax.scatter(
                            wps[-1, 0],
                            wps[-1, 1],
                            c="red",
                            s=80,
                            zorder=21,
                            edgecolors="black",
                            label="Goal",
                        )

                        # Pre-goal (second last)
                        if len(wps) > 2:
                            ax.scatter(
                                wps[-2, 0],
                                wps[-2, 1],
                                c="orange",
                                s=70,
                                zorder=21,
                                edgecolors="black",
                                label="Pre-goal",
                            )

                        # Nice title
                        ax.set_title(
                            f"{lot} – Bay {bay_id}\n"
                            f"Waypoints: {len(wps)}"
                        )

                        # Optional: small legend
                        ax.legend(fontsize=6, loc="upper right")

                        # 6) Save PNG
                        filename = os.path.join(output_dir, f"{lot}_{bay_id}.png")
                        fig.savefig(filename, dpi=120, bbox_inches="tight")
                        print(f"✓ saved to {filename}")
                    else:
                        print("❌ no waypoints generated (env.waypoints empty)")
                else:
                    print("❌ env has no matplotlib axis (env.unwrapped.ax is None)")

        finally:
            env.close()


if __name__ == "__main__":
    visualize_all_paths()
