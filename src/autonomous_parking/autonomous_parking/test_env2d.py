#!/usr/bin/env python3
"""
Simple smoke test for the 2D parking environment.

Usage:
    ros2 run autonomous_parking test_env2d
"""

import argparse

from autonomous_parking.env2d.parking_env import ParkingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lot",
        default="lot_a",
        help="Parking lot name from bays.yaml (default: lot_a)",
    )
    args, _ = parser.parse_known_args()

    print(f"Creating 2D environment for lot: {args.lot}")
    env = ParkingEnv(lot_name=args.lot)

    env.reset()
    env.render()
    print("Close the window to exit.")
    # keep window open until user closes it
    import matplotlib.pyplot as plt

    plt.show()
    env.close()
    print("Environment closed cleanly.")


if __name__ == "__main__":
    main()
