#!/usr/bin/env python3
"""
Quick diagnostic to check yaw alignment issue.
"""
import numpy as np
from autonomous_parking.config_loader import load_parking_config

# Load bay configuration
config = load_parking_config("lot_a")
bays = config["bays"]

# Check bay A1
bay_a1 = [b for b in bays if b["id"] == "A1"][0]

print("=" * 60)
print("BAY A1 CONFIGURATION")
print("=" * 60)
print(f"Position: ({bay_a1['x']}, {bay_a1['y']})")
print(f"Yaw (from config): {bay_a1['yaw']} rad = {np.degrees(bay_a1['yaw']):.1f}°")
print(f"")
print("INTERPRETATION:")
print(f"  yaw = 0.0 means bay rectangle is VERTICAL")
print(f"  Car should park with yaw = 0.0 (facing NORTH)")
print(f"")
print("ENTRANCE CALCULATION (current code):")
goal_yaw = bay_a1["yaw"]  # 0.0
dist_entrance = 2.0
cos_yaw = np.cos(goal_yaw)
sin_yaw = np.sin(goal_yaw)
entrance_local_x = 0.0
entrance_local_y = -dist_entrance  # 2m in local -Y direction

# Rotate to world frame
entrance_x = bay_a1["x"] + (cos_yaw * entrance_local_x - sin_yaw * entrance_local_y)
entrance_y = bay_a1["y"] + (sin_yaw * entrance_local_x + cos_yaw * entrance_local_y)

print(f"  Goal: ({bay_a1['x']}, {bay_a1['y']}, {goal_yaw})")
print(f"  Entrance: ({entrance_x:.2f}, {entrance_y:.2f}, {goal_yaw})")
print(f"  Direction: Entrance is {entrance_y - bay_a1['y']:.2f}m in Y from goal")
print(f"")
print("EXPECTED vs ACTUAL:")
print(f"  Bay A1 at y={bay_a1['y']}, entrance should be at road (y=0)")
print(f"  Entrance calculated at y={entrance_y:.2f}")
print(f"  Distance from goal to entrance: {dist_entrance}m")
print("=" * 60)
