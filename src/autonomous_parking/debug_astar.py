"""Debug script to trace A* path planning for bay A1"""
import numpy as np
from autonomous_parking.config_loader import load_parking_config
from autonomous_parking.planning.astar import AStarPlanner, create_obstacle_grid

# Load lot_a configuration
config = load_parking_config("lot_a")
bays = config["bays"]
roads = config["roads"]

# Get bay A1
bay_a1 = [b for b in bays if b["id"] == "A1"][0]
print(f"Bay A1: {bay_a1}")

# Goal position
goal_x = bay_a1["x"]
goal_y = bay_a1["y"]
goal_yaw = bay_a1["yaw"]
print(f"Goal: x={goal_x}, y={goal_y}, yaw={goal_yaw}")

# Staging waypoint (manually calculated)
staging_x = goal_x
staging_y = 0.0  # Road
staging_yaw = goal_yaw
print(f"Staging: x={staging_x}, y={staging_y}, yaw={staging_yaw}")

# Entrance waypoint (2m from goal)
dist_entrance = 2.0
entrance_x = goal_x - dist_entrance * np.cos(goal_yaw)
entrance_y = goal_y - dist_entrance * np.sin(goal_yaw)
print(f"Entrance: x={entrance_x}, y={entrance_y}")
print(f"  (goal_yaw={goal_yaw}, cos={np.cos(goal_yaw)}, sin={np.sin(goal_yaw)})")

# Create obstacle grid
obstacles = create_obstacle_grid(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.5,
    bays=[],  # No other bays for simplicity
    margin=1.0,
    roads=roads,
    goal_bay=bay_a1
)

print(f"\nObstacle grid shape: {obstacles.shape}")

# Check grid around bay A1
resolution = 0.5
x_min, y_min = -25, -25

# Bay center in grid coords
gx = int((goal_x - x_min) / resolution)
gy = int((goal_y - y_min) / resolution)
print(f"Bay center grid coords: gx={gx}, gy={gy}")

# Check entrance area (should be free)
# Entrance is at y=~4.5 (2m south of bay at y=6.5)
entrance_gy = int((entrance_y - y_min) / resolution)
print(f"Entrance grid coords: gy={entrance_gy}")

# Check if entrance is free
print(f"\nChecking grid around entrance (x={goal_x}, y={entrance_y}):")
for dy in range(-2, 3):
    gy_check = entrance_gy + dy
    gx_check = gx
    world_y = y_min + gy_check * resolution
    is_free = obstacles[gy_check, gx_check] == 0
    print(f"  gy={gy_check} (world y≈{world_y:.1f}): {'FREE' if is_free else 'BLOCKED'}")

# Check south of bay (toward road)
print(f"\nChecking grid south of bay (toward road at y=0):")
for dy in range(-8, 1):
    gy_check = gy + dy
    world_y = y_min + gy_check * resolution
    is_free = obstacles[gy_check, gx] == 0
    print(f"  gy={gy_check} (world y≈{world_y:.1f}): {'FREE' if is_free else 'BLOCKED'}")

# Check north of bay (away from road)
print(f"\nChecking grid north of bay (away from road):")
for dy in range(0, 8):
    gy_check = gy + dy
    world_y = y_min + gy_check * resolution
    is_free = obstacles[gy_check, gx] == 0
    print(f"  gy={gy_check} (world y≈{world_y:.1f}): {'FREE' if is_free else 'BLOCKED'}")
