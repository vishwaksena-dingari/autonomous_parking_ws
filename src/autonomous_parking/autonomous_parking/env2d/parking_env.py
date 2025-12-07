#!/usr/bin/env python3
"""
2D Parking Environment - Updated for Gymnasium

Modern Gymnasium-compatible environment for autonomous parking.
"""

import math
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib
# Headless backend: no GUI windows during training / video capture
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
import matplotlib.transforms as transforms  # noqa: F401 (reserved for future use)

from ..config_loader import load_parking_config
from ..sensors.lidar import EnhancedLidar
from ..sensors.occupied_bays import OccupiedBayManager  # v38.9: Parked car detection


class ParkingEnv(gym.Env):
    """
    2D kinematic-car parking environment.

    State (internal): [x, y, yaw, v]  (absolute pose + velocity)
    Action: [v_cmd, steer_cmd]        (m/s, rad)
    Observation (69D):
        [local_x, local_y, yaw_err, v, dist, lidar_0, ..., lidar_63]
    
    v38.9: Lidar consistency fix - now uses all 64 rays.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    # ======================= INIT =======================

    def __init__(
        self,
        lot_name: str = "lot_a",
        dt: float = 0.1,
        max_steps: int = 300,
        render_mode: str = "human",
    ):
        """
        Initialize parking environment.

        Args:
            lot_name: Name of parking lot ('lot_a' or 'lot_b')
            dt: Time step for physics simulation (seconds)
            max_steps: Maximum steps per episode
            render_mode: 'human' for interactive (if not using Agg),
                         anything else for pure headless/video.
        """
        self.debug = False  # set True for verbose parking debug prints
        self.lot_name = lot_name
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        # ---- Car dimensions (compact car, fits comfortably in bay) ----
        self.car_length = 4.2  # m  (slightly shorter than bay depth)
        self.car_width = 1.9   # m
        self.wheelbase = 2.6   # m
        self.collision_penalty = -500.0  # v40: Harsh penalty for hitting physical objects

        # ---- Parking bay dimensions (match SDF) ----
        self.bay_length = 5.5  # m (depth)
        self.bay_width = 2.7   # m (width)

        # Visual size of goal highlight (slightly smaller than bay)
        self.goal_length = 0.9 * self.bay_length
        self.goal_width = 0.9 * self.bay_width

        # ---- Control limits ----
        self.max_speed = 3.0              # m/s
        self.max_steer = math.radians(45)  # ~45 degrees (Super Agile: R ~ 2.6m)

        # Random state for reproducibility (decoupled from Gym's own RNG)
        self.random_state = np.random.RandomState()

        # ---- Speed limit zones (realistic parking lot rules) ----
        self.speed_limit_general = 1.5  # m/s (~5.4 km/h)
        self.speed_limit_near_bay = 1.0  # m/s (~3.6 km/h)
        self.speed_limit_final = 0.5     # m/s (~1.8 km/h)

        # ---- Nominal success criteria (not directly used, but kept for clarity) ----
        self.pos_tol = 1.5   # m
        self.yaw_tol = 0.3   # rad

        # ---- Gymnasium spaces ----
        # Action space: [v_cmd, steer_cmd]
        self.action_space = spaces.Box(
            low=np.array([-self.max_speed, -self.max_steer], dtype=np.float32),
            high=np.array([self.max_speed, self.max_steer], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: [local_x, local_y, yaw_err, v, dist, lidar_0, ..., lidar_63]
        # Total: 69D (5 state + 64 lidar)
        # v38.9: Changed from 32 to 64 lidar rays for consistency with EnhancedLidar
        obs_low = np.array(
            [-50.0, -50.0, -np.pi, -5.0, 0.0] + [0.0] * 64,
            dtype=np.float32,
        )
        obs_high = np.array(
            [50.0, 50.0, np.pi, 5.0, 50.0] + [20.0] * 64,  # 20.0 = lidar max_range
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        # ---- Road compliance parameters ----
        self.road_center_y = 0.0   # Center of main aisle (lot_a)
        self.road_center_x = None  # For vertical roads (lot_b V-bays)
        self.road_orientation = "horizontal"  # lot_a default
        self.road_width = 6.0  # m (approx driveable width)

        # Load parking lot configuration
        cfg = load_parking_config(lot_name)
        self.entrance = cfg["entrance"]
        self.bays = cfg["bays"]
        self.roads = cfg.get("roads", [])

        # ---- Enhanced Lidar Sensor ----
        # Lidar sensor (CRITICAL FIX: Increased from 32 to 64 rays for precision parking)
        self.lidar = EnhancedLidar(
            num_rays=64,  # FIXED: was 32
            max_range=20.0,
            noise_std=0.02,
            min_range=0.1,
        )

        # v41.2: Override fields for micro-curriculum
        self.max_spawn_dist_override = None     # Existing
        self.spawn_side_override = None         # Existing
        self.aligned_spawn_override = False     # NEW: For straight-in baby parking
        self.lateral_offset_override = None     # NEW: For offset training

        # ---- Occupied Bay Manager (v38.9: Parked car detection) ----
        # Manages which bays have parked cars (visible to lidar)
        self.occupied_manager = OccupiedBayManager(
            all_bays=self.bays,
            occupancy_rate=0.3,  # 30% of bays have parked cars
        )

        # ---- Episode state ----
        self.state = None          # [x, y, yaw, v]
        self.goal_bay = None
        self.occupied_bays = []
        self.parked_patches = [] # v40: Parked car patches
        self.episode_count = 0
        self.episode_start_time = 0.0

        self.last_steering = 0.0
        # v15 attributes max_spawn_dist_override/spawn_side_override removed (handled in v41 block above)

        # Dynamic tolerances driven by curriculum
        self.current_tol_pos = 0.5
        self.current_tol_yaw = 0.5

        # ---- Rendering state ----
        self.fig = None
        self.ax = None
        self.car_patch = None
        self.car_front_stripe = None
        self.goal_patch = None
        self.goal_arrow = None
        self.bay_patches = []

    # ======================= HELPERS =======================

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (theta + math.pi) % (2 * math.pi) - math.pi

    # NOTE: _pick_goal_bay() was removed (v41 cleanup).
    # It had a +90° yaw offset that was inconsistent with reset().
    # Bay selection now happens directly in reset().

    def _sample_start_pose(self):
        """
        Sample starting position near entrance with noise.
        Returns: (x, y, yaw)
        """
        ent = self.entrance
        x = ent["x"] + random.uniform(-1.0, 1.0)
        y = ent["y"] + random.uniform(-1.0, 1.0)
        yaw = ent["yaw"] + random.uniform(-0.2, 0.2)
        return x, y, yaw

    def _get_obs(self):
        """
        Compute observation in robot-centric frame.

        Returns:
            [local_x, local_y, yaw_err, v, dist, lidar_0, ..., lidar_63]  (69D)
        """
        x, y, yaw, v = self.state
        gx = self.goal_bay["x"]
        gy = self.goal_bay["y"]
        gyaw = self.goal_bay["yaw"]

        # Goal in world frame
        dx = gx - x
        dy = gy - y

        # Transform to robot frame
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy

        # Heading error
        yaw_err = self._wrap_angle(gyaw - yaw)

        # Distance to goal from car center
        center_x = x + (self.car_length / 2.0) * math.cos(yaw)
        center_y = y + (self.car_length / 2.0) * math.sin(yaw)
        dx_center = gx - center_x
        dy_center = gy - center_y
        dist = math.hypot(dx_center, dy_center)

        # Enhanced 64-ray lidar scan (v38.9: use all 64 rays, not truncated to 32)
        world_bounds = (-25.0, 25.0, -25.0, 25.0)
        lidar_ranges = self.lidar.scan(
            robot_pose=np.array([x, y, yaw]),
            world_bounds=world_bounds,
            bays=self.bays,
            occupied_bays=self.occupied_bays,
            dynamic_obstacles=None,
        )

        # v38.9: Use all 64 rays from lidar (no truncation)
        lidar = np.asarray(lidar_ranges, dtype=np.float32)

        obs = np.array(
            [local_x, local_y, yaw_err, v, dist, *lidar],
            dtype=np.float32,
        )
        return obs

    # ======================= LOT / CURRICULUM =======================

    def _load_lot(self, lot_name: str):
        """
        Load parking lot configuration dynamically.

        Args:
            lot_name: Name of the lot to load ("lot_a" or "lot_b")
        """
        self.lot_name = lot_name
        cfg = load_parking_config(lot_name)
        self.entrance = cfg["entrance"]
        self.bays = cfg["bays"]
        self.roads = cfg.get("roads", [])

        # Update road geometry based on lot
        if lot_name == "lot_a":
            self.road_center_y = 0.0
            self.road_center_x = None
            self.road_orientation = "horizontal"
        elif lot_name == "lot_b":
            # Lot B has BOTH horizontal (H-bays) and vertical (V-bays) roads
            self.road_center_y = 10.0  # H-bay horizontal road
            self.road_center_x = -4.0   # V-bay vertical road
            self.road_orientation = "mixed"
        
        # v38.9: Reinitialize OccupiedBayManager for new lot
        self.occupied_manager = OccupiedBayManager(
            all_bays=self.bays,
            occupancy_rate=0.3,
        )

    def _reset_with_curriculum_spawn(self, seed, options, bay_id):
        """Curriculum-based spawn logic when max_spawn_dist_override is set."""
        level = "v15 Curriculum"
        self.current_tol_pos = 0.5
        self.current_tol_yaw = 0.5

        # ---- v41.2: Read overrides ----
        aligned = getattr(self, "aligned_spawn_override", False)
        lateral_offset = getattr(self, "lateral_offset_override", None)

        gx = self.goal_bay["x"]
        gy = self.goal_bay["y"]
        gyaw = self.goal_bay["yaw"]

        # v41.3: FIX UnboundLocalError - Determine road geometry upfront
        is_horizontal = True
        road_y = 0.0
        road_x = 0.0

        if self.lot_name == "lot_b":
            # Detect bay type by ID prefix (H1-H5 vs V1-V6)
            bay_id = self.goal_bay.get("id", "")
            if bay_id.startswith("H"):  # H-bays (horizontal road)
                is_horizontal = True
                road_y = 10.0
            else:  # V-bays (vertical road)
                is_horizontal = False
                road_x = -4.0
        
        max_dist = self.max_spawn_dist_override if self.max_spawn_dist_override else 20.0
        max_dist = min(max_dist, 8.0)

        # ✅ SPECIAL CASE: aligned straight-in spawn (S0/S1 baby stages)
        if aligned:
            # Give the car some run-up distance (v41.2: ensure > 3.0m)
            d_min = 3.0
            d_max = max(4.0, max_dist)
            d = self.random_state.uniform(d_min, d_max)

            # v42 FIX: Spawn at distance d BACKWARD from bay center
            # In bay local frame: (-d, 0) where -X is toward entrance
            # Transform to world frame
            spawn_x = gx - d * math.cos(gyaw)
            spawn_y = gy - d * math.sin(gyaw)
            spawn_yaw = gyaw  # Face into bay (same as parking direction)

            # Optional lateral offset (for S1 training)
            if lateral_offset is not None:
                # Lateral = along bay width (Y-axis in bay frame)
                spawn_x += lateral_offset * (-math.sin(gyaw))
                spawn_y += lateral_offset * math.cos(gyaw)
        else:
            # 🔁 Existing road-based logic (using pre-calculated geometry)
            dist = self.random_state.uniform(5.0, max_dist)

            if self.spawn_side_override == "left":
                direction = -1
            elif self.spawn_side_override == "right":
                direction = 1
            else:
                direction = self.random_state.choice([-1, 1])

            if is_horizontal:
                spawn_x = gx + dist * direction
                spawn_y = road_y + self.random_state.uniform(-0.5, 0.5)
                spawn_yaw = 0.0 if direction < 0 else math.pi
            else:
                spawn_x = road_x + self.random_state.uniform(-0.5, 0.5)
                spawn_y = gy + dist * direction
                spawn_yaw = math.pi / 2 if direction < 0 else 3 * math.pi / 2

            spawn_yaw += self.random_state.uniform(-0.1, 0.1)

        # Debugging Orientation
        print(f"[DEBUG_SPAWN] Bay={self.goal_bay.get('id')} Aligned={aligned} H={is_horizontal} GYaw={gyaw:.2f} SpawnYaw={spawn_yaw:.2f}")

        # STRICT ROAD CLAMPING (Updated for 4-corner safety)
        # Road half-width = 3.75m. Car half-width = 0.95m. -> Lateral limit +/- 2.0m
        # Road length = +/- 25.0m. Car half-length = 2.1m. -> Long. limit +/- 22.0m
        
        if self.lot_name == "lot_a":
            spawn_x = np.clip(spawn_x, -22.0, 22.0) # Safe from road ends
            spawn_y = np.clip(spawn_y, -2.0, 2.0)   # Safe from road sides
        elif self.lot_name == "lot_b":
            if is_horizontal: # H-bays (Road y=10)
                spawn_x = np.clip(spawn_x, -22.0, 22.0)
                spawn_y = np.clip(spawn_y, 8.0, 11.0) # Changed from 12.0 to 11.0
            else: # V-bays (Road x=0)
                spawn_x = np.clip(spawn_x, -6.0, -2.0)
                # Vertical road [-25, 10] + Intersection up to 13.75
                # Max safe Y = 13.75 - 2.1 = 11.65. We use 11.0.
                spawn_y = np.clip(spawn_y, -22.0, 11.0)

        # Set state
        self.state = np.array([spawn_x, spawn_y, spawn_yaw, 0.0], dtype=np.float32)
        self.last_steering = 0.0
        self.min_dist_to_goal = float("inf")

        obs = self._get_obs()
        info = {"level": level}
        return obs, info

    # ======================= MAIN GYM API =======================

    def reset(self, seed=None, options=None, bay_id=None):
        """
        Reset environment.

        Args:
            seed: optional seed
            options: Gymnasium options (unused)
            bay_id: optional fixed bay id (for evaluation/testing)

        Returns:
            obs, info
        """
        super().reset(seed=seed)

        # Allow bay_id to be passed via options dict (Gymnasium convention)
        if options is not None and "bay_id" in options and bay_id is None:
            bay_id = options["bay_id"]

        # 🔧 Tie our RNG to Gym's seed for reproducibility
        if seed is not None:
            self.random_state.seed(seed)
        else:
            # If Gymnasium already created self.np_random, use it as source
            if hasattr(self, "np_random"):
                self.random_state.seed(self.np_random.integers(0, 2**32 - 1))

        self.steps = 0
        self.episode_count += 1

        # Select goal bay
        if bay_id is not None:
            matches = [b for b in self.bays if b["id"] == bay_id]
            if matches:
                self.goal_bay = matches[0]
            else:
                print(f"Warning: Bay ID {bay_id} not found, using random bay.")
                self.goal_bay = self.random_state.choice(self.bays)
        else:
            self.goal_bay = self.random_state.choice(self.bays)

        # v41: Baby Parking Support - Disable Obstacles
        disable_obstacles = False
        if options is not None and "disable_obstacles" in options:
             disable_obstacles = options["disable_obstacles"]

        if disable_obstacles:
            # Clear all obstacles for C0/C1 stages
            self.occupied_manager.occupied_bay_ids.clear()
            self.occupied_manager.occupied_bay_objects.clear()
            self.occupied_bays = []
            if self.lot_name == "lot_a":
                print(f"[ParkingEnv] 🧹 Obstacles DISABLED (Baby Parking)")
        else:
            # Regular logic: Randomize occupied bays
            # v38.9: Randomize occupied bays (parked cars) each episode
            # Ensures lidar detects parked cars realistically
            self.occupied_manager.randomize_occupancy()
            
            # CRITICAL: Goal bay must NEVER be occupied
            if self.occupied_manager.is_bay_occupied(self.goal_bay["id"]):
                self.occupied_manager.occupied_bay_ids.remove(self.goal_bay["id"])
                self.occupied_manager.occupied_bay_objects = [
                    b for b in self.occupied_manager.occupied_bay_objects
                    if b["id"] != self.goal_bay["id"]
                ]
            
            # Update occupied_bays for lidar detection
            self.occupied_bays = self.occupied_manager.get_all_parked_cars()

        goal_x = self.goal_bay["x"]
        goal_y = self.goal_bay["y"]

        # ========== v13.0: AUTO-CURRICULUM ==========
        # L1: close & aligned
        # L2: medium dist & random yaw
        # L3: full navigation from entrance

        valid_spawn = False
        for _ in range(100):
            # v15: if curriculum override is active, use special reset
            if (
                hasattr(self, "max_spawn_dist_override")
                and self.max_spawn_dist_override is not None
            ):
                return self._reset_with_curriculum_spawn(seed, options, bay_id)

            if self.episode_count < 500:
                # LEVEL 1: EASY
                level = "L1 (Easy)"
                self.current_tol_pos = 0.8
                self.current_tol_yaw = 1.5

                # Lot-specific spawn
                if self.lot_name == "lot_a":
                    road_y = 0.0

                min_offset = 4.0
                max_offset = 8.0
                offset_x = (
                    self.random_state.uniform(min_offset, max_offset)
                    * self.random_state.choice([-1, 1])
                )

                if self.lot_name == "lot_a":
                    road_y = 0.0
                    spawn_x = goal_x + offset_x
                    spawn_y = road_y + self.random_state.uniform(-1.0, 1.0)
                    spawn_yaw = 0.0 if offset_x < 0 else math.pi

                elif self.lot_name == "lot_b":
                    if self.goal_bay["id"].upper().startswith("H"):
                        road_y = 10.0
                        offset_x = (
                            self.random_state.uniform(4.0, 8.0)
                            * self.random_state.choice([-1, 1])
                        )
                        spawn_x = goal_x + offset_x
                        spawn_y = road_y + self.random_state.uniform(-1.0, 1.0)
                        spawn_yaw = 0.0 if offset_x < 0 else math.pi
                    else:
                        road_x = 0.0
                        offset_y = (
                            self.random_state.uniform(4.0, 8.0)
                            * self.random_state.choice([-1, 1])
                        )
                        spawn_x = road_x + self.random_state.uniform(-1.0, 1.0)
                        spawn_y = goal_y + offset_y
                        spawn_yaw = (
                            math.pi / 2 if offset_y < 0 else 3 * math.pi / 2
                        )

                spawn_yaw += self.random_state.uniform(-0.1, 0.1)

            elif self.episode_count < 1000:
                # LEVEL 2: MEDIUM
                level = "L2 (Medium)"
                self.current_tol_pos = 0.4
                self.current_tol_yaw = 0.4

                if self.lot_name == "lot_a":
                    road_y = 0.0
                    max_offset = min(15.0, 18.0 - abs(goal_x))
                    offset_x = (
                        self.random_state.uniform(8.0, max_offset)
                        * self.random_state.choice([-1, 1])
                    )
                    spawn_x = goal_x + offset_x
                    spawn_y = road_y + self.random_state.uniform(-1.0, 1.0)
                    spawn_yaw = self.random_state.uniform(0, 2 * math.pi)

                elif self.lot_name == "lot_b":
                    if self.goal_bay["id"].upper().startswith("H"):
                        road_y = 10.0
                        max_offset = min(15.0, 18.0 - abs(goal_x))
                        offset_x = (
                            self.random_state.uniform(8.0, max_offset)
                            * self.random_state.choice([-1, 1])
                        )
                        spawn_x = goal_x + offset_x
                        spawn_y = road_y + self.random_state.uniform(-1.0, 1.0)
                        spawn_yaw = self.random_state.uniform(0, 2 * math.pi)
                    else:
                        road_x = 0.0
                        max_offset = min(15.0, 18.0 - abs(goal_y))
                        offset_y = (
                            self.random_state.uniform(8.0, max_offset)
                            * self.random_state.choice([-1, 1])
                        )
                        spawn_x = road_x + self.random_state.uniform(-1.0, 1.0)
                        spawn_y = goal_y + offset_y
                        spawn_yaw = self.random_state.uniform(0, 2 * math.pi)

            else:
                # LEVEL 3: HARD (Full Task)
                level = "L3 (Hard)"
                self.current_tol_pos = 0.3
                self.current_tol_yaw = 0.3

                spawn_x = self.entrance["x"] + self.random_state.uniform(-1.5, 1.5)
                spawn_y = self.entrance["y"] + self.random_state.uniform(-1.5, 1.5)
                spawn_yaw = self.entrance["yaw"] + self.random_state.uniform(-0.5, 0.5)

                valid_spawn = True
                break

            # ---- VALIDATION CHECKS ----

            # 1. World bounds
            if abs(spawn_x) > 25.0 or abs(spawn_y) > 25.0:
                continue

            # 2. Strict road check (all 4 corners on road)
            half_l = self.car_length / 2.0
            half_w = self.car_width / 2.0
            corners_local = [
                (half_l, half_w),
                (half_l, -half_w),
                (-half_l, -half_w),
                (-half_l, half_w),
            ]

            on_road = True
            for lx, ly in corners_local:
                cx = spawn_x + lx * math.cos(spawn_yaw) - ly * math.sin(spawn_yaw)
                cy = spawn_y + lx * math.sin(spawn_yaw) + ly * math.cos(spawn_yaw)

                if self.lot_name == "lot_a":
                    # Horizontal road at y=0, width 7.5m → safe band ±2.75m
                    if abs(cy - 0.0) > 2.75:
                        on_road = False
                        break
                elif self.lot_name == "lot_b":
                    if self.goal_bay["id"].upper().startswith("H"):
                        if abs(cy - 10.0) > 2.75:
                            on_road = False
                            break
                    else:
                        if abs(cx - 0.0) > 2.75:
                            on_road = False
                            break

            if not on_road:
                continue

            # 3. Avoid spawning too close to bays
            collision = False
            for bay in self.bays:
                d = math.hypot(spawn_x - bay["x"], spawn_y - bay["y"])
                if d < 3.5:  # too close
                    collision = True
                    break

            if not collision:
                valid_spawn = True
                break

        if not valid_spawn:
            print("Warning: Could not find valid spawn, using Entrance.")
            spawn_x = self.entrance["x"] + self.random_state.uniform(-1.0, 1.0)
            spawn_y = self.entrance["y"] + self.random_state.uniform(-1.0, 1.0)
            spawn_yaw = self.entrance["yaw"]
            level = "L3 (Hard) - Fallback"

        # Final noise + clamping
        x = np.clip(spawn_x + self.random_state.uniform(-0.5, 0.5), -19.5, 19.5)
        y = np.clip(spawn_y + self.random_state.uniform(-0.5, 0.5), -19.5, 19.5)
        yaw = spawn_yaw
        v = 0.0

        self.state = np.array([x, y, yaw, v], dtype=np.float32)
        self.last_steering = 0.0
        self.min_dist_to_goal = float("inf")

        if self.episode_count in [1, 500, 1000]:
            print(f"*** CURRICULUM LEVEL UP: {level} ***")

        obs = self._get_obs()
        info = {"level": level}
        return obs, info

    def step(self, action):
        """
        One simulation step with bicycle model kinematics.

        Args:
            action: [v_cmd, steer_cmd] (velocity m/s, steering rad)

        Returns:
            obs, reward, terminated, truncated, info
        """
        done = False
        success = False
        collision = False
        info = {}

        v_cmd, steer_cmd = action

        # Clip commands
        v = float(np.clip(v_cmd, -self.max_speed, self.max_speed))
        delta = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))

        # Unpack state
        x, y, yaw, _ = self.state

        # Bicycle model (rear-axle reference)
        x += v * math.cos(yaw) * self.dt
        y += v * math.sin(yaw) * self.dt
        yaw += (v / self.wheelbase) * math.tan(delta) * self.dt
        yaw = self._wrap_angle(yaw)

        self.state = np.array([x, y, yaw, v], dtype=np.float32)
        self.steps += 1

        # Observation
        obs = self._get_obs()
        local_x, local_y, yaw_err, v_obs, dist, *sensors = obs

        # ========== REWARD SHAPING ==========
        
        # Initialize base reward (e.g. small time penalty)
        reward = -0.01

        # Smoothness penalty
        steering_change = abs(delta - self.last_steering)
        if steering_change > 0.01:
            reward -= 0.1 * steering_change
        self.last_steering = delta

        # v41.2: Removed duplicate shaping rewards (alignment, road compliance, near-miss)
        # to ensure WaypointEnv is the single source of truth for reward shaping.
        # Only base functionality (collision check, OOB) remains here.

        # v41.2: Removed redundant road compliance and wrong-bay penalties.
        # Strict OOB (25m) and OBB collision checks are handled below.

        # Track minimum distance
        if hasattr(self, "min_dist_to_goal"):
            self.min_dist_to_goal = min(self.min_dist_to_goal, dist)

        # Center-based success check
        if dist < 3.0:
            gx, gy, gyaw = (
                self.goal_bay["x"],
                self.goal_bay["y"],
                self.goal_bay["yaw"],
            )

            center_x = x + (self.car_length / 2.0) * math.cos(yaw)
            center_y = y + (self.car_length / 2.0) * math.sin(yaw)

            dx_c = center_x - gx
            dy_c = center_y - gy
            cos_g = math.cos(-gyaw)
            sin_g = math.sin(-gyaw)
            cx_bay = cos_g * dx_c - sin_g * dy_c
            cy_bay = sin_g * dx_c + cos_g * dy_c

            # v41: STRICT SUCCESS CRITERIA
            # 1. Position Error < 0.2m (Lateral/Longitudinal in bay frame)
            # 2. Yaw Error < 0.1 rad (approx 6 deg)
            # 3. Speed < 0.1 m/s (Stopped)
            
            strict_tol_pos = 0.2
            strict_tol_yaw = 0.1
            strict_tol_speed = 0.1 # m/s
            
            lateral_error = abs(cx_bay)
            long_error = abs(cy_bay)
            heading_error = abs(yaw_err)
            speed_error = abs(self.state[3])
            
            is_centered = lateral_error < strict_tol_pos and long_error < strict_tol_pos
            is_aligned = heading_error < strict_tol_yaw
            is_stopped = speed_error < strict_tol_speed

            if is_centered and is_aligned and is_stopped:
                # v41.3: strict success reward moved to WaypointEnv to avoid double-counting
                # reward += 100.0 
                done = True
                success = True
                if self.debug:
                    print(
                        f"✓ STRICT SUCCESS! lat={lateral_error:.2f}, long={long_error:.2f}, "
                        f"yaw={heading_error:.2f}, v={speed_error:.2f}"
                    )

            elif dist < 1.0 and self.steps % 10 == 0:
                if self.debug:
                    print(
                        f"DEBUG: Close - lat={lateral_error:.2f}, long={long_error:.2f}, "
                        f"yaw={heading_error:.2f}, v={speed_error:.2f}"
                    )

        # Out-of-bounds
        if abs(x) > 25.0 or abs(y) > 25.0:
            reward -= 20.0
            done = True
            collision = True

        # Timeout (v41: use truncated, not terminated, for Gymnasium semantics)
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        info.update({
            "success": success,
            "collision": collision,
            "dist": dist,
            "yaw_err": yaw_err,
            "steps": self.steps,
        })
        # v40: Check physical collision with parked cars
        is_collision = self._check_collision()
        
        # Check termination (true terminal states: success, collision, OOB)
        terminated = done  # Start with existing 'done' status (success/collision/OOB)
        if is_collision:
            terminated = True
            collision = True  # v41: Unify collision flags
            info["collision"] = True
            info["crash"] = True
            reward += self.collision_penalty

        return obs, reward, terminated, truncated, info

    # ======================= RENDERING =======================

    def _draw_roads(self):
        """
        Draw simple asphalt roads that roughly match the Gazebo layouts.

        - lot_a: one horizontal road between two rows of bays
        - lot_b: T-shaped road (vertical for V1-V5, horizontal for H1-H5)
        """
        road_color = (0.18, 0.18, 0.18)
        road_width = 6.0  # ~6 m total width

        if self.lot_name == "lot_a":
            xs = [b["x"] for b in self.bays]
            x_min = min(xs) - self.bay_width
            x_max = max(xs) + self.bay_width

            road = Rectangle(
                (x_min, -road_width / 2.0),
                x_max - x_min,
                road_width,
                facecolor=road_color,
                edgecolor="none",
                zorder=0.25,
            )
            self.ax.add_patch(road)

        elif self.lot_name == "lot_b":
            v_bays = [b for b in self.bays if b["id"].upper().startswith("V")]
            h_bays = [b for b in self.bays if b["id"].upper().startswith("H")]

            # Horizontal leg (in front of H1..H5)
            front_y = None
            if h_bays:
                h_y = min(b["y"] for b in h_bays)
                front_y = h_y - self.bay_length / 2.0

                x_min = min(b["x"] for b in h_bays) - self.bay_width / 2.0
                x_max = max(b["x"] for b in h_bays) + self.bay_width / 2.0

                road_y_min = front_y - road_width
                horiz_road = Rectangle(
                    (x_min, road_y_min),
                    x_max - x_min,
                    road_width,
                    facecolor=road_color,
                    edgecolor="none",
                    zorder=0.25,
                )
                self.ax.add_patch(horiz_road)

            # Vertical leg (beside V1..V5)
            if v_bays:
                y_min = min(b["y"] for b in v_bays) - self.bay_length / 2.0
                if front_y is not None:
                    y_max = front_y
                else:
                    y_max = max(b["y"] for b in v_bays) + self.bay_length / 2.0

                v_x = max(b["x"] for b in v_bays)
                road_x_min = v_x + self.bay_length / 2.0

                vert_road = Rectangle(
                    (road_x_min, y_min),
                    road_width,
                    y_max - y_min,
                    facecolor=road_color,
                    edgecolor="none",
                    zorder=0.25,
                )
                self.ax.add_patch(vert_road)

    def _setup_render(self):
        """Initialize matplotlib figure for visualization."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_title(f"Parking Environment: {self.lot_name}")
        self.ax.grid(True, alpha=0.3)

        # Ground
        ground = Rectangle(
            (-25, -25),
            50,
            50,
            facecolor=(0.25, 0.25, 0.25),
            edgecolor="none",
            zorder=0,
        )
        self.ax.add_patch(ground)

        # Roads
        self._draw_roads()

        # Bay outlines
        for bay in self.bays:
            bx = bay["x"]
            by = bay["y"]
            byaw = bay["yaw"]

            # v42 FIX: Swap width/length for new yaw convention
            # New convention: yaw points INTO bay, local +X is depth, +Y is width
            half_width = self.bay_width / 2   # 1.35m
            half_length = self.bay_length / 2  # 2.75m

            corners_local = np.array(
                [
                    [-half_length, -half_width],  # entrance left (SWAPPED!)
                    [-half_length, half_width],   # entrance right (SWAPPED!)
                    [half_length, half_width],    # back right (SWAPPED!)
                    [half_length, -half_width],   # back left (SWAPPED!)
                ]
            )

            cos_yaw = math.cos(byaw)
            sin_yaw = math.sin(byaw)
            rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

            corners = corners_local @ rotation.T + np.array([bx, by])

            # Entrance edge (green)
            entrance_start = corners[0]
            entrance_end = corners[1]
            self.ax.plot(
                [entrance_start[0], entrance_end[0]],
                [entrance_start[1], entrance_end[1]],
                "lime",
                linewidth=3,
                zorder=1,
            )

            # Other edges (white dashed)
            for i in range(1, 4):
                start = corners[i]
                end = corners[(i + 1) % 4]
                self.ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    "w--",
                    linewidth=1.5,
                    zorder=1,
                )

            self.bay_patches.append(None)

            # Bay ID label
            self.ax.text(
                bx,
                by,
                bay["id"],
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                zorder=2,
            )

        # Goal highlight
        self.goal_patch = Rectangle(
            (0, 0),
            self.goal_length,
            self.goal_width,
            fill=True,
            facecolor="green",
            alpha=0.3,
            edgecolor="green",
            linewidth=2,
            zorder=1,
        )
        self.ax.add_patch(self.goal_patch)
        self.goal_arrow = None  # v42: Dynamic arrow

        # Car body - use physics dimensions (set in __init__)
        # v38.7: FIXED - was overwriting car_length to 4.5, causing physics/render mismatch
        self.car_patch = Rectangle(
            (0, 0),
            self.car_length,
            self.car_width,
            facecolor="blue",
            edgecolor="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=3,
        )
        self.ax.add_patch(self.car_patch)

        # Car front stripe
        stripe_width = 0.8
        self.car_front_stripe = Rectangle(
            (0, 0),
            stripe_width,
            self.car_width,
            facecolor="red",
            edgecolor="none",
            alpha=0.95,
            zorder=4,
        )
        self.ax.add_patch(self.car_front_stripe)

    # ======================= PHYSICS & COLLISION (v40) =======================

    def _check_collision(self) -> bool:
        """
        Check if ego car collides with any occupied bay (parked car).
        Uses OBB (Oriented Bounding Box) Separating Axis Theorem (SAT).
        """
        # 1. Get Ego Car OBB
        x, y, yaw, _ = self.state

        # v41 FIX: Rear-axle → geometric center of the car
        center_x = x + (self.car_length / 2.0) * math.cos(yaw)
        center_y = y + (self.car_length / 2.0) * math.sin(yaw)

        ego_bbox = {
            "x": center_x,
            "y": center_y,
            "yaw": yaw,
            "length": self.car_length,
            "width": self.car_width,
        }
        ego_corners = self._get_obb_corners(ego_bbox)
        
        # 2. Check against all occupied bays
        for bay in self.occupied_bays:
            parked_corners = self._get_obb_corners(bay)
            
            if self._check_obb_intersection(ego_corners, parked_corners):
                return True
                
        return False

    def _get_obb_corners(self, bbox: dict) -> np.ndarray:
        """Calculate 4 corners of an OBB from center pose and dims."""
        cx, cy, yaw = bbox['x'], bbox['y'], bbox['yaw']
        l2 = bbox['length'] / 2.0
        w2 = bbox['width'] / 2.0
        
        c, s = math.cos(yaw), math.sin(yaw)
        
        # Rotated relative vectors
        # FL, FR, RR, RL
        corners = np.array([
            [ l2,  w2],
            [ l2, -w2],
            [-l2, -w2],
            [-l2,  w2]
        ])
        
        # Rotate and translate
        rotated = np.dot(corners, np.array([[c, s], [-s, c]]))
        return rotated + np.array([cx, cy])

    def _check_obb_intersection(self, corners1: np.ndarray, corners2: np.ndarray) -> bool:
        """
        Check intersection using Separating Axis Theorem (SAT).
        Returns True if overlapping.
        """
        # Axes to test: Normals of all edges (2 from box1, 2 from box2)
        axes = []
        
        # Get edges and normals for box1
        for i in range(4):
            p1 = corners1[i]
            p2 = corners1[(i+1)%4]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal / np.linalg.norm(normal))
            
        # Get edges and normals for box2 (only need 2 unique axes for rectangle really, but 4 is safe)
        for i in range(4):
            p1 = corners2[i]
            p2 = corners2[(i+1)%4]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal / np.linalg.norm(normal))
            
        # SAT Check
        for axis in axes:
            if not self._overlap_on_axis(axis, corners1, corners2):
                return False  # Separating axis found -> No collision
                
        return True # No separating axis found -> Collision

    def _overlap_on_axis(self, axis, corners1, corners2) -> bool:
        """Project corners onto axis and check overlap."""
        # Project corners1
        proj1 = np.dot(corners1, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project corners2
        proj2 = np.dot(corners2, axis)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        return not (max1 < min2 or max2 < min1)

    def render(self):
        """Update visualization with current state."""
        if self.fig is None:
            self._setup_render()

        x, y, yaw, v = self.state

        # Car body position (rear-axle → rectangle lower-left)
        dx = (self.car_length / 2) * math.cos(yaw) - (self.car_width / 2) * math.sin(
            yaw
        )
        dy = (self.car_length / 2) * math.sin(yaw) + (self.car_width / 2) * math.cos(
            yaw
        )
        llx = x - dx
        lly = y - dy

        self.car_patch.set_xy((llx, lly))
        self.car_patch.angle = math.degrees(yaw)

        # Front stripe
        stripe_offset_x = (self.car_length - 0.3) / 2
        stripe_dx = stripe_offset_x * math.cos(yaw) - (
            self.car_width / 2
        ) * math.sin(yaw)
        stripe_dy = stripe_offset_x * math.sin(yaw) + (
            self.car_width / 2
        ) * math.cos(yaw)
        stripe_llx = x - stripe_dx
        stripe_lly = y - stripe_dy

        self.car_front_stripe.set_xy((stripe_llx, stripe_lly))
        self.car_front_stripe.angle = math.degrees(yaw)

        # Goal bay highlight
        if self.goal_bay is not None:
            gx = self.goal_bay["x"]
            gy = self.goal_bay["y"]
            gyaw = self.goal_bay["yaw"]

            # dx_g = (self.goal_width / 2) * math.cos(gyaw) - (
            #     self.goal_length / 2
            # ) * math.sin(gyaw)
            # dy_g = (self.goal_width / 2) * math.sin(gyaw) + (
            #     self.goal_length / 2
            # ) * math.cos(gyaw)
            dx_g = (self.goal_length / 2) * math.cos(gyaw) - (self.goal_width / 2) * math.sin(gyaw)
            dy_g = (self.goal_length / 2) * math.sin(gyaw) + (self.goal_width / 2) * math.cos(gyaw)
            llx_g = gx - dx_g
            lly_g = gy - dy_g

            self.goal_patch.set_xy((llx_g, lly_g))
            self.goal_patch.angle = math.degrees(gyaw)

            # v42: Render Goal Orientation Arrow
            if self.goal_arrow is not None:
                self.goal_arrow.remove()
                self.goal_arrow = None

            arrow_len = 2.0
            dx_a = arrow_len * math.cos(gyaw)
            dy_a = arrow_len * math.sin(gyaw)
            
            # Draw arrow from center
            self.goal_arrow = Arrow(
                gx, gy, dx_a, dy_a,
                width=0.8,
                color='white',
                alpha=0.9,
                zorder=6
            )
            self.ax.add_patch(self.goal_arrow)

        # v40: Render Occupied Bays (Parked Cars) as GREY rectangles
        # Crucial for visual debugging of collisions
        if hasattr(self, 'occupied_bays'):
            # Remove old patches if they exist (simple cleanup)
            # In efficient animate, we'd update them. 
            # For simplicity in this codebase structure, we might just draw them as patches on the ax.
            # But render() re-draws/updates patches.
            
            # Check if we need to initialize patches pool
            if not hasattr(self, 'parked_patches'):
                self.parked_patches = []
                
            # Create patches for new occupied bays if count mismatch (simple sync)
            while len(self.parked_patches) < len(self.occupied_bays):
                patch = Rectangle(
                    (0, 0), self.car_length, self.car_width,
                    linewidth=1.5, edgecolor='darkred', facecolor='#CC4444', alpha=0.85, zorder=5  # v41.1: Red parked cars
                )
                self.ax.add_patch(patch)
                self.parked_patches.append(patch)
            
            # Update patches
            for i, bay in enumerate(self.occupied_bays):
                patch = self.parked_patches[i]
                
                # Setup pose
                bx, by, byaw = bay['x'], bay['y'], bay['yaw']
                
                dx_b = (self.car_length / 2) * math.cos(byaw) - (self.car_width / 2) * math.sin(byaw)
                dy_b = (self.car_length / 2) * math.sin(byaw) + (self.car_width / 2) * math.cos(byaw)
                
                llx_b = bx - dx_b
                lly_b = by - dy_b
                
                patch.set_xy((llx_b, lly_b))
                patch.angle = math.degrees(byaw)
                patch.set_visible(True)
                
            # Hide unused patches
            for i in range(len(self.occupied_bays), len(self.parked_patches)):
                self.parked_patches[i].set_visible(False)

        # For Agg / video capture
        self.fig.canvas.draw()

        # Optional pause for interactive backends
        if self.render_mode == "human":
            try:
                plt.pause(0.001)
            except Exception:
                pass
    
    def render_v34_overlays(self):
        """
        Render v34 visualization overlays for training videos.
        Shows: corridor boundaries, 8-point bay reference, waypoint path.
        
        This method is called by WaypointEnv after the base render.
        """
        if self.ax is None:
            return
            
        # Only render if we have waypoints (WaypointEnv sets this)
        if not hasattr(self, 'waypoints') or self.waypoints is None or len(self.waypoints) == 0:
            return
        
        try:
            from ..utils.v34_visualization import (
                compute_path_tangents,
                calculate_corridor_boundaries,
                calculate_8_point_bay_reference
            )
            
            # 1. Draw corridor boundaries (red dashed lines)
            waypoints_corrected = compute_path_tangents(self.waypoints)
            left_boundary, right_boundary = calculate_corridor_boundaries(
                waypoints_corrected, self.goal_bay, corridor_width=3.0
            )
            
            if left_boundary and len(left_boundary) > 0:
                left_x, left_y = zip(*left_boundary)
                self.ax.plot(left_x, left_y, 'r--', linewidth=1.5, alpha=0.5, zorder=2)
            
            if right_boundary and len(right_boundary) > 0:
                right_x, right_y = zip(*right_boundary)
                self.ax.plot(right_x, right_y, 'r--', linewidth=1.5, alpha=0.5, zorder=2)
            
            # 2. Draw 8-point bay reference (cyan dots)
            bay_points = calculate_8_point_bay_reference(self.goal_bay)
            if bay_points and len(bay_points) > 0:
                cyan_x, cyan_y = zip(*bay_points)
                self.ax.scatter(cyan_x, cyan_y, c='cyan', s=15, zorder=22, 
                              edgecolors='black', linewidth=0.5, alpha=0.7)
            
            # 3. Draw waypoint path (yellow dots)
            wps = np.array(self.waypoints)
            self.ax.scatter(wps[:, 0], wps[:, 1], c='yellow', s=20, zorder=20,
                          edgecolors='black', linewidth=0.5, alpha=0.7)
        
        except Exception as e:
            # Silently fail if v34 visualization fails (don't break training)
            pass

    def close(self):
        """Clean up rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.car_patch = None
            self.car_front_stripe = None
            self.goal_patch = None
            self.bay_patches = []
