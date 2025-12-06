#!/usr/bin/env python3
"""
# Waypoint-Following Environment for Hierarchical RL (Final v17 - Dense Path + Smart Subsampling)

ALL FIXES APPLIED:
✅ Enhanced observation (43D with goal bay info + progress indicators)
✅ Distance-normalized waypoint bonus (fair rewards)
✅ Balanced phase rewards (nav/approach/park scaled properly)
✅ Phase-aware anti-freeze (only when far from goal)
✅ Adaptive success thresholds (curriculum-based precision)
✅ Extended, orientation-adaptive timeout
✅ Dual stuck detection (no progress + time near goal)
✅ Fixed B-spline smoothing (tighter fit, shorter phantom, no oversampling)
✅ Finer A* grid (0.25m resolution instead of 0.5m)
✅ v17: Dense path generation + Smart subsampling + Distance-normalized rewards

READY FOR TRAINING - PATHS ARE NOW OPTIMAL
"""

from typing import List, Tuple, Optional

import numpy as np
# from scipy.interpolate import splprep, splev  # ❌ OLD: Embedded smoothing - moved to planning.smoothing module
from gymnasium import spaces
from shapely.geometry import Polygon  # v40: For accurate bay overlap calculation
from .parking_env import ParkingEnv
from ..planning.astar import AStarPlanner, create_obstacle_grid
from ..planning.smoothing import smooth_path_bspline  # ✅ NEW: Modular B-spline smoothing
from ..curriculum import CurriculumManager
from ..rewards import (  # ✅ NEW: Modular reward calculators
    WaypointRewardCalculator,
    ParkingRewardCalculator,
    calculate_bay_entry_bonus,
    calculate_goal_progress_reward,
    get_curriculum_thresholds,
)


class WaypointEnv(ParkingEnv):
    """
    Environment where the agent follows A* + B-spline waypoints,
    then performs a final parking maneuver inside a bay.
    """
    REWARD_BUDGET = 300.0

    def __init__(
        self,
        lot_name: str = "lot_a",
        multi_lot: bool = False,
        enable_curriculum: bool = False,
        verbose: bool = True,
        use_reeds_shepp: bool = False,
        reeds_shepp_turning_radius: float = 5.0,
        # v38.8: Hyperparameter Tuning Args
        align_w: float = 50.0,
        success_bonus: float = 50.0,
        bay_entry_bonus: float = 60.0,
        corridor_penalty: float = 0.05,  # v38.9: Relaxed from 0.5 for early training
        vel_reward_w: float = 0.05,
        **kwargs,
    ):
        """
        Args:
            lot_name: Default lot if multi_lot is False.
            multi_lot: If True, randomly pick from ["lot_a", "lot_b"].
            enable_curriculum: If True, use CurriculumManager for staged difficulty.
            verbose: If False, suppress per-episode / per-step prints (for long runs).
            use_reeds_shepp: Use Reeds-Shepp path planning.
            reeds_shepp_turning_radius: Turning radius for Reeds-Shepp.
            align_w: Weight for alignment reward.
            success_bonus: Bonus for successful parking.
            bay_entry_bonus: Bonus for entering the bay.
            corridor_penalty: Penalty for hitting corridor boundaries.
            vel_reward_w: Weight for velocity reward.
            **kwargs: Passed through to ParkingEnv.
        """
        super().__init__(lot_name, **kwargs)
        self.verbose = verbose
        self.use_reeds_shepp = use_reeds_shepp
        self.reeds_shepp_turning_radius = reeds_shepp_turning_radius
        self.episode_id: int = 0     

        # Store tuning params
        self.corridor_penalty_weight = corridor_penalty
        self.bay_entry_bonus_val = bay_entry_bonus
        
        # v40: Smoothness tracking
        self.prev_steer = 0.0

        # ==== Override observation_space for 98D waypoint obs ====
        # Observation space (FIXED: was 66D, now 98D for 64 lidar rays)
        # 11 (waypoint/goal) + 1 (goal_side) + 16 (8 bay points) + 6 (lookahead) + 64 (lidar) = 98
        obs_low = np.concatenate([
            np.full(34, -1.0, dtype=np.float32),         # 11 + 1 + 16 + 6 normalized features
            np.zeros(64, dtype=np.float32),              # lidar >= 0 (FIXED: was 32)
        ])
        obs_high = np.concatenate([
            np.ones(34, dtype=np.float32),               # conservative upper bound
            np.full(64, 20.0, dtype=np.float32),         # lidar max range (FIXED: was 32)
        ])
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # Action space: [steering_norm, accel_norm] in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ==== END BLOCK ====

        # Lot selection
        self.multi_lot = multi_lot
        self.available_lots = ["lot_a", "lot_b"] if multi_lot else [lot_name]
        self.current_lot = lot_name

        # Curriculum
        # v38.9 FIX: CurriculumManager doesn't accept 'enable' param - conditionally create
        self.enable_curriculum = enable_curriculum
        self.curriculum = CurriculumManager() if enable_curriculum else None
        self._episode_steps: int = 0
        self.max_spawn_dist_override: Optional[float] = None

        # PATCH: Finer A* grid for smoother paths
        self.planner = AStarPlanner(
            world_bounds=(-25, 25, -25, 25),
            resolution=0.25,  # Changed from 0.5 to 0.25
        )

        # Waypoint state
        self.waypoints: List[Tuple[float, float, float]] = []
        self.full_path: List[Tuple[float, float, float]] = []  # v17: Dense path
        self.current_waypoint_idx: int = 0
        self.prev_dist_to_waypoint: Optional[float] = None
        self.total_path_length: float = 0.0  # v17
        self.reward_per_meter: float = 0.0   # v17

        # Goal pose cache (bay center + yaw)
        self.goal_x: float = 0.0
        self.goal_y: float = 0.0
        self.goal_yaw: float = 0.0

        # Final approach / parking tracking
        self.best_dist_to_goal: float = float("inf")
        self.no_progress_steps: int = 0

        # Stuck detection: cumulative time near goal
        self.time_near_goal: int = 0
        self.max_time_near_goal: int = 250  # Will be set per episode

        # Success tracking for monitoring progress
        self.recent_successes: List[bool] = []  # Track last 100 episodes
        self.success_window: int = 100
        self._last_success: bool = False  # Store for next reset

        # ✅ NEW: Initialize modular reward calculators
        # Waypoint-based reward calculator (v26+)
        # FIXED: Removed reward_budget parameter (scaled rewards in v34)
        self.waypoint_rewards = WaypointRewardCalculator(
            velocity_reward_weight=vel_reward_w,   # v38.8: Tuned param
            low_velocity_penalty=0.01,
            anti_freeze_penalty=0.02,
        )
        self.parking_rewards = ParkingRewardCalculator(
            bay_length=5.5,
            bay_width=2.7,
            alignment_reward_weight=align_w,  # v38.8: Tuned param
            success_bonus=success_bonus,      # v38.8: Tuned param
        )
        
        # Bay entry tracking (for bonus)
        self._entered_bay: bool = False
        self._prev_goal_dist: Optional[float] = None  # For goal progress reward

    # ------------------------------------------------------------------ #
    # Reset + path generation
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None, bay_id=None):
        """Reset environment and generate A* + smoothed waypoints."""
        # Track success from previous episode (for rolling success rate)
        self.recent_successes.append(self._last_success)
        if len(self.recent_successes) > self.success_window:
            self.recent_successes.pop(0)
            
        # Log success rate every 10 episodes
        if self.verbose and len(self.recent_successes) > 0 and len(self.recent_successes) % 10 == 0:
            rate = sum(self.recent_successes) / len(self.recent_successes)
            print(f"[Success Rate] {rate:.1%} (last {len(self.recent_successes)} eps)")
            
        self._last_success = False  # Reset for new episode
        
        # ✅ NEW: Reset tracking variables for modular reward calculations
        self._entered_bay = False  # Bay entry bonus flag
        self._prev_goal_dist = None  # Goal progress tracking
        
        self._episode_steps = 0
        self.best_dist_to_goal = float("inf")  # ✅ BUGFIX: Reset stuck detection
        self.no_progress_steps = 0
        self.prev_dist_to_waypoint = None
        self._prev_goal_dist = None  # For goal progress reward
        self._entered_bay = False     # For bay entry bonus
        self.time_near_goal = 0

        # v42: Reset action smoothing state
        self.prev_action_norm = np.zeros(2, dtype=np.float32)

        # ----- Curriculum: choose lot, bay, spawn distance -----
        # v38.9 FIX: Sample scenario ONCE and reuse throughout reset()
        scenario = None
        if self.enable_curriculum and self.curriculum is not None:
            scenario = self.curriculum.sample_scenario()
            self.current_lot = scenario["lot"]

            # Reload config if lot changed
            if self.current_lot != self.lot_name:
                self._load_lot(self.current_lot)

            # Choose bay if not forced
            if bay_id is None:
                allowed_bays = scenario.get("allowed_bays")
                allowed_orientations = scenario.get("allowed_orientations")

                eligible_bays: List[str] = []
                if allowed_bays:
                    eligible_bays = [
                        b["id"] for b in self.bays if b["id"] in allowed_bays
                    ]
                elif allowed_orientations:
                    for bay in self.bays:
                        for yaw_allowed in allowed_orientations:
                            if abs(self._wrap_angle(bay["yaw"] - yaw_allowed)) < 0.1:
                                eligible_bays.append(bay["id"])
                                break
                else:
                    eligible_bays = [b["id"] for b in self.bays]

                if eligible_bays:
                    bay_id = self.random_state.choice(eligible_bays)
                else:
                    bay_id = self.random_state.choice([b["id"] for b in self.bays])

            # Optional: override spawn distance
            self.max_spawn_dist_override = scenario.get("max_spawn_dist", None)
            self.spawn_side_override = scenario.get("spawn_side", None)
            self.aligned_spawn_override = scenario.get("aligned_spawn", False)  # v41.2: Pass aligned flag
            self.lateral_offset_override = scenario.get("lateral_offset", None) # v41.2: Pass lateral offset
        else:
            # Simple multi-lot logic without curriculum
            if self.multi_lot:
                self.current_lot = self.random_state.choice(self.available_lots)
                if self.current_lot != self.lot_name:
                    self._load_lot(self.current_lot)
            self.max_spawn_dist_override = None

        # v41: Inject disable_obstacles into options if specified by curriculum
        if self.enable_curriculum and scenario is not None and scenario.get("disable_obstacles", False):
            if options is None:
                options = {}
            options["disable_obstacles"] = True

        # ----- Call parent reset (sets goal_bay, spawn, etc.) -----
        obs, info = super().reset(seed=seed, options=options, bay_id=bay_id)

        self.goal_x = self.goal_bay["x"]
        self.goal_y = self.goal_bay["y"]
        self.goal_yaw = self.goal_bay["yaw"]

        # v41.3: DISABLED - ParkingEnv now handles aligned spawn correctly via _reset_with_curriculum_spawn
        # This old v38.5 logic was overwriting the corrected spawn_yaw calculation
        # Keeping the block commented for reference
        
        # if self.enable_curriculum and self.curriculum is not None and scenario is not None:
        #     if scenario.get("aligned_spawn", False):
        #         spawn_dist = self.max_spawn_dist_override if self.max_spawn_dist_override else 12.0
        #         spawn_x = self.goal_x - spawn_dist * np.sin(self.goal_yaw)
        #         spawn_y = self.goal_y - spawn_dist * np.cos(self.goal_yaw)
        #         dx = self.goal_x - spawn_x
        #         dy = self.goal_y - spawn_y
        #         spawn_yaw = self.goal_yaw  # BUG: This was 0 when it should be 90
        #         lat_offset = scenario.get("lateral_offset")
        #         if lat_offset is not None:
        #             dx_lat = lat_offset * np.sin(spawn_yaw)
        #             dy_lat = -lat_offset * np.cos(spawn_yaw)
        #             spawn_x += dx_lat
        #             spawn_y += dy_lat
        #         self.state = np.array([spawn_x, spawn_y, spawn_yaw, 0.0], dtype=np.float32)  # OVERWRITES!


        # PATCH: Set timeout based on bay orientation difficulty
        yaw_norm = abs(self._wrap_angle(self.goal_yaw))
        if abs(yaw_norm - np.pi/2) < 0.3 or abs(yaw_norm + np.pi/2) < 0.3:
            # 90° or 270° bays are harder
            self.max_time_near_goal = 300  # ~15 seconds
        else:
            # 0° or 180° bays are easier
            self.max_time_near_goal = 250  # ~12.5 seconds

        start = (self.state[0], self.state[1], self.state[2])
        goal = (self.goal_x, self.goal_y, self.goal_yaw)

        if self.verbose:
            print(
                f"[WaypointEnv.reset] lot={self.current_lot}, "
                f"goal_bay={self.goal_bay.get('id', 'NA')}, "
                f"goal=({self.goal_x:.1f}, {self.goal_y:.1f}, "
                f"{np.degrees(self.goal_yaw):.1f}°), "
                f"timeout={self.max_time_near_goal}"
            )

        # v41.3: SIMPLE PATH for aligned spawns (S0/S1 baby parking)
        if self.enable_curriculum and scenario is not None and scenario.get("aligned_spawn", False):
            # For perfectly aligned spawns, generate a simple straight line
            # No A*, no guidance waypoints, no smoothing needed
            start_x, start_y, start_yaw = start
            
            # Create 4 simple waypoints: start -> 1/3 -> 2/3 -> goal
            self.waypoints = [
                (start_x, start_y, start_yaw),
                (start_x + (self.goal_x - start_x) * 0.33, 
                 start_y + (self.goal_y - start_y) * 0.33, 
                 start_yaw),
                (start_x + (self.goal_x - start_x) * 0.67, 
                 start_y + (self.goal_y - start_y) * 0.67, 
                 start_yaw),
                (self.goal_x, self.goal_y, start_yaw),
            ]
            
            self.full_path = self.waypoints  # No dense path needed
            self.total_path_length = np.hypot(self.goal_x - start_x, self.goal_y - start_y)
            self.reward_per_meter = self.REWARD_BUDGET / max(self.total_path_length, 1.0)
            self.current_waypoint_idx = 1
            
            # Dynamic timeout
            travel_steps = int((self.total_path_length / 0.5) / 0.1)
            self.max_steps = max(300, min(2000, travel_steps + 250))
            
            if self.verbose:
                print(
                    f"[WaypointEnv.reset] SIMPLE ALIGNED PATH: "
                    f"waypoints={len(self.waypoints)}, len={self.total_path_length:.1f}m, "
                    f"reward={self.reward_per_meter:.1f}/m, max_steps={self.max_steps}"
                )
            
            self.prev_dist_to_waypoint = None
            self._prev_goal_dist = None
            self._entered_bay = False
            
            # Skip the complex path planning below
        else:

            # ----- Plan path: start -> staging -> entrance -> pre-goal -> goal -----
            staging = self._create_staging_waypoint(goal)

            # Road-aware obstacle grid (uses finer 0.25m resolution now)
            obstacles = create_obstacle_grid(
                world_bounds=(-25, 25, -25, 25),
                resolution=0.25,  # Match planner resolution
                bays=self.occupied_bays,
                margin=1.0,
                roads=self.roads,
                goal_bay=self.goal_bay,
            )

            # ----- A* on road (with straight launch from the start pose) -----
            sx, sy, syaw = start

            # 1) Force a short straight segment along the car's current heading
            launch_dist = 2.0  # meters
            forward_wp = (
                sx + launch_dist * np.cos(syaw),
                sy + launch_dist * np.sin(syaw),
                syaw,
            )

            # 2) Plan from the forward launch point to the staging waypoint
            #    so A* never introduces a sideways move right at the car.
            road_path = self.planner.plan(forward_wp, staging, obstacles)
            if road_path is None or len(road_path) < 2:
                # Fall back: straight launch + staging
                road_path = [forward_wp, staging]

            # 3) Prepend the true start pose so the full path is:
            #    start -> forward_wp -> A* path -> (later) entrance/pregoal/goal
            road_path = [start] + road_path

            # ----- Generate Guidance Waypoints (Entrance -> Deep Bay) -----
            # Assuming goal_yaw points INTO the bay (towards back wall) based on dist_entrance logic
            # -y is OUT (aisle), +y is IN (deep bay)
            
            guidance_points = []
            
            # 1. Entrance Line (Start of bay)
            # Bay length 5.5m -> Half length 2.75m
            # Entrance is at -2.75m relative to center
            guidance_points.append({"name": "entrance", "dist": 2.75})
            
            # 2. Mid-Approach (Halfway to center)
            # Helps smooth the transition from entrance to goal
            guidance_points.append({"name": "mid_bay", "dist": 1.4})
            
            # 3. Goal (Center)
            guidance_points.append({"name": "goal", "dist": 0.0})
            
            # 4. Deep Target (3/4th depth) - REQUESTED BY USER
            # Pulls the agent fully into the bay.
            # 3/4th of 5.5m from entrance = 4.125m travel
            # Start at -2.75 -> End at -2.75 + 4.125 = +1.375m
            guidance_points.append({"name": "deep_target", "dist": -1.375})
            
            final_waypoints = []
            cos_yaw = np.cos(self.goal_yaw)
            sin_yaw = np.sin(self.goal_yaw)
            
            for pt in guidance_points:
                # Note: dist is "distance from goal OUTWARDS". 
                # So local_y = -dist.
                d = pt["dist"]
                local_x, local_y = 0.0, -d
                
                wx = self.goal_x + (cos_yaw * local_x - sin_yaw * local_y)
                wy = self.goal_y + (sin_yaw * local_x + cos_yaw * local_y)
                
                # CRITICAL FIX: Don't set heading yet - will be computed from path tangents
                # This prevents entrance waypoints from pointing INTO the bay prematurely
                wh = 0.0  # Placeholder, will be recomputed
                
                final_waypoints.append((wx, wy, wh))

            # Combine: Road Path -> Guidance Waypoints
            full_path = road_path + final_waypoints
            
            # CRITICAL FIX: Recompute orientations from path tangents
            # This ensures waypoints follow the actual path direction, not goal_yaw
            from ..planning.corridor import compute_path_tangents
            full_path = compute_path_tangents(full_path)

            # ----- Smooth path: B-spline (default) or Reeds-Shepp (optional) -----
            if self.use_reeds_shepp:
                from ..planning.smoothing import smooth_path_reeds_shepp
                smooth_dense = smooth_path_reeds_shepp(
                    full_path,
                    turning_radius=self.reeds_shepp_turning_radius,
                    step_size=0.5,
                )
            else:
                # v17: B-spline smoothing (default)
                smooth_dense = self._smooth_path(full_path)
            
            self.full_path = smooth_dense
            
            # v17: Intelligently subsample for RL agent
            self.waypoints = self._smart_subsample(smooth_dense)

            # v17: Calculate path length and reward scaling
            self.total_path_length = sum(
                np.linalg.norm([self.waypoints[i+1][0] - self.waypoints[i][0],
                               self.waypoints[i+1][1] - self.waypoints[i][1]])
                for i in range(len(self.waypoints) - 1)
            )
            
            # Fixed budget (e.g. 300) distributed over path length
            # This ensures fair rewards regardless of path length
            # self.REWARD_BUDGET = 300.0
            self.reward_per_meter = self.REWARD_BUDGET / max(self.total_path_length, 1.0)
            if self.verbose:
                print(
                    f"[WaypointEnv.reset] waypoints: dense={len(self.full_path)} -> sparse={len(self.waypoints)}, "
                    f"len={self.total_path_length:.1f}m, reward={self.reward_per_meter:.1f}/m"
                )

            # Initialize waypoint index (safe guard for short paths)
            self.current_waypoint_idx = 1 if len(self.waypoints) > 1 else 0

            # v17.2: Dynamic Episode Timeout
            # Calculate steps needed based on path length and conservative speed (0.5 m/s)
            # dt = 0.1s. Steps = (Dist / Speed) / dt
            travel_steps = int((self.total_path_length / 0.5) / 0.1)
            parking_buffer = 250 # Extra steps for final parking precision
            
            self.max_steps = travel_steps + parking_buffer
            
            # Safety bounds: Min 300 (for very short paths), Max 2000 (aligned with user config)
            self.max_steps = max(300, min(2000, self.max_steps))
            
            if self.verbose:
                print(f"[Timeout] Dynamic max_steps set to {self.max_steps} (Path: {self.total_path_length:.1f}m)")
            self.prev_dist_to_waypoint = None
            self._prev_goal_dist = None
            self._entered_bay = False

        return self._get_waypoint_obs(), info

    # ------------------------------------------------------------------ #
    # Helpers: staging waypoint and spline smoothing
    # ------------------------------------------------------------------ #

    def _calculate_turn_angle(self, p1, p2, p3):
        """Calculate angle between two path segments."""
        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0
        
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        return np.arccos(cos_angle)

        
    #     for i in range(1, len(dense_path) - 1):
    #         angle = self._calculate_turn_angle(
    #             dense_path[i-1], dense_path[i], dense_path[i+1]
    #         )
            
    #         dist = np.linalg.norm([
    #             dense_path[i][0] - selected[-1][0],
    #             dense_path[i][1] - selected[-1][1]
    #         ])
            
    #         # Keep if sharp turn (>15 deg) or sufficient distance
    #         if angle > np.radians(15) or dist >= min_spacing:
    #             selected.append(dense_path[i])
                
    #         # Cap at max_waypoints
    #         if len(selected) >= max_waypoints - 1:
    #             break
        
    #     selected.append(dense_path[-1])
    #     return selected
    
    def _smart_subsample(self, dense_path, min_spacing=1.2, max_waypoints=24):
        """
        Keep important waypoints (turns), remove redundant ones (straights).

        Tweaks:
        - Always keep the first TWO points (start + straight launch).
        - Slightly lower min_spacing so early bends get at least one waypoint.
        - Fewer total waypoints than the dense spline, but more than just 4–5
          so the curve looks smoother.
        """
        if len(dense_path) <= 3:
            return dense_path

        selected = [dense_path[0]]

        for i in range(1, len(dense_path) - 1):
            # Always keep the 2nd point (the launch waypoint)
            if i == 1:
                selected.append(dense_path[i])
                continue

            angle = self._calculate_turn_angle(
                dense_path[i - 1], dense_path[i], dense_path[i + 1]
            )

            dist = np.linalg.norm([
                dense_path[i][0] - selected[-1][0],
                dense_path[i][1] - selected[-1][1],
            ])

            # Keep if sharp turn (>15°) or we've moved far enough
            if angle > np.radians(15) or dist >= min_spacing:
                selected.append(dense_path[i])

            if len(selected) >= max_waypoints - 1:
                break

        # Always end exactly at the last point (goal)
        selected.append(dense_path[-1])
        return selected


    def _create_staging_waypoint(self, goal: Tuple[float, float, float]):
        """
        Place a staging waypoint on the road in front of the bay.
        
        This version uses the ORIGINAL orientation-based logic which was correct.
        """
        goal_x, goal_y, goal_theta = goal
        import math

        theta_norm = (goal_theta + math.pi) % (2 * math.pi) - math.pi

        # Facing south (down) - 0° bays
        if abs(theta_norm) < math.pi / 4:
            if goal_y > 10.0:  # lot_b horizontal bays
                staging_x, staging_y = goal_x, 10.0
            else:              # lot_a top row (A bays)
                staging_x, staging_y = goal_x, 0.0
            staging_theta = goal_theta

        # Facing north (up) - 180° bays  
        elif (
            abs(theta_norm - math.pi) < math.pi / 4
            or abs(theta_norm + math.pi) < math.pi / 4
        ):
            # lot_a bottom row (B bays)
            staging_x, staging_y = goal_x, 0.0
            staging_theta = goal_theta

        # Facing east (right) - 90° bays
        elif abs(theta_norm - math.pi / 2) < math.pi / 4:
            # lot_b vertical bays should use the main road at x=0
            if self.current_lot == "lot_b":
                staging_x = 0.0  # Main road in lot_b runs vertically at x=0
                staging_y = goal_y
            else:
                # Fallback for other lots or if road isn't at x=0
                staging_distance = 3.0
                staging_x = goal_x - staging_distance * np.cos(goal_theta)
                staging_y = goal_y - staging_distance * np.sin(goal_theta)
            staging_theta = goal_theta

        else:
            # Fallback: simple offset backwards along bay heading
            staging_distance = 3.0
            staging_x = goal_x - staging_distance * np.cos(goal_theta)
            staging_y = goal_y - staging_distance * np.sin(goal_theta)
            staging_theta = goal_theta

        return (staging_x, staging_y, staging_theta)

    # def _smooth_path(
    #     self, waypoints: List[Tuple[float, float, float]]
    # ) -> List[Tuple[float, float, float]]:
    #     """
    #     Smooth (x, y) with a B-spline and reconstruct yaw.

    #     PATCHES APPLIED:
    #     - Reduced smoothing parameter (s=0.1 instead of 2.0)
    #     - Shorter phantom distance (0.5m instead of 1.0m)
    #     - No oversampling (num_samples = len(waypoints) instead of 1.5x)
        
    #     Result: 20-40% shorter paths, fewer unnecessary curves.
    #     """
    #     if len(waypoints) < 3:
    #         return waypoints

    #     try:
    #         pts = np.array([[w[0], w[1]] for w in waypoints])
    #         goal_x, goal_y, goal_theta = waypoints[-1]

    #         # PATCH: Shorter phantom point (0.5m instead of 1.0m)
    #         phantom_dist = 0.0
    #         phantom_x = goal_x + phantom_dist * np.cos(goal_theta)
    #         phantom_y = goal_y + phantom_dist * np.sin(goal_theta)

    #         pts_with_phantom = np.vstack([pts, [[phantom_x, phantom_y]]])

    #         # PATCH: Reduced smoothing (s=0.1 instead of 2.0)
    #         tck, _ = splprep(
    #             [pts_with_phantom[:, 0], pts_with_phantom[:, 1]],
    #             s=0.1,  # Tighter fit to waypoints
    #             k=min(3, len(pts_with_phantom) - 1),
    #         )

    #         # PATCH: No oversampling (was int(len * 1.5))
    #         num_samples = len(waypoints)
    #         # u_max = len(pts) / len(pts_with_phantom)
    #         # u_new = np.linspace(0, u_max, num_samples)
    #         u_new = np.linspace(0, 1, num_samples)
    #         sx, sy = splev(u_new, tck)

    #         smooth: List[Tuple[float, float, float]] = []
    #         for i in range(len(sx)):
    #             if i < len(sx) - 1:
    #                 dx = sx[i + 1] - sx[i]
    #                 dy = sy[i + 1] - sy[i]
    #                 theta = np.arctan2(dy, dx)
    #             else:
    #                 theta = goal_theta
    #             smooth.append((sx[i], sy[i], theta))

    #         return smooth

    #     except Exception as e:
    #         print(f"[WaypointEnv] WARNING: spline failed ({e}), using raw waypoints")
    #         return waypoints
    
    # def _smooth_path(
    #     self, waypoints: List[Tuple[float, float, float]]
    # ) -> List[Tuple[float, float, float]]:
    #     """
    #     Smooth (x, y) with a B-spline and reconstruct yaw.

    #     PATCHES APPLIED:
    #     - Reduced smoothing parameter (s=0.1 instead of 2.0)
    #     - No oversampling (num_samples = len(waypoints) instead of 1.5x)
    #     - No phantom end-point (spline goes exactly from start to goal)

    #     Result: shorter, smoother paths that still end exactly at the bay center.
    #     """
    #     if len(waypoints) < 3:
    #         return waypoints

    #     try:
    #         pts = np.array([[w[0], w[1]] for w in waypoints])
    #         goal_x, goal_y, goal_theta = waypoints[-1]

    #         # Fit spline directly through the original points (no phantom)
    #         tck, _ = splprep(
    #             [pts[:, 0], pts[:, 1]],
    #             s=0.1,  # tight fit
    #             k=min(3, len(pts) - 1),
    #         )

    #         # Same number of samples as original waypoints,
    #         # but now over the full parameter range [0, 1]
    #         num_samples = len(waypoints)
    #         u_new = np.linspace(0.0, 1.0, num_samples)
    #         sx, sy = splev(u_new, tck)

    #         smooth: List[Tuple[float, float, float]] = []
    #         for i in range(len(sx)):
    #             if i < len(sx) - 1:
    #                 dx = sx[i + 1] - sx[i]
    #                 dy = sy[i + 1] - sy[i]
    #                 theta = np.arctan2(dy, dx)
    #             else:
    #                 # Force final orientation to match bay yaw
    #                 theta = goal_theta
    #             smooth.append((sx[i], sy[i], theta))

    #         return smooth

    #     except Exception as e:
    #         print(f"[WaypointEnv] WARNING: spline failed ({e}), using raw waypoints")
    #         return waypoints

    # ❌ OLD: Embedded _smooth_path method (lines 587-636) - now using modular smoothing
    # def _smooth_path(self, waypoints):
    #     """Smooth (x, y) with a B-spline... [OLD CODE COMMENTED OUT]"""
    #     ...scipy splprep/splev code...
    
    def _smooth_path(
        self, waypoints: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Smooth path using modular B-spline smoother from planning.smoothing module.
        
        ✅ NEW: Uses smooth_path_bspline() from planning/smoothing.py
        - Same algorithm as before (s=0.1, densification 3x)
        - Now centralized for reuse across project
        
        Fallback: Returns original waypoints if smoothing fails
        """
        return smooth_path_bspline(
            waypoints=waypoints,
            smoothness=0.1,  # Tight fit (same as original)
            densification_factor=3.0,  # 3x denser sampling (same as original)
            validate_collision_free=False,  # Disable for now (can enable later)
            obstacles=None,  # Would need to pass obstacle grid
            world_to_grid_fn=None,  # Would need to pass planner.world_to_grid
        )
    
    # ❌ OLD EMBEDDED CODE (COMMENTED OUT - using modular version above):
    # def _smooth_path(
    #     self, waypoints: List[Tuple[float, float, float]]
    # ) -> List[Tuple[float, float, float]]:
    #     """
    #     Smooth (x, y) with a B-spline and reconstruct yaw.

    #     PATCHES APPLIED:
    #     - Reduced smoothing parameter (s=0.1 instead of 2.0)
    #     - No oversampling (num_samples = len(waypoints) instead of 1.5x)
    #     - No phantom end-point (spline goes exactly from start to goal)

    #     Result: shorter, smoother paths that still end exactly at the bay center.
    #     """
    #     if len(waypoints) < 3:
    #         return waypoints

    #     try:
    #         pts = np.array([[w[0], w[1]] for w in waypoints])
    #         goal_x, goal_y, goal_theta = waypoints[-1]

    #         # Fit spline directly through the original points (no phantom)
    #         tck, _ = splprep(
    #             [pts[:, 0], pts[:, 1]],
    #             s=0.1,  # tight fit
    #             k=min(3, len(pts) - 1),
    #         )

    #         # Same number of samples as original waypoints,
    #         # but now over the full parameter range [0, 1]
    #         # Denser sampling for a visually smoother curve
    #         num_samples = max(len(waypoints) * 3, len(waypoints) + 2)
    #         u_new = np.linspace(0.0, 1.0, num_samples)
    #         sx, sy = splev(u_new, tck)

    #         smooth: List[Tuple[float, float, float]] = []
    #         for i in range(len(sx)):
    #             if i < len(sx) - 1:
    #                 dx = sx[i + 1] - sx[i]
    #                 dy = sy[i + 1] - sy[i]
    #                 theta = np.arctan2(dy, dx)
    #             else:
    #                 # Force final orientation to match bay yaw
    #                 theta = goal_theta
    #             smooth.append((sx[i], sy[i], theta))

    #         return smooth

    #     except Exception as e:
    #         print(f"[WaypointEnv] WARNING: spline failed ({e}), using raw waypoints")
    #         return waypoints

    # ------------------------------------------------------------------ #
    # Observation: waypoint + goal bay info (98D)
    # ------------------------------------------------------------------ #

    def _get_waypoint_obs(self) -> np.ndarray:
        """
        Observation with both waypoint and goal-bay context.

        Layout (98D):
            Core (12):
              [local_dx_wp, local_dy_wp, dtheta_wp, v, dist_to_wp,
               cx_bay, cy_bay, yaw_err_goal, dist_to_goal,
               waypoint_progress, is_near_goal, goal_side]
            Bay geometry (16):
              8 bay-frame points (corners + edge midpoints),
              each as (x_car, y_car) in car frame.
            Lookahead waypoints (6):
              2 next waypoints, each as (local_dx, local_dy, dtheta).
            Lidar (64):
              lidar_0 .. lidar_63
        """
        x, y, yaw, v = self.state

        # Current waypoint (or goal fallback)
        if self.current_waypoint_idx < len(self.waypoints):
            wx, wy, wtheta = self.waypoints[self.current_waypoint_idx]
        else:
            wx, wy, wtheta = self.goal_x, self.goal_y, self.goal_yaw

        # Waypoint-relative
        dx_wp = wx - x
        dy_wp = wy - y
        dist_to_wp = np.sqrt(dx_wp**2 + dy_wp**2)
        dtheta_wp = self._wrap_angle(wtheta - yaw)

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        local_dx_wp = cos_yaw * dx_wp + sin_yaw * dy_wp
        local_dy_wp = -sin_yaw * dx_wp + cos_yaw * dy_wp

        # Goal-bay-relative (car center)
        center_x = x + (self.car_length / 2.0) * np.cos(yaw)
        center_y = y + (self.car_length / 2.0) * np.sin(yaw)

        dx_goal = self.goal_x - center_x
        dy_goal = self.goal_y - center_y
        dist_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)

        cos_goal = np.cos(self.goal_yaw)
        sin_goal = np.sin(self.goal_yaw)
        cx_bay = cos_goal * dx_goal + sin_goal * dy_goal
        cy_bay = -sin_goal * dx_goal + cos_goal * dy_goal
        yaw_err_goal = self._wrap_angle(yaw - self.goal_yaw)

        # Progress indicators
        waypoint_progress = (
            self.current_waypoint_idx / max(len(self.waypoints), 1)
        )
        is_near_goal = 1.0 if dist_to_goal < 5.0 else 0.0

        # Lidar from parent obs (ensure fixed length to avoid shape mismatches)
        parent_obs = super()._get_obs()
        lidar_raw = parent_obs[5:]  # [local_x, local_y, yaw_err, v, dist, lidar...]

        MAX_LIDAR = 64  # FIXED: was 32, now matches actual lidar resolution
        lidar = np.zeros(MAX_LIDAR, dtype=np.float32)
        n = min(MAX_LIDAR, lidar_raw.shape[0])
        lidar[:n] = lidar_raw[:n]


        # Normalize all observations to similar scales for better learning
        # Positions/distances: normalize to [-1, 1] or [0, 1]
        # Angles: normalize to [-1, 1] by dividing by π
        # Velocities: normalize to [-1, 1] by dividing by max_speed
        
        # v29: LOOKAHEAD WAYPOINTS
        # Get next 2 waypoints for path context
        next_wps = []
        for i in range(1, 3):  # +1, +2
            idx = self.current_waypoint_idx + i
            if idx < len(self.waypoints):
                nwx, nwy, nwtheta = self.waypoints[idx]
            else:
                # Fallback to goal if no more waypoints
                nwx, nwy, nwtheta = self.goal_x, self.goal_y, self.goal_yaw
                
            # Relative to car
            ndx = nwx - x
            ndy = nwy - y
            n_local_dx = cos_yaw * ndx + sin_yaw * ndy
            n_local_dy = -sin_yaw * ndx + cos_yaw * ndy
            n_dtheta = self._wrap_angle(nwtheta - yaw)
            
            # FIXED: Use world bounds for normalization (was / 20.0 with clip)
            max_dist = 25.0
            next_wps.extend([
                n_local_dx / max_dist,
                n_local_dy / max_dist,
                n_dtheta / np.pi
            ])
        
        # v32: GOAL SIDE INDICATOR
        # Explicit signal for which side the goal is on
        goal_side = 1.0 if self.goal_y > 0 else -1.0  # +1 = A-row (top), -1 = B-row (bottom)
        
        # v32: 8-POINT BAY REFERENCE SYSTEM
        # Instead of just bay center, give agent all 8 key points:
        # 4 corners + 4 edge midpoints for precise spatial awareness
        
        # Bay dimensions (v41: use actual env parameters for consistency)
        bay_width = self.bay_width
        bay_length = self.bay_length
        half_w = bay_width / 2.0
        half_l = bay_length / 2.0
        
        # 8 points in bay frame (relative to bay center, bay orientation)
        # Corners: TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left)
        # Midpoints: T, R, B, L (top, right, bottom, left)
        bay_points_local = [
            (-half_l, half_w),   # Top-left corner
            (-half_l, -half_w),  # Top-right corner  
            (half_l, -half_w),   # Bottom-right corner
            (half_l, half_w),    # Bottom-left corner
            (-half_l, 0),        # Top edge midpoint
            (0, -half_w),        # Right edge midpoint
            (half_l, 0),         # Bottom edge midpoint
            (0, half_w),         # Left edge midpoint
        ]
        
        # Transform to car frame
        bay_points_car = []
        for bx_local, by_local in bay_points_local:
            # Bay frame to world frame
            bx_world = self.goal_x + bx_local * cos_goal - by_local * sin_goal
            by_world = self.goal_y + bx_local * sin_goal + by_local * cos_goal
            
            # World frame to car frame
            dx = bx_world - center_x
            dy = by_world - center_y
            bx_car = cos_yaw * dx + sin_yaw * dy
            by_car = -sin_yaw * dx + cos_yaw * dy
            
            # FIXED: Use world bounds (was / 10.0 with clip)
            max_dist = 25.0
            bay_points_car.extend([
                bx_car / max_dist,
                by_car / max_dist,
            ])
        
        # FIXED: Use world bounds for normalization (was / 10.0 with clip)
        max_dist = 25.0
        return np.array(
            [
                local_dx_wp / max_dist,      # Normalize local waypoint position
                local_dy_wp / max_dist,
                dtheta_wp / np.pi,                        # Normalize angle
                v / max(self.max_speed, 1e-6),           # Normalize velocity using env limit
                np.clip(dist_to_wp / 20.0, 0, 1),        # Normalize distance
                np.clip(cx_bay / 10.0, -1, 1),           # Normalize bay-frame x (center)
                np.clip(cy_bay / 10.0, -1, 1),           # Normalize bay-frame y (center)
                yaw_err_goal / np.pi,                     # Normalize goal yaw error
                np.clip(dist_to_goal / 20.0, 0, 1),      # Normalize goal distance
                waypoint_progress,                        # Already [0, 1]
                is_near_goal,                             # Already binary {0, 1}
                goal_side,                                # v32: +1 (A-row) or -1 (B-row)
                *bay_points_car,                          # v32: 8 points x 2 coords = 16 dims
                *next_wps,                                # v29: Next 2 waypoints (6 dims)
                *np.clip(lidar / 20.0, 0.0, 1.0),        # v41: Normalized lidar (0-1 range)
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Waypoint threshold
    # ------------------------------------------------------------------ #

    def get_progressive_threshold(self, waypoint_idx: int) -> float:
        """
        All intermediate waypoints loose (4m), final one tighter (1m).
        """
        total = len(self.waypoints)
        if waypoint_idx < total - 1:
            return 4.0
        return 1.0
    
    def get_recent_success_rate(self) -> float:
        """Get success rate over recent episode window."""
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)

    # ------------------------------------------------------------------ #
    # Step: blended reward (navigation → approach → parking)
    # ------------------------------------------------------------------ #

    def step(self, action):
        """
        Fully modular step function using rewards.py exclusively.
        
        Environment responsibilities:
        - Compute state geometry (distances, angles, bay frame)
        - Track waypoint progression
        - Detect termination conditions (success, stuck, OOB)
        - Call reward calculators
        
        Reward responsibilities (delegated to rewards.py):
        - All continuous shaping (navigation, velocity, parking)
        - All discrete bonuses (waypoints, bay entry, goal progress)
        - Curriculum-adaptive thresholds
        """
        # v40 FIX: Action mapping [steer, accel] -> [v, steer]
        # WaypointEnv action: [steering_norm, accel_norm]
        # ParkingEnv expects: [v_cmd, steer_cmd]

        # v42: Exponential Action Smoothing (Simulated Input Lag)
        if not hasattr(self, 'prev_action_norm'):
            self.prev_action_norm = np.zeros(2, dtype=np.float32)
        
        alpha = 0.7  # Smoothing factor (0.0=full smooth, 1.0=raw input)
        current_action = np.array(action, dtype=np.float32)
        smoothed_action = alpha * current_action + (1.0 - alpha) * self.prev_action_norm
        self.prev_action_norm = smoothed_action
        
        # 1. Unpack smoothed normalized action
        steer_norm = float(smoothed_action[0])
        accel_norm = float(smoothed_action[1])
        
        # 2. Map to physical limits
        steer_cmd = steer_norm * self.max_steer
        # v_cmd logic: simplified direct mapping for now
        v_cmd = accel_norm * self.max_speed
        
        # 3. Call physics step with correct order [v, steer]
        physics_action = np.array([v_cmd, steer_cmd], dtype=np.float32)
        obs, physics_reward, terminated, truncated, info = super().step(physics_action)
        self.current_obs = obs  # v40: Capture for lidar reward logic
        self._episode_steps += 1

        x, y, yaw, v = self.state

        # ===== Goal geometry & bay frame =====
        goal_x, goal_y, goal_theta = self.goal_x, self.goal_y, self.goal_yaw

        center_x = x + (self.car_length / 2.0) * np.cos(yaw)
        center_y = y + (self.car_length / 2.0) * np.sin(yaw)

        dx_goal = goal_x - center_x
        dy_goal = goal_y - center_y
        dist_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)

        cos_goal = np.cos(goal_theta)
        sin_goal = np.sin(goal_theta)
        cx_bay = cos_goal * dx_goal + sin_goal * dy_goal
        cy_bay = -sin_goal * dx_goal + cos_goal * dy_goal
        yaw_err = abs(self._wrap_angle(yaw - goal_theta))

        # ===== Current waypoint distance =====
        if self.current_waypoint_idx < len(self.waypoints):
            wx, wy, _ = self.waypoints[self.current_waypoint_idx]
        else:
            # Fallback: treat goal as final waypoint
            wx, wy, _ = self.goal_x, self.goal_y, self.goal_yaw
        
        dist_to_wp_raw = np.linalg.norm([x - wx, y - wy])
        # Normalize to [0,1] to match old behavior (dist / 20m, clipped)
        dist_to_wp_norm = np.clip(dist_to_wp_raw / 20.0, 0.0, 1.0)

        # ===== MODULAR REWARD CALCULATION =====
        reward_terms = {}  # Track components for debugging
        # v41.3: Ignore base env scalar reward; keep only physics events via info flags.
        # reward_terms["physics_base"] = physics_reward
        
        # 1. Navigation reward (distance, progress, velocity, anti-freeze)
        # v38.7: FIXED - use RAW distance, not normalized (reward was tuned for meters)
        nav_reward = self.waypoint_rewards.calculate_navigation_reward(
            dist_to_waypoint=dist_to_wp_raw,   # Use RAW distance in meters
            prev_dist_to_waypoint=self.prev_dist_to_waypoint,
            velocity=v,
            dist_to_goal=dist_to_goal,
            yaw_error=yaw_err,
            y_position=y,
            goal_y=self.goal_y,
        )
        reward_terms["navigation"] = nav_reward
        self.prev_dist_to_waypoint = dist_to_wp_raw  # Track RAW distance

        # 2. Goal progress reward (separate from waypoint)
        goal_progress_reward = calculate_goal_progress_reward(
            dist_to_goal=dist_to_goal,
            prev_dist_to_goal=self._prev_goal_dist,
            weight=1.0,
        )
        reward_terms["goal_progress"] = goal_progress_reward
        self._prev_goal_dist = dist_to_goal

        # 3. Bay entry bonus (one-time)
        bay_bonus, self._entered_bay = calculate_bay_entry_bonus(
            cx_bay=cx_bay,
            cy_bay=cy_bay,
            bay_length=self.parking_rewards.bay_length,
            bay_width=self.parking_rewards.bay_width,
            entered_bay_flag=self._entered_bay,
            bonus_amount=self.bay_entry_bonus_val,  # v38.8: Tuned param
        )
        reward_terms["bay_entry"] = bay_bonus

        # 4. Phase-blended parking reward
        # v38.7: DISABLED - was causing reward explosion, Gaussian reward is sufficient
        blended_reward = 0.0
        reward_terms["parking_blend"] = blended_reward

        # 5. Time penalty (environment-specific)
        time_penalty = -0.05
        reward_terms["time_penalty"] = time_penalty

        # ===== Waypoint progression & bonus =====
        waypoint_bonus = 0.0
        if self.current_waypoint_idx < len(self.waypoints):
            thr = self.get_progressive_threshold(self.current_waypoint_idx)

            if dist_to_wp_raw < thr:  # Use raw distance for threshold check
                # Calculate bonus BEFORE incrementing index
                if 0 < self.current_waypoint_idx < len(self.waypoints):
                    prev_wp = self.waypoints[self.current_waypoint_idx - 1]
                    curr_wp = self.waypoints[self.current_waypoint_idx]
                    segment_len = np.linalg.norm([
                        curr_wp[0] - prev_wp[0],
                        curr_wp[1] - prev_wp[1]
                    ])
                    
                    progress_ratio = self.current_waypoint_idx / max(len(self.waypoints), 1)
                    
                    # ✅ MODULAR: Use reward calculator
                    waypoint_bonus = self.waypoint_rewards.calculate_waypoint_bonus(
                        segment_length=segment_len,
                        reward_per_meter=self.reward_per_meter,
                        progress_ratio=progress_ratio,
                    )

                # Move to next waypoint
                self.current_waypoint_idx += 1
                
                if self.verbose and self.current_waypoint_idx < len(self.waypoints):
                    print(
                        f"✓ Waypoint {self.current_waypoint_idx}/"
                        f"{len(self.waypoints) - 1} (bonus {waypoint_bonus:.1f})"
                    )

        reward_terms["waypoint_bonus"] = waypoint_bonus

        # ===== Get curriculum thresholds (needed for continuous parking reward) =====
        success = info.get("success", False)
        
        if self.enable_curriculum and self.curriculum is not None:
            stage = getattr(self.curriculum, "current_stage_idx", None)
        else:
            stage = None

        success_cy, success_yaw = get_curriculum_thresholds(
            curriculum_stage=stage,
            enable_curriculum=self.enable_curriculum,
        )

        # ===== Continuous parking alignment reward (CRITICAL FIX) =====
        # v24: CONTINUOUS PARKING REWARD (proximity-scaled)
        # Get steering angle from action for straight entry penalty
        # v38.9 FIX: action[0] is steering, action[1] is acceleration
        # v38.9 FIX: Use physical steering command computed above
        steering_angle = steer_cmd
        
        continuous_parking_reward, alignment_score = \
            self.parking_rewards.calculate_continuous_parking_reward(
                cx_bay=cx_bay,
                cy_bay=cy_bay,
                yaw_err=yaw_err,
                dist_to_goal=dist_to_goal,
                success_cy=success_cy,
                success_yaw=success_yaw,
                steering_angle=steering_angle,  # NEW: For straight entry penalty
            )
        reward_terms["parking_continuous"] = continuous_parking_reward
        
        # v33: PHASED PARKING REWARD
        # Guide car through natural parking phases
        phased_reward, parking_phase = self.parking_rewards.calculate_phased_parking_reward(
            car_pos=(x, y),
            car_yaw=yaw,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            goal_yaw=self.goal_yaw,
            dist_to_goal=dist_to_goal,
        )
        reward_terms["phased_parking"] = phased_reward
        
        # Store phase for debugging
        info["parking_phase"] = parking_phase

        # v31: PATH DEVIATION PENALTY
        # Continuous feedback to stay on the path line
        path_deviation_penalty = self.waypoint_rewards.calculate_path_deviation_penalty(
            current_pos=(x, y),
            waypoints=self.waypoints,
            current_wp_idx=self.current_waypoint_idx,
        )
        reward_terms["path_deviation"] = path_deviation_penalty
        
        # v35: FIXED CORRIDOR CONSTRAINT (perpendicular distance + car corners)
        # CRITICAL FIX: Now uses perpendicular distance to path segment, not point distance
        # Checks ALL 4 car corners, not just center
        # v35.1: REDUCED penalty weight (was 0.5, now 0.05) - too harsh for untrained agent
        corridor_constraint, corridor_violation = self.waypoint_rewards.calculate_corridor_constraint_reward(
            car_x=x,
            car_y=y,
            car_yaw=yaw,  # NEW: needed for corner positions
            waypoints=self.waypoints,
            current_wp_idx=self.current_waypoint_idx,  # NEW: use current segment
            goal_bay=self.goal_bay,
            corridor_width=4.0,  # v36.1: WIDENED to 1.5x bay width (was 2.2m, too tight)
            penalty_weight=self.corridor_penalty_weight,  # v38.8: Tuned param
            car_length=self.car_length,
            car_width=getattr(self, 'car_width', 1.9),
        )
        reward_terms["corridor_constraint"] = corridor_constraint
        
        # v38.6: DISABLE corridor termination temporarily
        # The 100x penalty is still applied, but episode doesn't end
        # Agent needs to learn to MOVE first, then we can enable termination
        # if corridor_violation:
        #     terminated = True
        #     info["terminated_corridor_violation"] = True
        
        # ====================================================================
        # v36: BULLETPROOF OFF-PATH DETECTION
        # Simple, guaranteed-to-work check: distance to nearest waypoint
        # Catches cases where agent wanders 5-10m off-path
        # ====================================================================
        min_dist_to_path = min(
            np.hypot(x - wp[0], y - wp[1]) 
            for wp in self.waypoints
        )
        info["min_dist_to_path"] = min_dist_to_path
        
        # GRADUATED PENALTIES (v36.1 ADJUSTED for 4.0m corridor):
        # 0-3.0m: No penalty (inside wider corridor)
        # 3.0-10.0m: Growing penalty  
        # >10.0m: TERMINATE (hopelessly lost)
        # v38.7: Relaxed thresholds for early training
        if min_dist_to_path > 4.0:  # Start penalty at 4m (was 3m)
            excess = min_dist_to_path - 4.0
            off_path_penalty = -2.0 * (excess ** 2)  # Reduced from -5.0
            reward_terms["off_path_penalty"] = off_path_penalty
        
        if min_dist_to_path > 10.0:  # Increased from 6m to 10m
            # TERMINATE - car is hopelessly lost
            terminated = True
            info["terminated_off_path"] = True
            reward_terms["off_path_termination"] = -50.0
            if self.verbose:
                print(
                    f"⚠️ OFF PATH TERMINATION: {min_dist_to_path:.1f}m "
                    f"from nearest waypoint at step {self._episode_steps}"
                )

        # Settling reward removed in v24 (ineffective)
        # reward_terms["settling"] = ...

        # ===== Sum all reward components =====
        reward = sum(reward_terms.values())

        # ===== Stuck detection (environment logic) =====
        if dist_to_goal < self.best_dist_to_goal - 0.05:
            self.best_dist_to_goal = dist_to_goal
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        if dist_to_goal < 5.0:
            self.time_near_goal += 1
        else:
            self.time_near_goal = 0

        # ===== Success Alignment Check =====
        well_aligned = (
            abs(cy_bay) < success_cy
            and yaw_err < success_yaw
            and abs(cx_bay) < 3.0
        )

        # CRITICAL FIX: Add velocity check to ensure agent stops (not just passes through)
        # BUT: Make it optional if agent is VERY close and aligned (to prevent infinite episodes)
        is_stopped = abs(v) < 0.3  # m/s (~1 km/h)
        very_close = dist_to_goal < 1.0  # Within 1m
        
        # Success if: (aligned AND close AND stopped) OR (aligned AND VERY close)
        if well_aligned and ((dist_to_goal < 2.0 and is_stopped) or very_close):
            # v38: SUCCESS BONUS (SCALED: was 10.0, now 50.0)
            # Must be significantly larger than the final waypoint bonus (~5)
            success_bonus = self.parking_rewards.success_bonus
            reward += success_bonus
            reward_terms["success_bonus"] = success_bonus
            success = True
            terminated = True
            info["success"] = True
            self._last_success = True
            info["parking_quality"] = {
                "lateral_offset": float(abs(cy_bay)),
                "yaw_error": float(yaw_err),
                "depth": float(abs(cx_bay)),
            }
            if self.verbose:
                print(
                    f"✅ PARK SUCCESS: cy={abs(cy_bay):.3f}m, "
                    f"yaw_err={yaw_err:.3f}rad ({np.degrees(yaw_err):.1f}°), "
                    f"cx={abs(cx_bay):.3f}m"
                )

        # ===== Termination conditions (environment-specific) =====
        # Stuck: no progress
        if self.no_progress_steps > 100 and dist_to_goal < 10.0:
            stuck_penalty = -50.0
            reward += stuck_penalty
            reward_terms["stuck_penalty"] = stuck_penalty
            terminated = True
            info["terminated_stuck"] = True
            if self.verbose:
                print(
                    f"⚠️  STUCK(no_progress): {self.no_progress_steps} steps, "
                    f"dist={dist_to_goal:.2f}m"
                )

        # Stuck: near goal timeout
        if self.time_near_goal >= self.max_time_near_goal and not success:
            stuck_near_penalty = -50.0
            reward += stuck_near_penalty
            reward_terms["stuck_near_penalty"] = stuck_near_penalty
            terminated = True
            info["terminated_stuck_near_goal"] = True
            if self.verbose:
                print(
                    f"⚠️  STUCK(near_goal): {self.time_near_goal}/{self.max_time_near_goal} steps, "
                    f"dist={dist_to_goal:.2f}m"
                )
        
        # Check collision from parent step (it's in the info dict, not an attribute)
        # The parent class step() returns collision in info["collision"]
        collision = info.get("collision", False)
        oob = abs(x) > 25.0 or abs(y) > 25.0

        # Collision
        if collision:
            collision_penalty = -50.0  # v41.3: Primary collision penalty (base env penalty ignored)
            reward += collision_penalty
            reward_terms["collision"] = collision_penalty
            terminated = True
            info["terminated_collision"] = True

        # Out of bounds
        if oob:
            oob_penalty = -0.5 # SCALED: was -20.0
            reward += oob_penalty
            reward_terms["oob_penalty"] = oob_penalty
            terminated = True
            info["terminated_oob"] = True

        # Timeout
        done = terminated or truncated
        if self.steps >= self.max_steps:
            truncated = True
            done = True

        # ===== Observation =====
        obs = self._get_waypoint_obs()

        # ===== Store metrics =====
        info["waypoint_idx"] = self.current_waypoint_idx
        info["total_waypoints"] = len(self.waypoints)
        info["dist_to_goal"] = dist_to_goal
        info["reward_terms"] = reward_terms  # For debugging/analysis

        # Set failure flag if terminated without success
        if (terminated or truncated) and not info.get("success", False):
            self._last_success = False

        # v38.7: EP_METRICS logging for hyperparameter tuning
        if done and self.verbose:
            path_completion = self.current_waypoint_idx / max(len(self.waypoints) - 1, 1)
            coll = 1 if info.get("terminated_collision", False) else 0
            offpath = 1 if info.get("terminated_off_path", False) else 0
            print(f"[EP_METRICS] rew={reward:.1f} path={path_completion:.2f} "
                  f"succ={int(success)} dist={dist_to_goal:.2f} coll={coll} offpath={offpath}")

        # ===== Curriculum update =====
        if (
            done
            and self.enable_curriculum
            and self.curriculum is not None
        ):
            self.curriculum.update_after_episode(
                success=bool(success),
                steps=self._episode_steps,
            )

        # 6. Smoothness Reward (v40)
        # Penalize jerky steering to encourage smooth trajectories
        # v40 FIX: Use physical steering command, not raw action[1] (which is accel)
        current_steer = steer_cmd
        steer_diff = abs(current_steer - self.prev_steer)
        # self.max_steer is in radians (approx 0.78 for 45 deg)
        # Normalize diff by max range (2*max_steer) roughly
        smoothness_penalty = -1.0 * steer_diff  # Tunable weight
        reward_terms["smoothness"] = smoothness_penalty
        self.prev_steer = current_steer

        # 7. Near-Obstacle Penalty (v40)
        # Continuous penalty for being too close to obstacles (safety buffer)
        # info["lidar"] contains 64 rays. Min distance is closest obstacle.
        # ParkingEnv.step() computes lidar but doesn't return it in info by default?
        # It lives in obs. Let's extract from observer state or similar.
        # Actually ParkingEnv computes it in _get_obs().
        # We can re-access self.lidar.last_scan if available or re-compute min.
        # For efficiency, let's look at 'obs'.
        # obs structure: [local_x, local_y, yaw_err, v, dist, lidar_0 ... lidar_63]
        # Lidar starts at index 5 (after 5 state vars)
        lidar_data = self.current_obs[5:] # v40 FIX: Correct index (was 34)
        min_lidar_dist = np.min(lidar_data)
        
        # Buffer zone: 0.5m
        if min_lidar_dist < 0.5:
            # Exponential penalty as we get closer
            # at 0.5m -> 0
            # at 0.1m -> -High
            obstacle_penalty = -5.0 * (0.5 - min_lidar_dist)
        else:
            obstacle_penalty = 0.0
        reward_terms["obstacle_safety"] = obstacle_penalty

        # 8. Continuous Bay Overlap Reward (v40)
        # Replaces simple "entry bonus" with dense IoU-like progress
        # Calculate intersection area between Car and Goal Bay
        car_poly = self._get_car_polygon(x, y, yaw)
        # Goal bay polygon (static, could cache but cheap to compute)
        goal_poly = self._get_bay_polygon(self.goal_bay)
        
        intersection_area = car_poly.intersection(goal_poly).area
        car_area = self.car_length * self.car_width
        overlap_ratio = intersection_area / car_area
        
        # Scale reward: Max overlap is ~1.0. 
        # Weighted to be significant but not overpowering.
        # Say max +5.0 per step? Or is this a bonus?
        # If it's per step, it accumulates huge.
        # Should be a POTENTIAL-based reward (diff from prev) OR a small density.
        # Let's make it a dense reward for *being* in the bay.
        overlap_reward = 2.0 * overlap_ratio * overlap_ratio # quadratic for peak centering
        reward_terms["bay_overlap"] = overlap_reward

        # Sum up new v40 rewards
        reward += smoothness_penalty + obstacle_penalty + overlap_reward

        # Update info with v40 metrics
        info["reward_terms"] = reward_terms
        info["overlap_ratio"] = overlap_ratio

        # v40: Respect ParkingEnv collision termination
        # (It already set terminated=True and processed crash penalty in super().step())
        # We just ensure we don't accidentally overwrite it with False if we are "in bay"
        # But usually discrete logic sets terminated=True on success.
        # If crashed, we are done.
        
        # Final reward composition
        return obs, reward, terminated, truncated, info

    def _get_car_polygon(self, x, y, yaw):
        """Get shapely Polygon for car, using geometric center."""
        # v41 FIX: Convert rear-axle (x,y) -> geometric center
        center_x = x + (self.car_length / 2.0) * np.cos(yaw)
        center_y = y + (self.car_length / 2.0) * np.sin(yaw)

        l2, w2 = self.car_length / 2.0, self.car_width / 2.0
        corners = [
            ( l2,  w2),
            ( l2, -w2),
            (-l2, -w2),
            (-l2,  w2),
        ]
        c, s = np.cos(yaw), np.sin(yaw)
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * c - cy * s + center_x
            ry = cx * s + cy * c + center_y
            rotated_corners.append((rx, ry))
        return Polygon(rotated_corners)

    def _get_bay_polygon(self, bay):
        """Get shapely Polygon for parking bay."""
        x, y, yaw = bay['x'], bay['y'], bay['yaw']
        l2 = self.bay_length / 2.0
        w2 = self.bay_width / 2.0
        corners = [
            ( l2,  w2),
            ( l2, -w2),
            (-l2, -w2),
            (-l2,  w2),
        ]
        c, s = np.cos(yaw), np.sin(yaw)
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * c - cy * s + x
            ry = cx * s + cy * c + y
            rotated_corners.append((rx, ry))
        return Polygon(rotated_corners)
    
    def render(self):
        """
        Override parent render to add v34 corridor/bay overlays for training videos.
        """
        # Call parent render first
        result = super().render()
        
        # Add v34 overlays if we have an axis and waypoints
        if hasattr(self, 'ax') and self.ax is not None and hasattr(self, 'waypoints') and self.waypoints is not None and len(self.waypoints) > 0:
            try:
                from ..planning.corridor import (
                    compute_path_tangents,
                    calculate_corridor_boundaries,
                    calculate_8_point_bay_reference
                )
                
                # CRITICAL FIX: Clear previous overlays to prevent piling up
                # Remove all lines and scatter plots that were added by previous render calls
                # Keep only the base environment elements (parking lot, car, obstacles)
                artists_to_remove = []
                for artist in self.ax.lines + self.ax.collections:
                    # Check if this is an overlay (has specific properties we set)
                    if hasattr(artist, '_v34_overlay'):
                        artists_to_remove.append(artist)
                for artist in artists_to_remove:
                    artist.remove()
                
                # 1. Draw corridor boundaries (red dashed lines)
                waypoints_corrected = compute_path_tangents(self.waypoints)
                left_boundary, right_boundary = calculate_corridor_boundaries(
                    waypoints_corrected, self.goal_bay, corridor_width=4.0  # v36.1: Widened corridor
                )
                
                if left_boundary and len(left_boundary) > 0:
                    left_x, left_y = zip(*left_boundary)
                    line_left = self.ax.plot(left_x, left_y, 'r--', linewidth=1.5, alpha=0.5, zorder=2)[0]
                    line_left._v34_overlay = True  # Mark for removal next frame
                
                if right_boundary and len(right_boundary) > 0:
                    right_x, right_y = zip(*right_boundary)
                    line_right = self.ax.plot(right_x, right_y, 'r--', linewidth=1.5, alpha=0.5, zorder=2)[0]
                    line_right._v34_overlay = True  # Mark for removal next frame
                
                # 2. Draw 8-point bay reference (cyan dots)
                bay_points = calculate_8_point_bay_reference(self.goal_bay)
                if bay_points and len(bay_points) > 0:
                    cyan_x, cyan_y = zip(*bay_points)
                    scatter_bay = self.ax.scatter(cyan_x, cyan_y, c='cyan', s=15, zorder=22, 
                                  edgecolors='black', linewidth=0.5, alpha=0.7)
                    scatter_bay._v34_overlay = True  # Mark for removal next frame
                
                # 3. Draw waypoint path (yellow dots)
                wps = np.array(self.waypoints)
                scatter_wps = self.ax.scatter(wps[:, 0], wps[:, 1], c='yellow', s=20, zorder=20,
                              edgecolors='black', linewidth=0.5, alpha=0.7)
                scatter_wps._v34_overlay = True  # Mark for removal next frame
                
                # Redraw canvas with overlays
                if hasattr(self, 'fig') and self.fig is not None:
                    self.fig.canvas.draw()
            
            except Exception as e:
                # Silently fail if v34 visualization fails (don't break training)
                pass
        
        return result