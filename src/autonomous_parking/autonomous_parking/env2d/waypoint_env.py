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
from scipy.interpolate import splprep, splev
from gymnasium import spaces
from .parking_env import ParkingEnv
from ..planning.astar import AStarPlanner, create_obstacle_grid
from ..curriculum import CurriculumManager


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
        **kwargs,
    ):
        """
        Args:
            lot_name: Default lot if multi_lot is False.
            multi_lot: If True, randomly pick from ["lot_a", "lot_b"].
            enable_curriculum: If True, use CurriculumManager for staged difficulty.
            verbose: If False, suppress per-episode / per-step prints (for long runs).
            **kwargs: Passed through to ParkingEnv.
        """
        super().__init__(lot_name, **kwargs)
        self.verbose = verbose
        self.episode_id: int = 0     

        # ==== NEW: Override observation_space to match 43-D waypoint obs ====
        # Layout (43D):
        #   [local_dx_wp, local_dy_wp, dtheta_wp, v, dist_to_wp,      # 5
        #    cx_bay, cy_bay, yaw_err_goal, dist_to_goal,              # 4
        #    waypoint_progress, is_near_goal,                         # 2
        #    lidar_0..lidar_31]                                       # 32
        #
        # All the first 11 values are normalized to [-1, 1] or [0, 1],
        # lidar comes from parent and lives in [0, 10].
        obs_low = np.concatenate([
            np.full(11, -1.0, dtype=np.float32),         # conservative lower bound
            np.zeros(32, dtype=np.float32),              # lidar >= 0
        ])
        obs_high = np.concatenate([
            np.ones(11, dtype=np.float32),               # conservative upper bound
            np.full(32, 10.0, dtype=np.float32),         # lidar max range
        ])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        # ==== END NEW BLOCK ====

        # Lot selection
        self.multi_lot = multi_lot
        self.available_lots = ["lot_a", "lot_b"] if multi_lot else [lot_name]
        self.current_lot = lot_name

        # Curriculum
        self.enable_curriculum = enable_curriculum
        self.curriculum: Optional[CurriculumManager] = (
            CurriculumManager() if enable_curriculum else None
        )
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
        
        self._episode_steps = 0
        self.best_dist_to_goal = float("inf")
        self.no_progress_steps = 0
        self.prev_dist_to_waypoint = None
        self.time_near_goal = 0

        # ----- Curriculum: choose lot, bay, spawn distance -----
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
        else:
            # Simple multi-lot logic without curriculum
            if self.multi_lot:
                self.current_lot = self.random_state.choice(self.available_lots)
                if self.current_lot != self.lot_name:
                    self._load_lot(self.current_lot)
            self.max_spawn_dist_override = None

        # ----- Call parent reset (sets goal_bay, spawn, etc.) -----
        obs, info = super().reset(seed=seed, options=options, bay_id=bay_id)

        self.goal_x = self.goal_bay["x"]
        self.goal_y = self.goal_bay["y"]
        self.goal_yaw = self.goal_bay["yaw"]

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


        cos_yaw = np.cos(self.goal_yaw)
        sin_yaw = np.sin(self.goal_yaw)

        # Entrance 2m outside bay
        dist_entrance = 2.0
        ex_local_x, ex_local_y = 0.0, -dist_entrance
        entrance_x = self.goal_x + (cos_yaw * ex_local_x - sin_yaw * ex_local_y)
        entrance_y = self.goal_y + (sin_yaw * ex_local_x + cos_yaw * ex_local_y)
        entrance = (entrance_x, entrance_y, self.goal_yaw)

        # Pre-goal 1m outside bay
        dist_pregoal = 1.0
        pg_local_x, pg_local_y = 0.0, -dist_pregoal
        pregoal_x = self.goal_x + (cos_yaw * pg_local_x - sin_yaw * pg_local_y)
        pregoal_y = self.goal_y + (sin_yaw * pg_local_x + cos_yaw * pg_local_y)
        pregoal = (pregoal_x, pregoal_y, self.goal_yaw)

        full_path = road_path + [entrance, pregoal, goal]

        # ----- Smooth path with B-spline (PATCHED for straighter paths) -----
        # v17: Generate dense smooth path first
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

    # def _smart_subsample(self, dense_path, min_spacing=2.5, max_waypoints=18):
    #     """Keep important waypoints (turns), remove redundant ones (straights)."""
    #     if len(dense_path) <= 3:
    #         return dense_path
        
    #     selected = [dense_path[0]]
        
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

    def _smooth_path(
        self, waypoints: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        Smooth (x, y) with a B-spline and reconstruct yaw.

        PATCHES APPLIED:
        - Reduced smoothing parameter (s=0.1 instead of 2.0)
        - No oversampling (num_samples = len(waypoints) instead of 1.5x)
        - No phantom end-point (spline goes exactly from start to goal)

        Result: shorter, smoother paths that still end exactly at the bay center.
        """
        if len(waypoints) < 3:
            return waypoints

        try:
            pts = np.array([[w[0], w[1]] for w in waypoints])
            goal_x, goal_y, goal_theta = waypoints[-1]

            # Fit spline directly through the original points (no phantom)
            tck, _ = splprep(
                [pts[:, 0], pts[:, 1]],
                s=0.1,  # tight fit
                k=min(3, len(pts) - 1),
            )

            # Same number of samples as original waypoints,
            # but now over the full parameter range [0, 1]
            # Denser sampling for a visually smoother curve
            num_samples = max(len(waypoints) * 3, len(waypoints) + 2)
            u_new = np.linspace(0.0, 1.0, num_samples)
            sx, sy = splev(u_new, tck)

            smooth: List[Tuple[float, float, float]] = []
            for i in range(len(sx)):
                if i < len(sx) - 1:
                    dx = sx[i + 1] - sx[i]
                    dy = sy[i + 1] - sy[i]
                    theta = np.arctan2(dy, dx)
                else:
                    # Force final orientation to match bay yaw
                    theta = goal_theta
                smooth.append((sx[i], sy[i], theta))

            return smooth

        except Exception as e:
            print(f"[WaypointEnv] WARNING: spline failed ({e}), using raw waypoints")
            return waypoints

    # ------------------------------------------------------------------ #
    # Observation: waypoint + goal bay info (43D)
    # ------------------------------------------------------------------ #

    def _get_waypoint_obs(self) -> np.ndarray:
        """
        Observation with both waypoint and goal-bay context.

        Layout (43D):
            [local_dx_wp, local_dy_wp, dtheta_wp, v, dist_to_wp,      # 5
             cx_bay, cy_bay, yaw_err_goal, dist_to_goal,              # 4
             waypoint_progress, is_near_goal,                          # 2
             lidar_0..lidar_31]                                        # 32
        """
        x, y, yaw, v = self.state

        # Current waypoint (or goal fallback)
        if self.verbose and self.current_waypoint_idx < len(self.waypoints):
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

        MAX_LIDAR = 32
        lidar = np.zeros(MAX_LIDAR, dtype=np.float32)
        n = min(MAX_LIDAR, lidar_raw.shape[0])
        lidar[:n] = lidar_raw[:n]


        # Normalize all observations to similar scales for better learning
        # Positions/distances: normalize to [-1, 1] or [0, 1]
        # Angles: normalize to [-1, 1] by dividing by π
        # Velocities: normalize to [-1, 1] by dividing by max_speed
        return np.array(
            [
                np.clip(local_dx_wp / 10.0, -1, 1),      # Normalize local waypoint position
                np.clip(local_dy_wp / 10.0, -1, 1),
                dtheta_wp / np.pi,                        # Normalize angle
                v / max(self.max_speed, 1e-6),           # Normalize velocity using env limit
                np.clip(dist_to_wp / 20.0, 0, 1),        # Normalize distance
                np.clip(cx_bay / 10.0, -1, 1),           # Normalize bay-frame x
                np.clip(cy_bay / 10.0, -1, 1),           # Normalize bay-frame y
                yaw_err_goal / np.pi,                     # Normalize goal yaw error
                np.clip(dist_to_goal / 20.0, 0, 1),      # Normalize goal distance
                waypoint_progress,                        # Already [0, 1]
                is_near_goal,                             # Already binary {0, 1}
                *lidar,                                   # Lidar already normalized in parent
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
        1) ParkingEnv dynamics + safety.
        2) Waypoint progression + distance-normalized bonus.
        3) Blended phase reward (navigation → approach → parking).
        4) Dual stuck detection (no progress + time near goal).
        5) Adaptive success threshold based on curriculum stage.
        6) Curriculum update at end.
        """
        _, _, terminated, truncated, info = super().step(action)
        self._episode_steps += 1

        x, y, yaw, v = self.state

        # ===== Waypoint progression =====
        waypoint_bonus = 0.0
        if self.current_waypoint_idx < len(self.waypoints):
            wx, wy, _ = self.waypoints[self.current_waypoint_idx]
            dist_to_wp = np.linalg.norm([x - wx, y - wy])
            thr = self.get_progressive_threshold(self.current_waypoint_idx)

            if dist_to_wp < thr:
                # v17: Distance-normalized reward with progressive multiplier
                # Calculate bonus BEFORE incrementing index to avoid out-of-bounds
                waypoint_bonus = 0.0
                
                if 0 < self.current_waypoint_idx < len(self.waypoints):
                    prev_wp = self.waypoints[self.current_waypoint_idx - 1]
                    curr_wp = self.waypoints[self.current_waypoint_idx]
                    segment_len = np.linalg.norm([
                        curr_wp[0] - prev_wp[0],
                        curr_wp[1] - prev_wp[1]
                    ])
                    
                    # Progressive multiplier (1.0 -> 1.5) to encourage finishing
                    progress_ratio = self.current_waypoint_idx / max(len(self.waypoints), 1)
                    multiplier = 1.0 + 0.5 * progress_ratio
                    
                    waypoint_bonus = segment_len * self.reward_per_meter * multiplier

                # Now move to next waypoint index
                self.current_waypoint_idx += 1
                
                if self.current_waypoint_idx < len(self.waypoints):
                    print(
                        f"✓ Waypoint {self.current_waypoint_idx}/"
                        f"{len(self.waypoints) - 1} (bonus {waypoint_bonus:.1f})"
                    )

        # Observation
        obs = self._get_waypoint_obs()
        dist_to_target = float(obs[4])

        # Base reward: distance to current waypoint
        reward = -0.2 * dist_to_target

        # Progress reward
        if self.prev_dist_to_waypoint is not None:
            progress = self.prev_dist_to_waypoint - dist_to_target
            reward += 2.0 * progress
        self.prev_dist_to_waypoint = dist_to_target

        # Velocity shaping
        if v > 0.1:
            reward += 0.3 * min(v, 2.0)
        if v < 0.2:
            reward -= 0.2

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

        # Stuck detection: track best distance
        if dist_to_goal < self.best_dist_to_goal - 0.05:
            self.best_dist_to_goal = dist_to_goal
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        # Stuck detection: cumulative time near goal
        if dist_to_goal < 5.0:
            self.time_near_goal += 1
        else:
            self.time_near_goal = 0

        # PATCH: Phase-aware anti-freeze (only when truly far from goal AND not aligning)
        if dist_to_goal > 5.0 and v < 0.1 and yaw_err > 0.5:
            reward -= 0.3  # Reduced from -1.0 (was too harsh)

        # ===== Phase-blended reward (PATCHED: balanced scaling) =====
        w_nav = float(np.clip((dist_to_goal - 2.0) / 3.0, 0.0, 1.0))
        w_park = float(np.clip((2.0 - dist_to_goal) / 2.0, 0.0, 1.0))
        w_approach = 1.0 - max(w_nav, w_park)

        # Navigation: go closer
        nav_term = -0.15 * dist_to_goal

        # Approach: distance + yaw alignment
        approach_term = -0.2 * dist_to_goal - 0.8 * yaw_err

        # Parking: bay-frame alignment
        target_depth = 2.0
        depth_err = abs(abs(cx_bay) - target_depth)
        park_term = -0.8 * abs(cy_bay) - 0.6 * yaw_err - 0.3 * depth_err

        reward += w_nav * nav_term
        reward += w_approach * approach_term
        reward += w_park * park_term

        # Time penalty
        reward -= 0.05

        # Waypoint bonus
        reward += waypoint_bonus

        # ===== Success condition (PATCH: curriculum-adaptive thresholds) =====
        success = info.get("success", False)

        # Determine thresholds based on curriculum stage
        if self.enable_curriculum and self.curriculum is not None:
            # current_stage = getattr(self.curriculum, 'current_stage', 0)
            # if current_stage < 3:
            #     success_cy = 1.5   # 1.5m lateral - very forgiving
            #     success_yaw = 0.5  # ~28° - allows rough alignment
            # elif current_stage < 6:
            #     success_cy = 0.8   # 80cm - moderate
            #     success_yaw = 0.3  # ~17° - tighter
            # else:
            #     success_cy = 0.15  # 15cm - your v14.20 precision
            #     success_yaw = 0.1  # ~5.7° - tight alignment
            current_stage = getattr(self.curriculum, 'current_stage_idx', 0)
            if not isinstance(current_stage, int):
                current_stage = int(getattr(self.curriculum, 'current_stage_idx', 0))

            if current_stage < 3:
                success_cy = 1.5
                success_yaw = 0.5
            elif current_stage < 6:
                success_cy = 0.8
                success_yaw = 0.3
            else:
                success_cy = 0.15
                success_yaw = 0.1

        else:
            # No curriculum: use moderate defaults
            success_cy = 0.5   # 50cm
            success_yaw = 0.2  # ~11°

        well_aligned = (
            abs(cy_bay) < success_cy
            and yaw_err < success_yaw
            and abs(cx_bay) < 3.0
        )

        if well_aligned and dist_to_goal < 2.0:
            reward += 500.0  # Reduced from 2000 for better reward balance
            success = True
            terminated = True
            info["success"] = True
            self._last_success = True  # Store for next reset
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

        # ===== Termination conditions =====
        # Stuck: no progress (v17.3: Reduced penalty from -200 to -50)
        if self.no_progress_steps > 60 and dist_to_goal < 10.0:
            reward -= 50.0  # Gentler penalty to allow learning final parking
            terminated = True
            info["terminated_stuck"] = True
            if self.verbose:
                print(
                    f"⚠️  STUCK(no_progress): {self.no_progress_steps} steps, "
                    f"dist={dist_to_goal:.2f}m"
                )

        # PATCH: Stuck near goal (extended timeout, orientation-adaptive)
        # v17.3: Reduced penalty from -200 to -50
        if self.time_near_goal >= self.max_time_near_goal and not success:
            reward -= 50.0  # Gentler penalty to allow learning final parking
            terminated = True
            info["terminated_stuck_near_goal"] = True
            if self.verbose:
                print(
                    f"⚠️  STUCK(near_goal): {self.time_near_goal}/{self.max_time_near_goal} steps, "
                    f"dist={dist_to_goal:.2f}m"
                )

        # Out of bounds
        if abs(x) > 25.0 or abs(y) > 25.0:
            reward -= 20.0
            terminated = True
            info["terminated_oob"] = True

        # Timeout
        done = terminated or truncated
        if self.steps >= self.max_steps:
            truncated = True
            done = True

        # Store metrics
        info["waypoint_idx"] = self.current_waypoint_idx
        info["total_waypoints"] = len(self.waypoints)
        info["dist_to_goal"] = dist_to_goal

        # Set failure flag if terminated without success
        if (terminated or truncated) and not info.get("success", False):
            self._last_success = False

        # Curriculum update at episode end
        if (
            done
            and self.enable_curriculum
            and self.curriculum is not None
        ):
            self.curriculum.update_after_episode(
                success=bool(success),
                steps=self._episode_steps,
            )

        return obs, reward, terminated, truncated, info