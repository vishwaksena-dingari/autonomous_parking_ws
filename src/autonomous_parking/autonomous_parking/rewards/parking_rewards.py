"""
Reward Calculation Module for Autonomous Parking

Centralized reward functions for hierarchical RL parking agent.
Supports continuous reward shaping for PPO optimization.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List


# ============================================================================
# Waypoint-Following Rewards (Hierarchical RL - Low Level)
# ============================================================================

class WaypointRewardCalculator:
    """
    Calculates rewards for waypoint-following behavior.
    
    Encourages:
    - Progress along A* path
    - Smooth velocity control
    - Waypoint reaching with distance-normalized bonuses
    """
    
    def __init__(
        self,
        velocity_reward_weight: float = 0.05,   # v38.5: INCREASED 10x (was 0.005) to fix oscillation
        low_velocity_penalty: float = 0.01,     # SCALED: was 1.0
        anti_freeze_penalty: float = 0.02,      # SCALED: was 2.0
    ):
        """
        Waypoint-based reward calculator.
        
        Args:
            velocity_reward_weight: Weight for velocity reward
            low_velocity_penalty: Penalty for being too slow
            anti_freeze_penalty: Penalty for freezing in place
        """
        self.velocity_reward_weight = velocity_reward_weight
        self.low_velocity_penalty = low_velocity_penalty
        self.anti_freeze_penalty = anti_freeze_penalty
    
    def calculate_waypoint_bonus(
        self,
        segment_length: float,
        reward_per_meter: float,
        progress_ratio: float,
    ) -> float:
        """
        Distance-normalized waypoint reaching bonus with progressive multiplier.
        
        Args:
            segment_length: Distance of current segment (meters)
            reward_per_meter: reward_budget / total_path_length
            progress_ratio: Current waypoint index / total waypoints
            
        Returns:
            Bonus reward for reaching waypoint
        """
        # v29: MASSIVE bonus for final waypoints to create "irresistible pull"
        # Early waypoints: 1.0x - 1.5x (normal)
        # Final 30%: Exponentially increases up to 10x
        if progress_ratio > 0.7:
            # 0.7 -> 1.0x
            # 0.8 -> 2.0x
            # 0.9 -> 5.0x
            # 1.0 -> 10.0x
            multiplier = 1.0 + 9.0 * ((progress_ratio - 0.7) / 0.3) ** 2
        else:
            multiplier = 1.0 + 0.5 * (progress_ratio / 0.7)
        return segment_length * reward_per_meter * multiplier
    
    def calculate_navigation_reward(
        self,
        dist_to_waypoint: float,
        prev_dist_to_waypoint: Optional[float],
        velocity: float,
        dist_to_goal: float,
        yaw_error: float,
        y_position: float = 0.0,
        goal_y: float = 0.0,  # v30.1: Added to determine target side
    ) -> float:
        """
        Combined navigation reward for waypoint following.
        """
        reward = 0.0
        
        # Base penalty: distance to waypoint (SCALED: was -0.2)
        reward += -0.002 * dist_to_waypoint
        
        # # Progress reward: getting closer to waypoint (v38.5: INCREASED 5x, was 0.02)
        # if prev_dist_to_waypoint is not None:
        #     progress = prev_dist_to_waypoint - dist_to_waypoint
        #     reward += 0.1 * progress
        if prev_dist_to_waypoint is not None:
            progress = prev_dist_to_waypoint - dist_to_waypoint
            if progress > 0 and velocity > 0.1:
                reward += 0.1 * progress
        
        # v28: PATH-FOLLOWING reward (replaces generic forward velocity)
        # Reward velocity component ALONG the path to waypoint, not just "forward"
        # This encourages following the path shape, not just driving straight
        # if prev_dist_to_waypoint is not None and dist_to_waypoint > 0.01:
        #     # Calculate path-aligned velocity component
        #     # If moving towards waypoint: positive reward
        #     # If moving away: negative reward (implicit via progress term above)
        #     path_velocity = (prev_dist_to_waypoint - dist_to_waypoint) / 0.1  # dt = 0.1s
        #     if path_velocity > 0.1:  # Moving towards waypoint
        #         reward += self.velocity_reward_weight * min(path_velocity, 2.0)
        
        if prev_dist_to_waypoint is not None and dist_to_waypoint > 0.01:
            path_velocity = (prev_dist_to_waypoint - dist_to_waypoint) / 0.1
            if path_velocity > 0.1 and velocity > 0.1:
                reward += self.velocity_reward_weight * min(path_velocity, 2.0)
            elif path_velocity > 0.1 and velocity < -0.1:
                reward -= 1.0

        # FIX: Only penalize low SPEED when FAR from goal (> 3.0m)
        if abs(velocity) < 0.2 and dist_to_goal > 3.0:
            reward -= self.low_velocity_penalty
            
        # v30.1: SMART ROAD-STAYING PENALTY
        # Only penalize going to the WRONG side of the road.
        # Allow entering the correct bay area.
        if dist_to_goal > 3.0:
            # If target is Top (y > 0), penalize Bottom (y < -3.0) (SCALED: was 1.0)
            if goal_y > 0 and y_position < -3.0:
                reward -= 0.01 * (abs(y_position) - 3.0)
            # If target is Bottom (y < 0), penalize Top (y > 3.0) (SCALED: was 1.0)
            elif goal_y < 0 and y_position > 3.0:
                reward -= 0.01 * (abs(y_position) - 3.0)
        
        # v38.7: Anti-freeze penalty (balanced scale)
        # Penalty for not moving when far from goal
        # v38.9 FIX: Use configurable self.anti_freeze_penalty instead of hardcoded 0.02
        # if dist_to_goal > 3.0 and velocity < 0.3:
        #     reward -= self.anti_freeze_penalty  # Small but consistent push to move
        if dist_to_goal > 3.0 and abs(velocity) < 0.3:
            reward -= self.anti_freeze_penalty
        
        # v42: BACKWARD MOTION PENALTY (CRITICAL FIX)
        if velocity < -0.1:
            backward_penalty = -2.0 * abs(velocity)
            reward += backward_penalty
        elif velocity < 0.0:
            reward -= 0.5

        return reward
    
    def calculate_path_deviation_penalty(
        self,
        current_pos: Tuple[float, float],
        waypoints: List[Tuple[float, float, float]],
        current_wp_idx: int,
    ) -> float:
        """
        v31: PATH DEVIATION PENALTY
        Penalize perpendicular distance from the current path segment.
        
        This provides continuous feedback to keep the agent on the path line,
        not just rewarding waypoint arrival.
        
        Args:
            current_pos: (x, y) current position
            waypoints: List of (x, y, theta) waypoints
            current_wp_idx: Index of current target waypoint
            
        Returns:
            Penalty (negative reward) for deviating from path
        """
        if current_wp_idx >= len(waypoints) or current_wp_idx < 1:
            return 0.0
        
        # Get current segment: previous waypoint -> current waypoint
        p1 = waypoints[current_wp_idx - 1][:2]  # (x, y)
        p2 = waypoints[current_wp_idx][:2]
        
        x, y = current_pos
        x1, y1 = p1
        x2, y2 = p2
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx*dx + dy*dy
        
        if length_sq < 1e-6:
            # Degenerate segment (waypoints too close), no penalty
            return 0.0
        
        # Project current position onto line segment
        # t=0 at p1, t=1 at p2
        t = ((x - x1) * dx + (y - y1) * dy) / length_sq
        t = max(0.0, min(1.0, t))  # Clamp to segment
        
        # Closest point on segment
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # Perpendicular distance
        perp_dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
        
        # Penalty: -0.01 per meter of deviation (SCALED: was -1.0)
        # This is strong enough to guide but not dominate waypoint bonuses
        penalty = -0.01 * perp_dist
        
        return penalty
    
    
    def _get_perpendicular_distance_to_segment(
        self,
        point: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Calculate perpendicular distance from point to line segment p1->p2.
        Returns (perp_distance, signed_distance) where positive = left of path.
        """
        x, y = point
        x1, y1 = p1
        x2, y2 = p2
        
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx*dx + dy*dy
        
        if length_sq < 1e-6:
            dist = np.hypot(x - x1, y - y1)
            return dist, dist
        
        # Project point onto line segment
        t = ((x - x1) * dx + (y - y1) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # Perpendicular distance
        perp_dist = np.hypot(x - proj_x, y - proj_y)
        
        # Signed distance (positive = left of path, negative = right)
        cross = dx * (y - y1) - dy * (x - x1)
        signed_dist = perp_dist if cross > 0 else -perp_dist
        
        return perp_dist, signed_dist
    
    def _get_car_corners(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        car_length: float = 4.5,
        car_width: float = 1.9,
    ) -> List[Tuple[float, float]]:
        """Get the 4 corners of the car in world coordinates.
        
        v41 FIX: car_x, car_y are rear-axle coordinates (from state).
        We convert to geometric center before computing corners.
        """
        # v41: Convert rear-axle to geometric center
        center_x = car_x + (car_length / 2.0) * np.cos(car_yaw)
        center_y = car_y + (car_length / 2.0) * np.sin(car_yaw)
        
        half_l = car_length / 2.0
        half_w = car_width / 2.0
        
        cos_y = np.cos(car_yaw)
        sin_y = np.sin(car_yaw)
        
        # Corners in local frame: [front_left, front_right, rear_right, rear_left]
        corners_local = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, half_w),
        ]
        
        # Transform to world frame (from center, not rear-axle)
        corners_world = []
        for lx, ly in corners_local:
            wx = center_x + lx * cos_y - ly * sin_y
            wy = center_y + lx * sin_y + ly * cos_y
            corners_world.append((wx, wy))
        
        return corners_world
    
    def calculate_corridor_constraint_reward(
        self,
        car_x: float,
        car_y: float,
        waypoints: List[Tuple[float, float, float]],
        goal_bay: Dict,
        corridor_width: float = 3.0,
        penalty_weight: float = 1.0,
        car_yaw: float = 0.0,  # NEW: needed for corner checking
        current_wp_idx: int = 1,  # NEW: use current waypoint segment
        car_length: float = 4.5,
        car_width: float = 1.9,
    ) -> Tuple[float, bool]:
        """
        v35: FIXED corridor constraint using perpendicular distance and car corners.
        
        CRITICAL FIXES:
        1. Uses perpendicular distance to path SEGMENT (not point distance)
        2. Checks ALL 4 car corners (not just center)
        3. Asymmetric: outer = 10x harsh, inner = 1x soft
        4. Terminates if ANY corner > 2m outside
        """
        # Need at least 2 waypoints to define a segment
        if current_wp_idx < 1 or current_wp_idx >= len(waypoints):
            return 0.0, False
        
        # Get current path segment
        p1 = waypoints[current_wp_idx - 1][:2]
        p2 = waypoints[current_wp_idx][:2]
        
        # Get car corners
        corners = self._get_car_corners(car_x, car_y, car_yaw, car_length, car_width)
        
        corridor_half_width = corridor_width / 2.0
        total_penalty = 0.0
        should_terminate = False
        
        for corner in corners:
            perp_dist, signed_dist = self._get_perpendicular_distance_to_segment(corner, p1, p2)
            
            # Check if outside corridor
            if perp_dist > corridor_half_width:
                violation_dist = perp_dist - corridor_half_width
                
                # v38: STRICT ASYMMETRIC PENALTY
                # Positive signed_dist = LEFT of path (typically outer/oncoming lane)
                # Negative signed_dist = RIGHT of path (typically inner/cutting corner)
                if signed_dist > 0:
                    # OUTER BOUNDARY: Strong penalty (10x) - was 100x, too harsh
                    # v38.6: Reduced from 100x to 10x to prevent reward explosion
                    corner_penalty = -penalty_weight * 10.0 * violation_dist
                    total_penalty += corner_penalty
                    
                    # TERMINATE immediately if ANY outer violation > 0.5m
                    if violation_dist > 0.5:
                        should_terminate = True
                else:
                    # INNER BOUNDARY: NO PENALTY (0x) - allow corner cutting
                    # This is intentional - agent can cut corners to optimize path
                    corner_penalty = 0.0
        
        return total_penalty, should_terminate


# ============================================================================
# Parking Alignment Rewards (Final Maneuver)
# ============================================================================

class ParkingRewardCalculator:
    """
    Calculates continuous rewards for parking alignment.
    
    Provides smooth gradient for PPO learning instead of binary success/fail.
    """
    
    def __init__(
        self,
        bay_length: float = 5.5,
        bay_width: float = 2.7,
        alignment_reward_weight: float = 50.0,  # v38: REDUCED from 300.0 to balance with penalties
        success_bonus: float = 50.0,  # v38: REDUCED from 200.0 to balance with penalties
    ):
        """
        Args:
            bay_length: Parking bay depth (meters)
            bay_width: Parking bay width (meters)
            alignment_reward_weight: Max continuous reward for perfect alignment
            success_bonus: Additional bonus when success threshold met
        """
        self.bay_length = bay_length
        self.bay_width = bay_width
        self.alignment_reward_weight = alignment_reward_weight
        self.success_bonus = success_bonus
    
    def calculate_continuous_parking_reward(
        self,
        cx_bay: float,
        cy_bay: float,
        yaw_err: float,
        dist_to_goal: float,
        success_cy: float,  # Unused in Gaussian formulation, kept for API compatibility
        success_yaw: float, # Unused in Gaussian formulation
        steering_angle: float = 0.0,
    ) -> Tuple[float, float]:
        """
        v37: GAUSSIAN (EXPONENTIAL) PARKING REWARD
        
        Replaces linear thresholds with smooth exponential kernels (Gaussian).
        Reward = W * exp(-d_lat²/σ_lat) * exp(-d_long²/σ_long) * exp(-θ²/σ_theta)
        
        Benefits:
        1. No "dead zones" (gradient exists everywhere).
        2. Exponentially stronger signal near 0 error (high precision).
        3. Multiplication ensures agent fixes ALL errors (cannot trade off).
        """
        
        # 1. LATERAL (cy): The most critical component.
        # sigma=0.5 means ~60% reward at 0.5m error, ~13% at 1.0m.
        # Very steep gradient pulling into the center line.
        lat_reward = np.exp(- (cy_bay ** 2) / (2 * 0.5 ** 2))
        
        # 2. HEADING (yaw): 
        # sigma=0.3 rad (~17 deg). 
        # ~60% reward at 17 deg error. Rapid drop-off after that.
        # This explicitly enforces the "Angle should become 0" rule.
        yaw_reward = np.exp(- (yaw_err ** 2) / (2 * 0.3 ** 2))
        
        # 3. LONGITUDINAL (cx) / DEPTH:
        # We want the agent to go deep into the bay.
        # sigma=1.0m (gentler slope to encourage moving in).
        long_reward = np.exp(- (cx_bay ** 2) / (2 * 1.0 ** 2))
        
        # 4. ALIGNMENT SCORE (0.0 to 1.0)
        # Multiply them: if ANY part is bad, the score drops significantly.
        # This forces the agent to align position AND angle simultaneously.
        alignment_score = lat_reward * yaw_reward * long_reward
        
        # 5. SCALE BY DISTANCE (The "Attractor")
        # Standard exponential decay for distance from goal.
        # Reward is high only if: Alignment is good AND Distance is small.
        dist_factor = np.exp(- (dist_to_goal ** 2) / (2 * 4.0 ** 2))
        
        # Final calculation
        # Weight = 50.0 (v38: reduced from 300.0 to balance with penalties)
        continuous_reward = self.alignment_reward_weight * alignment_score * dist_factor
        
        # --- BONUS: Deep Target Incentive ---
        # If perfectly aligned laterally and angularly, give extra push to go deep
        if lat_reward > 0.8 and yaw_reward > 0.8:
            # Add a linear push to drive to the back of the bay
            # cx_bay is negative when going deep.
            continuous_reward += 10.0 * long_reward

        return continuous_reward, alignment_score
    
    def calculate_phased_parking_reward(
        self,
        car_pos: Tuple[float, float],
        car_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        dist_to_goal: float,
    ) -> Tuple[float, str]:
        """
        v33.1: DETAILED 8-POINT PHASED PARKING REWARD
        
        Uses ALL 8 reference points to guide car through parking:
        
        Phase 1 (APPROACH): Attract to top 3 points (TL, T, TR)
          - Reward getting close to top edge
          - Reward lateral alignment (car centered between TL and TR)
          - Simultaneous attraction + alignment
          
        Phase 2 (ALIGN): Use all 8 points for orientation
          - Reward when left/right midpoints are level (L.y ≈ R.y in car frame)
          - Reward when top/bottom are aligned (T.x ≈ B.x in car frame)
          - This ensures car is parallel to bay
          
        Phase 3 (ENTER): Guide using side midpoints (L, R)
          - Reward moving forward while keeping L.y ≈ R.y (staying centered)
          - Reward reducing distance to center
          
        Phase 4 (SETTLE): Use all 4 corners for final positioning
          - Reward when all corners are equidistant
          - Reward precise center alignment
        """
        if dist_to_goal > 8.0:
            return 0.0, "TOO_FAR"
        
        # Bay dimensions (v41: match ParkingEnv values)
        bay_width = 2.7
        bay_length = 5.5
        half_w = bay_width / 2.0
        half_l = bay_length / 2.0
        
        # Calculate all 8 points in world frame
        cos_g = np.cos(goal_yaw)
        sin_g = np.sin(goal_yaw)
        
        def bay_to_world(bx, by):
            wx = goal_x + bx * cos_g - by * sin_g
            wy = goal_y + bx * sin_g + by * cos_g
            return wx, wy
        
        def world_to_car(wx, wy):
            dx = wx - car_pos[0]
            dy = wy - car_pos[1]
            cos_c = np.cos(car_yaw)
            sin_c = np.sin(car_yaw)
            cx = cos_c * dx + sin_c * dy
            cy = -sin_c * dx + cos_c * dy
            return cx, cy
        
        # 8 points in car frame
        TL_w = bay_to_world(-half_l, half_w)
        TR_w = bay_to_world(-half_l, -half_w)
        BR_w = bay_to_world(half_l, -half_w)
        BL_w = bay_to_world(half_l, half_w)
        T_w = bay_to_world(-half_l, 0)
        R_w = bay_to_world(0, -half_w)
        B_w = bay_to_world(half_l, 0)
        L_w = bay_to_world(0, half_w)
        
        TL_c = world_to_car(*TL_w)
        TR_c = world_to_car(*TR_w)
        T_c = world_to_car(*T_w)
        L_c = world_to_car(*L_w)
        R_c = world_to_car(*R_w)
        B_c = world_to_car(*B_w)
        
        # Distance to top edge (entrance)
        dist_to_entrance = np.sqrt(T_c[0]**2 + T_c[1]**2)
        
        # Lateral alignment: how centered is car between TL and TR?
        # If car is centered, TL_c[1] ≈ -TR_c[1] (symmetric)
        lateral_symmetry = abs(TL_c[1] + TR_c[1]) / 2.0  # 0 = perfect, >0 = off-center
        lateral_alignment = max(0.0, 1.0 - lateral_symmetry / half_w)
        
        # Orientation alignment: are L and R at same y-level in car frame?
        # If yes, car is parallel to bay
        lr_level_diff = abs(L_c[1] - R_c[1])
        orientation_alignment = max(0.0, 1.0 - lr_level_diff / bay_width)
        
        # Yaw error
        yaw_err = abs(self._wrap_angle(car_yaw - goal_yaw))
        
        # ===== PHASE DETERMINATION =====
        
        if dist_to_entrance > 3.0:
            # PHASE 1: APPROACH + LATERAL ALIGN
            # Attract to top edge while aligning laterally
            phase = "APPROACH"
            
            # Attraction reward (get closer to entrance)
            approach_reward = 50.0 * max(0.0, 1.0 - dist_to_entrance / 8.0)
            
            # Lateral alignment reward (center between TL and TR)
            lateral_reward = 100.0 * lateral_alignment
            
            # Combined: approach while aligning
            reward = approach_reward + lateral_reward
            
        elif dist_to_entrance > 1.5 or yaw_err > 0.3:
            # PHASE 2: ORIENT (near entrance, but not aligned)
            # Use all 8 points to match bay orientation
            phase = "ALIGN"
            
            # Orientation reward (L and R at same level)
            orient_reward = 150.0 * orientation_alignment
            
            # Lateral maintenance (stay centered)
            lateral_reward = 50.0 * lateral_alignment
            
            # Position maintenance (stay near entrance)
            if dist_to_entrance < 2.5:
                position_reward = 50.0
            else:
                position_reward = 0.0
            
            reward = orient_reward + lateral_reward + position_reward
            
        elif dist_to_goal > 1.0:
            # PHASE 3: ENTER (aligned, now drive in)
            # Keep L and R level while moving toward center
            phase = "ENTER"
            
            # Entry progress reward
            entry_progress = 1.0 - (dist_to_goal / 3.0)
            progress_reward = 200.0 * entry_progress
            
            # Maintain orientation while entering
            orient_maintain = 100.0 * orientation_alignment
            
            # Maintain lateral centering
            lateral_maintain = 50.0 * lateral_alignment
            
            reward = progress_reward + orient_maintain + lateral_maintain
            
        else:
            # PHASE 4: SETTLE (use all 4 corners)
            # Fine-tune using corner distances
            phase = "SETTLE"
            
            # Ideally all 4 corners should be equidistant
            # (This is handled by existing continuous_parking_reward)
            reward = 0.0
            
        return reward, phase
    
    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    # Removed calculate_settling_reward as it was ineffective

    def check_success_threshold(
        self,
        cx_bay: float,
        cy_bay: float,
        yaw_err: float,
        success_cy: float,
        success_yaw: float,
    ) -> bool:
        """
        Check if parking meets success criteria.
        
        Returns:
            True if within thresholds
        """
        return (
            abs(cy_bay) < success_cy
            and abs(yaw_err) < success_yaw  # v38.7: CRITICAL FIX - was missing abs()!
            and abs(cx_bay) < 3.0
        )
    
    def calculate_phase_blended_reward(
        self,
        dist_to_goal: float,
        cx_bay: float,
        cy_bay: float,
        yaw_err: float,
    ) -> float:
        """
        Phase-adaptive reward blending (navigation -> approach -> parking).
        
        Args:
            dist_to_goal: Distance to goal bay center
            cx_bay: Longitudinal offset in bay frame
            cy_bay: Lateral offset in bay frame
            yaw_err: Heading error
            
        Returns:
            Blended reward
        """
        # Phase weights (smooth transitions)
        w_nav = float(np.clip((dist_to_goal - 2.0) / 3.0, 0.0, 1.0))
        w_park = float(np.clip((2.0 - dist_to_goal) / 2.0, 0.0, 1.0))
        w_approach = 1.0 - max(w_nav, w_park)
        
        # Navigation: get closer
        nav_term = -0.15 * dist_to_goal
        
        # Approach: distance + yaw alignment
        approach_term = -0.2 * dist_to_goal - 0.8 * yaw_err
        
        # Parking: bay-frame precision
        target_depth = 2.0
        depth_err = abs(abs(cx_bay) - target_depth)
        park_term = -0.8 * abs(cy_bay) - 0.6 * yaw_err - 0.3 * depth_err
        
        # Weighted blend
        reward = (
            w_nav * nav_term +
            w_approach * approach_term +
            w_park * park_term
        )
        
        return reward


# ============================================================================
# Bay Entry Bonus (Encourages Entering Goal Bay)
# ============================================================================

def calculate_bay_entry_bonus(
    cx_bay: float,
    cy_bay: float,
    bay_length: float,
    bay_width: float,
    entered_bay_flag: bool,
    bonus_amount: float = 150.0,
) -> Tuple[float, bool]:
    """
    One-time bonus for entering the goal bay.
    
    Args:
        cx_bay: Longitudinal offset in bay frame
        cy_bay: Lateral offset in bay frame
        bay_length: Bay depth (meters)
        bay_width: Bay width (meters)
        entered_bay_flag: Has bay been entered this episode?
        bonus_amount: Reward for first entry
        
    Returns:
        (bonus_reward, new_entered_bay_flag)
    """
    inside_bay = (
        abs(cx_bay) < bay_length / 2 and
        abs(cy_bay) < bay_width / 2
    )
    
    if inside_bay and not entered_bay_flag:
        return bonus_amount, True  # Award bonus and set flag
    
    return 0.0, entered_bay_flag  # No bonus or already awarded


# ============================================================================
# Goal Progress Reward (Separate from Waypoint)
# ============================================================================

def calculate_goal_progress_reward(
    dist_to_goal: float,
    prev_dist_to_goal: Optional[float],
    weight: float = 1.0,
) -> float:
    """
    Separate reward for approaching the final goal (independent of waypoints).
    
    Args:
        dist_to_goal: Current distance to goal
        prev_dist_to_goal: Previous step's distance
        weight: Reward scaling factor
        
    Returns:
        Progress reward (positive if getting closer)
    """
    if prev_dist_to_goal is None:
        return 0.0
    
    goal_progress = prev_dist_to_goal - dist_to_goal
    
    if goal_progress > 0:  # Moving closer
        return weight * goal_progress
    
    return 0.0  # No penalty for moving away (waypoint logic handles that)


# ============================================================================
# Curriculum-Adaptive Success Thresholds
# ============================================================================

def get_curriculum_thresholds(
    curriculum_stage: Optional[int],
    enable_curriculum: bool,
) -> Tuple[float, float]:
    """
    Get success thresholds based on curriculum stage.
    
    Returns progressively tighter tolerances as agent improves.
    
    Args:
        curriculum_stage: Current stage index (0-14)
        enable_curriculum: Whether curriculum is enabled
        
    Returns:
        (success_cy, success_yaw) in meters and radians
    """
    if not enable_curriculum or curriculum_stage is None:
        return 0.5, 0.2  # Moderate defaults
    
    stage = int(curriculum_stage)
    
    if stage < 3:
        # Early stages: tighter to prevent wrong-bay success
        return 0.8, 0.4  # 80cm lateral, ~23° yaw (was 1.5m, 0.5)
    elif stage < 6:
        # Mid stages: moderate
        return 0.5, 0.25  # 50cm lateral, ~14° yaw
    else:
        # Advanced stages: precise
        return 0.15, 0.1  # 15cm lateral, ~5.7° yaw
