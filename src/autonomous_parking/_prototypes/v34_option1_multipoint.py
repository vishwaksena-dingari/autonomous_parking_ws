"""
v34 OPTION 1: Multi-Point Offset Waypoints
Prototype implementation for review.

Each waypoint generates 4 offset targets for car corners.
Car corners must reach their respective offset targets.
"""

import numpy as np
from typing import Tuple, List, Optional

class MultiPointWaypointReward:
    """
    Reward calculator for multi-point path following.
    
    Instead of just tracking car center, tracks all 4 corners.
    """
    
    def __init__(
        self,
        car_length: float = 4.5,
        car_width: float = 2.0,
        corner_weight: float = 0.5,  # Weight for each corner penalty
    ):
        self.car_length = car_length
        self.car_width = car_width
        self.half_length = car_length / 2.0
        self.half_width = car_width / 2.0
        self.corner_weight = corner_weight
    
    def calculate_car_corners(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
    ) -> List[Tuple[float, float]]:
        """
        Calculate positions of all 4 car corners.
        
        Returns:
            [(FL_x, FL_y), (FR_x, FR_y), (RL_x, RL_y), (RR_x, RR_y)]
        """
        cos_yaw = np.cos(car_yaw)
        sin_yaw = np.sin(car_yaw)
        
        # Front-Left corner
        FL_x = car_x + self.half_length * cos_yaw - self.half_width * sin_yaw
        FL_y = car_y + self.half_length * sin_yaw + self.half_width * cos_yaw
        
        # Front-Right corner
        FR_x = car_x + self.half_length * cos_yaw + self.half_width * sin_yaw
        FR_y = car_y + self.half_length * sin_yaw - self.half_width * cos_yaw
        
        # Rear-Left corner
        RL_x = car_x - self.half_length * cos_yaw - self.half_width * sin_yaw
        RL_y = car_y - self.half_length * sin_yaw + self.half_width * cos_yaw
        
        # Rear-Right corner
        RR_x = car_x - self.half_length * cos_yaw + self.half_width * sin_yaw
        RR_y = car_y - self.half_length * sin_yaw - self.half_width * cos_yaw
        
        return [(FL_x, FL_y), (FR_x, FR_y), (RL_x, RL_y), (RR_x, RR_y)]
    
    def calculate_offset_waypoints(
        self,
        waypoint_x: float,
        waypoint_y: float,
        waypoint_theta: float,
    ) -> List[Tuple[float, float]]:
        """
        Calculate 4 offset target positions for car corners.
        
        Offsets are in the WAYPOINT's frame (perpendicular to waypoint direction).
        This ensures that when all corners reach targets, car is aligned.
        
        Returns:
            [(FL_target_x, FL_target_y), (FR_target_x, FR_target_y), 
             (RL_target_x, RL_target_y), (RR_target_x, RR_target_y)]
        """
        cos_theta = np.cos(waypoint_theta)
        sin_theta = np.sin(waypoint_theta)
        
        # Front-Left: +half_length along theta, +half_width perpendicular
        FL_target_x = waypoint_x + self.half_length * cos_theta - self.half_width * sin_theta
        FL_target_y = waypoint_y + self.half_length * sin_theta + self.half_width * cos_theta
        
        # Front-Right: +half_length along theta, -half_width perpendicular
        FR_target_x = waypoint_x + self.half_length * cos_theta + self.half_width * sin_theta
        FR_target_y = waypoint_y + self.half_length * sin_theta - self.half_width * cos_theta
        
        # Rear-Left: -half_length along theta, +half_width perpendicular
        RL_target_x = waypoint_x - self.half_length * cos_theta - self.half_width * sin_theta
        RL_target_y = waypoint_y - self.half_length * sin_theta + self.half_width * cos_theta
        
        # Rear-Right: -half_length along theta, -half_width perpendicular
        RR_target_x = waypoint_x - self.half_length * cos_theta + self.half_width * sin_theta
        RR_target_y = waypoint_y - self.half_length * sin_theta - self.half_width * cos_theta
        
        return [
            (FL_target_x, FL_target_y),
            (FR_target_x, FR_target_y),
            (RL_target_x, RL_target_y),
            (RR_target_x, RR_target_y),
        ]
    
    def calculate_multi_point_reward(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        waypoint_x: float,
        waypoint_y: float,
        waypoint_theta: float,
    ) -> Tuple[float, dict]:
        """
        Calculate reward based on all 4 corners reaching their targets.
        
        Returns:
            (total_reward, debug_info)
        """
        # Get actual car corner positions
        car_corners = self.calculate_car_corners(car_x, car_y, car_yaw)
        
        # Get target positions for corners
        target_corners = self.calculate_offset_waypoints(
            waypoint_x, waypoint_y, waypoint_theta
        )
        
        # Calculate distance from each corner to its target
        corner_distances = []
        for actual, target in zip(car_corners, target_corners):
            dx = actual[0] - target[0]
            dy = actual[1] - target[1]
            dist = np.sqrt(dx**2 + dy**2)
            corner_distances.append(dist)
        
        # Total penalty: sum of all corner distances
        total_distance = sum(corner_distances)
        
        # Reward: negative penalty (closer = better)
        reward = -self.corner_weight * total_distance
        
        # Debug info
        debug_info = {
            "FL_dist": corner_distances[0],
            "FR_dist": corner_distances[1],
            "RL_dist": corner_distances[2],
            "RR_dist": corner_distances[3],
            "total_dist": total_distance,
            "avg_dist": total_distance / 4.0,
        }
        
        return reward, debug_info
