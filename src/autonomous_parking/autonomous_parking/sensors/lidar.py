#!/usr/bin/env python3
"""
Enhanced 64-Ray Lidar Sensor for 2D Parking Environment

Matches Gazebo sensor fidelity with:
- 360° scanning coverage
- Configurable ray count (default 64)
- Realistic noise model
- Detection of:
  * World boundaries
  * Parking bay edges
  * Parked cars (occupied bays)
  * Dynamic obstacles
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional


class EnhancedLidar:
    """
    Production-grade 2D lidar sensor matching Gazebo specifications.
    
    Performs ray-casting to detect obstacles in 360° around robot.
    """
    
    def __init__(
        self,
        num_rays: int = 64,
        max_range: float = 10.0,
        noise_std: float = 0.02,
        min_range: float = 0.1
    ):
        """
        Initialize lidar sensor.
        
        Args:
            num_rays: Number of rays in 360° scan (default: 64)
            max_range: Maximum detection range in meters
            noise_std: Standard deviation of Gaussian noise
            min_range: Minimum detection range (dead zone)
        """
        self.num_rays = num_rays
        self.max_range = max_range
        self.noise_std = noise_std
        self.min_range = min_range
        
        # Pre-compute ray angles for efficiency
        self.angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    
    def scan(
        self,
        robot_pose: np.ndarray,
        world_bounds: Tuple[float, float, float, float],
        bays: List[Dict],
        occupied_bays: List[Dict],
        dynamic_obstacles: List = None
    ) -> np.ndarray:
        """
        Perform complete 360° lidar scan.
        
        Args:
            robot_pose: [x, y, yaw] robot position
            world_bounds: (x_min, x_max, y_min, y_max) boundaries
            bays: List of all parking bay dicts with {x, y, yaw, ...}
            occupied_bays: List of occupied bay dicts (parked cars)
            dynamic_obstacles: List of dynamic obstacle objects
        
        Returns:
            np.array of shape (num_rays,) with distances in meters
        """
        ranges = np.full(self.num_rays, self.max_range, dtype=np.float32)
        
        x, y, yaw = robot_pose[0], robot_pose[1], robot_pose[2]
        
        for i, angle in enumerate(self.angles):
            # Ray direction in world frame
            ray_angle = yaw + angle
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            
            min_dist = self.max_range
            
            # 1. Check world boundaries
            bound_dist = self._ray_world_intersection(
                origin=(x, y),
                direction=ray_dir,
                bounds=world_bounds
            )
            min_dist = min(min_dist, bound_dist)
            
            # 2. Check parking bay boundaries
            for bay in bays:
                bay_dist = self._ray_bay_intersection(
                    origin=(x, y),
                    direction=ray_dir,
                    bay=bay,
                    bay_width=2.7,
                    bay_length=5.5
                )
                min_dist = min(min_dist, bay_dist)
            
            # 3. Check occupied bays (parked cars)
            for occupied_bay in occupied_bays:
                car_dist = self._ray_box_intersection(
                    origin=(x, y),
                    direction=ray_dir,
                    box_center=(occupied_bay['x'], occupied_bay['y']),
                    box_yaw=occupied_bay['yaw'],
                    box_length=4.2,  # Car length
                    box_width=1.9    # Car width
                )
                min_dist = min(min_dist, car_dist)
            
            # 4. Check dynamic obstacles
            if dynamic_obstacles:
                for obstacle in dynamic_obstacles:
                    obs_dist = self._ray_circle_intersection(
                        origin=(x, y),
                        direction=ray_dir,
                        circle_center=obstacle.pos,
                        circle_radius=obstacle.radius
                    )
                    min_dist = min(min_dist, obs_dist)
            
            # Apply min range (sensor dead zone)
            if min_dist < self.min_range:
                min_dist = self.max_range
            
            # Add realistic Gaussian noise
            noisy_dist = min_dist + np.random.normal(0, self.noise_std)
            ranges[i] = np.clip(noisy_dist, self.min_range, self.max_range)
        
        return ranges
    
    def _ray_world_intersection(
        self,
        origin: Tuple[float, float],
        direction: np.ndarray,
        bounds: Tuple[float, float, float, float]
    ) -> float:
        """
        Compute distance to world boundary rectangle.
        
        Args:
            origin: (x, y) ray origin
            direction: [dx, dy] normalized ray direction
            bounds: (x_min, x_max, y_min, y_max)
        
        Returns:
            Distance to nearest boundary
        """
        x, y = origin
        dx, dy = direction
        x_min, x_max, y_min, y_max = bounds
        
        min_t = self.max_range
        
        # Check all 4 walls
        # Left wall (x = x_min)
        if abs(dx) > 1e-6:
            t = (x_min - x) / dx
            if t > 0:
                y_hit = y + t * dy
                if y_min <= y_hit <= y_max:
                    min_t = min(min_t, t)
        
        # Right wall (x = x_max)
        if abs(dx) > 1e-6:
            t = (x_max - x) / dx
            if t > 0:
                y_hit = y + t * dy
                if y_min <= y_hit <= y_max:
                    min_t = min(min_t, t)
        
        # Bottom wall (y = y_min)
        if abs(dy) > 1e-6:
            t = (y_min - y) / dy
            if t > 0:
                x_hit = x + t * dx
                if x_min <= x_hit <= x_max:
                    min_t = min(min_t, t)
        
        # Top wall (y = y_max)
        if abs(dy) > 1e-6:
            t = (y_max - y) / dy
            if t > 0:
                x_hit = x + t * dx
                if x_min <= x_hit <= x_max:
                    min_t = min(min_t, t)
        
        return min_t if min_t < self.max_range else self.max_range
    
    def _ray_bay_intersection(
        self,
        origin: Tuple[float, float],
        direction: np.ndarray,
        bay: Dict,
        bay_width: float,
        bay_length: float
    ) -> float:
        """
        Compute distance to parking bay boundary (treated as oriented rectangle).
        
        Args:
            origin: (x, y) ray origin
            direction: [dx, dy] ray direction
            bay: Dict with {x, y, yaw}
            bay_width: Width of bay (meters)
            bay_length: Length/depth of bay (meters)
        
        Returns:
            Distance to bay boundary
        """
        return self._ray_box_intersection(
            origin=origin,
            direction=direction,
            box_center=(bay['x'], bay['y']),
            box_yaw=bay['yaw'],
            box_length=bay_length,
            box_width=bay_width
        )
    
    def _ray_box_intersection(
        self,
        origin: Tuple[float, float],
        direction: np.ndarray,
        box_center: Tuple[float, float],
        box_yaw: float,
        box_length: float,
        box_width: float
    ) -> float:
        """
        Ray-oriented bounding box intersection (separating axis theorem).
        
        Args:
            origin: (x, y) ray origin
            direction: [dx, dy] ray direction
            box_center: (x, y) center of box
            box_yaw: Orientation of box (radians)
            box_length: Length along yaw direction
            box_width: Width perpendicular to yaw
        
        Returns:
            Distance to box edge (or max_range if no hit)
        """
        # Transform ray to box-local coordinates
        dx = origin[0] - box_center[0]
        dy = origin[1] - box_center[1]
        
        cos_yaw = np.cos(-box_yaw)
        sin_yaw = np.sin(-box_yaw)
        
        local_origin_x = cos_yaw * dx - sin_yaw * dy
        local_origin_y = sin_yaw * dx + cos_yaw * dy
        
        local_dir_x = cos_yaw * direction[0] - sin_yaw * direction[1]
        local_dir_y = sin_yaw * direction[0] + cos_yaw * direction[1]
        
        # Ray-AABB intersection in local space
        half_length = box_length / 2.0
        half_width = box_width / 2.0
        
        t_min = -np.inf
        t_max = np.inf
        
        # X-axis slab
        if abs(local_dir_x) > 1e-6:
            t1 = (-half_length - local_origin_x) / local_dir_x
            t2 = (half_length - local_origin_x) / local_dir_x
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        elif abs(local_origin_x) > half_length:
            return self.max_range
        
        # Y-axis slab
        if abs(local_dir_y) > 1e-6:
            t1 = (-half_width - local_origin_y) / local_dir_y
            t2 = (half_width - local_origin_y) / local_dir_y
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
        elif abs(local_origin_y) > half_width:
            return self.max_range
        
        # Check intersection validity
        if t_max < t_min or t_max < 0:
            return self.max_range
        
        # Return first hit (entry point)
        t_hit = t_min if t_min > 0 else t_max
        return t_hit if t_hit < self.max_range else self.max_range
    
    def _ray_circle_intersection(
        self,
        origin: Tuple[float, float],
        direction: np.ndarray,
        circle_center: np.ndarray,
        circle_radius: float
    ) -> float:
        """
        Ray-circle intersection (for dynamic obstacles).
        
        Args:
            origin: (x, y) ray origin
            direction: [dx, dy] ray direction
            circle_center: [x, y] center of circle
            circle_radius: Radius in meters
        
        Returns:
            Distance to circle (or max_range if no hit)
        """
        # Vector from origin to circle center
        to_center = circle_center - np.array(origin)
        
        # Project onto ray direction
        proj_length = np.dot(to_center, direction)
        
        # If behind ray, no intersection
        if proj_length < 0:
            return self.max_range
        
        # Distance from circle center to ray
        closest_point = np.array(origin) + proj_length * direction
        dist_to_ray = np.linalg.norm(circle_center - closest_point)
        
        # Check if ray intersects circle
        if dist_to_ray > circle_radius:
            return self.max_range
        
        # Compute intersection distance
        half_chord = np.sqrt(circle_radius**2 - dist_to_ray**2)
        dist = proj_length - half_chord
        
        return dist if dist > 0 else self.max_range
