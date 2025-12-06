"""
v34 OPTION 2: Path Corridor Constraint
Prototype implementation for review.

Defines a corridor around the path centerline.
All car points must stay within corridor.
"""

import numpy as np
from typing import Tuple, List, Optional

class PathCorridorReward:
    """
    Reward calculator for path corridor constraint.
    
    Ensures entire car body stays within a corridor around the path.
    """
    
    def __init__(
        self,
        car_length: float = 4.5,
        car_width: float = 2.0,
        corridor_width: float = 3.0,
        penalty_weight: float = 10.0,
        num_check_points: int = 5,  # Points along car to check
    ):
        self.car_length = car_length
        self.car_width = car_width
        self.corridor_width = corridor_width
        self.penalty_weight = penalty_weight
        self.num_check_points = num_check_points
    
    def calculate_car_check_points(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
    ) -> List[Tuple[float, float]]:
        """
        Calculate positions of multiple points along the car body.
        
        Checks front, center, rear, and corners.
        
        Returns:
            List of (x, y) positions
        """
        cos_yaw = np.cos(car_yaw)
        sin_yaw = np.sin(car_yaw)
        half_length = self.car_length / 2.0
        half_width = self.car_width / 2.0
        
        check_points = []
        
        # 4 corners
        corners = [
            (half_length, half_width),    # FL
            (half_length, -half_width),   # FR
            (-half_length, half_width),   # RL
            (-half_length, -half_width),  # RR
        ]
        
        for lx, ly in corners:
            px = car_x + lx * cos_yaw - ly * sin_yaw
            py = car_y + lx * sin_yaw + ly * cos_yaw
            check_points.append((px, py))
        
        # Center point
        check_points.append((car_x, car_y))
        
        # Front center
        fx = car_x + half_length * cos_yaw
        fy = car_y + half_length * sin_yaw
        check_points.append((fx, fy))
        
        # Rear center
        rx = car_x - half_length * cos_yaw
        ry = car_y - half_length * sin_yaw
        check_points.append((rx, ry))
        
        return check_points
    
    def point_to_line_distance(
        self,
        point: Tuple[float, float],
        line_p1: Tuple[float, float],
        line_p2: Tuple[float, float],
    ) -> float:
        """
        Calculate perpendicular distance from point to line segment.
        
        Returns:
            Perpendicular distance (always positive)
        """
        px, py = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx*dx + dy*dy
        
        if length_sq < 1e-6:
            # Degenerate line, use distance to p1
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        # Project point onto line
        t = ((px - x1) * dx + (py - y1) * dy) / length_sq
        t = max(0.0, min(1.0, t))  # Clamp to segment
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Perpendicular distance
        perp_dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        
        return perp_dist
    
    def calculate_corridor_reward(
        self,
        car_x: float,
        car_y: float,
        car_yaw: float,
        path_segment_start: Tuple[float, float],
        path_segment_end: Tuple[float, float],
    ) -> Tuple[float, dict]:
        """
        Calculate reward based on car staying within corridor.
        
        Args:
            car_x, car_y, car_yaw: Car state
            path_segment_start: (x, y) of current path segment start
            path_segment_end: (x, y) of current path segment end
        
        Returns:
            (total_reward, debug_info)
        """
        # Get all check points on car
        check_points = self.calculate_car_check_points(car_x, car_y, car_yaw)
        
        # Calculate perpendicular distance for each point
        violations = []
        max_allowed = self.corridor_width / 2.0
        
        for point in check_points:
            perp_dist = self.point_to_line_distance(
                point, path_segment_start, path_segment_end
            )
            
            # Check if outside corridor
            if perp_dist > max_allowed:
                violation = perp_dist - max_allowed
                violations.append(violation)
            else:
                violations.append(0.0)
        
        # Total violation
        total_violation = sum(violations)
        num_violating = sum(1 for v in violations if v > 0)
        
        # Penalty for violations
        penalty = -self.penalty_weight * total_violation
        
        # Bonus for staying well within corridor
        if total_violation == 0:
            # All points within corridor
            avg_dist = sum(
                self.point_to_line_distance(p, path_segment_start, path_segment_end)
                for p in check_points
            ) / len(check_points)
            
            # Bonus for being centered (closer to centerline)
            centering_quality = 1.0 - (avg_dist / max_allowed)
            bonus = 10.0 * centering_quality
        else:
            bonus = 0.0
        
        reward = penalty + bonus
        
        # Debug info
        debug_info = {
            "total_violation": total_violation,
            "num_violating_points": num_violating,
            "max_violation": max(violations) if violations else 0.0,
            "penalty": penalty,
            "bonus": bonus,
        }
        
        return reward, debug_info
