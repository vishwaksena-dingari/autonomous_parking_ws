"""
Hybrid A* Planner

Combines grid-based A* search with Reeds-Shepp curves to generate
kinematically feasible paths that respect obstacles.

Key differences from standard A*:
- State space: (x, y, θ) instead of just (x, y)
- Motion primitives: Arcs (forward/backward) instead of straight lines
- Analytic expansion: Uses full Reeds-Shepp when close to goal
"""

import math
import heapq
from typing import List, Tuple, Optional, Set
import numpy as np

from .reeds_shepp import reeds_shepp_path_planning


class Node:
    """Node in Hybrid A* search space (x, y, θ)"""
    
    def __init__(self, x: float, y: float, theta: float, 
                 g: float = 0.0, h: float = 0.0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta  # radians
        self.g = g  # cost from start
        self.h = h  # heuristic to goal
        self.f = g + h
        self.parent = parent
        
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return (self.x, self.y, self.theta) == (other.x, other.y, other.theta)
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))


class HybridAStarPlanner:
    """
    Hybrid A* planner for kinematically feasible parking paths.
    
    Usage:
        planner = HybridAStarPlanner(world_bounds=(-25, 25, -25, 25), resolution=0.5)
        path = planner.plan(start, goal, obstacle_grid)
    """
    
    def __init__(self, 
                 world_bounds=(-25, 25, -25, 25), 
                 resolution: float = 0.5,
                 heading_resolution: float = math.radians(5),  # 5 degrees
                 max_curvature: float = 0.2,  # 1/min_radius
                 step_size: float = 0.5):  # meters per primitive
        """
        Args:
            world_bounds: (x_min, x_max, y_min, y_max) in meters
            resolution: Grid cell size for collision checking
            heading_resolution: Discretization of heading angle (radians)
            max_curvature: Maximum curvature (1/R) for motion primitives
            step_size: Distance traveled per motion primitive
        """
        self.x_min, self.x_max, self.y_min, self.y_max = world_bounds
        self.resolution = float(resolution)
        self.heading_resolution = heading_resolution
        self.max_curvature = max_curvature
        self.step_size = step_size
        
        # Grid dimensions
        self.grid_width = int((self.x_max - self.x_min) / self.resolution)
        self.grid_height = int((self.y_max - self.y_min) / self.resolution)
        
        # Motion primitives: [steering_angle, direction]
        # steering_angle: -max_curv, 0, +max_curv
        # direction: 1 (forward), -1 (backward)
        self.primitives = [
            (0.0, 1),  # Straight forward
            (self.max_curvature, 1),  # Left forward
            (-self.max_curvature, 1),  # Right forward
            (0.0, -1),  # Straight backward
            (self.max_curvature, -1),  # Left backward
            (-self.max_curvature, -1),  # Right backward
        ]
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)
        return gx, gy
    
    def discretize_heading(self, theta: float) -> float:
        """Discretize heading to nearest bin"""
        return round(theta / self.heading_resolution) * self.heading_resolution
    
    def normalize_angle(self, theta: float) -> float:
        """Normalize angle to [-π, π]"""
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta
    
    def apply_primitive(self, node: Node, curvature: float, direction: int) -> Node:
        """
        Apply a motion primitive to generate a new node.
        
        Args:
            node: Current node
            curvature: Steering curvature
            direction: 1 (forward) or -1 (backward)
            
        Returns:
            New node after applying primitive
        """
        x, y, theta = node.x, node.y, node.theta
        dist = direction * self.step_size
        
        if abs(curvature) < 1e-4:  # Straight line
            x_new = x + dist * math.cos(theta)
            y_new = y + dist * math.sin(theta)
            theta_new = theta
        else:  # Arc
            radius = 1.0 / abs(curvature)
            dtheta = dist * curvature
            
            # Center of rotation
            cx = x - radius * math.sin(theta) * np.sign(curvature)
            cy = y + radius * math.cos(theta) * np.sign(curvature)
            
            # New position
            theta_new = theta + dtheta
            x_new = cx + radius * math.sin(theta_new) * np.sign(curvature)
            y_new = cy - radius * math.cos(theta_new) * np.sign(curvature)
        
        theta_new = self.normalize_angle(theta_new)
        theta_new_discrete = self.discretize_heading(theta_new)
        
        cost = abs(dist)
        if direction < 0:  # Penalty for reversing
            cost *= 1.5
        
        new_node = Node(
            x_new, y_new, theta_new_discrete,
            g=node.g + cost,
            parent=node
        )
        
        return new_node
    
    def is_collision_free(self, node: Node, obstacles: np.ndarray) -> bool:
        """Check if a node is collision-free"""
        gx, gy = self.world_to_grid(node.x, node.y)
        
        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False
        
        return obstacles[gy, gx] == 0
    
    def is_path_collision_free(self, node1: Node, node2: Node, obstacles: np.ndarray) -> bool:
        """Check if the path between two nodes is collision-free"""
        # Sample points along the path
        num_samples = int(math.hypot(node2.x - node1.x, node2.y - node1.y) / (self.resolution * 0.5))
        num_samples = max(num_samples, 2)
        
        for i in range(num_samples + 1):
            alpha = i / num_samples
            x = node1.x + alpha * (node2.x - node1.x)
            y = node1.y + alpha * (node2.y - node1.y)
            
            gx, gy = self.world_to_grid(x, y)
            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                return False
            if obstacles[gy, gx] != 0:
                return False
        
        return True
    
    def heuristic(self, node: Node, goal: Node) -> float:
        """Heuristic: Euclidean distance (non-holonomic distance would be better)"""
        dx = goal.x - node.x
        dy = goal.y - node.y
        return math.hypot(dx, dy)
    
    def try_analytic_expansion(self, node: Node, goal: Node, obstacles: np.ndarray) -> Optional[List[Node]]:
        """
        Try to connect current node to goal using Reeds-Shepp curve.
        Returns path if successful and collision-free, None otherwise.
        """
        # Use Reeds-Shepp to generate path
        xs, ys, yaws, _, _ = reeds_shepp_path_planning(
            node.x, node.y, node.theta,
            goal.x, goal.y, goal.theta,
            maxc=self.max_curvature,
            step_size=self.resolution * 0.5
        )
        
        if xs is None:
            return None
        
        # Check collision along the path
        for x, y in zip(xs, ys):
            gx, gy = self.world_to_grid(x, y)
            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                return None
            if obstacles[gy, gx] != 0:
                return None
        
        # Convert to nodes
        path = []
        for x, y, yaw in zip(xs, ys, yaws):
            path.append(Node(x, y, yaw))
        
        return path
    
    def reconstruct_path(self, node: Node) -> List[Tuple[float, float, float]]:
        """Reconstruct path from goal node to start"""
        path = []
        current = node
        while current is not None:
            path.append((current.x, current.y, current.theta))
            current = current.parent
        return list(reversed(path))
    
    def plan(self,
             start: Tuple[float, float, float],
             goal: Tuple[float, float, float],
             obstacles: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan a path from start to goal using Hybrid A*.
        
        Args:
            start: (x, y, theta) starting pose
            goal: (x, y, theta) goal pose
            obstacles: 2D grid (0=free, 1=occupied)
            
        Returns:
            List of (x, y, theta) waypoints, or None if no path found
        """
        # Sync grid dimensions
        self.grid_height, self.grid_width = obstacles.shape
        
        # Create start and goal nodes
        start_node = Node(start[0], start[1], self.discretize_heading(start[2]), g=0.0)
        goal_node = Node(goal[0], goal[1], self.discretize_heading(goal[2]))
        
        # Check if start/goal are valid
        if not self.is_collision_free(start_node, obstacles):
            print("[Hybrid A*] Start position in collision!")
            return None
        if not self.is_collision_free(goal_node, obstacles):
            print("[Hybrid A*] Goal position in collision!")
            return None
        
        # Initialize search
        open_set = []
        heapq.heappush(open_set, start_node)
        
        closed_set: Set[Tuple[int, int, float]] = set()
        
        # For duplicate detection, use discretized (x, y, θ)
        def get_state_key(node: Node) -> Tuple[int, int, float]:
            gx, gy = self.world_to_grid(node.x, node.y)
            return (gx, gy, node.theta)
        
        iterations = 0
        max_iterations = 50000  # Increased for complex parking scenarios
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_set)
            current_key = get_state_key(current)
            
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            # Check if close enough to goal for analytic expansion
            dist_to_goal = math.hypot(current.x - goal_node.x, current.y - goal_node.y)
            if dist_to_goal < 8.0:  # Try analytic expansion within 8m
                analytic_path = self.try_analytic_expansion(current, goal_node, obstacles)
                if analytic_path is not None:
                    # Success! Reconstruct full path
                    path_to_current = self.reconstruct_path(current)
                    full_path = path_to_current + [(n.x, n.y, n.theta) for n in analytic_path[1:]]
                    print(f"[Hybrid A*] Path found in {iterations} iterations (analytic expansion)")
                    return full_path
            
            # Expand using motion primitives
            for curvature, direction in self.primitives:
                neighbor = self.apply_primitive(current, curvature, direction)
                
                # Check bounds and collision
                if not self.is_collision_free(neighbor, obstacles):
                    continue
                
                if not self.is_path_collision_free(current, neighbor, obstacles):
                    continue
                
                neighbor_key = get_state_key(neighbor)
                if neighbor_key in closed_set:
                    continue
                
                # Set heuristic (with weight for faster search)
                neighbor.h = 2.0 * self.heuristic(neighbor, goal_node)  # Weighted A*
                neighbor.f = neighbor.g + neighbor.h
                
                heapq.heappush(open_set, neighbor)
        
        print(f"[Hybrid A*] No path found after {iterations} iterations")
        return None
