#!/usr/bin/env python3
"""
Dynamic Obstacle Classes for Realistic Parking Scenarios

Implements moving obstacles:
- Pedestrians (walking)
- Shopping carts (rolling)
- Moving cars (driving through lot)
"""

import random
import numpy as np
from typing import Tuple


class DynamicObstacle:
    """
    Base class for moving obstacles in parking lot.
    
    All dynamic obstacles have:
    - Position (x, y)
    - Velocity (vx, vy)
    - Collision radius
    """
    
    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float,
        bounce: bool = True
    ):
        """
        Initialize dynamic obstacle.
        
        Args:
            x, y: Initial position (meters)
            vx, vy: Velocity components (m/s)
            radius: Collision radius (meters)
            bounce: Whether to bounce off boundaries
        """
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([vx, vy], dtype=np.float32)
        self.radius = radius
        self.bounce = bounce
    
    def update(self, dt: float, bounds: Tuple[float, float, float, float]):
        """
        Update obstacle position and handle boundary collisions.
        
        Args:
            dt: Time step (seconds)
            bounds: (x_min, x_max, y_min, y_max) boundaries
        """
        # Update position
        self.pos += self.vel * dt
        
        # Handle boundaries
        x_min, x_max, y_min, y_max = bounds
        
        if self.bounce:
            # Bounce off walls
            if self.pos[0] < x_min or self.pos[0] > x_max:
                self.vel[0] *= -1
                self.pos[0] = np.clip(self.pos[0], x_min, x_max)
            
            if self.pos[1] < y_min or self.pos[1] > y_max:
                self.vel[1] *= -1
                self.pos[1] = np.clip(self.pos[1], y_min, y_max)
        else:
            # Wrap around or clamp
            self.pos[0] = np.clip(self.pos[0], x_min, x_max)
            self.pos[1] = np.clip(self.pos[1], y_min, y_max)
    
    def get_state(self) -> dict:
        """
        Get obstacle state for rendering.
        
        Returns:
            Dict with pos, vel, radius
        """
        return {
            'pos': self.pos.copy(),
            'vel': self.vel.copy(),
            'radius': self.radius,
            'type': self.__class__.__name__
        }


class Pedestrian(DynamicObstacle):
    """
    Walking pedestrian obstacle.
    
    Characteristics:
    - Walking speed: 1.0-1.5 m/s
    - Collision radius: 0.3m
    - Random walking pattern
    """
    
    def __init__(self, x: float, y: float):
        """
        Create pedestrian at position with random walking direction.
        
        Args:
            x, y: Initial position
        """
        # Random walking direction
        speed = random.uniform(1.0, 1.5)
        angle = random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        super().__init__(x, y, vx, vy, radius=0.3, bounce=True)
    
    def update(self, dt: float, bounds: Tuple[float, float, float, float]):
        """
        Update with occasional random direction changes.
        
        Args:
            dt: Time step
            bounds: World boundaries
        """
        super().update(dt, bounds)
        
        # Randomly change direction occasionally
        if random.random() < 0.02:  # 2% chance per update
            speed = np.linalg.norm(self.vel)
            new_angle = random.uniform(0, 2 * np.pi)
            self.vel[0] = speed * np.cos(new_angle)
            self.vel[1] = speed * np.sin(new_angle)


class ShoppingCart(DynamicObstacle):
    """
    Rolling shopping cart obstacle.
    
    Characteristics:
    - Rolling speed: 0.3-0.8 m/s (slower than pedestrian)
    - Collision radius: 0.4m
    - Gradual slowdown (friction)
    """
    
    def __init__(self, x: float, y: float):
        """
        Create shopping cart at position with random rolling direction.
        
        Args:
            x, y: Initial position
        """
        # Random rolling direction (slower than pedestrian)
        speed = random.uniform(0.3, 0.8)
        angle = random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        super().__init__(x, y, vx, vy, radius=0.4, bounce=True)
    
    def update(self, dt: float, bounds: Tuple[float, float, float, float]):
        """
        Update with friction-like slowdown.
        
        Args:
            dt: Time step
            bounds: World boundaries
        """
        super().update(dt, bounds)
        
        # Apply friction (gradual slowdown)
        friction = 0.98
        self.vel *= friction


class MovingCar(DynamicObstacle):
    """
    Car driving through parking lot.
    
    Characteristics:
    - Driving speed: 2.0-3.0 m/s (~7-11 km/h)
    - Collision radius: 2.0m (length/2)
    - Follows aisle (horizontal or vertical)
    """
    
    def __init__(
        self,
        x: float,
        y: float,
        direction: str = 'horizontal'
    ):
        """
        Create moving car following aisle.
        
        Args:
            x, y: Initial position
            direction: 'horizontal' or 'vertical' aisle following
        """
        speed = random.uniform(2.0, 3.0)
        
        if direction == 'horizontal':
            # Drive along x-axis
            vx = random.choice([-speed, speed])
            vy = 0.0
        else:  # vertical
            # Drive along y-axis
            vx = 0.0
            vy = random.choice([-speed, speed])
        
        super().__init__(x, y, vx, vy, radius=2.0, bounce=False)
        self.direction = direction
    
    def update(self, dt: float, bounds: Tuple[float, float, float, float]):
        """
        Update with despawn/respawn at boundaries.
        
        Args:
            dt: Time step
            bounds: World boundaries
        """
        super().update(dt, bounds)
        
        # Respawn at opposite side when leaving bounds
        x_min, x_max, y_min, y_max = bounds
        margin = 2.0
        
        if self.direction == 'horizontal':
            if self.pos[0] < x_min - margin:
                self.pos[0] = x_max + margin
            elif self.pos[0] > x_max + margin:
                self.pos[0] = x_min - margin
        else:  # vertical
            if self.pos[1] < y_min - margin:
                self.pos[1] = y_max + margin
            elif self.pos[1] > y_max + margin:
                self.pos[1] = y_min - margin
