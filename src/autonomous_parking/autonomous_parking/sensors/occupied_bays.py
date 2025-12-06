#!/usr/bin/env python3
"""
Occupied Bay Management System

Handles which parking bays contain parked cars, supporting:
- Random occupancy generation
- Free bay selection
-  Parked car bounding boxes for lidar detection
- Configurable occupancy rates
"""

import random
import math  # v40 FIX: Needed for pi/2 rotation
from typing import List, Dict, Optional


class OccupiedBayManager:
    """
    Manages occupied parking bays with realistic occupancy rates.
    
    Integrates with lidar to detect parked cars as obstacles.
    """
    
    def __init__(
        self,
        all_bays: List[Dict],
        occupancy_rate: float = 0.3
    ):
        """
        Initialize occupied bay manager.
        
        Args:
            all_bays: List of all parking bay dicts with {id, x, y, yaw}
            occupancy_rate: Base occupation rate (0.0-1.0)
        """
        self.all_bays = all_bays
        self.base_occupancy_rate = occupancy_rate
        self.occupied_bay_ids = []
        self.occupied_bay_objects = []
    
    def randomize_occupancy(
        self,
        occupancy_rate: Optional[float] = None
    ) -> List[Dict]:
        """
        Randomly occupy bays based on occupancy rate.
        
        Args:
            occupancy_rate: Override base rate (if None, use base rate)
        
        Returns:
            List of occupied bay dicts
        """
        rate = occupancy_rate if occupancy_rate is not None else self.base_occupancy_rate
        rate = max(0.0, min(1.0, rate))  # Clamp to [0, 1]
        
        num_occupied = int(len(self.all_bays) * rate)
        
        # Randomly select bays to occupy
        occupied = random.sample(self.all_bays, k=num_occupied)
        
        self.occupied_bay_ids = [bay['id'] for bay in occupied]
        self.occupied_bay_objects = occupied
        
        return occupied
    
    def get_free_bays(self) -> List[Dict]:
        """
        Get list of unoccupied (free) parking bays.
        
        Returns:
            List of free bay dicts
        """
        return [
            bay for bay in self.all_bays
            if bay['id'] not in self.occupied_bay_ids
        ]
    
    def is_bay_occupied(self, bay_id: str) -> bool:
        """
        Check if specific bay is occupied.
        
        Args:
            bay_id: Bay identifier string
        
        Returns:
            True if bay is occupied
        """
        return bay_id in self.occupied_bay_ids
    
    def get_parked_car_bbox(self, bay: Dict) -> Dict:
        """
        Get bounding box of parked car in occupied bay.
        
        Used for lidar collision detection.
        
        Args:
            bay: Bay dict with {x, y, yaw}
        
        Returns:
            Dict with {x, y, yaw, length, width} of parked car
        """
        # Car dimensions (slightly smaller than bay to fit)
        return {
            'x': bay['x'],
            'y': bay['y'],
            'yaw': bay['yaw'] + (math.pi / 2), # v40 FIX: Bays are Vertical at 0, Cars are Horizontal at 0. Add 90 deg.
            'length': 4.2,  # m (compact car)
            'width': 1.9    # m
        }
    
    def get_all_parked_cars(self) -> List[Dict]:
        """
        Get bounding boxes for all parked cars.
        
        Returns:
            List of parked car bbox dicts
        """
        return [
            self.get_parked_car_bbox(bay)
            for bay in self.occupied_bay_objects
        ]
    
    def set_specific_occupancy(self, occupied_bay_ids: List[str]):
        """
        Set specific bays as occupied (for deterministic scenarios).
        
        Args:
            occupied_bay_ids: List of bay ID strings to occupy
        """
        self.occupied_bay_ids = occupied_bay_ids
        self.occupied_bay_objects = [
            bay for bay in self.all_bays
            if bay['id'] in occupied_bay_ids
        ]
    
    def clear_occupancy(self):
        """Clear all occupied bays (empty lot)."""
        self.occupied_bay_ids = []
        self.occupied_bay_objects = []
    
    def get_occupancy_rate(self) -> float:
        """
        Get current occupancy rate.
        
        Returns:
            Fraction of bays occupied (0.0-1.0)
        """
        if len(self.all_bays) == 0:
            return 0.0
        return len(self.occupied_bay_ids) / len(self.all_bays)
