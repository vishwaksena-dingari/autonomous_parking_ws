"""
Reward Module for Autonomous Parking

Centralized reward calculation for hierarchical RL.
"""

from .parking_rewards import (
    WaypointRewardCalculator,
    ParkingRewardCalculator,
    calculate_bay_entry_bonus,
    calculate_goal_progress_reward,
    get_curriculum_thresholds,
)

__all__ = [
    "WaypointRewardCalculator",
    "ParkingRewardCalculator",
    "calculate_bay_entry_bonus",
    "calculate_goal_progress_reward",
    "get_curriculum_thresholds",
]
