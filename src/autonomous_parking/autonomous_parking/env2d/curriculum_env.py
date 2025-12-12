"""
Legacy Curriculum Environment (5-Stage)
---------------------------------------
⚠️ DEPRECATED: This environment implements the older 5-stage curriculum.
For the new hierarchical curriculum, use `CurriculumManager` with `WaypointEnv`.

This environment is maintained for backward compatibility with pure RL experiments.
Curriculum Learning Environment for Pure RL Approach.

Stages of increasing difficulty to learn parking from scratch.
"""

import numpy as np
from typing import Optional, Tuple
from .parking_env import ParkingEnv


class CurriculumParkingEnv(ParkingEnv):
    """
    Parking environment with curriculum learning stages (legacy helper).

    Notes:
    - `should_progress()` only RETURNS whether we *should* move to the next stage.
    - The caller is responsible for actually calling `set_stage(...)` based on this flag.
    
    Stages:
    1. Close start (2-3m from goal)
    2. Medium start (5-7m from goal)
    3. Far start (10-12m from goal)
    4. Random position in same lot
    5. Random in same lot (doc says 'multi-lot' but current impl is single-lot)
    """
    
    def __init__(self, lot_name="lot_a", **kwargs):
        super().__init__(lot_name, **kwargs)
        
        self.current_stage = 1
        self.success_history = []
        self.success_window = 100  # Track last 100 episodes
        
        # Stage thresholds for auto-progression
        self.stage_thresholds = {
            1: 0.80,  # 80% success to progress from stage 1
            2: 0.70,  # 70% for stage 2
            3: 0.60,  # 60% for stage 3
            4: 0.50,  # 50% for stage 4
        }
        
    def set_stage(self, stage: int):
        """Set curriculum stage (1-5)."""
        self.current_stage = max(1, min(5, stage))
        print(f"🎯 Curriculum Stage: {self.current_stage}")
        
    def get_success_rate(self) -> float:
        """Get recent success rate."""
        if len(self.success_history) < 10:
            return 0.0
        recent = self.success_history[-self.success_window:]
        return sum(recent) / len(recent)
    
    def should_progress(self) -> bool:
        """Check if should progress to next stage."""
        if self.current_stage >= 5:
            return False
        if len(self.success_history) < self.success_window:
            return False
        
        threshold = self.stage_thresholds.get(self.current_stage, 0.5)
        success_rate = self.get_success_rate()
        return success_rate >= threshold
    
    def reset(self, *, seed=None, options=None, bay_id=None):
        """Reset with curriculum-based initial position."""
        # Pass seed to parent for reproducibility
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        
        # Select goal bay (support both index and string ID)
        if bay_id is None:
            bay_idx = int(self.random_state.choice(len(self.bays)))
        elif isinstance(bay_id, int):
            bay_idx = bay_id
        else:
            # Treat as bay ID string, e.g. "A1"
            matches = [i for i, b in enumerate(self.bays) if b.get("id") == bay_id]
            if not matches:
                available = [b.get("id") for b in self.bays]
                raise ValueError(f"Bay '{bay_id}' not found. Available: {available}")
            bay_idx = matches[0]
        
        self.goal_bay = self.bays[bay_idx]
        goal_x = self.goal_bay["x"]
        goal_y = self.goal_bay["y"]
        goal_theta = self.goal_bay["yaw"]  # Using 'yaw' key from config
        
        # Determine starting position based on stage
        if self.current_stage == 1:
            # Close start: 2-3m from goal
            distance = self.random_state.uniform(2.0, 3.0)
        elif self.current_stage == 2:
            # Medium start: 5-7m from goal
            distance = self.random_state.uniform(5.0, 7.0)
        elif self.current_stage == 3:
            # Far start: 10-12m from goal
            distance = self.random_state.uniform(10.0, 12.0)
        else:
            # Stage 4-5: Random anywhere in lot
            distance = self.random_state.uniform(5.0, 20.0)
        
        # Random angle offset from goal
        angle_offset = self.random_state.uniform(-np.pi, np.pi)
        
        # Place robot at distance and angle from goal
        x = goal_x + distance * np.cos(angle_offset)
        y = goal_y + distance * np.sin(angle_offset)
        
        # Random initial heading (but biased toward goal for easier stages)
        if self.current_stage <= 2:
            # Bias toward goal
            to_goal = np.arctan2(goal_y - y, goal_x - x)
            heading_noise = self.random_state.uniform(-np.pi/4, np.pi/4)
            theta = to_goal + heading_noise
        else:
            # Random heading
            theta = self.random_state.uniform(-np.pi, np.pi)
        
        # Clamp to world bounds
        x = np.clip(x, -20.0, 20.0)
        y = np.clip(y, -20.0, 20.0)
        
        # Initialize state
        self.state = np.array([x, y, theta, 0.0], dtype=np.float32)
        self.steps = 0
        
        #  FIX: Set tolerance attributes required by parent's step() method
        if self.current_stage == 1:
            self.current_tol_pos = 0.8
            self.current_tol_yaw = 0.5
        elif self.current_stage == 2:
            self.current_tol_pos = 0.6
            self.current_tol_yaw = 0.4
        elif self.current_stage == 3:
            self.current_tol_pos = 0.5
            self.current_tol_yaw = 0.3
        else:
            self.current_tol_pos = 0.3
            self.current_tol_yaw = 0.3
        
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        """Step with success tracking."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track success (done = terminated OR truncated)
        done = terminated or truncated
        if done:
            success = info.get("success", False)
            self.success_history.append(1 if success else 0)
            
            # Keep history bounded
            if len(self.success_history) > self.success_window * 2:
                self.success_history = self.success_history[-self.success_window:]
        
        # Add curriculum info
        info["curriculum_stage"] = self.current_stage
        info["success_rate"] = self.get_success_rate()
        info["should_progress"] = self.should_progress()
        
        return obs, reward, terminated, truncated, info
