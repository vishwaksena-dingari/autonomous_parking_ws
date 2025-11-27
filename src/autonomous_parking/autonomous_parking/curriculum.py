# autonomous_parking/curriculum.py
"""
v15 Micro-Curriculum for Autonomous Parking
15-stage progressive learning with anti-catastrophic-forgetting replay

Production-ready implementation with dataclasses, internal tracking,
and robust stage advancement logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict
import math
import numpy as np


@dataclass
class CurriculumStage:
    """One curriculum stage definition"""
    name: str
    # Which lot(s): "lot_a", "lot_b", or "both"
    lots: Sequence[str]
    # Allowed bay IDs (None = filter by orientation only)
    allowed_bays: Optional[Sequence[str]]
    # Allowed goal orientations in radians (None = all)
    allowed_orientations: Optional[Sequence[float]]
    # Max spawn distance to goal in meters (None = env default)
    max_spawn_dist: Optional[float]
    # When to advance: total env steps
    advance_at_steps: int
    # Replay probability from earlier stages
    replay_prob: float = 0.0
    # Optional note
    note: str = ""


class CurriculumManager:
    """
    v15 Micro-Curriculum Manager
    
    Responsibilities:
    - Manages 15 progressive stages
    - Tracks total steps and episodes
    - Maintains rolling success window
    - Automatically advances stages
    - Implements replay mechanism to prevent forgetting
    """
    
    def __init__(self) -> None:
        self.stages: List[CurriculumStage] = self._build_stages()
        self.current_stage_idx: int = 0
        
        # Global training stats
        self.total_steps: int = 0
        self.total_episodes: int = 0
        
        # Rolling success window (for optional success-based progression)
        self._window_success: List[bool] = []
        self._window_size: int = 200
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_idx]
    
    def sample_scenario(self) -> Dict:
        """
        Sample training scenario for next episode
        
        Returns dict with:
            - stage_idx: int (current curriculum stage)
            - effective_stage_idx: int (stage actually being used, may be earlier due to replay)
            - stage_name: str
            - lot: "lot_a" | "lot_b"  
            - allowed_bays: list[str] | None
            - allowed_orientations: list[float] | None
            - max_spawn_dist: float | None
        """
        current_idx = self.current_stage_idx
        stage = self.current_stage
        effective_idx = current_idx
        
        # Implement replay: sample from earlier stage
        if stage.replay_prob > 0 and np.random.random() < stage.replay_prob:
            replay_idx = self._sample_replay_stage()
            stage = self.stages[replay_idx]
            effective_idx = replay_idx
        
        # Choose lot
        lots = list(stage.lots)
        if "both" in lots:
            lots = ["lot_a", "lot_b"]
        lot = np.random.choice(lots) if len(lots) > 0 else "lot_a"
        
        scenario = {
            "stage_idx": current_idx,
            "effective_stage_idx": effective_idx,
            "stage_name": stage.name,
            "lot": lot,
            "allowed_bays": list(stage.allowed_bays) if stage.allowed_bays else None,
            "allowed_orientations": (list(stage.allowed_orientations) 
                                    if stage.allowed_orientations is not None else None),
            "max_spawn_dist": stage.max_spawn_dist,
        }
        return scenario
    
    def update_after_episode(self, success: bool, steps: int) -> None:
        """
        Update curriculum state after episode completes
        
        Args:
            success: Whether parking succeeded
            steps: Number of steps in episode
        """
        self.total_episodes += 1
        self.total_steps += steps
        
        # Update rolling success window
        self._window_success.append(success)
        if len(self._window_success) > self._window_size:
            self._window_success.pop(0)
        
        # Check advancement
        self._maybe_advance_stage()
    
    def _maybe_advance_stage(self) -> None:
        """Advance to next stage if conditions met"""
        if self.current_stage_idx >= len(self.stages) - 1:
            return  # Final stage
        
        stage = self.current_stage
        
        # Primary: step threshold
        if self.total_steps < stage.advance_at_steps:
            return
        
        # Optional: require minimum recent success (loose 5% to avoid stuck)
        success_rate = (sum(self._window_success) / len(self._window_success) 
                       if self._window_success else 0.0)
        # if success_rate < 0.05:
        #     return
        if success_rate < 0.01 and self.total_steps < stage.advance_at_steps * 1.5:
            return  # Only block if <1% success AND not too far past threshold
        
        # Advance!
        self.current_stage_idx += 1
        next_stage = self.current_stage
        print(f"\n{'*' * 70}")
        print(f"📚 CURRICULUM STAGE ADVANCE")
        print(f"   From: {stage.name}")
        print(f"   To:   {next_stage.name} (Stage {self.current_stage_idx + 1}/15)")
        print(f"   Steps: {self.total_steps:,} | Episodes: {self.total_episodes:,}")
        print(f"   Recent Success: {success_rate:.1%}")
        print(f"   Replay Prob: {next_stage.replay_prob:.0%}")
        print(f"{'*' * 70}\n")
    
    def _sample_replay_stage(self) -> int:
        """Sample earlier stage (weighted toward recent)"""
        if self.current_stage_idx == 0:
            return 0
        
        # Exponential decay weights
        weights = np.array([0.5 ** (self.current_stage_idx - i) 
                           for i in range(self.current_stage_idx)])
        weights = weights / weights.sum()
        
        return np.random.choice(self.current_stage_idx, p=weights)
    
    def _build_stages(self) -> List[CurriculumStage]:
        """Build 15-stage micro-curriculum"""
        
        def rad(deg: float) -> float:
            return math.radians(deg)
        
        stages = [
            # ===== PHASE 1: SINGLE BAY MASTERY (0-80k) =====
            CurriculumStage(
                name="S1: Single bay, aligned, close",
                lots=["lot_a"],
                allowed_bays=["A1"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=12.0,
                advance_at_steps=40_000,
                replay_prob=0.0,
                note="Warm-up: one aligned bay, nearby spawn"
            ),
            
            CurriculumStage(
                name="S2: Single bay, aligned, medium",
                lots=["lot_a"],
                allowed_bays=["A1"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=18.0,
                advance_at_steps=60_000,
                replay_prob=0.2
            ),
            
            CurriculumStage(
                name="S3: Single bay, aligned, far",
                lots=["lot_a"],
                allowed_bays=["A1"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=24.0,
                advance_at_steps=80_000,
                replay_prob=0.25
            ),
            
            # ===== PHASE 2: MULTI-BAY SAME ORIENTATION (80-140k) =====
            CurriculumStage(
                name="S4: Two bays, aligned",
                lots=["lot_a"],
                allowed_bays=["A1", "A2"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=20.0,
                advance_at_steps=100_000,
                replay_prob=0.3
            ),
            
            CurriculumStage(
                name="S5: Three bays, aligned",
                lots=["lot_a"],
                allowed_bays=["A1", "A2", "A3"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=22.0,
                advance_at_steps=120_000,
                replay_prob=0.3
            ),
            
            CurriculumStage(
                name="S6: All bays, single orientation",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=24.0,
                advance_at_steps=140_000,
                replay_prob=0.35
            ),
            
            # ===== PHASE 3: ADD PERPENDICULAR (140-230k) =====
            CurriculumStage(
                name="S7: Perpendicular only",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(90.0)],
                max_spawn_dist=18.0,
                advance_at_steps=170_000,
                replay_prob=0.4
            ),
            
            CurriculumStage(
                name="S8: Two orientations, easy",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0)],
                max_spawn_dist=20.0,
                advance_at_steps=200_000,
                replay_prob=0.4
            ),
            
            CurriculumStage(
                name="S9: Two orientations, full",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0)],
                max_spawn_dist=26.0,
                advance_at_steps=230_000,
                replay_prob=0.45
            ),
            
            # ===== PHASE 4: ALL 4 ORIENTATIONS (230-370k) =====
            CurriculumStage(
                name="S10: Three orientations",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0), rad(180.0)],
                max_spawn_dist=24.0,
                advance_at_steps=280_000,
                replay_prob=0.5
            ),
            
            CurriculumStage(
                name="S11: Four orientations, close",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0), rad(180.0), rad(-90.0)],
                max_spawn_dist=22.0,
                advance_at_steps=320_000,
                replay_prob=0.5
            ),
            
            CurriculumStage(
                name="S12: Four orientations, far",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=None,  # All orientations
                max_spawn_dist=28.0,
                advance_at_steps=370_000,
                replay_prob=0.5
            ),
            
            # ===== PHASE 5: MULTI-LOT (370-500k) =====
            CurriculumStage(
                name="S13: Lot B intro",
                lots=["lot_b"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0)],
                max_spawn_dist=20.0,
                advance_at_steps=420_000,
                replay_prob=0.6
            ),
            
            CurriculumStage(
                name="S14: Mixed lots, medium",
                lots=["both"],
                allowed_bays=None,
                allowed_orientations=None,
                max_spawn_dist=26.0,
                advance_at_steps=480_000,
                replay_prob=0.6
            ),
            
            # ===== PHASE 6: FINAL MASTERY (480k+) =====
            CurriculumStage(
                name="S15: Full deployment regime",
                lots=["both"],
                allowed_bays=None,
                allowed_orientations=None,
                max_spawn_dist=None,  # No limit
                advance_at_steps=10**9,  # No auto-advance
                replay_prob=0.7,
                note="Production distribution"
            ),
        ]
        
        return stages
    
    def log_status(self) -> None:
        """Print current curriculum status"""
        stage = self.current_stage
        success_rate = (sum(self._window_success) / len(self._window_success) 
                       if self._window_success else 0.0)
        
        print(f"Curriculum: Stage {self.current_stage_idx + 1}/15 - {stage.name}")
        print(f"  Steps: {self.total_steps:,} | Episodes: {self.total_episodes:,}")
        print(f"  Success: {success_rate:.1%} | Replay: {stage.replay_prob:.0%}")
