# autonomous_parking/curriculum.py
"""
v42 IMPROVED Micro-Curriculum for Autonomous Parking
16-stage progressive learning with anti-catastrophic-forgetting replay

Key improvements over v15:
- Fixed duplicate stage names and overlapping thresholds  
- Monotonically increasing complexity with strategic spawn reductions
- Smoother orientation transitions (no catastrophic forgetting)
- Added generalization test before bay specialization
- Earlier multi-lot exposure with simpler scenarios

Production-ready implementation with dataclasses, internal tracking,
and robust stage advancement logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict
import math
import numpy as np


@dataclass
class CurriculumStage:
    """One curriculum stage definition"""
    name: str
    # Which lot(s): "lot_a", "lot_b", or "both"
    lots: Sequence[str]
    # When to advance: total env steps
    advance_at_steps: int
    # Allowed bay IDs (None = filter by orientation only)
    allowed_bays: Optional[Sequence[str]]
    # Allowed goal orientations in radians (None = all)
    allowed_orientations: Optional[Sequence[float]]
    # Max spawn distance to goal in meters (None = env default)
    max_spawn_dist: Optional[float]
    # Spawn side: "left" or "right" (None = random)
    spawn_side: Optional[str]

    # v38.5: Force aligned spawn for straight-in stages
    aligned_spawn: bool = False
    # Replay probability from earlier stages
    replay_prob: float = 0.0
    # v41: Baby Parking Support
    disable_obstacles: bool = False
    lateral_offset: Optional[float] = None # Lateral offset in meters for aligned spawn

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
            "spawn_side": stage.spawn_side,
            "aligned_spawn": stage.aligned_spawn,  # v38.5: Pass flag to env
            # v41: New fields
            "disable_obstacles": stage.disable_obstacles,
            "lateral_offset": stage.lateral_offset,
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
        
        # 1. Minimum steps requirement (don't advance too fast)
        if self.total_steps < stage.advance_at_steps:
            return
        
        # 2. Success rate requirement (Competence-based)
        success_rate = (sum(self._window_success) / len(self._window_success) 
                       if self._window_success else 0.0)
        
        # Require 80% success to advance (User Request: "Only further if success")
        if success_rate < 0.80:
            return
        
        # Advance!
        self.current_stage_idx += 1
        next_stage = self.current_stage
        print(f"\n{'*' * 70}")
        print(f"📚 CURRICULUM STAGE ADVANCE")
        print(f"   From: {stage.name}")
        print(f"   To:   {next_stage.name} (Stage {self.current_stage_idx + 1}/{len(self.stages)})")
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
        """
        v42 IMPROVED Curriculum - Fixes for logical progression issues

        Key improvements over v15:
        1. Fixed duplicate stage names and overlapping thresholds  
        2. Monotonically increasing complexity with strategic spawn reductions
        3. Smoother orientation transitions (no catastrophic forgetting)
        4. Added generalization test (S4) before bay specialization
        5. Earlier multi-lot exposure with simpler scenarios (S12)
        6. More logical bay progression: A2+A3 → A2+A3+A4 → A1-A5 → All
        """
        
        def rad(deg: float) -> float:
            return math.radians(deg)
        
        stages = [
            # ===== PHASE 0: BABY PARKING (0-40k) =====
            CurriculumStage(
                name="S0: Baby - Straight aligned, no obstacles",
                lots=["lot_a"],
                allowed_bays=["A2", "A3"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=6.0,
                spawn_side=None,
                aligned_spawn=True,
                disable_obstacles=True,
                advance_at_steps=20_000,
                replay_prob=0.0,
                note="Learn brake + alignment without distractions"
            ),

            CurriculumStage(
                name="S1: Baby - Lateral offset, no obstacles",
                lots=["lot_a"],
                allowed_bays=["B3", "B4"],
                allowed_orientations=[rad(180.0)],
                max_spawn_dist=6.0,
                spawn_side=None,
                aligned_spawn=True,
                disable_obstacles=True,
                lateral_offset=1.0,
                advance_at_steps=40_000,
                replay_prob=0.1,
                note="Learn steering correction from 1m offset"
            ),

            # ===== PHASE 1: SINGLE BAY MASTERY (40-115k) =====
            CurriculumStage(
                name="S2: A3 only - Close spawn (4m), obstacles ON",
                lots=["lot_a"],
                allowed_bays=["A3"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=4.0,
                spawn_side=None,
                advance_at_steps=70_000,  # 30k steps on this stage
                replay_prob=0.1,
                note="Introduce obstacles with easy spawn"
            ),
            
            CurriculumStage(
                name="S3: A3 only - Medium spawn (8m)",
                lots=["lot_a"],
                allowed_bays=["A3"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=8.0,
                spawn_side=None,  # Both sides
                advance_at_steps=95_000,  # 25k steps
                replay_prob=0.15,
                note="Extend navigation distance"
            ),
            
            # NEW: Test generalization to nearby bay before specializing elsewhere
            CurriculumStage(
                name="S4: A2+A3 - Test generalization (8m)",
                lots=["lot_a"],
                allowed_bays=["A2", "A3"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=8.0,
                spawn_side=None,
                advance_at_steps=115_000,  # 20k steps
                replay_prob=0.2,
                note="Verify A3 learning transfers to adjacent bay"
            ),
            
            # ===== PHASE 2: MULTI-BAY EXPANSION (115-195k) =====
            CurriculumStage(
                name="S5: Three middle bays (A2,A3,A4) - 12m",
                lots=["lot_a"],
                allowed_bays=["A2", "A3", "A4"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=12.0,
                spawn_side=None,
                advance_at_steps=140_000,  # 25k steps
                replay_prob=0.25,
                note="Expand to central cluster"
            ),
            
            CurriculumStage(
                name="S6: Five bays (A1-A5) - 16m",
                lots=["lot_a"],
                allowed_bays=["A1", "A2", "A3", "A4", "A5"],
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=16.0,
                spawn_side=None,
                advance_at_steps=165_000,  # 25k steps
                replay_prob=0.3,
                note="Near-complete bay coverage"
            ),
            
            CurriculumStage(
                name="S7: All bays - 20m - Single orientation mastery",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0)],
                max_spawn_dist=20.0,
                spawn_side=None,
                advance_at_steps=195_000,  # 30k steps
                replay_prob=0.3,
                note="Complete 0° mastery before adding orientations"
            ),
            
            # ===== PHASE 3: ADD PERPENDICULAR (195-260k) =====
            # FIXED: Keep 0° while introducing 90° to prevent forgetting
            CurriculumStage(
                name="S8: 0°+90° mixed - 16m - Dual orientation intro",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0)],
                max_spawn_dist=16.0,  # Easier spawn when adding complexity
                spawn_side=None,
                advance_at_steps=225_000,  # 30k steps
                replay_prob=0.35,
                note="Introduce perpendicular while maintaining straight-in"
            ),
            
            CurriculumStage(
                name="S9: 0°+90° mixed - 22m - Extended dual practice",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0)],
                max_spawn_dist=22.0,
                spawn_side=None,
                advance_at_steps=260_000,  # 35k steps
                replay_prob=0.4,
                note="Solidify two-orientation competence"
            ),
            
            # ===== PHASE 4: ADD REVERSE ORIENTATIONS (260-345k) =====
            CurriculumStage(
                name="S10: Three orientations (0°,90°,180°) - 20m",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0), rad(180.0)],
                max_spawn_dist=20.0,  # Slightly easier when adding 3rd orientation
                spawn_side=None,
                advance_at_steps=300_000,  # 40k steps
                replay_prob=0.45,
                note="Add reverse orientation"
            ),
            
            CurriculumStage(
                name="S11: All four orientations - 22m",
                lots=["lot_a"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0), rad(180.0), rad(-90.0)],
                max_spawn_dist=22.0,
                spawn_side=None,
                advance_at_steps=345_000,  # 45k steps
                replay_prob=0.5,
                note="Complete orientation coverage"
            ),
            
            # ===== PHASE 5: MULTI-LOT INTRODUCTION (345-445k) =====
            # IMPROVED: Introduce Lot B earlier with simpler orientations
            CurriculumStage(
                name="S12: Lot B only - 0°+90° - 18m - New environment",
                lots=["lot_b"],
                allowed_bays=None,
                allowed_orientations=[rad(0.0), rad(90.0)],
                max_spawn_dist=18.0,
                spawn_side=None,
                advance_at_steps=385_000,  # 40k steps
                replay_prob=0.55,
                note="Transfer to new lot with familiar orientations"
            ),
            
            CurriculumStage(
                name="S13: Both lots - All orientations - 24m",
                lots=["both"],
                allowed_bays=None,
                allowed_orientations=None,
                max_spawn_dist=24.0,
                spawn_side=None,
                advance_at_steps=445_000,  # 60k steps
                replay_prob=0.6,
                note="Multi-lot integration"
            ),
            
            # ===== PHASE 6: FINAL MASTERY (445k+) =====
            CurriculumStage(
                name="S14: Full deployment - 28m - Final stretch",
                lots=["both"],
                allowed_bays=None,
                allowed_orientations=None,
                max_spawn_dist=28.0,
                spawn_side=None,
                advance_at_steps=500_000,  # 55k steps
                replay_prob=0.65,
                note="Near-production challenge level"
            ),
            
            CurriculumStage(
                name="S15: Production distribution - No limits",
                lots=["both"],
                allowed_bays=None,
                allowed_orientations=None,
                max_spawn_dist=None,
                spawn_side=None,
                advance_at_steps=10**9,
                replay_prob=0.7,
                note="Full deployment regime"
            ),
        ]
        
        return stages
    
    def log_status(self) -> None:
        """Print current curriculum status"""
        stage = self.current_stage
        success_rate = (sum(self._window_success) / len(self._window_success) 
                       if self._window_success else 0.0)
        
        print(f"Curriculum: Stage {self.current_stage_idx + 1}/{len(self.stages)} - {stage.name}")
        print(f"  Steps: {self.total_steps:,} | Episodes: {self.total_episodes:,}")
        print(f"  Success: {success_rate:.1%} | Replay: {stage.replay_prob:.0%}")
