# Autonomous Parking Agent - Complete Project Documentation

**Last Updated:** 2025-11-26  
**Current Version:** v17.2 🚀 **DYNAMIC TIMEOUT + SPAWN SAFETY + VIDEO ENHANCEMENTS**  
**Project Status:** Production Ready (All Critical Fixes Applied)  
**Architecture:** Hierarchical RL (A* + B-spline + PPO) with Curriculum Learning  
**Key Features:** Normalized observations (43D), Distance-normalized rewards, Smart waypoint subsampling, Dynamic episode timeouts

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Environment Design](#3-environment-design)
4. [Waypoint Generation System](#4-waypoint-generation-system)
5. [Reward Function System](#5-reward-function-system)
6. [Curriculum Learning](#6-curriculum-learning)
7. [Training System](#7-training-system)
8. [Evaluation System](#8-evaluation-system)
9. [File Structure](#9-file-structure)
10. [Current Implementation (v17.2)](#10-current-implementation-v172)
11. [Version History](#11-version-history)
12. [Usage Guide](#12-usage-guide)
13. [Troubleshooting](#13-troubleshooting)
14. [Development Roadmap](#14-development-roadmap)

---

## 🎉 What's New in v17.2

### Major Improvements
1. **✅ Dynamic Episode Timeout** - Episode length now scales with path distance (300-2000 steps). Fair for all spawn locations.
2. **✅ Spawn Safety Guarantee** - Strict 4-corner road clamping ensures 100% valid spawns. Verified with 15,000 resets.
3. **✅ Video Overlay Enhancements** - Training videos now show spawn position, target bay, and episode number in title.
4. **✅ Video Sync Fix** - Fixed multi-env desync bug where videos showed mid-episode frames.

### Critical Fixes (v17.2)
1. **🐛 FIXED:** Off-road spawns in Lot B (V-bays could overshoot intersection).
2. **🐛 FIXED:** Video recorder triggered by wrong environment (any env vs env[0]).
3. **🐛 FIXED:** Fixed timeout too short for long paths (was 300, now dynamic).
4. **🐛 FIXED:** Lateral spawn clamps too loose (car corners could clip road edges).

### Previous Improvements (v17.0-v17.1)
1. **✅ Dense Path Generation** - A* produces high-fidelity dense paths for geometric accuracy.
2. **✅ Smart Subsampling** - Intelligent filtering keeps only key waypoints (turns) for the RL agent.
3. **✅ Distance-Normalized Rewards** - Fixed reward budget (300.0) distributed over path length.
4. **✅ Exact Goal Reaching** - B-spline now samples full [0, 1] range, guaranteeing path ends exactly at bay center.
5. **🐛 FIXED:** Crash at last waypoint (IndexError in `step()`).
6. **🐛 FIXED:** Double success tracking in `reset()`.

### Expected Impact
- **Stable Training:** No crashes, no off-road spawns, no video artifacts.
- **Fair Learning:** Agent gets appropriate time for all path lengths.
- **Easy Debugging:** Video titles show exactly what the agent was asked to do.
2. **🐛 FIXED:** Double success tracking in `reset()`.
3. **🐛 FIXED:** Phantom point removal in smoothing (cleaner, more robust).
4. **🐛 FIXED:** Edge case safety for short paths.

### Expected Impact
- **Stable Training:** No more reward explosions or crashes.
- **Fair Rewards:** Agent learns efficiency, not path lengthening.
- **Geometric Precision:** Paths lead exactly to the goal.

---

## 1. Project Overview

### 1.1 Goal
Develop an autonomous parking agent capable of navigating complex parking lots (lot_a and lot_b) and parking in designated bays with high precision using hierarchical reinforcement learning with curriculum progression.

### 1.2 Key Challenges
1. **Long-horizon task** - Navigate 10-25m to bay, requires up to 2000 steps
2. **Complex path planning** - Must stay on roads, avoid obstacles, enter correct bay
3. **Precision parking** - Final alignment requires <0.8m position, <0.5 rad yaw accuracy
4. **Multi-lot generalization** - Must work across different parking lot layouts
5. **Multi-orientation support** - Bays facing different directions (0°, 90°, 180°, 270°)
6. **Sparse rewards** - Success only at end of episode

### 1.3 Solution Approach
**Hierarchical RL with Curriculum:**
- **High-level:** A* path planning on road-aware grid
- **Mid-level:** B-spline smoothing for kinematically feasible paths  
- **Low-level:** PPO trained to follow waypoints and execute final parking
- **Curriculum:** Progressive difficulty from easy (close spawn, few orientations) to hard (far spawn, all bays)
- **Parking mode:** Specialized bay-frame alignment rewards for final maneuver

---

## 2. System Architecture

### 2.1 Component Diagram
```
┌─────────────────────────────────────────────────────┐
│            CurriculumManager (v15)                  │
│  - Stage selection (S1-S15)                         │
│  - Lot sampling (lot_a, lot_b)                      │
│  - Bay filtering                                    │
│  - Difficulty progression                           │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│              PPO Agent (Stable-Baselines3)          │
│  - Policy Network (MLP 256x256)                     │
│  - Value Network                                    │
│  - 4 Parallel Envs                                  │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│            WaypointEnv (v15.1)                      │
│  - A* Planning (road-aware grid)                    │
│  - B-spline Smoothing (phantom point)               │
│  - Waypoint Following Mode                          │
│  - Final Approach Mode (v14.20)                     │
│  - Parking Mode (v15.1 - bay-frame)                 │
│  - Stuck Detection                                  │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│            ParkingEnv (Base)                        │
│  - Bicycle Model Physics                            │
│  - 32-beam LIDAR                                    │
│  - Collision Detection                              │
│  - Multi-lot Support                                │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│          Configuration (YAML)                       │
│  - lot_a: 12 bays (A1-A6, B1-B6)                    │
│  - lot_b: 10 bays (H1-H5, V1-V5)                    │
│  - Road definitions                                 │
│  - Bay positions & orientations                     │
└─────────────────────────────────────────────────────┘
```

### 2.2 Data Flow (Episode Lifecycle)
```
Episode Start
    ↓
[Curriculum] Select scenario: (lot, bays, orientations, spawn_dist)
    ↓
[Reset] Load lot config, spawn car, select goal bay
    ↓
[A*] Plan path: start → staging (road) → entrance → goal
    ↓
[B-spline] Smooth path with phantom point for C1 continuity
    ↓
┌─ STEP LOOP (max 2000 steps) ──────────────────────┐
│                                                    │
│  [Obs] Waypoint-relative position + LIDAR + state │
│    ↓                                               │
│  [PPO] Action → [v_cmd, steer_cmd]                │
│    ↓                                               │
│  [Physics] Update state [x, y, yaw, v]            │
│    ↓                                               │
│  [Waypoint Progress] Check distance to current WP │
│  - If < threshold: advance to next waypoint       │
│  - Award exponential bonuses                       │
│    ↓                                               │
│  [Mode Selection]                                  │
│  ┌─ Normal Mode (far from goal)                   │
│  │  - Waypoint following rewards                  │
│  │  - Velocity incentives                          │
│  │                                                 │
│  ├─ Final Approach (dist<2m, yaw<0.7 rad)         │
│  │  - Alignment improvement rewards                │
│  │  - Backing-friendly logic                       │
│  │  - Combo rewards                                │
│  │                                                 │
│  └─ Parking Mode (dist<8m, last 2 waypoints)      │
│     - Bay-frame penalties (cx, cy, yaw)            │
│     - Success check (relaxed thresholds)           │
│     - Stuck detection (50-step timeout)            │
│    ↓                                               │
│  [Success Check]                                   │
│  - |cy_bay| < 0.8m AND |yaw_err| < 0.5 rad        │
│    → +1500 reward, done=True                       │
│    ↓                                               │
│  [Termination]                                     │
│  - Success | Timeout | Collision | Stuck | OOB    │
│                                                    │
└────────────────────────────────────────────────────┘
    ↓
[Curriculum] Update stage based on success/steps
    ↓
[PPO] Update policy every 2048 steps
```

---

## 3. Environment Design

### 3.1 State Space

#### 3.1.1 Observation (43D vector - v16 NORMALIZED)
```python
obs = [
    # Waypoint guidance (5D) - NORMALIZED
    local_dx / 10.0,      # Waypoint x in car frame [-1, 1]
    local_dy / 10.0,      # Waypoint y in car frame [-1, 1]
    dtheta / π,           # Yaw difference to waypoint [-1, 1]
    v / 2.0,              # Current velocity [-1, 1]
    dist / 20.0,          # Distance to current waypoint [0, 1]
    
    # Goal-bay coordinates (4D) - NEW in v16 - NORMALIZED
    cx_bay / 10.0,        # Bay-frame x (longitudinal) [-1, 1]
    cy_bay / 10.0,        # Bay-frame y (lateral) [-1, 1]
    yaw_err / π,          # Yaw error to goal [-1, 1]
    dist_to_goal / 20.0,  # Distance to goal [0, 1]
    
    # Progress indicators (2D) - NEW in v16
    waypoint_progress,    # Fraction of waypoints completed [0, 1]
    is_near_goal,         # Binary flag (dist < 5m) {0, 1}
    
    # LIDAR (32D) - 360° coverage
    lidar[0:31] / 10.0,   # Distance readings (normalized) [0, 1]
]
```

**Key v16 Change:** All features normalized to similar scales for 2x faster learning!

#### 3.1.2 Action Space (2D continuous)
```python
action = [
    v_cmd,      # Velocity [-2.0, +2.0] m/s (backing allowed!)
    steer_cmd   # Steering [-0.6, +0.6] rad (~34°)
]
```

### 3.2 Physics Model
**Kinematic Bicycle Model (dt=0.1s):**
```python
x_new = x + v * cos(yaw) * dt
y_new = y + v * sin(yaw) * dt
yaw_new = yaw + (v / wheelbase) * tan(steer) * dt
v_new = v_cmd  # Direct velocity control

# Parameters:
car_length = 4.5m
car_width = 2.0m
wheelbase = 2.7m
```

### 3.3 Sensors
**32-Beam LIDAR:**
- Range: 0-10m
- FOV: 360°
- Detects: obstacles, occupied bays, target bay walls

---

## 4. Waypoint Generation System (v15.1)

### 4.1 Overview
The waypoint generation system uses a three-stage approach:
1. **A* Grid Planning** - Discrete path on road-aware grid
2. **Multi-stage approach points** - Staging → Entrance → Pre-goal → Goal
3. **B-spline Smoothing** - Kinematically feasible smooth curve

### 4.2 A* Planning (`astar.py`)

#### 4.2.1 Obstacle Grid Creation
```python
create_obstacle_grid(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.5m,
    bays=occupied_bays,      # All bays EXCEPT goal
    roads=road_definitions,
    goal_bay=target_bay      # Explicitly opened
)

# Grid marking logic:
# 1. START: Everything = OBSTACLE (1)
# 2. Mark ROADS = FREE (0)
# 3. Mark NON-GOAL BAYS = OBSTACLE (1)
# 4. Mark GOAL BAY interior = FREE (0)
# 5. OPEN goal bay entrance side based on yaw
```

**Key Innovation:** Roads are explicitly marked as free space, forcing A* to stay on roads until entering the goal bay.

#### 4.2.2 Path Planning
```python
# Plan from spawn to staging point (on road in front of bay)
road_path = planner.plan(start, staging, obstacles)

# Staging point logic:
# - lot_a top row (yaw=0): staging at (bay_x, 0.0)
# - lot_a bottom row (yaw=π): staging at (bay_x, 0.0)
# - lot_b H bays (yaw=0): staging at (bay_x, 10.0)
# - lot_b V bays (yaw=π/2): staging at (0.0, bay_y)
```

### 4.3 Multi-Stage Approach
```python
# 1. Road path (A* result)
road_path = [(x1, y1, θ1), ..., (staging_x, staging_y, staging_θ)]

# 2. Entrance waypoint (2m from bay mouth)
entrance = goal + 2m * rotate(-yaw)  # 2m back along bay normal

# 3. Pre-goal waypoint (1m from bay mouth)
pregoal = goal + 1m * rotate(-yaw)   # 1m back along bay normal

# 4. Goal (bay center)
goal = (goal_x, goal_y, goal_yaw)

# Complete path
complete_path = road_path + [entrance, pregoal, goal]
```

### 4.4 B-spline Smoothing
```python
def _smooth_path(waypoints):
    """
    Smooth path using cubic B-spline (v17.1).
    
    Key technique: 
    - Fit spline to dense A* points
    - Sample full range [0, 1] for exact start-to-goal connection
    - No phantom point needed (s=0.1 ensures alignment)
    """
    # Fit spline directly
    tck, u = splprep([pts_x, pts_y], s=0.1, k=3)
    
    # Sample full range
    u_new = linspace(0, 1.0, num_pts)
    smooth_x, smooth_y = splev(u_new, tck)
    
    return smooth_waypoints
```

**Result:** Dense, smooth path ending exactly at the goal.
**Smart Subsampling:** The dense path is then filtered to keep only key turning points for the RL agent.

---

## 5. Reward Function System (v15.1)

### 5.1 Mode-Based Reward Structure

The agent operates in THREE modes with different reward structures:

#### 5.1.1 Normal Waypoint Following Mode
**Triggers:** Default mode until final approach conditions met

```python
# Base distance penalty
reward = -dist_to_waypoint

# Progress reward
if prev_dist is not None:
    progress = prev_dist - dist_to_waypoint
    reward += 3.0 * progress

# Velocity shaping
if v > 0.1:
    reward += 0.5 * v     # Encourage movement
if v < 0.3:
    reward -= 0.5         # Penalize stopping

# Waypoint bonuses (exponential)
# Waypoint bonuses (Distance-Normalized - v17)
# Fixed budget (300.0) distributed over total path length
reward_per_meter = 300.0 / total_path_length
multiplier = 1.0 + 0.5 * progress_ratio  # Progressive bonus (1.0 -> 1.5)

waypoint_bonus = segment_len * reward_per_meter * multiplier

# Time penalty
reward -= 0.05
```

#### 5.1.2 Final Approach Mode (v14.20)
**Triggers:** `dist_to_goal < 2.0m AND abs(yaw_err) < 0.7 rad`

This mode was developed through extensive iteration and is the key to achieving successful parkings.

```python
# Freeze waypoint progression
current_waypoint_idx = len(waypoints)

# Calculate improvements
progress = prev_dist_to_goal - dist_to_goal
yaw_improve = abs(prev_yaw_err) - abs(yaw_err)
cx_improve = abs(prev_cx_bay) - abs(cx_bay)
cy_improve = abs(prev_cy_bay) - abs(cy_bay)

alignment_gain = 0.5*yaw_improve + 0.25*cx_improve + 0.25*cy_improve

# Backing-friendly logic
if progress < 0.0 and alignment_gain <= 0.0:
    # Moving away AND getting worse → bad
    reward += 4.0 * progress  # Negative penalty
else:
    # Either getting closer OR improving alignment → good!
    reward += 4.0 * progress
    reward += 8.0 * alignment_gain  # KEY INNOVATION

# Extra yaw focus very close
if dist_to_goal < 2.5:
    reward += 7.5 * yaw_improve

# Adaptive speed penalty
if v_linear > 0.5:
    penalty_scale = 0.8
    if dist_to_goal < 3.0 and abs(yaw_err) < 0.7:
        penalty_scale = 0.4  # Reduce when doing well
    reward -= penalty_scale * v_linear

# COMBO REWARD (both distance AND yaw good)
if dist_to_goal < 3.0 and abs(yaw_err) < 0.7:
    combo = 3.0 * (1 - dist/3.0) * (1 - yaw/0.7)
    reward += combo  # Max +3.0

# Anti-freeze penalty
if dist_to_goal > 0.8 and v_linear < 0.05:
    reward -= 0.01

# Success-region soft shaping (gradients near success)
if dist_to_goal < 1.5:
    reward += 0.1 * (1 - dist/1.5)
    if abs(yaw_err) < 1.8:
        reward += 0.05 * (1 - yaw/1.8)
    if abs(cy_bay) < 1.2:
        reward += 0.05 * (1 - cy/1.2)
```

#### 5.1.3 Parking Mode (v15.1) - NEW
**Triggers:** `dist_to_goal < 8.0m AND reached last 2 waypoints`

This mode runs ON TOP OF final approach and focuses on bay-frame alignment.

```python
# Track progress for stuck detection
if dist_to_goal < best_dist_to_goal - 0.05:
    best_dist_to_goal = dist_to_goal
    no_progress_steps = 0
else:
    no_progress_steps += 1

# Bay-frame coordinate penalties (GENTLE for untrained model)
reward -= 1.0 * abs(cy_bay)   # Lateral offset
reward -= 1.0 * abs(yaw_err)  # Yaw error

# Depth penalty (too deep/shallow in bay)
bay_length = 5.5m
target_depth = bay_length * 0.4  # 40% into bay
depth_error = abs(abs(cx_bay) - target_depth)
reward -= 0.5 * depth_error

# SUCCESS CHECK (relaxed thresholds for curriculum start)
well_aligned = (
    abs(cy_bay) < 0.8m AND      # 80cm lateral tolerance
    abs(yaw_err) < 0.5 rad AND  # ~29° yaw tolerance
    abs(cx_bay) < 0.6 * bay_length
)

if well_aligned:
    reward += 1500.0
    terminated = True
    info["success"] = True
    info["parking_quality"] = {
        "lateral_offset": abs(cy_bay),
        "yaw_error": abs(yaw_err),
        "depth": abs(cx_bay)
    }

# STUCK DETECTION
if no_progress_steps > 50:
    reward -= 200.0
    terminated = True
    info["terminated_stuck"] = True

# Encourage entering bay vs hovering outside
if dist_to_goal > 6.0:
    reward -= 2.0

inside_bay = (abs(cx_bay) < bay_length/2 and
              abs(cy_bay) < bay_width/2)
if inside_bay and not success:
    reward += 5.0  # Commitment bonus
```

### 5.2 Success / Failure Rewards
```python
# Success (discovered by parent's strict check OR parking mode)
if success:
    reward += 5000.0  # From parent class check
    # OR +1500.0 from parking mode check
    
# Out of bounds
if abs(x) > 25 or abs(y) > 25:
    reward -= 20.0
    done = True
    
# Timeout
if steps >= 2000:
    done = True
```

---

## 6. Curriculum Learning (v15)

### 6.1 Curriculum Stages

The curriculum progressively increases difficulty across 15 stages:

```python
# Stage definitions (curriculum.py)
S1 = CurriculumStage(
    lots=["lot_a"],
    allowed_bays=["A1", "A2"],      # Only 2 easy bays
    allowed_orientations=[0.0],      # Only one orientation
    max_spawn_dist=10.0,             # Close spawns
    advance_at_steps=50000,
    replay_prob=0.0
)

S2 = CurriculumStage(
    lots=["lot_a"],
    allowed_bays=["A1", "A2", "A3", "A4"],  # 4 bays
    allowed_orientations=[0.0],
    max_spawn_dist=12.0,
    advance_at_steps=100000,
    replay_prob=0.1
)

# ... progressive stages ...

S12 = CurriculumStage(
    lots=["lot_a", "lot_b"],         # Both lots
    allowed_bays=None,                # All bays
    allowed_orientations=None,        # All orientations
    max_spawn_dist=None,              # No limit
    advance_at_steps=1200000,
    replay_prob=0.3
)

# Final stages (S13-S15) maintain full difficulty
```

### 6.2 Curriculum Manager

```python
class CurriculumManager:
    def sample_scenario(self) -> Dict:
        """
        Sample scenario from current stage (with optional replay).
        
        Returns:
            {
                "stage_idx": int,
                "effective_stage_idx": int,  # May be earlier if replaying
                "stage_name": str,
                "lot": str,
                "allowed_bays": List[str] or None,
                "allowed_orientations": List[float] or None,
                "max_spawn_dist": float or None
            }
        """
        # With probability replay_prob, sample from earlier stage
        if random() < current_stage.replay_prob:
            replay_stage = sample_from_earlier_stages()
        else:
            replay_stage = current_stage
            
        # Sample lot
        lot = random.choice(replay_stage.lots)
        
        return scenario_dict
    
    def update_after_episode(self, success: bool, steps: int):
        """Update curriculum after episode completes."""
        total_env_steps += steps
        
        # Check for stage advancement
        if total_env_steps >= current_stage.advance_at_steps:
            advance_to_next_stage()
```

### 6.3 Integration with WaypointEnv

```python
# In waypoint_env.py reset():
if self.enable_curriculum and self.curriculum is not None:
    scenario = self.curriculum.sample_scenario()
    self.current_lot = scenario["lot"]
    
    # Reload lot config if changed
    if self.current_lot != self.lot_name:
        self._load_lot(self.current_lot)
    
    # Filter bays by curriculum constraints
    if bay_id is None:
        allowed_bays = scenario.get("allowed_bays")
        allowed_orientations = scenario.get("allowed_orientations")
        
        # Build eligible_bays list based on constraints
        # ...
        
        bay_id = random.choice(eligible_bays)
    
    # Apply spawn distance override
    max_spawn = scenario.get("max_spawn_dist")
    self.max_spawn_dist_override = max_spawn
```

---

## 7. Training System

### 7.1 Training Configuration

```python
# PPO Hyperparameters
ppo_params = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "policy_kwargs": {
        "net_arch": [256, 256]
    }
}

# Environment settings
env_params = {
    "max_episode_steps": 2000,  # Increased for complex paths
    "n_envs": 4,                 # Parallel environments
    "enable_curriculum": True    # Use curriculum learning
}
```

### 7.2 Training Commands

**Fresh Training (1.5M steps recommended for v15.1):**
```bash
cd ~/autonomous_parking_ws/src/autonomous_parking

../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \
  --total-steps 1500000 \
  --n-envs 4 \
  --max-episode-steps 2000 \
  --use-curriculum \
  --run-name hierarchical_v15_1_full \
  --save-freq 50000 \
  | tee logs/train_v15_1_$(date +%Y%m%d_%H%M%S).log
```

**Quick Test (100k steps):**
```bash
../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \
  --total-steps 100000 \
  --n-envs 4 \
  --use-curriculum \
  --run-name hierarchical_v15_1_test
```

### 7.3 Expected Training Timeline

| Timesteps | Curriculum Stage | Expected Behavior | Success Rate |
|-----------|------------------|-------------------|--------------|
| 0-50k | S1 (lot_a, 2 bays, close) | Learning basic parking | 5-10% |
| 50k-200k | S2-S4 (more bays) | Improving precision | 15-25% |
| 200k-500k | S5-S8 (both orientations) | Multi-orientation mastery | 30-45% |
| 500k-1M | S9-S11 (both lots) | Generalization | 45-65% |
| 1M-1.5M | S12-S15 (full difficulty) | Final polish | **60-80%** |

---

## 8. Evaluation System

### 8.1 Evaluation Commands

**Standard Evaluation (20 episodes):**
```bash
cd ~/autonomous_parking_ws/src/autonomous_parking

../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_a \
  --model-dir results/ppo_hierarchical/hierarchical_v15_1_full \
  --episodes 20 \
  --max-episode-steps 2000
```

**Cross-lot Evaluation:**
```bash
# Test lot_a
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_a \
  --model-dir results/ppo_hierarchical/hierarchical_v15_1_full \
  --episodes 50

# Test lot_b
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_b \
  --model-dir results/ppo_hierarchical/hierarchical_v15_1_full \
  --episodes 50
```

### 8.2 Evaluation Metrics

```python
# Per-episode metrics
{
    "success": bool,
    "min_distance": float,
    "final_yaw_error": float,
    "steps_taken": int,
    "waypoints_reached": int,
    "parking_mode_activated": bool,
    "stuck_detected": bool,
    "parking_quality": {
        "lateral_offset": float,
        "yaw_error": float,
        "depth": float
    }
}

# Summary statistics
{
    "success_rate": float,
    "mean_min_distance": float,
    "median_final_yaw": float,
    "parking_mode_activation_rate": float,
    "mean_episode_length": float
}
```

---

## 9. File Structure

```
autonomous_parking_ws/
├── src/autonomous_parking/autonomous_parking/
│   ├── env2d/
│   │   ├── parking_env.py           # Base physics environment
│   │   ├── waypoint_env.py          # ⭐ v15.1 Hierarchical env
│   │   └── curriculum_env.py        # Legacy (not used)
│   ├── planning/
│   │   ├── astar.py                 # ⭐ A* planner (road-aware)
│   │   └── __init__.py
│   ├── curriculum.py                # ⭐ v15 Curriculum manager
│   ├── sensors/
│   │   ├── lidar.py                 # 32-beam LIDAR
│   │   └── __init__.py
│   ├── worlds/
│   │   ├── lot_a_empty.yaml        # lot_a configuration
│   │   ├── lot_b_empty.yaml        # lot_b configuration
│   │   └── bays.yaml               # ⭐ Unified bay definitions
│   ├── config_loader.py             # YAML loader
│   ├── sb3_train_hierarchical.py   # ⭐ Training script
│   ├── sb3_eval_hierarchical.py    # ⭐ Evaluation script
│   └── __init__.py
├── visualize_all_paths.py          # ⭐ Path visualization tool
├── results/ppo_hierarchical/        # Saved models
│   └── hierarchical_v15_1_full/
│       ├── ppo_parking_final.zip
│       └── checkpoints/
├── logs/                            # Training logs
└── AUTONOMOUS_PARKING_DOCUMENTATION.md  # ⭐ THIS FILE
```

---

## 10. Current Implementation (v15.1)

### 10.1 Version Timeline

| Version | Key Features | Status |
|---------|-------------|--------|
| v14.16 | First final approach | Failed (never triggered) |
| v14.17 | Tighter thresholds | Failed (too strict) |
| v14.18 | 6m trigger, backing-friendly | ✅ First successes |
| v14.19 | Combo rewards | ✅ 1.5% success |
| v14.20 | Alignment-gated trigger | ✅ 9 successes (500k) |
| **v15.0** | **Curriculum system** | ✅ Framework ready |
| **v15.1** | **Parking mode + stuck detection** | ✅ **CURRENT** |

### 10.2 v15.1 Key Features

1. **Multi-lot Support**
   - Seamless switching between lot_a and lot_b
   - Automatic configuration reloading
   - 22 total bays (12 + 10)

2. **Curriculum Learning**
   - 15 progressive stages
   - Automatic difficulty scaling
   - Experience replay from earlier stages
   - Lot/bay/orientation filtering

3. **Advanced Waypoint System**
   - A* road-aware planning
   - B-spline with phantom point
   - Multi-stage approach (staging→entrance→pregoal→goal)
   - 15-25 smooth waypoints per episode

4. **Three-Mode Reward System**
   - Normal: Waypoint following
   - Final Approach: v14.20 alignment logic
   - Parking: Bay-frame penalties + stuck detection

5. **Robust Termination**
   - Success (relaxed: 0.8m lateral, 0.5 rad yaw)
   - Stuck detection (50 steps no progress)
   - Timeout (2000 steps)
   - Standard collision/OOB

### 10.3 Known Limitations

**Current v15.1 with 700k v14.20 model:**
- ❌ 0% success rate (model trained with OLD rewards)
- ❌ Agent gets stuck ~3.5m from goal
- ❌ Parking mode triggers but then agent retreats

**Requires:**
- ✅ Fresh training from scratch (1.5M steps)
- ✅ Curriculum progression
- ✅ New reward structure learning

---

## 11. Version History

### Complete Evolution

**Phase 1: Foundation (v1-v10)**
- Basic A* + PPO integration
- Simple waypoint following
- No final approach behavior

**Phase 2: First Final Approach (v14.16-v14.17)**
- Attempted final approach at <3m: FAILED
- Tightened to <1.5m: FAILED (too close)

**Phase 3: Breakthrough (v14.18)**
- **6m trigger** - KEY INSIGHT
- Backing-friendly rewards
- 71% yaw improvement observed
- First successful parkings!

**Phase 4: Refinement (v14.19-v14.20)**
- Combo rewards (distance AND yaw)
- Adaptive speed penalties
- Alignment-gated final approach trigger
- **Result: 1.5% success, 0.04m precision**

**Phase 5: Curriculum System (v15.0)**
- 15-stage curriculum designed
- Multi-lot support added
- Bay/orientation filtering
- Automatic difficulty progression

**Phase 6: Parking Mode (v15.1) - CURRENT**
- Bay-frame coordinate rewards
- Stuck detection (50-step)
- Relaxed success thresholds
- Inside-bay encouragement
- **Ready for 1.5M training**

---

## 12. Usage Guide

### 12.1 Quick Start

**1. Visualize Paths:**
```bash
cd ~/autonomous_parking_ws
python visualize_all_paths.py
# Output: path_visualizations/lot_*.png
```

**2. Start Training:**
```bash
cd ~/autonomous_parking_ws/src/autonomous_parking

../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \
  --total-steps 1000000 \
  --n-envs 4 \
  --use-curriculum \
  --run-name my_training_run \
  | tee logs/train_$(date +%Y%m%d_%H%M%S).log
```

**3. Monitor Progress:**
```bash
# In another terminal
../../.venv/bin/python -m tensorboard.main \
  --logdir results/ppo_hierarchical/my_training_run
```

**4. Evaluate:**
```bash
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_a \
  --model-dir results/ppo_hierarchical/my_training_run \
  --episodes 50
```

### 12.2 Understanding Output

**Training Logs:**
```
*** CURRICULUM LEVEL UP: L3 (Medium-Easy) ***
[WaypointEnv.reset] lot=lot_a, goal_bay=A3, waypoints=18
✓ Waypoint 3/6 reached (bonus: 112.5)
🅿️  PARKING MODE ACTIVATED at step 245, dist=5.61m
✅ PARKING SUCCESS! cy=0.345m, yaw_err=0.218rad
Episode finished: success=True, steps=267
```

**Evaluation Summary:**
```
=== EVAL SUMMARY ===
Episodes:        50
Success rate:    72.0% (36/50)
Mean min dist:   0.52m
Mean steps:      412
Parking mode:    94% activation
Stuck:           4% (2/50)
```

---

## 13. Troubleshooting

### 13.1 Known Issues (v16)

**✅ FIXED in v16: Video recording crash on macOS**
- **Was:** `AttributeError: 'FigureCanvasMac' has no 'tostring_rgb'`
- **Fix Applied:** Now uses `tostring_argb()` with proper ARGB handling
- **Status:** ✅ RESOLVED

**✅ FIXED in v16: Over-harsh anti-freeze penalty**
- **Was:** `-1.0` penalty at `dist > 3.0m` discouraged valid slow maneuvers
- **Fix Applied:** Reduced to `-0.3`, triggered only at `dist > 5.0m` AND `yaw_err > 0.5`
- **Status:** ✅ RESOLVED

**✅ FIXED in v16: Unbalanced reward scales**
- **Was:** Waypoint bonuses (55-97) overwhelmed per-step rewards
- **Fix Applied:** Reduced to (11-19), success bonus 2000→500
- **Status:** ✅ RESOLVED

### 13.2 Common Issues

**Issue: Agent stops 3-5m from goal**
- **Cause:** Model trained with old rewards (pre-v16)
- **Fix:** Retrain from scratch with v16

**Issue: "No module named 'autonomous_parking'"**
- **Cause:** Wrong working directory
- **Fix:** `cd ~/autonomous_parking_ws/src/autonomous_parking`

**Issue: Curriculum not advancing**
- **Cause:** Not enough successful episodes at current stage
- **Fix:** Adjust `advance_at_steps` in curriculum.py or train longer

**Issue: Agent oscillates near bay**
- **Cause:** Stuck detection not triggering fast enough
- **Fix:** Already has dual stuck detection in v16 (60 steps + time near goal)

---

## 13.4 Spawn Verification (v17.2)

### Purpose
Verify that 100% of spawns are strictly on the road (all 4 car corners within road boundaries).

### Verification Script
Location: `src/autonomous_parking/verify_spawns.py`

**Run Verification:**
```bash
cd ~/autonomous_parking_ws/src/autonomous_parking
../../.venv/bin/python verify_spawns.py
```

**What it does:**
- Tests 5,000 spawns per scenario (15,000 total)
- Checks Lot A (all bays)
- Checks Lot B (H-bays and V-bays separately)
- Generates visual heatmaps showing spawn distributions
- Reports min/max X and Y coordinates

**Expected Output:**
```
Testing Lot A (All Bays)...
  [STATS] X Range: [-22.00, 22.00]
  [STATS] Y Range: [-0.50, 0.50]
  ✅ SUCCESS: Spawns strictly within +/- 2.0m Y.

Testing Lot B (H-Bays)...
  [STATS] X Range: [-20.00, 20.00]
  [STATS] Y Range: [9.50, 10.50]
  ✅ SUCCESS: H-Bay spawns strictly within [8.0, 12.0] Y.

Testing Lot B (V-Bays)...
  [STATS] X Range: [-0.50, 0.50]
  [STATS] Y Range: [-22.00, 11.00]
  ✅ SUCCESS: V-Bay spawns strictly within road bounds.
```

**Generated Files:**
- `spawn_verify_lot_a_all.png` - Heatmap of Lot A spawns
- `spawn_verify_lot_b_H3.png` - Heatmap of H-bay spawns
- `spawn_verify_lot_b_V3.png` - Heatmap of V-bay spawns

### Spawn Safety Guarantees (v17.2)

**Road Geometry:**
- Lot A: Road width = 7.5m (y ∈ [-3.75, 3.75])
- Lot B H-Road: Width = 7.5m (y ∈ [6.25, 13.75])
- Lot B V-Road: Width = 7.5m (x ∈ [-3.75, 3.75])

**Car Dimensions:**
- Length: 4.2m (half-length: 2.1m)
- Width: 1.9m (half-width: 0.95m)

**Safety Margins:**
- Lateral: Road half-width (3.75m) - Car half-width (0.95m) - Buffer (0.8m) = **±2.0m safe zone**
- Longitudinal: Road end (25.0m) - Car half-length (2.1m) - Buffer (0.9m) = **±22.0m safe zone**

**Clamping Logic:**
```python
# Lot A
spawn_x = np.clip(spawn_x, -22.0, 22.0)  # Longitudinal
spawn_y = np.clip(spawn_y, -2.0, 2.0)    # Lateral

# Lot B (H-bays)
spawn_x = np.clip(spawn_x, -22.0, 22.0)
spawn_y = np.clip(spawn_y, 8.0, 11.0)    # Road at y=10

# Lot B (V-bays)
spawn_x = np.clip(spawn_x, -2.0, 2.0)    # Road at x=0
spawn_y = np.clip(spawn_y, -22.0, 11.0)  # Prevents intersection overshoot
```

### 13.3 Debugging Tools

**Print Current Mode:**
```python
# Add to waypoint_env.py step()
if self.steps % 50 == 0:
    print(f"Mode: {'Parking' if parking_mode else 'Final' if in_final_approach else 'Normal'}")
    print(f"Dist: {dist_to_goal:.2f}m, Yaw: {yaw_err:.3f}rad")
```

**Visualize Episode:**
```bash
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_a \
  --model-dir results/ppo_hierarchical/my_run \
  --episodes 1 \
  --render    # Shows matplotlib animation
```

---

## 14. Development Roadmap

### 14.1 Completed ✅
- ✅ Base environment with physics
- ✅ A* planner (road-aware)
- ✅ B-spline smoothing
- ✅ Waypoint following rewards
- ✅ Final approach mode (v14.20)
- ✅ Curriculum system (v15.0)
- ✅ Parking mode (v15.1)
- ✅ Multi-lot support
- ✅ Stuck detection
- ✅ Dynamic episode timeout (v17.2)
- ✅ Spawn safety verification (v17.2)
- ✅ Video overlay enhancements (v17.2)

### 14.2 In Progress 🔄
- 🔄 Fresh 1.5M training run with v17.2
- 🔄 Cross-lot generalization testing
- 🔄 Final evaluation for report

### 14.3 Future Enhancements 🔮
- Reverse parking (currently forward-only)
- Dynamic obstacles (moving cars)
- Weather/lighting variations
- Different vehicle models
- Real-world deployment (ROS integration)

---

## Quick Reference

### Training (v17.2)
```bash
cd ~/autonomous_parking_ws/src/autonomous_parking

../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \
  --total-steps 1500000 \
  --n-envs 4 \
  --max-episode-steps 2000 \
  --use-curriculum \
  --run-name hier_v17_2_full \
  --record-video \
  --video-freq 25 \
  --eval-freq 50000 \
  --save-freq 50000 \
  --checkpoint-freq 100000 \
  --log-interval 10 \
  --quiet-env \
  | tee logs/train_hier_v17_2_full_$(date +%Y%m%d_%H%M%S).log
```

### Evaluation
```bash
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical --lot lot_a --model-dir results/ppo_hierarchical/hier_v17_2_full --episodes 50
```

### Spawn Verification
```bash
../../.venv/bin/python verify_spawns.py
```

### Visualization
```bash
cd ~/autonomous_parking_ws
python visualize_all_paths.py
```

### TensorBoard
```bash
../../.venv/bin/python -m tensorboard.main --logdir results/ppo_hierarchical/tb
```

---

**End of Documentation**  
**Version:** v17.2  
**Last Updated:** 2025-11-26  
**Status:** Production Ready - All Systems Go 🚀

