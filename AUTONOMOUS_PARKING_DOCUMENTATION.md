# Autonomous Parking Agent - Complete Project Documentation

**Last Updated:** 2025-12-04  
**Current Version:** v35 🚀 **CORRIDOR CONSTRAINT FIX + CRITICAL IMPROVEMENTS**  
**Project Status:** Active Development (Major Corridor Detection Bug Fixed)  
**Architecture:** Hierarchical RL (A* + B-spline + PPO) with Curriculum Learning + Corridor Constraints  
**Key Features:** v35 perpendicular distance corridor checking, 4-corner car validation, asymmetric penalties, steering control, proximity scaling

---

## 🎉 What's New in v35 (December 2024)

### CRITICAL BUG FIX: Corridor Detection Completely Rewritten

**The Problem (v34):**
- Corridor constraint calculated distance to boundary **POINTS**, not perpendicular distance to boundary **LINES**
- Only checked car **CENTER**, not corners
- Agent could go 5-7 meters off-path with minimal penalty
- Episodes never terminated for corridor violations

**The Solution (v35):**
- ✅ **Perpendicular distance to path segments** - Mathematically correct distance calculation
- ✅ **All 4 car corners checked** - Front-left, front-right, rear-left, rear-right
- ✅ **Asymmetric enforcement** - Outer boundary 10x harsher than inner
- ✅ **Proper termination** - Episode ends if ANY corner >2m outside corridor
- ✅ **Simpler logic** - Uses current waypoint segment directly

### All Critical Fixes Implemented (5/5)

1. ✅ **Reward Scaling** - All rewards scaled down ~100x (300.0 → 3.0) for PPO stability
2. ✅ **Observation Normalization** - Dynamic normalization prevents clipping (max_dist = 25.0m)
3. ✅ **Lidar Resolution** - Increased from 32 → 64 rays for better spatial awareness
4. ✅ **Waypoint Orientation** - Fixed to use path tangents (not premature goal_yaw)
5. ✅ **Success Velocity Check** - Agent must stop (v < 0.3 m/s) to succeed, with relaxed condition when very close

### Additional v34-v35 Features

6. ✅ **Corridor Constraint Reward** - Penalizes going outside planned corridor
7. ✅ **Asymmetric Corridor Penalties** - Outer 10x harsh, inner 1x soft (allows corner cutting)
8. ✅ **Steering Penalty** - Forces straight entry when <2m from goal (±0.2 rad threshold)
9. ✅ **Proximity Scaling** - Rewards 6-15x stronger when close to goal
10. ✅ **Tighter Corridor** - 2.2m width (car 1.9m + 0.3m margin) vs old 3.0m
11. ✅ **Video Overlay System** - Training videos show corridor boundaries, waypoints, bay reference points
12. ✅ **Corridor Visualization** - Real-time overlay of planned path corridor during training

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Environment Design](#3-environment-design)
4. [Waypoint Generation System](#4-waypoint-generation-system)
5. [Reward Function System](#5-reward-function-system)
6. [Corridor Constraint System (v34-v35)](#6-corridor-constraint-system-v34-v35)
7. [Curriculum Learning](#7-curriculum-learning)
8. [Training System](#8-training-system)
9. [Evaluation System](#9-evaluation-system)
10. [File Structure](#10-file-structure)
11. [Current Implementation (v35)](#11-current-implementation-v35)
12. [Future Improvements](#12-future-improvements)
13. [Version History](#13-version-history)
14. [Usage Guide](#14-usage-guide)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

### 1.1 Goal
Develop an autonomous parking agent capable of navigating complex parking lots and parking in designated bays with high precision using hierarchical reinforcement learning with corridor constraints and curriculum progression.

### 1.2 Key Challenges
1. **Long-horizon task** - Navigate 10-25m to bay, requires up to 2000 steps
2. **Complex path planning** - Must stay on roads AND within corridor boundaries
3. **Precision parking** - Final alignment requires <0.5m position, <0.3 rad yaw accuracy
4. **Multi-lot generalization** - Must work across different parking lot layouts
5. **Corridor following** - Must stay within 2.2m corridor (car is 1.9m wide)
6. **Sparse rewards** - Success only at end of episode

### 1.3 Solution Approach
**Hierarchical RL with Corridor Constraints:**
- **High-level:** A* path planning on road-aware grid
- **Mid-level:** B-spline smoothing + corridor boundary generation
- **Low-level:** PPO trained to follow waypoints within corridor and execute final parking
- **Corridor enforcement:** v35 perpendicular distance checking with 4-corner validation
- **Curriculum:** Progressive difficulty from easy to hard
- **Asymmetric penalties:** Strict outer boundary, lenient inner boundary

---

## 2. System Architecture

### 2.1 Component Diagram
```
┌─────────────────────────────────────────────────────┐
│            CurriculumManager                        │
│  - Stage selection (S1-S15)                         │
│  - Lot sampling (lot_a, lot_b)                      │
│  - Bay filtering                                    │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│              PPO Agent (Stable-Baselines3)          │
│  - Policy Network (MLP 256x256)                     │
│  - Value Network                                    │
│  - 6 Parallel Envs                                  │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│            WaypointEnv (v35)                        │
│  - A* Planning (road-aware grid)                    │
│  - B-spline Smoothing                               │
│  - Corridor Boundary Generation                     │
│  - v35 Corridor Constraint (perpendicular dist)     │
│  - Waypoint Following Mode                          │
│  - Parking Mode (proximity-scaled)                  │
│  - Steering Penalty (<2m from goal)                 │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│            ParkingEnv (Base)                        │
│  - Bicycle Model Physics                            │
│  - 64-beam LIDAR (upgraded from 32)                 │
│  - Collision Detection                              │
│  - Multi-lot Support                                │
└─────────────────────────────────────────────────────┘
```

---

## 3. Environment Design

### 3.1 State Space

#### 3.1.1 Observation (98D vector - v35)
```python
obs = [
    # Waypoint guidance (5D) - NORMALIZED
    local_dx / 25.0,      # Waypoint x in car frame
    local_dy / 25.0,      # Waypoint y in car frame  
    dtheta / π,           # Yaw difference to waypoint
    v / 2.0,              # Current velocity
    dist / 25.0,          # Distance to current waypoint
    
    # Goal-bay coordinates (4D) - NORMALIZED
    cx_bay / 25.0,        # Bay-frame x (longitudinal)
    cy_bay / 25.0,        # Bay-frame y (lateral)
    yaw_err / π,          # Yaw error to goal
    dist_to_goal / 25.0,  # Distance to goal
    
    # Progress indicators (2D)
    waypoint_progress,    # Fraction of waypoints completed [0, 1]
    is_near_goal,         # Binary flag (dist < 5m) {0, 1}
    
    # 8-point bay reference system (16D) - NEW in v34
    # 4 corners + 4 edge midpoints in car frame
    bay_point_1_x / 10.0, bay_point_1_y / 10.0,
    bay_point_2_x / 10.0, bay_point_2_y / 10.0,
    # ... (8 points total = 16 values)
    
    # Goal side indicator (1D) - NEW in v34
    goal_side,            # -1 (bottom) or +1 (top)
    
    # LIDAR (64D) - UPGRADED in v35
    lidar[0:63] / 10.0,   # Distance readings (normalized)
]
```

**Key Changes:**
- **v35:** Lidar upgraded to 64 rays (was 32)
- **v35:** Normalization uses max_dist=25.0 (was 10.0) to prevent clipping
- **v34:** Added 8-point bay reference system for spatial awareness
- **v34:** Added goal_side indicator for approach direction

#### 3.1.2 Action Space (2D continuous)
```python
action = [
    v_cmd,      # Velocity [-2.0, +2.0] m/s (backing allowed)
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
car_width = 1.9m   # Used for v35 corner checking
wheelbase = 2.7m
```

### 3.3 Sensors
**64-Beam LIDAR (v35 upgrade):**
- Range: 0-10m
- FOV: 360°
- Beams: 64 (upgraded from 32 for better resolution)
- Detects: obstacles, occupied bays, target bay walls

---

## 4. Waypoint Generation System

### 4.1 Overview
The waypoint generation system uses a three-stage approach:
1. **A* Grid Planning** - Discrete path on road-aware grid
2. **B-spline Smoothing** - Kinematically feasible smooth curve
3. **Corridor Boundary Generation** - v34 hybrid corridor (smooth approach + straight bay)

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
```

### 4.3 B-spline Smoothing
```python
def _smooth_path(waypoints):
    """
    Smooth path using cubic B-spline.
    
    Key technique:
    - Fit spline to dense A* points
    - Sample full range [0, 1] for exact start-to-goal connection
    - s=0.5 smoothing parameter for balance
    """
    tck, u = splprep([pts_x, pts_y], s=0.5, k=3)
    u_new = linspace(0, 1.0, 100)  # Dense sampling
    smooth_x, smooth_y = splev(u_new, tck)
    
    return smooth_waypoints
```

### 4.4 Waypoint Orientation Fix (v35 Critical Fix #4)

**Problem (OLD):**
```python
# All waypoints prematurely set to goal_yaw
for wp in waypoints:
    wp[2] = goal_yaw  # WRONG! Causes misaligned corridor
```

**Solution (v35):**
```python
# Use path tangents for correct orientation
def compute_path_tangents(waypoints):
    """Calculate orientation from actual path geometry."""
    for i, (x, y, _) in enumerate(waypoints):
        if i == 0:
            dx = waypoints[i+1][0] - x
            dy = waypoints[i+1][1] - y
        elif i == len(waypoints) - 1:
            dx = x - waypoints[i-1][0]
            dy = y - waypoints[i-1][1]
        else:
            # Average of incoming and outgoing directions
            dx = (waypoints[i+1][0] - waypoints[i-1][0]) / 2.0
            dy = (waypoints[i+1][1] - waypoints[i-1][1]) / 2.0
        
        theta = np.arctan2(dy, dx)
        corrected.append((x, y, theta))
```

---

## 5. Reward Function System (v35)

### 5.1 Reward Scaling (v35 Critical Fix #1)

**ALL rewards scaled down ~100x for PPO stability:**

| Component | Old Scale | New Scale (v35) |
|-----------|-----------|-----------------|
| Waypoint bonus | 50-500 | 0.5-5.0 |
| Success bonus | 1000 | 10.0 |
| Collision penalty | -100 | -1.0 |
| Corridor penalty | -20 per meter | -0.2 per meter |
| Parking alignment | 300 | 3.0 |

### 5.2 Core Reward Components

#### 5.2.1 Waypoint Following
```python
# Distance-normalized bonuses (v17 feature, v35 scaled)
reward_per_meter = 3.0 / total_path_length  # SCALED: was 300.0
multiplier = 1.0 + 0.5 * progress_ratio
waypoint_bonus = segment_len * reward_per_meter * multiplier

# Navigation reward
reward += -0.002 * dist_to_waypoint  # SCALED: was -0.2
reward += 0.02 * progress  # SCALED: was 2.0
```

#### 5.2.2 Parking Alignment (Proximity-Scaled)
```python
# v35: Proximity scaling (6-15x multiplier)
proximity_scale = 6.0 / max(dist_to_goal, 0.5)
proximity_scale = min(proximity_scale, 15.0)

# Tight denominators for sharp gradient
cy_quality = max(0.0, 1.0 - abs(cy_bay) / 1.5)  # Was 3.0
yaw_quality = max(0.0, 1.0 - yaw_err / 0.8)     # Was 1.0

alignment_score = cy_quality * 0.5 + yaw_quality * 0.3 + cx_quality * 0.2
continuous_reward = 3.0 * (alignment_score ** 2) * proximity_scale  # SCALED: was 300.0
```

#### 5.2.3 Steering Penalty (v35 NEW)
```python
# Force straight entry when close to goal
if dist_to_goal < 2.0:
    steering_threshold = 0.2  # ±0.2 rad ≈ ±11°
    if abs(steering_angle) > steering_threshold:
        steering_penalty = -0.5 * (abs(steering_angle) - steering_threshold)
        reward += steering_penalty

# Example penalties:
# steering=0.1 at 1.5m: 0.0 (within threshold)
# steering=0.3 at 1.5m: -0.05 (small correction OK)
# steering=0.5 at 1.5m: -0.15 (discouraged)
# steering=0.8 at 1.5m: -0.30 (heavily penalized)
```

### 5.3 Success Criteria (v35 Critical Fix #5)

```python
# Success requires stopping (with relaxed condition when very close)
is_stopped = abs(v) < 0.3  # m/s (~1 km/h)
very_close = dist_to_goal < 1.0  # Within 1m

# Success if: (aligned AND close AND stopped) OR (aligned AND VERY close)
if well_aligned and ((dist_to_goal < 2.0 and is_stopped) or very_close):
    success_bonus = 10.0  # SCALED: was 1000.0
    reward += success_bonus
    terminated = True
```

---

## 6. Corridor Constraint System (v34-v35)

### 6.1 Overview

The corridor constraint system ensures the agent follows the planned path by:
1. Generating corridor boundaries from waypoints
2. Checking if car corners are within corridor
3. Applying asymmetric penalties (outer harsh, inner soft)
4. Terminating episode for severe violations

### 6.2 Corridor Generation (`planning/corridor.py`)

#### 6.2.1 Hybrid Corridor Boundaries
```python
def calculate_corridor_boundaries(
    waypoints: List[Tuple[float, float, float]],
    goal_bay: Dict,
    corridor_width: float = 2.2  # v35: Tightened from 3.0m
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Generate hybrid corridor boundaries:
    1. Smooth corridor from B-spline path (approach phase)
    2. Straight boundaries aligned with bay (entry phase)
    """
    # Create dense smooth path (100 points)
    tck, u = splprep([x_sparse, y_sparse], s=0.5, k=3)
    u_dense = np.linspace(0, 1, 100)
    x_dense, y_dense = splev(u_dense, tck)
    
    # Compute tangents
    dx = np.gradient(x_dense)
    dy = np.gradient(y_dense)
    theta_dense = np.arctan2(dy, dx)
    
    # Find bay entrance point
    entrance_x = bay_x - (bay_depth / 2.0) * cos(bay_orientation)
    entrance_y = bay_y - (bay_depth / 2.0) * sin(bay_orientation)
    
    # Split path at entrance
    approach_path = dense_path[:entrance_idx]
    
    # Generate approach corridor (perpendicular offsets)
    offset = corridor_width / 2.0
    for x, y, theta in approach_path:
        perp_x = -sin(theta)
        perp_y = cos(theta)
        left_boundary.append((x + offset * perp_x, y + offset * perp_y))
        right_boundary.append((x - offset * perp_x, y - offset * perp_y))
    
    # Add straight bay boundaries
    # (Bay entrance corners + bay back corners)
    left_boundary.extend([front_left, back_left])
    right_boundary.extend([front_right, back_right])
    
    return left_boundary, right_boundary
```

#### 6.2.2 8-Point Bay Reference System (v34)
```python
def calculate_8_point_bay_reference(goal_bay: Dict):
    """
    Calculate 8-point bay reference (4 corners + 4 edge midpoints).
    Provides spatial awareness of bay boundaries.
    """
    # Bay orientation is perpendicular to goal_yaw
    bay_orientation = goal_yaw + π/2
    
    # 8 points in bay-aligned frame
    points_local = [
        (-half_depth, half_width),   # Corner 1 (front-left)
        (-half_depth, -half_width),  # Corner 2 (front-right)
        (half_depth, -half_width),   # Corner 3 (back-right)
        (half_depth, half_width),    # Corner 4 (back-left)
        (-half_depth, 0),            # Midpoint 1 (front)
        (0, -half_width),            # Midpoint 2 (right)
        (half_depth, 0),             # Midpoint 3 (back)
        (0, half_width),             # Midpoint 4 (left)
    ]
    
    # Transform to world frame
    for lx, ly in points_local:
        wx = bay_x + lx * cos(bay_orientation) - ly * sin(bay_orientation)
        wy = bay_y + lx * sin(bay_orientation) + ly * cos(bay_orientation)
        points_world.append((wx, wy))
    
    return points_world
```

### 6.3 Corridor Constraint Reward (v35 FIXED)

#### 6.3.1 The v34 Bug

**Problem:**
```python
# v34 BUGGY CODE:
left_x, left_y = left_boundary[corridor_idx]  # Gets ONE POINT
dist_to_left = np.hypot(car_x - left_x, car_y - left_y)  # Distance to POINT ❌
```

**Why it failed:**
- Calculated distance to a POINT on the boundary, not perpendicular distance to the LINE
- If car was between two boundary points, reported incorrect distance
- Agent could go 5-7m off-path with minimal penalty

#### 6.3.2 The v35 Fix

**Solution:**
```python
# v35 CORRECT CODE:
def _get_perpendicular_distance_to_segment(
    point: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Calculate perpendicular distance from point to line segment.
    Returns (perp_distance, signed_distance).
    """
    # Project point onto line segment
    t = ((x - x1) * dx + (y - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))  # Clamp to segment
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    # Perpendicular distance
    perp_dist = np.hypot(x - proj_x, y - proj_y)
    
    # Signed distance (positive = left of path, negative = right)
    cross = dx * (y - y1) - dy * (x - x1)
    signed_dist = perp_dist if cross > 0 else -perp_dist
    
    return perp_dist, signed_dist
```

#### 6.3.3 4-Corner Checking (v35 NEW)

```python
def _get_car_corners(
    car_x: float, car_y: float, car_yaw: float,
    car_length: float = 4.5, car_width: float = 1.9
) -> List[Tuple[float, float]]:
    """Get the 4 corners of the car in world coordinates."""
    half_l = car_length / 2.0
    half_w = car_width / 2.0
    
    cos_y = np.cos(car_yaw)
    sin_y = np.sin(car_yaw)
    
    # Corners: [front_left, front_right, rear_right, rear_left]
    corners_local = [
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ]
    
    # Transform to world frame
    corners_world = []
    for lx, ly in corners_local:
        wx = car_x + lx * cos_y - ly * sin_y
        wy = car_y + lx * sin_y + ly * cos_y
        corners_world.append((wx, wy))
    
    return corners_world
```

#### 6.3.4 Complete v35 Corridor Constraint

```python
def calculate_corridor_constraint_reward(
    car_x, car_y, car_yaw,
    waypoints, current_wp_idx,
    corridor_width=2.2,
    penalty_weight=0.5,
    car_length=4.5, car_width=1.9
) -> Tuple[float, bool]:
    """
    v35: FIXED corridor constraint.
    
    Returns: (penalty, should_terminate)
    """
    # Get current path segment
    p1 = waypoints[current_wp_idx - 1][:2]
    p2 = waypoints[current_wp_idx][:2]
    
    # Get car corners
    corners = _get_car_corners(car_x, car_y, car_yaw, car_length, car_width)
    
    corridor_half_width = corridor_width / 2.0
    total_penalty = 0.0
    should_terminate = False
    
    for corner in corners:
        perp_dist, signed_dist = _get_perpendicular_distance_to_segment(corner, p1, p2)
        
        # Check if outside corridor
        if perp_dist > corridor_half_width:
            violation_dist = perp_dist - corridor_half_width
            
            # ASYMMETRIC PENALTY
            if signed_dist > 0:
                # OUTER BOUNDARY (left of path): 10x harsh
                corner_penalty = -penalty_weight * 10.0 * violation_dist
            else:
                # INNER BOUNDARY (right of path): 1x soft (allow corner cutting)
                corner_penalty = -penalty_weight * 1.0 * violation_dist
            
            total_penalty += corner_penalty
            
            # TERMINATE if severe violation (>2m outside)
            if violation_dist > 2.0:
                should_terminate = True
    
    return total_penalty, should_terminate
```

### 6.4 Asymmetric Penalty Examples

| Scenario | Outer Penalty | Inner Penalty | Termination |
|----------|---------------|---------------|-------------|
| 1m outside | -5.0 | -0.5 | No |
| 2m outside | -10.0 | -1.0 | No |
| 3m outside | -15.0 | -1.5 | ✅ YES |
| 5m outside | -25.0 | -2.5 | ✅ YES |

**Key Insight:** Outer violations are 10x more expensive, encouraging safe driving while allowing efficient corner cutting on the inside.

### 6.5 Corridor Dimensions

```
Car width:      1.9m
Corridor width: 2.2m (v35: tightened from 3.0m)
Bay width:      2.7m

Clearances:
- Corridor: 0.15m per side (tight but realistic)
- Bay:      0.40m per side (standard parking)
```

**Why 2.2m?**
- Wider than car (1.9m) ✅ Can fit through
- Narrower than bay (2.7m) ✅ Smooth transition
- Tight enough to matter ✅ Agent must follow it

---

## 7. Curriculum Learning

### 7.1 Curriculum Stages

The curriculum progressively increases difficulty across 15 stages:

```python
S1 = CurriculumStage(
    lots=["lot_a"],
    allowed_bays=["A1", "A2"],
    allowed_orientations=[0.0],
    max_spawn_dist=10.0,
    advance_at_steps=50000
)

# ... progressive stages ...

S12 = CurriculumStage(
    lots=["lot_a", "lot_b"],
    allowed_bays=None,  # All bays
    allowed_orientations=None,  # All orientations
    max_spawn_dist=None,  # No limit
    advance_at_steps=1200000
)
```

---

## 8. Training System

### 8.1 Training Configuration (v35)

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
    "policy_kwargs": {"net_arch": [256, 256]}
}

# Environment settings
env_params = {
    "max_episode_steps": 2000,
    "n_envs": 6,  # v35: Increased from 4
    "enable_curriculum": True,
    "corridor_width": 2.2,  # v35: Tightened
    "penalty_weight": 0.5   # v35: Corridor penalty base weight
}
```

### 8.2 Training Commands

**v35 Training (with all fixes):**
```bash
cd ~/autonomous_parking_ws/src/autonomous_parking

nohup ../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \
  --total-steps 1500000 \
  --n-envs 6 \
  --max-episode-steps 2000 \
  --use-curriculum \
  --run-name v35_CORRIDOR_FIXED \
  --record-video \
  --video-freq 25 \
  > ../../logs/train_v35_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## 9. File Structure

```
autonomous_parking_ws/
├── src/autonomous_parking/autonomous_parking/
│   ├── env2d/
│   │   ├── parking_env.py           # Base physics environment
│   │   └── waypoint_env.py          # ⭐ v35 Hierarchical env with corridor
│   ├── planning/
│   │   ├── astar.py                 # A* planner (road-aware)
│   │   └── corridor.py              # ⭐ v34 Corridor boundary generation
│   ├── rewards/
│   │   └── parking_rewards.py       # ⭐ v35 Reward system (scaled, proximity, steering)
│   ├── sensors/
│   │   └── lidar.py                 # 64-beam LIDAR (v35 upgrade)
│   ├── utils/
│   │   └── v34_visualization.py    # ⭐ Corridor overlay rendering
│   ├── curriculum.py                # Curriculum manager
│   ├── sb3_train_hierarchical.py   # ⭐ Training script
│   └── sb3_eval_hierarchical.py    # Evaluation script
├── visualize_paths_v34_final.py    # ⭐ Corridor visualization tool
└── AUTONOMOUS_PARKING_DOCUMENTATION.md  # ⭐ THIS FILE
```

---

## 10. Current Implementation (v35)

### 10.1 Version Timeline

| Version | Key Features | Status |
|---------|-------------|--------|
| v14.20 | Final approach mode | ✅ 9 successes |
| v15.1 | Parking mode + curriculum | ✅ Framework |
| v17.2 | Dynamic timeout + spawn safety | ✅ Stable |
| **v34** | **Corridor constraint system** | ⚠️ Buggy distance calc |
| **v35** | **Perpendicular distance fix + 4-corner checking** | ✅ **CURRENT** |

### 10.2 v35 Complete Feature List

#### Critical Fixes (5/5)
1. ✅ Reward scaling (~100x reduction)
2. ✅ Observation normalization (max_dist=25.0)
3. ✅ Lidar resolution (64 rays)
4. ✅ Waypoint orientation (path tangents)
5. ✅ Success velocity check (must stop)

#### Corridor System
6. ✅ v35 perpendicular distance calculation
7. ✅ 4-corner car checking
8. ✅ Asymmetric penalties (10x outer, 1x inner)
9. ✅ Corridor termination (>2m violation)
10. ✅ Tighter corridor (2.2m)

#### Advanced Features
11. ✅ Steering penalty (<2m from goal)
12. ✅ Proximity scaling (6-15x)
13. ✅ 8-point bay reference system
14. ✅ Video overlay system
15. ✅ Curriculum learning (15 stages)

---

## 11. Future Improvements

### 11.1 Discussed But Not Yet Implemented

#### 11.1.1 Hybrid Corridor Approach
**Idea:** Apply v35 perpendicular distance math to pre-computed corridor boundaries (not raw waypoints)

**Pros:**
- Accounts for bay entrance geometry
- Uses smooth B-spline corridor
- More accurate than raw waypoint segments

**Cons:**
- More complex implementation
- Requires corridor segment finding logic
- May not be necessary if current approach works

**Status:** Deferred - v35 simpler approach working well

#### 11.1.2 Corridor Observation
**Idea:** Add corridor awareness to observation space

```python
# Add 3 new observations:
dist_to_left_boundary,   # 0=at edge, 1=centered
dist_to_right_boundary,  # 0=at edge, 1=centered
inside_corridor_flag,    # 1.0=inside, 0.0=outside

# Observation space: 98D → 101D
```

**Pros:**
- Agent explicitly "sees" corridor position
- May learn corridor following faster
- Provides interpretable signal

**Cons:**
- Increases observation dimension
- Adds complexity
- Current implicit learning may be sufficient

**Status:** Optional enhancement - can add if corridor following is slow

#### 11.1.3 Action Smoothing Wrapper
**Idea:** Limit steering change per step

```python
class SmoothedActionWrapper:
    def action(self, action):
        throttle, steering_target = action
        
        # Limit steering change
        steering_change = np.clip(
            steering_target - self.prev_steering,
            -0.1, 0.1  # Max ±0.1 rad/step
        )
        actual_steering = self.prev_steering + steering_change
        
        return [throttle, actual_steering]
```

**Pros:**
- Prevents jerky steering
- More realistic control
- May improve smoothness

**Cons:**
- Adds wrapper complexity
- PPO already learns smooth policies
- May slow learning initially

**Status:** Low priority - PPO entropy handles this

#### 11.1.4 Settling Reward
**Idea:** Reward stopping when aligned

```python
if alignment_score > 0.6 and abs(velocity) < 0.3:
    settling_reward = 15.0 * alignment_score
    reward += settling_reward
```

**Pros:**
- Encourages stopping when parked
- May reduce oscillation

**Cons:**
- Already have success velocity check
- Proximity scaling creates strong gradient
- May conflict with existing rewards

**Status:** Test without first - may not be needed

#### 11.1.5 Precision-Based Curriculum
**Idea:** Progress stages by precision, not just location

```python
Stage 1-3: tolerance 1.0m, 30° yaw
Stage 4-6: tolerance 0.5m, 15° yaw
Stage 7-9: tolerance 0.3m, 10° yaw
Stage 10+: tolerance 0.15m, 5° yaw
```

**Pros:**
- Forces progressive precision learning
- May achieve tighter final parking

**Cons:**
- More complex curriculum logic
- Current location-based curriculum working

**Status:** Future enhancement if precision is insufficient

### 11.2 Potential Optimizations

#### 11.2.1 VecNormalize Wrapper
**Idea:** Normalize rewards online

```python
from stable_baselines3.common.vec_env import VecNormalize

train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_reward=10.0
)
```

**Status:** Consider if training is unstable

#### 11.2.2 Increased Batch Size
**Idea:** Larger batches for stability

```python
batch_size = 256  # Was 64
n_steps = 4096    # Was 2048
```

**Status:** Try if precision task needs more stable updates

#### 11.2.3 Lower Entropy Coefficient
**Idea:** Reduce exploration once basic behavior learned

```python
# Start: ent_coef = 0.01
# After 500k steps: ent_coef = 0.005
# After 1M steps: ent_coef = 0.002
```

**Status:** Implement if agent is too random late in training

---

## 12. Version History

### v35 (December 2024) - CORRIDOR FIX
- ✅ Fixed corridor detection (perpendicular distance to segments)
- ✅ Added 4-corner car checking
- ✅ Implemented asymmetric penalties (10x outer, 1x inner)
- ✅ Added corridor termination (>2m violation)
- ✅ Tightened corridor (3.0m → 2.2m)
- ✅ All 5 critical fixes applied

### v34 (December 2024) - CORRIDOR SYSTEM
- ✅ Corridor boundary generation
- ✅ Corridor constraint reward
- ✅ 8-point bay reference system
- ✅ Video overlay system
- ⚠️ Buggy distance calculation (fixed in v35)

### v17.2 (November 2024)
- ✅ Dynamic episode timeout
- ✅ Spawn safety guarantee
- ✅ Video enhancements

### v15.1 (November 2024)
- ✅ Parking mode
- ✅ Curriculum learning
- ✅ Multi-lot support

---

## 13. Usage Guide

### 13.1 Training

```bash
# Standard training (v35)
cd ~/autonomous_parking_ws/src/autonomous_parking

nohup ../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \
  --total-steps 1500000 \
  --n-envs 6 \
  --max-episode-steps 2000 \
  --use-curriculum \
  --run-name v35_CORRIDOR_FIXED \
  --record-video \
  --video-freq 25 \
  > ../../logs/train_v35_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor training
tail -f ../../logs/train_v35_*.log
```

### 13.2 Evaluation

```bash
# Evaluate trained model
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_a \
  --model-dir results/ppo_hierarchical/v35_CORRIDOR_FIXED \
  --episodes 50 \
  --max-episode-steps 2000
```

### 13.3 Visualization

```bash
# Visualize corridor system
python visualize_paths_v34_final.py
```

---

## 14. Troubleshooting

### 14.1 Agent Going Outside Corridor

**Symptoms:** Training videos show agent 5-7m off path

**Diagnosis:**
- Check if using v35 (perpendicular distance)
- Verify corridor_width=2.2 in waypoint_env.py
- Confirm penalty_weight=0.5 (not 0.02)

**Solution:** Ensure v35 corridor constraint is active

### 14.2 Episodes Not Terminating

**Symptoms:** Episodes run to max_steps, no success

**Diagnosis:**
- Check success velocity condition
- Verify corridor termination logic
- Check if agent reaching final waypoint

**Solution:** Review success criteria and corridor termination

### 14.3 Reward Explosions

**Symptoms:** Rewards >1000, training unstable

**Diagnosis:**
- Check if rewards are scaled (should be ~3.0, not 300.0)
- Verify all reward components use v35 scaling

**Solution:** Apply reward scaling (Critical Fix #1)

---

## 15. Key Takeaways

### 15.1 What Works Well (v35)

1. ✅ **Perpendicular distance corridor checking** - Mathematically correct
2. ✅ **4-corner validation** - Realistic car geometry
3. ✅ **Asymmetric penalties** - Encourages safe + efficient driving
4. ✅ **Reward scaling** - Stable PPO training
5. ✅ **Proximity scaling** - Strong gradient near goal
6. ✅ **Steering penalty** - Forces straight entry

### 15.2 Critical Design Decisions

1. **Corridor width 2.2m** - Tight enough to matter, wide enough to navigate
2. **Outer 10x harsh, inner 1x soft** - Safety + efficiency balance
3. **Terminate at >2m violation** - Prevents wasted training time
4. **Check all 4 corners** - Realistic car geometry
5. **Perpendicular distance** - Correct mathematical approach

### 15.3 Future Directions

1. **Monitor corridor following** - May need hybrid approach if issues persist
2. **Consider corridor observation** - If learning is slow
3. **Precision curriculum** - If final parking not tight enough
4. **Action smoothing** - If steering is jerky

---

**End of Documentation**

*For questions or issues, refer to the troubleshooting section or review the version history for context on specific features.*
