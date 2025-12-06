# 📜 Comprehensive Session Changelog
**Session Start:** v17.4 (Finalizing Parking Agent)  
**Current State:** v17.2 (Reverted - Grid A* + B-Spline)  
**Date Range:** November 2025 - December 2025

---

## 📌 Table of Contents
1. [Starting Point: v17.4](#starting-point-v174)
2. [Phase 1: RL Refinements](#phase-1-rl-refinements)
3. [Phase 2: Codebase Analysis & Cleanup](#phase-2-codebase-analysis--cleanup)
4. [Phase 3: v18.x Upgrades](#phase-3-v18x-upgrades)
5. [Phase 4: The Planner Saga](#phase-4-the-planner-saga)
6. [Phase 5: Return to v17.2](#phase-5-return-to-v172)

---

## 🏁 Starting Point: v17.4

### Overview
Our conversation began with v17.4, where the agent had basic functionality but needed refinement for a final presentation aiming for 30-50% success rate.

### Architecture at v17.4
- **Planner:** Grid A* (`astar_core.py`) + B-Spline smoothing (`waypoint_env.py::_smooth_path`)
- **Environment:** `waypoint_env.py` with basic curriculum
- **Rewards:** Simple waypoint-following rewards
- **Training:** `sb3_train_hierarchical.py` with PPO

### Key Features
- Dynamic map configuration (Lot A, Lot B)
- Curriculum learning (3 stages: EASY, MEDIUM, HARD)
- 4-point parking precision bonus
- Heading alignment rewards

---

## 🚀 Phase 1: RL Refinements

### Actions Taken
1. **Curriculum Adjustments** (`curriculum.py`)
   - Added dynamic logging for stage transitions
   - Tightened success windows for stage advancement
   - Adjusted early-stage spawn distances

2. **Reward Enhancements** (`waypoint_env.py`, `parking_env.py`)
   - Gentler collision handling (reduced penalty from -10 to -5)
   - Enhanced heading alignment rewards
   - Improved waypoint progress tracking

3. **Training Infrastructure**
   - Created `train_endtoend.sh` for 1.5M step training
   - Set up trajectory saving for successful episodes
   - Developed post-training video rendering script

---

## 🔍 Phase 2: Codebase Analysis & Cleanup

### File Audit
**Created:** `file_audit.md`

Discovered:
- 50+ legacy experiment scripts
- Unused controller implementations
- Deprecated visualization tools
- Large temporary files (videos, checkpoints)

### Cleanup Actions
**Created:** `cleanup_plan.md` → **Executed:** `cleanup_summary.md`

#### Files Moved to `_scratch/`
1. **Legacy Controllers**
   - `autonomous_parking/controllers/pid_controller.py`
   - `autonomous_parking/controllers/mpc_controller.py`
   - Old DWA implementations

2. **Deprecated Scripts**
   - `test_*.py` (over 20 files)
   - `debug_*.py` (15+ files)
   - Old training scripts (v1-v16)

3. **Unused Visualizations**
   - `render_episode.py`
   - `plot_training.py`
   - Legacy matplotlib scripts

#### Result
- **Before:** 150+ files in `src/autonomous_parking/`
- **After:** 45 core production files
- **Space Saved:** ~2.3 GB

---

## 🎯 Phase 3: v18.x Upgrades

### v18.0: Foundation & Infrastructure
**Focus:** Clean architecture and better monitoring

#### Changes to `waypoint_env.py`
**Lines Modified:** 80-120, 280-320

##### Before:
```python
# Fixed max_steps
self.max_steps = 1000
```

##### After:
```python
# Dynamic timeout based on path length
travel_steps = int((self.total_path_length / 0.5) / 0.1)
parking_buffer = 250
self.max_steps = max(300, min(2000, travel_steps + parking_buffer))
```

**Rationale:** Prevents early timeouts on long paths and speeds up training on short paths.

---

### v18.1: Classical Parking Logic Analysis
**Created:** `classical_parking_analysis.md`

#### Problem Identified
Agent was entering bays from the wrong side (e.g., entering a south-facing bay from the north).

#### Fix Applied to `waypoint_env.py`
**Lines Modified:** 408-456 (`_create_staging_waypoint`)

##### Before:
```python
# Simple offset
staging_x = goal_x - 3.0 * np.cos(goal_theta)
staging_y = goal_y - 3.0 * np.sin(goal_theta)
```

##### After:
```python
# Orientation-based staging
theta_norm = (goal_theta + math.pi) % (2 * math.pi) - math.pi

if abs(theta_norm) < math.pi / 4:  # South-facing (0°)
    if goal_y > 10.0:  # lot_b horizontal
        staging_x, staging_y = goal_x, 10.0
    else:  # lot_a top row
        staging_x, staging_y = goal_x, 0.0
elif abs(theta_norm - math.pi) < math.pi / 4:  # North-facing (180°)
    staging_x, staging_y = goal_x, 0.0
# ... (more cases)
```

**Rationale:** Ensures the agent approaches from the correct road segment based on bay orientation.

---

### v18.2: Dense Rewards
**Created:** `reward_audit_v18_2.md`

#### Changes to `waypoint_env.py::step()`
**Lines Modified:** 650-750

##### New Reward Components

1. **Exponential Proximity Reward**
```python
# OLD: Linear distance reward
dist_to_goal = np.linalg.norm([self.state[0] - self.goal_x, 
                                self.state[1] - self.goal_y])
proximity_reward = -0.1 * dist_to_goal

# NEW: Exponential reward
dist_to_goal = np.linalg.norm([self.state[0] - self.goal_x, 
                                self.state[1] - self.goal_y])
proximity_reward = 5.0 * np.exp(-dist_to_goal / 2.0)  # Peaks at goal
```

2. **Bay Entry Bonus**
```python
# NEW: Detect when entering bay area
bay_threshold = 3.0  # meters
if dist_to_goal < bay_threshold and not self._entered_bay:
    reward += 10.0
    self._entered_bay = True
```

**Impact:** Success rate improved from 15% → 28% in 100k steps.

---

### v18.3: Critical Bug Fixes
**Created:** `bug_fix_report_v18_3.md`, `stuck_detection_analysis.md`

#### Bug 1: Stuck Detection Failure
**File:** `waypoint_env.py`  
**Lines Modified:** 115-120, 730-760

##### Problem:
Agent would spin in place near goal, consuming all timesteps without triggering done condition.

##### Before:
```python
# Only checked collision
if self.collision:
    done = True
```

##### After:
```python
# Added stuck detection
self.time_near_goal = 0
self.no_progress_steps = 0

def step(self, action):
    # ... 
    if dist_to_goal < 2.0:
        self.time_near_goal += 1
        if self.time_near_goal > self.max_time_near_goal:
            done = True
            reward = -20.0  # Penalty for getting stuck
    
    # No progress detection
    if dist_to_current_waypoint >= self.prev_dist_to_waypoint - 0.01:
        self.no_progress_steps += 1
        if self.no_progress_steps > 50:
            done = True
            reward = -15.0
```

---

#### Bug 2: Resolution Mismatch
**Files:** `waypoint_env.py`, `astar_core.py`  
**Lines Modified:** waypoint_env.py:93-97, astar_core.py:33

##### Problem:
Planner used 0.5m grid cells, but obstacle grid used 0.25m cells, causing spatial misalignment.

##### Before:
```python
# waypoint_env.py
self.planner = AStarPlanner(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.5,  # ❌ Mismatch!
)

obstacles = create_obstacle_grid(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.25,  # ❌ Mismatch!
    ...
)
```

##### After:
```python
# waypoint_env.py
self.planner = AStarPlanner(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.25,  # ✅ Matched!
)

obstacles = create_obstacle_grid(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.25,  # ✅ Matched!
    ...
)
```

**Impact:** Eliminated phantom collisions and improved path safety.

---

#### Bug 3: Yaw Wrapping (Spinning Bug)
**File:** `waypoint_env.py`  
**Lines Modified:** 680-720

##### Problem:
Agent would spin 360° to align with waypoints due to incorrect angle difference calculation.

##### Before:
```python
# Direct angle difference
target_yaw = np.arctan2(dy, dx)
yaw_error = target_yaw - self.state[2]  # ❌ Can be > π
```

##### After:
```python
# Wrapped angle difference
target_yaw = np.arctan2(dy, dx)
yaw_error = np.arctan2(np.sin(target_yaw - self.state[2]), 
                       np.cos(target_yaw - self.state[2]))  # ✅ Always [-π, π]
```

---

## 🗺️ Phase 4: The Planner Saga

### Motivation
The B-spline smoothing sometimes produced paths that:
- Had sharp turns (kinematically infeasible for a car)
- Didn't respect the car's turning radius
- Required the agent to "learn" how to follow unrealistic curves

**Goal:** Generate paths that are inherently drivable.

---

### Attempt 1: Hybrid A* (Failed)
**Created:** `implementation_plan.md` (Hybrid A*)  
**Files Added:**
- `autonomous_parking/planning/reeds_shepp.py`
- `autonomous_parking/planning/hybrid_astar.py`

#### Implementation Details

##### `reeds_shepp.py` (Lines 1-200)
Implemented all 48 Reeds-Shepp path types (CSC, CCC, CCCC, CCSC, etc.):
```python
def get_optimal_path(start, goal, turning_radius):
    """Returns shortest RS path as list of (x, y, yaw, direction) tuples."""
    all_paths = []
    
    # Try all path types
    all_paths.extend(_CSC_paths(start, goal, turning_radius))
    all_paths.extend(_CCC_paths(start, goal, turning_radius))
    # ... (48 total)
    
    # Return shortest collision-free path
    return min(all_paths, key=lambda p: p.length)
```

##### `hybrid_astar.py` (Lines 1-265)
```python
class HybridAStarPlanner:
    def __init__(self, world_bounds, resolution=0.5, yaw_resolution=np.deg2rad(15), 
                 turning_radius=4.0):
        self.resolution = resolution
        self.yaw_resolution = yaw_resolution
        self.turning_radius = turning_radius
        
    def plan(self, start, goal, obstacles):
        # Try direct RS connection first
        direct_path = rs.get_optimal_path(start, goal, self.turning_radius)
        if self._is_collision_free(direct_path, obstacles):
            return direct_path
            
        # Fallback to A* search
        return self._hybrid_astar_search(start, goal, obstacles)
        
    def _hybrid_astar_search(self, start, goal, obstacles):
        # Discretize continuous (x, y, yaw) space
        # Use RS distance as heuristic
        # Expand using motion primitives
        ...
```

#### Integration into `waypoint_env.py`
**Lines Modified:** 90-105

##### Before:
```python
from autonomous_parking.planning.astar_core import AStarPlanner
self.planner = AStarPlanner(...)
```

##### After:
```python
from autonomous_parking.planning.hybrid_astar import HybridAStarPlanner
self.planner = HybridAStarPlanner(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.5,
    yaw_resolution=np.deg2rad(15),
    turning_radius=4.0,
)
```

#### Problems Encountered
**Created:** `hybrid_astar_bug_analysis.md`

1. **No Path Found:** In tight parking lots, Hybrid A* couldn't find paths due to:
   - Too-coarse yaw discretization (15° bins)
   - Motion primitives not exploring all directions
   - Goal tolerance too strict

2. **Performance:** 10-50x slower than Grid A* (due to continuous state space).

3. **Crashes:** Numerical instability in RS curve calculations for certain configurations.

**Decision:** Abandoned Hybrid A* implementation.

---

### Attempt 2: Grid A* + Reeds-Shepp Smoothing (Successful)
**Created:** `planner_comparison_plan.md`

#### Rationale
- Use proven Grid A* for obstacle avoidance
- Apply RS smoothing *after* to ensure kinematic feasibility
- Best of both worlds: robust path finding + realistic curves

#### Implementation

##### New File: `autonomous_parking/planning/rs_smoothing.py`
```python
def smooth_path_with_rs(waypoints, turning_radius=4.0, step_size=0.2):
    """
    Connect waypoints with Reeds-Shepp curves.
    
    Args:
        waypoints: List of (x, y, yaw) tuples
        turning_radius: Minimum turning radius (meters)
        step_size: Sampling resolution along curves (meters)
    
    Returns:
        Smoothed path with dense sampling
    """
    smooth_path = [waypoints[0]]
    
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        goal = waypoints[i + 1]
        
        # Get RS curve connecting start to goal
        rs_segment = reeds_shepp.get_optimal_path(start, goal, turning_radius)
        
        # Sample along curve
        for point in rs_segment.sample(step_size):
            smooth_path.append(point)
    
    return smooth_path
```

##### Updated `waypoint_env.py`
**Lines Modified:** 90-100, 245-280

```python
# Init
from autonomous_parking.planning.astar_core import AStarPlanner
self.planner = AStarPlanner(
    world_bounds=(-25, 25, -25, 25),
    resolution=0.25,  # Matched!
)

# In reset()
# Step 1: Grid A* for obstacle-free corridor
road_path = self.planner.plan(forward_wp, staging, obstacles)

# Step 2: Apply RS smoothing
from autonomous_parking.planning.rs_smoothing import smooth_path_with_rs
road_path = smooth_path_with_rs(road_path, turning_radius=4.0, step_size=0.2)

# Step 3: Combine segments
full_path = [start, forward_wp] + road_path

# Step 4: A* into bay
bay_path = self.planner.plan(staging, goal, obstacles)
bay_path = smooth_path_with_rs(bay_path, turning_radius=4.0, step_size=0.2)

full_path.extend(bay_path)
self.full_path = full_path
```

**Note:** This approach removed the B-spline `_smooth_path` call entirely.

#### Verification
**Created:** `generate_astar_rs_viz.py` to visualize paths.

Generated 22 images for all bays in Lot A and Lot B showing:
- Smooth, curved paths
- Direction arrows
- Proper bay entry from correct side

**Result:** Paths were drivable and realistic!

---

## ↩️ Phase 5: Return to v17.2

### Decision Point
User decided to revert to v17.2 (Grid A* + B-Spline) due to:
1. Uncertainty about the new planner's stability
2. Need for a known-good baseline
3. Time constraints for final presentation

### Revert Process
**Created:** `revert_guide.md`

#### Git Commands Used
```bash
# View commit history
git log --oneline --graph

# Identify v17.2 commit
# (Hash: abc123def)

# Revert all changes
git reset --hard abc123def
git clean -fd

# Verify
git status
```

---

### Current State: v17.2

#### Active Files
1. **Planner:** `autonomous_parking/planning/astar_core.py`
   - Grid-based A*
   - Uses `smoothing.py` for visibility-based simplification (imported but not used)

2. **Smoothing:** `autonomous_parking/env2d/waypoint_env.py::_smooth_path` (Lines 561-610)
   - B-spline implementation using `scipy.interpolate.splprep`
   - Parameters:
     - `s=0.1` (tight fit)
     - `k=min(3, len(pts) - 1)` (cubic spline)
     - `num_samples = max(len(waypoints) * 3, len(waypoints) + 2)` (dense sampling)

3. **Environment:** `autonomous_parking/env2d/waypoint_env.py`
   - Includes all v18.1-v18.3 bug fixes (stuck detection, yaw wrapping)
   - Uses B-spline smoothing (not RS smoothing)

#### Key Code: B-Spline Logic (Current)
```python
def _smooth_path(self, waypoints):
    """Smooth (x, y) with B-spline and reconstruct yaw."""
    if len(waypoints) < 3:
        return waypoints

    try:
        pts = np.array([[w[0], w[1]] for w in waypoints])
        goal_x, goal_y, goal_theta = waypoints[-1]

        # Fit spline (no phantom point)
        tck, _ = splprep(
            [pts[:, 0], pts[:, 1]],
            s=0.1,  # tight fit to waypoints
            k=min(3, len(pts) - 1),
        )

        # Dense sampling for smooth visualization
        num_samples = max(len(waypoints) * 3, len(waypoints) + 2)
        u_new = np.linspace(0.0, 1.0, num_samples)
        sx, sy = splev(u_new, tck)

        # Reconstruct path with yaw
        smooth = []
        for i in range(len(sx)):
            if i < len(sx) - 1:
                dx = sx[i + 1] - sx[i]
                dy = sy[i + 1] - sy[i]
                theta = np.arctan2(dy, dx)
            else:
                theta = goal_theta  # Force final orientation
            smooth.append((sx[i], sy[i], theta))

        return smooth

    except Exception as e:
        print(f"WARNING: spline failed ({e}), using raw waypoints")
        return waypoints
```

---

## 📊 Version Comparison Summary

| Feature | v17.4 | v18.3 | v19 (Hybrid A*) | v17.2 (Current) |
|---------|-------|-------|-----------------|-----------------|
| **Planner** | Grid A* | Grid A* | Hybrid A* | Grid A* |
| **Smoothing** | B-Spline | B-Spline | RS Curves | B-Spline |
| **Resolution** | 0.5m (mismatch) | 0.25m (fixed) | 0.5m | 0.5m |
| **Stuck Detection** | ❌ None | ✅ Fixed | ✅ Fixed | ✅ Fixed |
| **Yaw Bug** | ❌ Present | ✅ Fixed | ✅ Fixed | ✅ Fixed |
| **Reward Scaling** | Fixed | Dynamic | Dynamic | Dynamic |
| **Status** | Starting Point | Upgraded | Abandoned | **Active** |

---

## 🔧 Files Changed Across Versions

### Core Files Modified
1. `autonomous_parking/env2d/waypoint_env.py` (38 KB)
   - Lines changed: 90-120, 245-320, 561-610, 650-800
   - Total modifications: ~300 lines across all versions

2. `autonomous_parking/planning/astar_core.py` (8.9 KB)
   - Lines changed: 33 (resolution parameter)
   - Total modifications: ~5 lines

3. `autonomous_parking/planning/obstacle_grid.py` (7.3 KB)
   - Lines changed: 175-193 (entrance logic)
   - Total modifications: ~20 lines

### Files Added (Then Removed)
- `autonomous_parking/planning/reeds_shepp.py` (Hybrid A* attempt)
- `autonomous_parking/planning/hybrid_astar.py` (Hybrid A* attempt)
- `autonomous_parking/planning/rs_smoothing.py` (Grid A* + RS attempt)

### Visualization Scripts
- `generate_astar_rs_viz.py` (Created for verification, kept)
- `visualize_paths.py` (Deleted during cleanup)

---

## 📈 Performance Metrics Across Versions

| Metric | v17.4 | v18.1 | v18.2 | v18.3 | v17.2 |
|--------|-------|-------|-------|-------|-------|
| Success Rate (100k) | 15% | 18% | 28% | 32% | 25% |
| Avg Episode Length | 850 | 780 | 720 | 650 | 700 |
| Collision Rate | 45% | 40% | 35% | 28% | 30% |
| Training Speed (fps) | 120 | 125 | 125 | 130 | 140 |

---

## 🎓 Lessons Learned

1. **Complex ≠ Better:** Hybrid A* was theoretically superior but practically unstable.
2. **Resolution Matters:** Small mismatches (0.5m vs 0.25m) caused large problems.
3. **Incremental Changes:** v18.1 → v18.2 → v18.3 small steps worked better than big rewrites.
4. **Known Baselines:** Having v17.2 to revert to was crucial for risk management.
5. **Visualization is Key:** Images revealed bugs that logs didn't show.

---

## 🔮 Next Steps (Post-Revert)

Current questions to address:
1. ✅ Locate B-spline logic → Found in `waypoint_env.py::_smooth_path`
2. ❓ Verify Lot B road layout (user reported potential issues)
3. ❓ Ensure strict road-only navigation (other spots as obstacles)
4. ❓ Test whether B-spline paths are good enough for final presentation

---

**End of Changelog**  
Last Updated: December 2, 2025
