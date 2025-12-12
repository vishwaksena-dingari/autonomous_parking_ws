# Autonomous Parking Agent - Complete Project Documentation

**Last Updated:** December 12, 2025
**Status:** Active Development (16-Stage Curriculum + 64-Ray Lidar)

---

## 📚 Table of Contents
1. [Project Overview](#1-project-overview)
2. [How It Works (System Architecture)](#2-how-it-works-system-architecture)
3. [Environment & Sensors](#3-environment--sensors)
4. [Rewards (How it Learns)](#4-rewards-how-it-learns)
5. [Curriculum (16 Parking Levels)](#5-curriculum-16-parking-levels)
6. [File Structure](#6-file-structure)
7. [Running the Project](#7-running-the-project)

---

## 1. Project Overview

### Goal
The goal is to train an AI car to park itself perfectly in a busy parking lot. It needs to:
*   Park in reverse or parallel.
*   Avoid hitting other parked cars or curbs.
*   Park straight (within 15cm error).

### The Challenge
Parking is hard because:
1.  **Precision:** You can't just be "close enough"; you must be perfectly aligned.
2.  **Tight Spaces:** The car is 1.9m wide, and the driving corridor is only 3.0m wide.
3.  **Dense Traffic:** We simulated a busy day where almost all other spots are full.

### The Solution
We use a method called **Hierarchical Reinforcement Learning** with a **16-Stage Curriculum**:
*   **Planner (The Calculator):** Finds a safe path on the map using A* and B-splines.
*   **Agent (The Driver):** A neural network (PPO) controls the steering and gas to follow that path.
*   **Curriculum (The Teacher):** Starts with simple empty lots and gradually adds obstacles, difficult angles, and multiple lots.

---

## 2. How It Works (System Architecture)

Instead of a complex diagram, here is the flow of data:

1.  **Curriculum Manager:** Decides the difficulty level (e.g., "Stage 5: Park in Bay A3 with obstacles").
2.  **Environment (Simulation):** 2D kinematic car model simulates motion + collisions via geometry checks (not full rigid-body physics).
3.  **Sensors:** The car "sees" using:
    *   **Lidar:** 64 laser beams measuring distance to obstacles.
    *   **GPS/Odometry:** Its own position and speed.
    *   **Goal Info:** Where the target parking spot is.
4.  **Planner:** Calculates a smooth "Green Line" path to the target.
5.  **Agent (Brain):** Looks at the sensor data and path, then outputs Steering and Gas commands.
6.  **Rewards:** The agent gets points for moving forward and parking well, and loses points for hitting things or reversing unnecessarily.

---

## 3. Environment & Sensors

### What the AI Sees (~100 Inputs)
The "brain" receives approximately 100 numbers (exact size depends on config) every split second:
*   **Navigation:** Distance/Angle to the next waypoint on the path.
*   **Target:** Where is the parking spot? (Distance, Angle).
*   **Lidar:** 64 numbers showing distance to nearest walls/cars (20m range).
*   **Car State:** Current speed and steering angle.
*   **Tracking:** Waypoint and corridor adherence features.

### Physics
*   **Car Size:** 4.2m long x 1.9m wide.
*   **Model:** Kinematic bicycle model (non-holonomic capabilities).
*   **Sensors:** 64-ray Lidar with realistic noise.
*   **Occupancy:** All non-target bays are filled with cars to make it realistic.

---

## 4. Rewards (How it Learns)

The agent learns by trying to maximize its score.

### ✅ Good Behavior (Points Added)
*   **Moving Forward:** Waypoint progress gives reward for advancing along the path (usually per-meter progress).
*   **Staying Centered:** Bonus for driving in the middle of the lane.
*   **Parking Deep:** Big bonus for reaching the back of the parking spot.
*   **Precision:** Target is tight parking precision (cm-level lateral + yaw tolerances), enforced by the environment’s success condition.

### ❌ Bad Behavior (Points deducted)
*   **Collisions:** HUGE penalty for hitting a wall or car (ends the try immediately).
*   **Freezing:** Penalty for standing still too long.
*   **Unnecessary Reversing:** Penalty for backing up when you should be going forward.
*   **Wandering:** Penalty for straying too far from the calculated path.

---

## 5. Curriculum (16 Parking Levels)

The AI goes through "School" with 16 grades. Progression is based on success rate thresholds over a moving window.

| Stage | Name | Description |
|:---:|:---|:---|
| **S0-S1** | **Baby Steps** | Empty lot. Just learn to drive straight and turn to a target. |
| **S2-S4** | **Single Bay** | Park in one specific spot (A3) with some obstacles nearby. |
| **S5-S7** | **Row Parking** | Park in any spot in the first row. Learning to find the right one. |
| **S8-S11** | **Angle Mastery** | Park from difficult starting angles (Perpendicular, Reverse entry). |
| **S12-S13** | **New Map** | Moving to "Lot B" with different layout to ensure generalized learning. |
| **S14-S15** | **Production** | Full difficulty. Random spawns, random targets, full traffic. |

---

## 6. File Structure

*   **`src/autonomous_parking/`**: The main code folder.
    *   **`env2d/`**: Physics and Simulation code.
    *   **`planning/`**: The A* pathfinder.
    *   **`rewards/`**: The scoring system code.
    *   **`autonomous_parking/curriculum.py`**: The "Teacher" (Level manager).
    *   **`autonomous_parking/sb3_train_hierarchical.py`**: The main training script.

---

## 7. Running the Project

**For the Easiest & Best experience, use Docker or the Local Script.**
Please see the **[README.md](README.md)** file for detailed instructions.

### Option A: Docker (Menu System)
Run without arguments to see the interactive menu (Train / Eval / Shell):
```bash
docker run -it --rm --net=host \
    -v $(pwd)/src:/root/autonomous_parking_ws/src \
    autonomous_parking_node
```

### Option B: Local Script (Mac/Linux)
```bash
chmod +x run_local.sh
./run_local.sh
```

### Manual / Advanced Configuration
*If you need to run the python module directly, use the Tuned "Gold Standard" configuration:*

```bash
python -m autonomous_parking.sb3_train_hierarchical \
    --total-steps 2000000 \
    --n-envs 4 \
    --max-episode-steps 1000 \
    --use-curriculum \
    --multi-lot \
    --run-name production_curriculum_final \
    --record-video \
    --video-freq 25 \
    --align-w 150.0 \
    --success-bonus 50.0 \
    --bay-entry-bonus 50.0 \
    --corridor-penalty 0.05 \
    --vel-reward-w 0.01 \
    --backward-penalty-weight 2.0 \
    --anti-freeze-penalty 0.01 \
    --ent-coef 0.005 \
    --learning-rate 0.0005 \
    --gamma 0.97 \
    --n-steps 512 \
    --batch-size 128 \
    --clip-range 0.3 \
    --n-epochs 15
```

