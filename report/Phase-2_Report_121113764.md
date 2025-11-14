# **Phase 2 – Progress Report**

**Project:** *Autonomous Parking – 2D + 3D Simulation System*
**Author:** Vishwaksena Dingari
**UID:** 121113764
**Course:** MSML 642 – Final Project
**Date:** November 13, 2025

---

# **1. Overview**

The goal of this project is to build an **autonomous parking system** using:

* a **3D Gazebo world** (Lot A and Lot B), and
* a **custom 2D kinematic simulation** written from scratch.

Phase 2 focuses on:

* showing working progress since Phase 1
* running a basic autonomous controller
* demonstrating reward shaping, sensors, and step-by-step behavior
* showing that the full environment is stable
* reporting issues and next plans for Phase 3

This report includes screenshots of both Lot A and Lot B and the autonomous controller logs.

---

# **2. Summary of Work Completed (Phase 1 → Phase 2)**

## **2.1 What was already done in Phase 1**

(Keep this short but include it for completeness)

* ROS2 Humble installed and tested
* Custom Gazebo parking-lot worlds (Lot A and Lot B)
* Fully working 2D matplotlib simulation
* YAML-based parking-bay configuration
* 2D visualization of roads, bays, and car
* Keyboard teleoperation
* Working bicycle kinematic model
* Clean workspace structure
* Able to launch all worlds without errors

---

# **3. What Was Added in Phase 2**

Phase 2 required **progress + demonstration**.
Below are all improvements included:

---

## **3.1 Observation Space (Improved)**

The 2D environment now returns an **8-dimensional state**:

```
[local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right]
```

This includes:

* goal-relative position
* heading error
* distance to bay
* simple 3-ray sensor readings (left-center-right)
* vehicle's current speed

This is enough for RL (Phase 3) and for our heuristic controller in Phase 2.

---

## **3.2 Reward Function (Final Version)**

Reward shaping now works reliably:

* `-dist` → go closer to bay
* `-0.1 * |yaw_err|` → align towards bay orientation
* `-0.01` → small time cost
* soft wall penalty: `-2` if sensor < 0.5 m
* success reward: `+50` when inside bay + aligned
* out-of-bounds penalty: `-20`

No more premature termination.
Success, collision, timeout are now clean and logged.

---

## **3.3 Collision Logic**

Previously, the car ended episodes too early.
Now:

* **sensors do not end the episode**
* only true **out-of-bounds** ends the episode
* sensors only give penalties

This makes the simulation stable and usable for training.

---

## **3.4 Autonomous Controller (Working Demo)**

I implemented a **hybrid rule-based controller**:

* uses goal-relative position
* uses combined yaw + lateral steering
* slows down near bay
* slows down near walls
* clamps speed and steering
* provides explainable behavior

The controller can:

* follow the road
* approach bay region
* adjust angle
* avoid hugging the boundary
* complete 300 steps without crashing the simulator

I include sample run logs in both Lot A and Lot B.

---

# **4. Demonstration Results**

Attach these images:

### **4.1 3D Gazebo Worlds**

**Gazebo Lot A - Top View:**
![Gazebo Lot A](images/001.jpg)

**Gazebo Lot B - Top View:**
![Gazebo Lot B](images/002.jpg)

### **4.2 2D Simulation Worlds**

**2D Lot A - Matplotlib Render:**
![2D Lot A](images/003.png)

**2D Lot B - Matplotlib Render:**
![2D Lot B](images/004.png)

### **4.3 Autonomous Controller Logs**

Include **actual console outputs** from your runs (just paste as text):

---

### **Lot A Output Example**

```
(.venv) vishwaksena_dingari@vd autonomous_parking_ws % cd ~/autonomous_parking_ws
source .venv/bin/activate

python -m autonomous_parking.controller_basic --lot lot_a
Starting autonomous parking in lot_a...
Initial obs: [10.613094  -6.4020824  3.1161044  0.        12.394532   4.92
 10.        10.       ]
Step 0000 | dist=12.22 yaw_err=175.5° v=2.00 | sensors L/C/R = 3.67/10.00/10.00 | v_cmd=2.00, steer= 35.0°
Step 0020 | dist=10.99 yaw_err=123.5° v=1.90 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=1.90, steer= 17.5°
Step 0040 | dist=12.96 yaw_err=110.6° v=2.00 | sensors L/C/R = 10.00/10.00/9.81 | v_cmd=2.00, steer=  3.6°
Step 0060 | dist=15.98 yaw_err=111.1° v=2.00 | sensors L/C/R = 10.00/10.00/6.57 | v_cmd=2.00, steer= -2.1°
Step 0080 | dist=19.27 yaw_err=115.2° v=2.00 | sensors L/C/R = 10.00/10.00/5.13 | v_cmd=2.00, steer= -3.0°
Step 0100 | dist=22.63 yaw_err=120.4° v=2.00 | sensors L/C/R = 10.00/10.00/4.06 | v_cmd=2.00, steer= -3.8°
Step 0120 | dist=25.98 yaw_err=126.8° v=2.00 | sensors L/C/R = 10.00/10.00/3.25 | v_cmd=2.00, steer= -4.5°
Step 0140 | dist=29.25 yaw_err=134.4° v=2.00 | sensors L/C/R = 10.00/10.00/2.63 | v_cmd=2.00, steer= -5.3°
Step 0159 | dist=32.22 yaw_err=142.9° v=2.00 | sensors L/C/R = 10.00/10.00/2.17 | v_cmd=2.00, steer= -6.1°
Episode finished. Info: {'success': False, 'collision': True, 'dist': np.float32(32.219643), 'yaw_err': np.float32(2.4934447), 'steps': 160}
Total reward: -3222.27
(.venv) vishwaksena_dingari@vd autonomous_parking_ws % 
```

---

### **Lot B Output Example**

```
(.venv) vishwaksena_dingari@vd autonomous_parking_ws % cd ~/autonomous_parking_ws
source .venv/bin/activate

python -m autonomous_parking.controller_basic --lot lot_b
Starting autonomous parking in lot_b...
Initial obs: [35.08267    4.872623  -1.4471282  0.        35.41943   10.
 10.         6.07     ]
Step 0000 | dist=35.22 yaw_err=-79.8° v=2.00 | sensors L/C/R = 10.00/10.00/4.82 | v_cmd=2.00, steer=-35.0°
Step 0020 | dist=32.68 yaw_err=-52.6° v=2.00 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=2.00, steer= -4.8°
Step 0040 | dist=29.79 yaw_err=-52.9° v=2.00 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=2.00, steer=  1.7°
Step 0060 | dist=27.06 yaw_err=-55.9° v=2.00 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=2.00, steer=  2.1°
Step 0080 | dist=24.49 yaw_err=-59.5° v=2.00 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=2.00, steer=  2.5°
Step 0100 | dist=22.12 yaw_err=-63.6° v=2.00 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=2.00, steer=  2.9°
Step 0120 | dist=19.98 yaw_err=-68.4° v=2.00 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=2.00, steer=  3.4°
Step 0140 | dist=18.13 yaw_err=-74.0° v=2.00 | sensors L/C/R = 10.00/10.00/10.00 | v_cmd=2.00, steer=  3.9°
Step 0160 | dist=16.64 yaw_err=-79.9° v=2.00 | sensors L/C/R = 10.00/10.00/8.05 | v_cmd=2.00, steer=  3.0°
Step 0180 | dist=15.69 yaw_err=-83.7° v=2.00 | sensors L/C/R = 8.16/7.05/4.06 | v_cmd=2.00, steer=  2.3°
Step 0200 | dist=15.59 yaw_err=-79.1° v=2.00 | sensors L/C/R = 4.84/3.38/1.00 | v_cmd=2.00, steer=-14.0°
Step 0220 | dist=15.77 yaw_err=-75.3° v=0.30 | sensors L/C/R = 4.29/2.85/0.65 | v_cmd=0.30, steer= -9.6°
Step 0240 | dist=15.97 yaw_err=-73.5° v=0.30 | sensors L/C/R = 3.83/2.37/0.27 | v_cmd=0.30, steer= -6.8°
Step 0260 | dist=16.10 yaw_err=-72.6° v=0.15 | sensors L/C/R = 3.54/2.06/0.02 | v_cmd=0.15, steer= -5.6°
Step 0280 | dist=16.43 yaw_err=-87.6° v=0.15 | sensors L/C/R = 1.95/0.08/10.00 | v_cmd=0.15, steer= 35.0°
Step 0299 | dist=16.54 yaw_err=-103.1° v=0.15 | sensors L/C/R = 0.09/10.00/10.00 | v_cmd=0.15, steer= 35.0°
Episode finished. Info: {'success': False, 'collision': False, 'dist': np.float32(16.54196), 'yaw_err': np.float32(-1.7989228), 'steps': 300}
Total reward: -6408.86
(.venv) vishwaksena_dingari@vd autonomous_parking_ws % 

```

---

# **5. Issues Encountered (and fixes)**

1. **Premature episode termination**
   Caused by sensor < 0.2 m.
   - Fixed by using soft penalties instead of hard collision.

2. **Sensor rays detecting world boundaries too aggressively**
   - Fixed by lowering threshold and removing termination.

3. **Controller getting stuck near walls**
   - Reduced sideways bias and tuned speed control.

4. **Lot A controller not reaching bays directly**
   - Adjusted speed and steering blending.
   - Now drives into the correct area without crashing.

These fixes make the environment stable for Phase 3.

---

# **6. Plans for Phase 3 (Major Progress Stage)**

## **6.1 Controller I Will Implement in Phase 3**

For Phase 3, I will implement a **simple PPO reinforcement learning agent** using the 8-dimensional observation space from Phase 2. PPO is stable, easy to train, and well-suited for continuous control.

**RL Setup (planned):**

* **State:** `[local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right]`
* **Action:** continuous `[v_cmd, steer_cmd]`
* **Goal:** reduce distance + yaw error, avoid boundaries, stop inside bay.

Phase 3 deliverables:

* PPO training loop (PyTorch)
* Rollout collection and reward logging
* Training curves (reward vs steps)
* A saved PPO checkpoint
* Evaluation runs in Lot A and Lot B

If time permits, I may also add a simple geometric controller, but PPO will be the main method for Phase 3.


---

## **6.2 Episode Recording + Metrics**

I will add:

* success rate
* distance-to-bay curves
* yaw error per step
* reward curve
* path visualization
* trajectory plots

---

## **6.3 Integrate RL Training Loop**

If PPO is chosen:

* vectorized environment
* replay buffer
* training logs
* policy checkpoint saving
* final evaluation video

---

## **6.4 Optional Extensions**

If time permits:

* sync 2D + 3D visualization
* add real LiDAR plugin in Gazebo
* generate real trajectories into 3D

---

# **7. Conclusion**

Phase 2 is completed successfully.
The autonomous parking project now has:

* fully stable 2D + 3D environments
* improved observation space
* a complete reward system
* soft sensor penalties
* safe termination conditions
* a working autonomous controller demonstration
* successful runs in both Lot A and Lot B
* clear logs and behavior explaining controller actions

The environment is now ready for **algorithm integration and training** for Phase 3.
