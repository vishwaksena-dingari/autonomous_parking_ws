# Autonomous Parking Agent

Hierarchical Reinforcement Learning agent for autonomous parking with 16-stage curriculum learning, developed for ROS 2 Humble.

[![ROS 2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![RL](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ⚡ TL;DR - Quick Start (Copy & Paste)

Here are the **exact commands** to run the project immediately.

### Option 1: Docker
*Run this from the folder where you extracted the project.*

```bash
# cd autonomous_parking_ws && \
docker build -t autonomous_parking_node . && \
docker run -it --rm --net=host \
    -v $(pwd)/src:/root/autonomous_parking_ws/src \
    autonomous_parking_node
```

### Option 2: Local Run
*Requires Python 3.8+ and ROS 2 Humble installed.*

```bash
# cd autonomous_parking_ws && \
chmod +x run_local.sh && \
./run_local.sh
```

---

## 🚀 Detailed Guide


Follow these steps to build and run the agent in a Docker container.

### 1. Requirements
*   **Docker Desktop** installed and running.
*   **OS:** Windows, Mac (Intel/Apple Silicon), or Linux.
    *   *Note: Apple Silicon (M1/M2/M3) users may see a platform warning. This is normal and harmless.*

### 2. Build the Docker Image
Navigate to the project root (`autonomous_parking_ws`) and run:

```bash
docker build -t autonomous_parking_node .
```
*(This may take a few minutes. If you see warnings about "platform linux/amd64", you can safely ignore them.)*

---

### 3. Run the Agent (Menu System)

We provide a unified interactive menu for both Training and Evaluation.

```bash
docker run -it --rm --net=host \
    -v $(pwd)/src:/root/autonomous_parking_ws/src \
    autonomous_parking_node
```

**Menu Options:**
1.  **Train:** Starts the 16-stage curriculum training (logs to `src/results/`).
2.  **Evaluate:** Runs the best model (`PROD_TUNED_2M`) and saves videos.
3.  **Shell:** Opens a bash shell for debugging.

### 4. Monitor Training (TensorBoard)
To visualize training progress (Reward, Curriculum Level, Entropy):

```bash
tensorboard --logdir src/results/
```
*Then open http://localhost:6006 in your browser.*

---

### 4. Advanced / Headless Usage (Optional)

If you prefer to run specific scripts directly without the menu:

**Run Evaluation:**
```bash
docker run -it --rm --net=host \
    -v $(pwd)/src:/root/autonomous_parking_ws/src \
    autonomous_parking_node \
    bash eval_latest.sh
```

**Run Training:**
```bash
docker run -it --rm --net=host \
    -v $(pwd)/src:/root/autonomous_parking_ws/src \
    autonomous_parking_node \
    bash train.sh
```

**✅ RESULT (TRAINING ARTIFACTS):** All files are saved to `src/results/`:
*   **Models:** `src/results/ppo_hierarchical/production_curriculum_final/*.zip`
*   **Logs:** Monitor with Tensorboard (`tensorboard --logdir src/results/`)
*   **Videos:** `src/results/ppo_hierarchical/production_curriculum_final/training_videos/`

---

## 💻 Option B: Run Locally (Mac/Linux/WSL)

If you prefer not to use Docker, we provide a unified script that handles environment setup, dependencies, and building.

**Prerequisites:**
*   Python 3.8+
*   ROS 2 Humble (installed on your system)

**How to Run:**

```bash
# 1. Make the script executable
chmod +x run_local.sh

# 2. Run the interactive menu
./run_local.sh
```

**What it does:**
1.  Creates a generic `.venv` (with access to ROS 2).
2.  Installs all pip dependencies.
3.  Builds the ROS 2 package.
4.  Sources the environment.
5.  Lets you choose between **Train** and **Evaluate**.

**What to expect:**
*   **Console Output:** `🚀 Starting hierarchical training...`, `Logging to ...`
*   **Monitoring:** The logs will stream to your console.
*   **Stopping:** Press `Ctrl+C` to stop (it is designed to run for millions of steps).

---

## 📚 Development & Debugging

### Interactive Shell
If you want to enter the container to explore files or run ROS 2 commands manually:

```bash
docker run -it --rm --net=host \
    -v $(pwd)/src:/root/autonomous_parking_ws/src \
    autonomous_parking_node \
    bash
```

### Troubleshooting / Known Warnings

| Warning Message | Explanation | Action |
| :--- | :--- | :--- |
| `InvalidBaseImagePlatform` | You are running an AMD64 (Intel) image on an ARM (Apple Silicon) chip. | **Ignore.** It works correctly via emulation. |
| `Unable to import Axes3D` | Matplotlib 3D plotting is disabled in headless Docker mode. | **Ignore.** 2D plotting still works. |
| `IMAGEIO FFMPEG_WRITER` | Video dimensions were slightly adjusted (e.g. +8px) for codec compatibility. | **Ignore.** The video is fine. |

---

## 📂 Project Structure

*   **/src**
    *   `autonomous_parking/`: Main ROS 2 package source code.
        *   `env2d/`: Gymnasium environments (ParkingEnv, WaypointEnv).
        *   `rewards/`: Modular reward components (Gaussian, Collision, Penalties).
        *   `planning/`: Hybrid A* path planner.
    *   `results/`: Storage for trained models, logs, and evaluation videos.
*   **Dockerfile**: Environment definition (ROS 2 Humble + RL libs).
*   **train.sh**: One-click script for curriculum training.
*   **eval_latest.sh**: One-click script for evaluation.
*   **requirements.txt**: Pinned Python dependencies (NumPy 2.x compatible).

## 📄 Documentation
For detailed system architecture, reward logic, and curriculum details, see:
👉 [AUTONOMOUS_PARKING_DOCUMENTATION.md](AUTONOMOUS_PARKING_DOCUMENTATION.md)
