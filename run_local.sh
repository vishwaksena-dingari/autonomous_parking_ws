#!/bin/bash
set -e

echo "========================================"
echo "   Autonomous Parking - Local Runner"
echo "========================================"

# 1. Create Virtual Environment (with access to system ROS packages)
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment (.venv)..."
    # --system-site-packages is crucial for ROS 2 (rclpy) visibility
    python3 -m venv .venv --system-site-packages
else
    echo "[INFO] Found existing .venv"
fi

# 2. Activate Virtual Env
echo "[INFO] Activating .venv..."
source .venv/bin/activate

# 3. Install Dependencies
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found!"
    exit 1
fi
echo "[INFO] Checking dependencies..."
# Upgrade pip first
pip install --upgrade pip
# Install requirements
pip install -r requirements.txt

# 4. Build ROS 2 Package
if command -v colcon &> /dev/null; then
    echo "[INFO] Building project with colcon..."
    colcon build --symlink-install
    source install/setup.bash
else
    echo "[WARN] 'colcon' not found. Assuming environment is already set up."
    # Add source to path manually if colcon isn't used
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src/autonomous_parking
fi

# 5. Menu
echo ""
echo "========================================"
echo "   Select Mode:"
echo "   1) Train (Start Production Curriculum)"
echo "   2) Evaluate (Run Latest Model)"
echo "========================================"
read -p "Enter choice [1 or 2]: " mode

if [ "$mode" == "1" ]; then
    echo "[INFO] Starting Production Training (2M steps, Tuned)..."
    bash train.sh

elif [ "$mode" == "2" ]; then
    echo "[INFO] Starting Evaluation (PROD_TUNED_2M)..."
    bash eval_latest.sh

else
    echo "[ERROR] Invalid choice."
fi
