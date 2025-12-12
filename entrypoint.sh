#!/bin/bash
source /opt/ros/humble/setup.bash
source /root/autonomous_parking_ws/install/setup.bash
# If no command is provided, show the menu (similar to run_local.sh)
if [ -z "$1" ]; then
    echo "========================================"
    echo "   Autonomous Parking - Docker Runner"
    echo "========================================"
    echo "   1) Train (Start Production Curriculum)"
    echo "   2) Evaluate (Run Latest Model)"
    echo "   3) Shell (Bash)"
    echo "========================================"
    read -p "Enter choice [1-3]: " mode

    if [ "$mode" == "1" ]; then
        echo "[INFO] Starting Production Training..."
        bash train.sh
    elif [ "$mode" == "2" ]; then
        echo "[INFO] Starting Evaluation..."
        bash eval_latest.sh
    elif [ "$mode" == "3" ]; then
        exec bash
    else
        echo "[ERROR] Invalid choice."
        exit 1
    fi
else
    # Otherwise execute the passed command
    exec "$@"
fi
