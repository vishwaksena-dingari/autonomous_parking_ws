### On either machine (before working):
```git
git pull origin master
```

### After edits (to upload latest):
```git
git add -A
git commit -m "sync"
git push origin master
```

## 🐧 Ubuntu - ROS + Gazebo + 2D

### 1. Setup (once per terminal)

```bash
source /opt/ros/humble/setup.bash
cd ~/autonomous_parking_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. Gazebo 3D parking lot (ROS)

```bash
# Lot A
ros2 launch autonomous_parking parking_lot_a.launch.py

# Lot B
# ros2 launch autonomous_parking parking_lot_b.launch.py
```

### 3. 2D world + keyboard (with ROS env)

```bash
cd ~/autonomous_parking_ws
source install/setup.bash

# 2D visualization test
python -m autonomous_parking.env2d.test_env2d

# 2D keyboard control
python -m autonomous_parking.keyboard_drive_2d --lot lot_a
```

---

## 🐧 Ubuntu - 2D only, **without** ROS (optional)

```bash
cd ~/autonomous_parking_ws

python -m venv .venv
source .venv/bin/activate

pip install -e src/autonomous_parking
pip install numpy pyglet matplotlib pyyaml

# 2D test
python -m autonomous_parking.env2d.test_env2d

# 2D keyboard control
python -m autonomous_parking.keyboard_drive_2d --lot lot_a
```

---

## 🍎 macOS - 2D only (no ROS)

```bash
cd ~/autonomous_parking_ws

python -m venv .venv
source .venv/bin/activate

pip install -e src/autonomous_parking
pip install numpy pyglet matplotlib pyyaml

# 2D test
python -m autonomous_parking.env2d.test_env2d

# 2D keyboard control
python -m autonomous_parking.keyboard_drive_2d --lot lot_a
```

This is for:
* **Ubuntu with ROS** → Gazebo + 2D + keyboard.
* **Ubuntu/macOS without ROS** → 2D + keyboard using the same code + `bays.yaml`.
