#!/usr/bin/env python3
"""
2D Kinematic Parking Environment for Reinforcement Learning

Features:
- Realistic bicycle model kinematics
- Configurable parking lots (lot_a, lot_b)
- Matplotlib visualization
- Gym-like API (reset, step, render)
"""

import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from autonomous_parking.config_loader import load_parking_config


class ParkingEnv:
    """
    2D kinematic-car parking environment.

    State: [x, y, yaw, v] (absolute pose + velocity)
    Action: [v_cmd, steer_cmd] (m/s, rad)
    Observation:
        [local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right]
    """

    def __init__(
        self,
        lot_name: str = "lot_a",
        dt: float = 0.1,
        max_steps: int = 300,
        render_mode: str = "human",
    ):
        """
        Initialize parking environment.

        Args:
            lot_name: Name of parking lot ('lot_a' or 'lot_b')
            dt: Time step for physics simulation (seconds)
            max_steps: Maximum steps per episode
            render_mode: (reserved for future use, e.g. 'human')
        """
        self.lot_name = lot_name
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        # ---- Car dimensions (compact car, fits comfortably in bay) ----
        self.car_length = 4.2  # m  (slightly shorter than bay depth)
        self.car_width = 1.9  # m
        self.wheelbase = 2.6  # m

        # ---- Parking bay dimensions (match SDF) ----
        self.bay_length = 5.5  # m (depth)
        self.bay_width = 2.7  # m (width)

        # Visual size of goal highlight (slightly smaller than bay)
        self.goal_length = 0.9 * self.bay_length
        self.goal_width = 0.9 * self.bay_width

        # Control limits
        self.max_speed = 3.0  # m/s
        self.max_steer = math.radians(35.0)  # rad

        # Success criteria
        self.pos_tol = 0.5  # m (distance to bay center)
        self.yaw_tol = math.radians(10)  # rad (heading alignment)

        # Load parking lot configuration
        cfg = load_parking_config(lot_name)
        self.entrance = cfg["entrance"]
        self.bays = cfg["bays"]

        # Episode state
        self.state = None  # [x, y, yaw, v]
        self.goal_bay = None
        self.steps = 0

        # Rendering
        self.fig = None
        self.ax = None
        self.car_patch = None
        self.goal_patch = None
        self.bay_patches = []

    # ======================= HELPERS =======================

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (theta + math.pi) % (2 * math.pi) - math.pi

    def _pick_goal_bay(self, bay_id=None):
        """Select target parking bay."""
        if bay_id is None:
            self.goal_bay = random.choice(self.bays)
        else:
            matches = [b for b in self.bays if b["id"] == bay_id]
            if not matches:
                raise ValueError(
                    f"Bay '{bay_id}' not found. Available: "
                    f"{[b['id'] for b in self.bays]}"
                )
            self.goal_bay = matches[0]

    def _sample_start_pose(self):
        """
        Sample starting position near entrance with noise.
        Returns: (x, y, yaw)
        """
        ent = self.entrance
        x = ent["x"] + random.uniform(-1.0, 1.0)
        y = ent["y"] + random.uniform(-1.0, 1.0)
        yaw = ent["yaw"] + random.uniform(-0.2, 0.2)
        return x, y, yaw

    def _get_obs(self):
        """
        Compute observation in robot-centric frame.
        Returns: [goal_x_local, goal_y_local, yaw_err, v, dist, sensor_left, sensor_center, sensor_right]]
        """
        x, y, yaw, v = self.state
        gx = self.goal_bay["x"]
        gy = self.goal_bay["y"]
        gyaw = self.goal_bay["yaw"]

        # World-frame goal vector
        dx = gx - x
        dy = gy - y

        # Transform to robot frame
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy

        # Heading error
        yaw_err = self._wrap_angle(gyaw - yaw)

        # Distance to goal
        dist = math.hypot(dx, dy)

        # 3 range sensors (left, center, right)
        ranges = self._compute_sensors()

        return np.array(
            [local_x, local_y, yaw_err, v, dist, *ranges],
            dtype=np.float32,
        )

    def _compute_sensors(self):
        """
        Very simple 3-ray range sensor model (left, center, right).

        Returns distances in meters, clipped to max_range.
        """
        # 3 range sensors (left, center, right)
        rays = [-math.pi / 6, 0.0, +math.pi / 6]
        max_range = 10.0
        readings = []

        x, y, yaw, _ = self.state

        for r in rays:
            ang = yaw + r
            dist = max_range

            # crude detection: distance to global square bounds [-20, 20]
            # (just to have some non-trivial readings)
            for bound in [-20.0, 20.0]:
                dx = bound - x
                dy = bound - y
                # project on ray direction
                proj = dx * math.cos(ang) + dy * math.sin(ang)
                if proj > 0.0:  # only obstacles in front
                    dist = min(dist, abs(proj))

            readings.append(round(dist, 2))

        return readings

    # ======================= GYM API =======================

    def reset(self, bay_id=None):
        """
        Reset environment and sample new episode.

        Args:
            bay_id: Specific bay to park in (None = random)

        Returns:
            obs: Initial observation [local_x, local_y, yaw_err, v, dist]
        """
        self._pick_goal_bay(bay_id)
        self.steps = 0

        # Sample start pose
        x, y, yaw = self._sample_start_pose()
        v = 0.0

        self.state = np.array([x, y, yaw, v], dtype=np.float32)

        return self._get_obs()

    def step(self, action):
        """
        Execute one simulation step with bicycle model kinematics.

        Args:
            action: [v_cmd, steer_cmd] - velocity (m/s) and steering angle (rad)

        Returns:
            obs: Observation [local_x, local_y, yaw_err, v, dist]
            reward: Scalar reward
            done: Episode termination flag
            info: Additional information dict
        """
        v_cmd, steer_cmd = action

        # Clip commands to physical limits
        v = float(np.clip(v_cmd, -self.max_speed, self.max_speed))
        delta = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))

        # Current state
        x, y, yaw, _ = self.state

        # Bicycle model kinematics (rear-axle reference point)
        x += v * math.cos(yaw) * self.dt
        y += v * math.sin(yaw) * self.dt
        yaw += (v / self.wheelbase) * math.tan(delta) * self.dt
        yaw = self._wrap_angle(yaw)

        # Update state
        self.state = np.array([x, y, yaw, v], dtype=np.float32)
        self.steps += 1

        # Compute observation
        obs = self._get_obs()
        local_x, local_y, yaw_err, v_obs, dist, *sensors = obs

        # Reward shaping
        reward = 0.0
        reward -= dist  # Closer is better
        reward -= 0.1 * abs(yaw_err)  # Better alignment
        reward -= 0.01  # Small time penalty

        done = False
        success = False

        # Success condition
        if dist < self.pos_tol and abs(yaw_err) < self.yaw_tol:
            reward += 50.0
            done = True
            success = True

        # Out-of-bounds check (simple rectangular bounds)
        if abs(x) > 25.0 or abs(y) > 25.0:
            reward -= 20.0
            done = True

        # Timeout
        if self.steps >= self.max_steps:
            done = True

        info = {
            "success": success,
            "dist": dist,
            "yaw_err": yaw_err,
            "steps": self.steps,
        }

        return obs, reward, done, info

    # ======================= RENDERING =======================

    # ======================= ROAD DRAWING =======================
    def _draw_roads(self):
        """
        Draw simple asphalt roads that roughly match the Gazebo layouts.

        - lot_a: one horizontal road between the two rows of bays
        - lot_b: T-shaped road (vertical for V1–V5, horizontal for H1–H5)
        """
        road_color = (0.18, 0.18, 0.18)
        road_width = 6.0  # ~6 m total width (2 × 3 m lanes)

        if self.lot_name == "lot_a":
            # One horizontal road between top (A*) and bottom (B*) rows
            xs = [b["x"] for b in self.bays]
            x_min = min(xs) - self.bay_width
            x_max = max(xs) + self.bay_width

            road = Rectangle(
                (x_min, -road_width / 2.0),
                x_max - x_min,
                road_width,
                facecolor=road_color,
                edgecolor="none",
                zorder=0.25,
            )
            self.ax.add_patch(road)

        elif self.lot_name == "lot_b":
            # Split bays into horizontal (H*) and vertical (V*) legs
            v_bays = [b for b in self.bays if b["id"].upper().startswith("V")]
            h_bays = [b for b in self.bays if b["id"].upper().startswith("H")]

            # --- Horizontal leg (in front of H1..H5) ---
            front_y = None
            if h_bays:
                # Bays are above the road, facing "down" toward it
                h_y = min(b["y"] for b in h_bays)  # all are 17.5
                front_y = h_y - self.bay_length / 2.0  # front edge of bays

                # Span slightly beyond first/last bay horizontally
                x_min = min(b["x"] for b in h_bays) - self.bay_width / 2.0
                x_max = max(b["x"] for b in h_bays) + self.bay_width / 2.0

                road_y_min = front_y - road_width  # road is below bays
                horiz_road = Rectangle(
                    (x_min, road_y_min),
                    x_max - x_min,
                    road_width,
                    facecolor=road_color,
                    edgecolor="none",
                    zorder=0.25,
                )
                self.ax.add_patch(horiz_road)

            # --- Vertical leg (beside V1..V5) ---
            if v_bays:
                # Vertical road should start just to the right of the V bays
                y_min = min(b["y"] for b in v_bays) - self.bay_length / 2.0
                if front_y is not None:
                    # Connect up to the front of the horizontal road
                    y_max = front_y
                else:
                    y_max = max(b["y"] for b in v_bays) + self.bay_length / 2.0

                # Rightmost V bay center (all V bays share x = -7.5)
                v_x = max(b["x"] for b in v_bays)
                # Front edge of V bays is at v_x + bay_length / 2 (they face east)
                road_x_min = v_x + self.bay_length / 2.0

                vert_road = Rectangle(
                    (road_x_min, y_min),
                    road_width,
                    y_max - y_min,
                    facecolor=road_color,
                    edgecolor="none",
                    zorder=0.25,
                )
                self.ax.add_patch(vert_road)

    def _setup_render(self):
        """Initialize matplotlib figure for visualization."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_title(f"Parking Environment: {self.lot_name}")
        self.ax.grid(True, alpha=0.3)

        # Draw ground/asphalt
        ground = Rectangle(
            (-25, -25),
            50,
            50,
            facecolor=(0.25, 0.25, 0.25),
            edgecolor="none",
            zorder=0,
        )
        self.ax.add_patch(ground)
        # Draw roads to match Gazebo layout
        self._draw_roads()

        # Draw parking bay outlines
        # NOTE: bay_width is across the row (x), bay_length is depth (y)
        for bay in self.bays:
            bx = bay["x"]
            by = bay["y"]
            byaw = bay["yaw"]

            # Compute lower-left corner for rotated rectangle
            # Compute lower-left corner for rotated rectangle
            # (center at (bx, by), width across row, length/depth along bay)
            dx = (self.bay_width / 2) * math.cos(byaw) - (
                self.bay_length / 2
            ) * math.sin(byaw)
            dy = (self.bay_width / 2) * math.sin(byaw) + (
                self.bay_length / 2
            ) * math.cos(byaw)
            llx = bx - dx
            lly = by - dy

            rect = Rectangle(
                (llx, lly),
                self.bay_width,  # width across row
                self.bay_length,  # depth
                angle=math.degrees(byaw),
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                linestyle="--",
                zorder=1,
            )
            self.ax.add_patch(rect)
            self.bay_patches.append(rect)

            # Add bay ID label, slightly inset into the bay
            # label_offset = 0.2 * self.bay_length  # move inside along bay heading
            # lx = bx + label_offset * math.cos(byaw)
            # ly = by + label_offset * math.sin(byaw)
            # Add bay ID label
            self.ax.text(
                bx,
                by,
                bay["id"],
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                zorder=2,
            )

        # Goal bay highlight (will be updated on reset)
        self.goal_patch = Rectangle(
            (0, 0),
            self.goal_width,
            self.goal_length,
            fill=True,
            facecolor="green",
            alpha=0.3,
            edgecolor="green",
            linewidth=2,
            zorder=1,
        )
        self.ax.add_patch(self.goal_patch)

        # Car patch
        self.car_patch = Rectangle(
            (0, 0),
            self.car_length,
            self.car_width,
            facecolor="blue",
            edgecolor="black",
            linewidth=1.5,
            alpha=0.9,
            zorder=3,
        )
        self.ax.add_patch(self.car_patch)

    def render(self):
        """Update visualization with current state."""
        if self.fig is None:
            self._setup_render()

        x, y, yaw, v = self.state

        # Update car position (rear-axle reference → rectangle lower-left)
        dx = (self.car_length / 2) * math.cos(yaw) - (self.car_width / 2) * math.sin(
            yaw
        )
        dy = (self.car_length / 2) * math.sin(yaw) + (self.car_width / 2) * math.cos(
            yaw
        )
        llx = x - dx
        lly = y - dy

        self.car_patch.set_xy((llx, lly))
        self.car_patch.angle = math.degrees(yaw)

        # Update goal bay highlight
        if self.goal_bay is not None:
            gx = self.goal_bay["x"]
            gy = self.goal_bay["y"]
            gyaw = self.goal_bay["yaw"]

            dx_g = (self.goal_width / 2) * math.cos(gyaw) - (
                self.goal_length / 2
            ) * math.sin(gyaw)
            dy_g = (self.goal_width / 2) * math.sin(gyaw) + (
                self.goal_length / 2
            ) * math.cos(gyaw)
            llx_g = gx - dx_g
            lly_g = gy - dy_g

            self.goal_patch.set_xy((llx_g, lly_g))
            self.goal_patch.angle = math.degrees(gyaw)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        """Clean up rendering resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.car_patch = None
            self.goal_patch = None
            self.bay_patches = []
