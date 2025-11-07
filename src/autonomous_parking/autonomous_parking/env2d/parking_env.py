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
    Observation: [goal_x_local, goal_y_local, yaw_err, v, dist]
    """

    def __init__(
        self,
        lot_name: str = "lot_a",
        dt: float = 0.1,
        max_steps: int = 300,
    ):
        """
        Initialize parking environment.
        
        Args:
            lot_name: Name of parking lot ('lot_a' or 'lot_b')
            dt: Time step for physics simulation (seconds)
            max_steps: Maximum steps per episode
        """
        self.lot_name = lot_name
        self.dt = dt
        self.max_steps = max_steps

        # Car dimensions (compact car approximation)
        self.car_length = 4.5   # m
        self.car_width = 2.0    # m
        self.wheelbase = 2.7    # m (distance between front and rear axles)

        # Control limits
        self.max_speed = 3.0    # m/s
        self.max_steer = math.radians(35.0)  # rad

        # Success criteria
        self.pos_tol = 0.5      # m (distance to bay center)
        self.yaw_tol = math.radians(10)  # rad (heading alignment)

        # Load parking lot configuration
        cfg = load_parking_config(lot_name)
        self.entrance = cfg["entrance"]
        self.bays = cfg["bays"]

        # Episode state
        self.state = None       # [x, y, yaw, v]
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
        Returns: [goal_x_local, goal_y_local, yaw_err, v, dist]
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

        return np.array([local_x, local_y, yaw_err, v, dist], dtype=np.float32)

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
        local_x, local_y, yaw_err, v_obs, dist = obs

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
            (-25, -25), 50, 50,
            facecolor=(0.25, 0.25, 0.25),
            edgecolor="none",
            zorder=0,
        )
        self.ax.add_patch(ground)

        # Draw parking bay outlines
        bay_length = 5.5
        bay_width = 2.7

        for bay in self.bays:
            bx = bay["x"]
            by = bay["y"]
            byaw = bay["yaw"]

            # Compute lower-left corner for rotated rectangle
            dx = (bay_length / 2) * math.cos(byaw) - (bay_width / 2) * math.sin(byaw)
            dy = (bay_length / 2) * math.sin(byaw) + (bay_width / 2) * math.cos(byaw)
            llx = bx - dx
            lly = by - dy

            rect = Rectangle(
                (llx, lly),
                bay_length,
                bay_width,
                angle=math.degrees(byaw),
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                linestyle="--",
                zorder=1,
            )
            self.ax.add_patch(rect)
            self.bay_patches.append(rect)

            # Add bay ID label
            self.ax.text(
                bx, by, bay["id"],
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                zorder=2,
            )

        # Goal bay highlight (will be updated on reset)
        self.goal_patch = Rectangle(
            (0, 0), bay_length, bay_width,
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

        # Update car position
        dx = (self.car_length / 2) * math.cos(yaw) - (self.car_width / 2) * math.sin(yaw)
        dy = (self.car_length / 2) * math.sin(yaw) + (self.car_width / 2) * math.cos(yaw)
        llx = x - dx
        lly = y - dy

        self.car_patch.set_xy((llx, lly))
        self.car_patch.angle = math.degrees(yaw)

        # Update goal bay highlight
        if self.goal_bay is not None:
            gx = self.goal_bay["x"]
            gy = self.goal_bay["y"]
            gyaw = self.goal_bay["yaw"]

            bay_length = 5.5
            bay_width = 2.7

            dx_g = (bay_length / 2) * math.cos(gyaw) - (bay_width / 2) * math.sin(gyaw)
            dy_g = (bay_length / 2) * math.sin(gyaw) + (bay_width / 2) * math.cos(gyaw)
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

# #!/usr/bin/env python3
# import math
# import random

# from autonomous_parking.config_loader import load_bays


# class ParkingEnv:
#     """
#     Simple 2D parking environment for RL.

#     - State: relative pose of robot to target bay (dx, dy, yaw_error, distance)
#     - Action: [v, w] (linear and angular velocity)
#     - Dynamics: differential-drive kinematics in 2D
#     """

#     def __init__(
#         self,
#         lot_name: str = "lot_a",
#         dt: float = 0.1,
#         max_v: float = 0.5,
#         max_w: float = 1.5,
#         pos_tol: float = 0.25,
#         yaw_tol: float = 0.25,
#         max_steps: int = 200,
#     ):
#         self.lot = load_bays(lot_name)
#         self.dt = dt
#         self.max_v = max_v
#         self.max_w = max_w
#         self.pos_tol = pos_tol      # success position tolerance (m)
#         self.yaw_tol = yaw_tol      # success yaw tolerance (rad)
#         self.max_steps = max_steps

#         # World state
#         self.x = 0.0
#         self.y = 0.0
#         self.yaw = 0.0

#         # Target bay
#         self.target_bay = None

#         self.step_count = 0

#     # -------------------------------------------------------
#     # Helpers
#     # -------------------------------------------------------

#     def _sample_start_pose(self):
#         """Spawn near entrance with small random noise."""
#         ent = self.lot["entrance"]
#         x = ent["x"] + random.uniform(-0.5, 0.5)
#         y = ent["y"] + random.uniform(-0.5, 0.5)
#         yaw = ent["yaw"] + random.uniform(-0.2, 0.2)
#         return x, y, yaw

#     def _choose_target_bay(self, bay_id=None):
#         if bay_id is not None:
#             for bay in self.lot["bays"]:
#                 if bay["id"] == bay_id:
#                     return bay
#             raise ValueError(f"Bay '{bay_id}' not found")
#         # default: pick random
#         return random.choice(self.lot["bays"])

#     def _relative_state(self):
#         """
#         Return state relative to the bay in the robot frame.

#         dx_r, dy_r = bay position expressed in robot frame.
#         yaw_err = shortest angle from robot heading to bay heading.
#         """
#         bx = self.target_bay["x"]
#         by = self.target_bay["y"]
#         byaw = self.target_bay["yaw"]

#         # Vector from robot to bay in world frame
#         dx = bx - self.x
#         dy = by - self.y

#         # Rotate into robot frame (robot yaw)
#         cos_y = math.cos(-self.yaw)
#         sin_y = math.sin(-self.yaw)
#         dx_r = cos_y * dx - sin_y * dy
#         dy_r = sin_y * dx + cos_y * dy

#         # Heading error
#         yaw_err = self._wrap_angle(byaw - self.yaw)

#         dist = math.sqrt(dx * dx + dy * dy)

#         return dx_r, dy_r, yaw_err, dist

#     @staticmethod
#     def _wrap_angle(a):
#         """Wrap angle to [-pi, pi]."""
#         while a > math.pi:
#             a -= 2 * math.pi
#         while a < -math.pi:
#             a += 2 * math.pi
#         return a

#     # -------------------------------------------------------
#     # Gym-like API
#     # -------------------------------------------------------

#     def reset(self, bay_id: str | None = None):
#         """Reset environment, pick target bay, randomize start pose."""
#         self.target_bay = self._choose_target_bay(bay_id)
#         self.x, self.y, self.yaw = self._sample_start_pose()
#         self.step_count = 0

#         state = self._relative_state()
#         return list(state)  # [dx_r, dy_r, yaw_err, dist]

#     def step(self, action):
#         """
#         Step the environment with action = [v, w].

#         Returns: state, reward, done, info
#         """
#         self.step_count += 1

#         v, w = action
#         # Clip to limits
#         v = max(-self.max_v, min(self.max_v, float(v)))
#         w = max(-self.max_w, min(self.max_w, float(w)))

#         # Simple kinematics
#         self.x += v * math.cos(self.yaw) * self.dt
#         self.y += v * math.sin(self.yaw) * self.dt
#         self.yaw = self._wrap_angle(self.yaw + w * self.dt)

#         # Compute new state
#         dx_r, dy_r, yaw_err, dist = self._relative_state()

#         # Reward shaping:
#         # - negative distance + alignment penalty
#         # - small step penalty
#         # - big bonus on success
#         reward = 0.0
#         reward -= dist         # closer is better (less negative)
#         reward -= 0.1 * abs(yaw_err)
#         reward -= 0.01         # time penalty

#         done = False
#         success = False

#         if dist < self.pos_tol and abs(yaw_err) < self.yaw_tol:
#             reward += 10.0
#             done = True
#             success = True

#         if self.step_count >= self.max_steps:
#             done = True

#         info = {"dist": dist, "yaw_err": yaw_err, "success": success}

#         state = [dx_r, dy_r, yaw_err, dist]
#         return state, reward, done, info

