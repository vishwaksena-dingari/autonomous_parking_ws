#!/usr/bin/env python3
"""
controller_pid.py
-----------------
Simple PID + geometric controller to auto-park the car in the 2D ParkingEnv.

Usage:
    python -m autonomous_parking.controller_pid --lot lot_a
    python -m autonomous_parking.controller_pid --lot lot_b
"""

import argparse
import math
import time
import numpy as np
from autonomous_parking.env2d.parking_env import ParkingEnv


class PIDController:
    """Basic PID for controlling yaw and speed."""

    def __init__(self, kp=1.2, ki=0.0, kd=0.25):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_err = 0.0
        self.int_err = 0.0

    def control(self, error, dt):
        """Compute PID output given error and timestep."""
        self.int_err += error * dt
        derivative = (error - self.prev_err) / dt
        self.prev_err = error
        return self.kp * error + self.ki * self.int_err + self.kd * derivative


def run_auto(lot: str = "lot_a", render=True):
    """Run an autonomous drive from entrance to target bay."""
    env = ParkingEnv(lot)
    obs, _ = env.reset()
    pid_yaw = PIDController(1.2, 0.0, 0.3)
    pid_speed = PIDController(0.6, 0.0, 0.1)

    dt = env.dt
    total_reward = 0.0

    print(f"Starting autonomous parking in {lot}...")

    for step in range(600):
        # Unpack current observation (x, y, yaw_err, v, dist)
        if len(obs) < 5:
            print("Warning: Observation length mismatch.")
            break

        # goal_x, goal_y, yaw_err, v, dist = obs
        # goal_x, goal_y, yaw_err, v, dist, lidar...
        # We only need the first 5 for the PID
        goal_x, goal_y, yaw_err, v, dist = obs[:5]

        # Compute steering and speed commands
        steer_cmd = np.clip(pid_yaw.control(yaw_err, dt), -env.max_steer, env.max_steer)
        v_target = np.clip(pid_speed.control(dist, dt), -1.5, 1.5)

        # Step simulation
        obs, reward, terminated, truncated, info = env.step([v_target, steer_cmd])
        done = terminated or truncated
        total_reward += reward

        if render:
            env.render()
        time.sleep(dt)

        if done:
            print(f"Reached goal in {step} steps. Reward = {total_reward:.2f}")
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous PID controller for ParkingEnv"
    )
    parser.add_argument(
        "--lot", type=str, default="lot_a", help="Parking lot: lot_a or lot_b"
    )
    args = parser.parse_args()
    run_auto(args.lot)
