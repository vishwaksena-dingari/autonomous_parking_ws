#!/usr/bin/env python3
"""
Basic autonomous parking controller for ParkingEnv.

- Uses the 2D kinematic ParkingEnv.
- Drives from spawn toward the selected goal bay.
- Uses goal-relative state + simple "sensor" ranges.
- Hybrid controller:
    * Geometric heading control toward goal
    * Simple proportional steering
    * Speed control based on distance + sensors

Run:

    cd ~/autonomous_parking_ws
    source .venv/bin/activate          # if using venv
    python -m autonomous_parking.controller_basic --lot lot_a

You can also try:

    python -m autonomous_parking.controller_basic --lot lot_b
"""

import math
import time
import argparse

import numpy as np

from autonomous_parking.env2d.parking_env import ParkingEnv


class BasicParkingController:
    """
    Simple hybrid controller for the ParkingEnv.

    Observation assumed to be:
        obs = [local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right]

    Where:
        local_x, local_y : goal-relative position (car in bay frame)
        yaw_err          : heading error vs. bay orientation (rad)
        v                : current velocity (m/s)
        dist             : distance to bay center (m)
        s_left           : left sensor range (m)
        s_center         : center sensor range (m)
        s_right          : right sensor range (m)
    """

    def __init__(self):
        # Steering / speed limits
        self.max_speed = 2.0  # m/s
        self.min_speed = 0.3  # don't crawl too slowly unless near goal
        self.max_steer = math.radians(35.0)  # steering angle clamp

        # Gains
        self.k_dist = 0.6  # how aggressively speed scales with distance
        self.k_yaw = 1.8  # steering from yaw error
        self.k_lat = 1.2  # steering from lateral offset

        # Safety / stopping thresholds
        self.goal_dist_thresh = 0.4  # within 40 cm in bay → consider parked
        self.goal_yaw_thresh = math.radians(10)  # ~10 deg alignment
        self.sensor_slow = 1.0  # slow down when obstacle closer than this
        self.sensor_stop = 0.2  # stop when obstacle very close

    def compute_control(self, obs):
        """
        Given the current observation, compute (v_cmd, steer_cmd).

        Modes (approx):
          - Far & straight  -> drive faster, small steering
          - Far & misaligned -> slower, stronger steering
          - Near bay        -> creep + focus on yaw
          - Very near bay   -> stop
        """
        local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right = obs

        yaw_deg = math.degrees(yaw_err)

        # --- 1) Check if we are basically parked ---
        if dist < self.goal_dist_thresh and abs(yaw_err) < self.goal_yaw_thresh:
            # very close + aligned -> full stop
            v_cmd = 0.0
            steer_cmd = 0.0
            return v_cmd, steer_cmd

        # --- 2) Decide "mode" based on distance & alignment ---

        # Far from bay (e.g. driving along main road)
        if dist > 8.0:
            if abs(yaw_deg) < 10 and abs(local_y) < 2.0:
                # ✅ Straight road: well aligned, small lateral error
                # -> go relatively fast, gentle steering
                base_speed = 1.5
                speed_gain = 0.2
            else:
                # ⚠️ Curvy / misaligned: bigger yaw or lateral offset
                # -> slower, focus more on steering
                base_speed = 0.8
                speed_gain = 0.1

        # Medium distance: approaching bay region
        elif dist > 3.0:
            # -> slow down, start preparing to turn in
            base_speed = 0.6
            speed_gain = 0.1

        # Very close to bay but not yet satisfied success condition
        else:
            # -> creep
            base_speed = 0.3
            speed_gain = 0.05

        # Nominal forward speed from distance + base
        v_cmd = base_speed + speed_gain * dist

        # --- 3) Steering control depending on "mode" like behavior ---

        # angle from car to bay center in local frame
        angle_to_goal = math.atan2(local_y, max(local_x, 0.1))

        # Weight yaw vs lateral differently depending on distance:
        if dist > 8.0:
            # Far: use more angle_to_goal (go toward bay center)
            k_yaw = 1.0
            k_lat = 1.2
        elif dist > 3.0:
            # Mid: balance between yaw and lateral
            k_yaw = 1.5
            k_lat = 1.0
        else:
            # Near bay: prioritize yaw alignment (straighten into bay)
            k_yaw = 2.0
            k_lat = 0.5

        steer_cmd = k_yaw * yaw_err + k_lat * angle_to_goal

        # --- 4) Simple obstacle logic with sensors (soft) ---
        min_sensor = min(s_left, s_center, s_right)

        # Quite close to something -> clamp speed down
        if min_sensor < self.sensor_slow:
            v_cmd = min(v_cmd, 0.3)

        # Extremely close -> crawl very slowly (but don't hard-stop)
        if min_sensor < self.sensor_stop:
            v_cmd = min(v_cmd, 0.15)

        # Small bias away from closer side
        side_diff = s_left - s_right  # if left closer => positive -> steer right
        steer_cmd += 0.2 * (-side_diff) / max(min_sensor, 1.0)

        # --- 5) Clamp commands (forward only for now) ---
        v_cmd = float(np.clip(v_cmd, 0.0, self.max_speed))
        steer_cmd = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))

        return v_cmd, steer_cmd

    # def compute_control(self, obs):
    #     """
    #     Given the current observation, compute (v_cmd, steer_cmd).
    #     """
    #     local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right = obs

    #     # --- 1) Check if we are basically parked ---
    #     if dist < self.goal_dist_thresh and abs(yaw_err) < self.goal_yaw_thresh:
    #         # we are inside the bay and aligned → full stop
    #         v_cmd = 0.0
    #         steer_cmd = 0.0
    #         return v_cmd, steer_cmd

    #     # --- 2) Compute nominal forward speed from distance to goal ---
    #     # More distance → higher speed (up to max_speed), but never below min_speed (when moving)
    #     if dist > self.goal_dist_thresh:
    #         v_cmd = self.k_dist * dist
    #         v_cmd = max(self.min_speed, min(v_cmd, self.max_speed))
    #     else:
    #         v_cmd = self.min_speed

    #     # Always drive forward for now (no reverse logic in this basic version)
    #     # v_cmd stays >= 0.

    #     # --- 3) Steering control ---
    #     # Lateral error from local frame:
    #     #   - local_x: forward/back relative to bay center (in bay coords)
    #     #   - local_y: left/right offset relative to bay center
    #     #
    #     # Simple geometric steering: combine yaw error + lateral position.
    #     #
    #     # "look-ahead" angle to goal center in local frame
    #     angle_to_goal = math.atan2(local_y, max(local_x, 0.1))

    #     # steering target combines:
    #     #   - yaw_err: difference between car heading and bay heading
    #     #   - angle_to_goal: where the bay center is relative to our forward axis
    #     steer_cmd = self.k_yaw * yaw_err + self.k_lat * angle_to_goal

    #     # # --- 4) Obstacle / wall avoidance using simple range sensors ---
    #     # min_sensor = min(s_left, s_center, s_right)

    #     # # If something is very close → brake
    #     # if min_sensor < self.sensor_stop:
    #     #     v_cmd = 0.0

    #     # # If something is moderately close → slow down
    #     # elif min_sensor < self.sensor_slow:
    #     #     v_cmd = min(v_cmd, self.min_speed)

    #     # # Small bias: if left is much closer than right, steer right a bit, and vice versa
    #     # side_diff = s_left - s_right
    #     # steer_cmd += 0.3 * (-side_diff) / max(min_sensor, 1.0)

    #     # --- 4) Obstacle / wall avoidance using simple range sensors ---
    #     min_sensor = min(s_left, s_center, s_right)

    #     # If something is extremely close → we treat this as "too close, back away"
    #     if min_sensor < self.sensor_stop:
    #         if dist > 2.0:
    #             # Far from goal but close to a wall → slowly reverse to escape
    #             v_cmd = -self.min_speed
    #         else:
    #             # Very close to goal and wall → just stop
    #             v_cmd = 0.0

    #     # If something is moderately close → slow down but keep moving
    #     elif min_sensor < self.sensor_slow:
    #         v_cmd = min(v_cmd, self.min_speed)

    #     # Small bias: if left is much closer than right, steer away from the closer side
    #     # (i.e., if left is closer, steer right, and vice versa)
    #     side_diff = s_left - s_right
    #     steer_cmd += 0.6 * (-side_diff) / max(min_sensor, 1.0)

    #     # --- 5) Clamp commands ---
    #     # v_cmd = float(np.clip(v_cmd, 0.0, self.max_speed))
    #     # steer_cmd = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))
    #     # Allow both forward and reverse (env will still clip to its own [-max_speed, max_speed])
    #     v_cmd = float(np.clip(v_cmd, -self.max_speed, self.max_speed))
    #     steer_cmd = float(np.clip(steer_cmd, -self.max_steer, self.max_steer))

    #     return v_cmd, steer_cmd


def run_auto(lot: str = "lot_a", max_steps: int = 600, sleep_dt: float = 0.05):
    """
    Run one autonomous parking episode in the given lot.
    """
    print(f"Starting autonomous parking in {lot}...")

    # IMPORTANT: use lot_name=lot, not lot=lot
    env = ParkingEnv(lot_name=lot, render_mode="human")
    controller = BasicParkingController()

    obs = env.reset()
    print("Initial obs:", obs)
    done = False
    episode_reward = 0.0

    for step in range(max_steps):
        # Compute control from observation
        v_cmd, steer_cmd = controller.compute_control(obs)

        # Step the environment
        obs, reward, done, info = env.step([v_cmd, steer_cmd])
        episode_reward += reward

        # Render (matplotlib animation)
        env.render()

        # Optional debug print every N steps
        if step % 20 == 0 or done:
            local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right = obs
            print(
                f"Step {step:04d} | "
                f"dist={dist:.2f} yaw_err={math.degrees(yaw_err):5.1f}° "
                f"v={v:.2f} | "
                f"sensors L/C/R = {s_left:.2f}/{s_center:.2f}/{s_right:.2f} | "
                f"v_cmd={v_cmd:.2f}, steer={math.degrees(steer_cmd):5.1f}°"
            )

        if done:
            print("Episode finished. Info:", info)
            break

        if sleep_dt is not None and sleep_dt > 0:
            time.sleep(sleep_dt)

    print(f"Total reward: {episode_reward:.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Basic autonomous parking demo.")
    parser.add_argument(
        "--lot",
        type=str,
        default="lot_a",
        choices=["lot_a", "lot_b"],
        help="Which parking lot layout to use.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Disable sleeping between steps (faster but choppier animation).",
    )

    args = parser.parse_args()
    sleep_dt = None if args.no_sleep else 0.05

    run_auto(lot=args.lot, max_steps=args.max_steps, sleep_dt=sleep_dt)


if __name__ == "__main__":
    main()
