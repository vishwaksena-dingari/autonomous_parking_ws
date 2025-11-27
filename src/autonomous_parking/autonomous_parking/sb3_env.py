#!/usr/bin/env python3
"""
Gymnasium-compatible wrapper for ParkingEnv to work with Stable-Baselines3.

This wrapper converts the old Gym API (used by ParkingEnv) to the new
Gymnasium API required by Stable-Baselines3 v2.x.
"""

import numpy as np

try:
    import gymnasium as gym  # SB3 v2 prefers gymnasium
except ImportError:
    import gym

from .env2d.parking_env import ParkingEnv


class ParkingSB3Env(gym.Env):
    """
    Gym-compatible wrapper around ParkingEnv for Stable-Baselines3 PPO.

    Action space:
        Box([-1, -1], [1, 1])  ->  [v_cmd, steer_cmd]
        v_cmd   in [0, max_speed]
        steer   in [-max_steer, max_steer]

    Observation space:
        8D vector from ParkingEnv (cast to float32).
        [local_x, local_y, yaw_err, v, dist, s_left, s_center, s_right]
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        lot_name: str = "lot_a",
        max_episode_steps: int = 600,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.lot_name = lot_name
        self._max_episode_steps = max_episode_steps
        self._render_mode = render_mode

        # Your existing env (Phase 2)
        # NOTE: keep this constructor consistent with your ParkingEnv
        self.env = ParkingEnv(lot_name=lot_name, render_mode=None)

        # Action: 2D continuous (scaled inside step)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 37D
        # Observation space: [local_x, local_y, yaw_err, v, dist, lidar_0, ..., lidar_31]
        # Total: 37 dimensions (5 state + 32 lidar)
        high = np.array(
            [
                50.0,  # local_x
                50.0,  # local_y
                np.pi,  # yaw_err
                5.0,  # v
                50.0,  # dist
                *([10.0] * 32),  # 32-ray lidar readings (max 10m each)
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32,
        )

        # Velocity / steering limits for scaling actions
        # Use values consistent with your Phase 2 controller / env
        self.max_speed = 1.5   # m/s (tweak if needed)
        self.max_steer = 0.7   # rad

        self._step_count = 0

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Map action in [-1, 1]^2 to [v_cmd, steer_cmd].

        v_cmd:   [0, max_speed]
        steer:   [-max_steer, max_steer]
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # action[0] in [-1, 1] -> v in [0, max_speed]
        v_cmd = (action[0] + 1.0) * 0.5 * self.max_speed

        # action[1] in [-1, 1] -> steer in [-max_steer, max_steer]
        steer_cmd = action[1] * self.max_steer

        return np.array([v_cmd, steer_cmd], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset environment (Gymnasium API).

        Returns:
            obs: Observation array
            info: Info dict
        """
        if seed is not None:
            np.random.seed(seed)
            
        self._step_count = 0
        # v15 FIX: Properly unpack tuple return
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one step (Gymnasium API).

        Returns:
            obs: Observation
            reward: Scalar reward
            terminated: True if success/collision
            truncated: True if timeout
            info: Info dict
        """
        self._step_count += 1

        # v15 FIX: Removed duplicate action scaling
        env_action = self._scale_action(np.asarray(action, dtype=np.float32))
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        
        # v15 FIX: Don't override parent's termination logic
        # Parent already returns correct terminated/truncated flags
        
        # Optional outer truncation guard in case wrapper max_episode_steps
        # is stricter than the inner env's own max_steps.
        if self._step_count >= self._max_episode_steps and not (terminated or truncated):
            truncated = True

        obs = np.asarray(obs, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """Delegate to 2D env renderer."""
        self.env.render()

    def close(self):
        """Clean up resources."""
        if hasattr(self.env, "close"):
            self.env.close()
