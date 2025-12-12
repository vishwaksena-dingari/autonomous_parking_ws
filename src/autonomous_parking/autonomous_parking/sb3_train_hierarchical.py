#!/usr/bin/env python3
"""
Hierarchical RL Training Script - Hybrid Approach

High-level: A* path planning
Low-level: PPO waypoint following
"""

import os
import argparse
import warnings
import json
try:
    import cv2
except ImportError:
    cv2 = None  # v41: Make cv2 optional for non-video runs
import numpy as np
from pathlib import Path


def to_tilde_path(p: Path) -> str:
    """Convert an absolute Path to a string starting with ~ if inside home."""
    p = p.resolve()
    home = Path.home().resolve()
    try:
        return "~" + str(p).split(str(home), 1)[1]
    except Exception:
        return str(p)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", message="Training and eval env are not of the same type")

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv  # Changed from SubprocVecEnv

from autonomous_parking.env2d.waypoint_env import WaypointEnv
from collections import deque

# class VideoRecorderCallback(BaseCallback):
#     """
#     Callback to record training episodes as video frames (silently).
#     Records every N episodes to avoid massive video files.
#     """
#     def __init__(self, video_dir: str, record_freq: int = 100, verbose: int = 0):
#         super().__init__(verbose)
#         self.video_dir = Path(video_dir)
#         self.video_dir.mkdir(parents=True, exist_ok=True)
#         self.record_freq = record_freq  # Record every N episodes
        
#         self.episode_count = 0
#         self.recording = False
#         self.current_video_path = None
#         self.video_writer = None
#         self.waypoint_drawn = False  # Track if waypoints have been drawn for current episode
#         # NEW: keep track of path artists so we can remove them
#         self._path_artists = []
#     def _on_step(self) -> bool:
#         # Check if we should start recording this episode
#         if len(self.locals.get("dones", [])) > 0:
#             if self.locals["dones"][0]:  # Only trigger if the RECORDED env (index 0) is done
#                 self.episode_count += 1
                
#                 # Save video if we were recording
#                 if self.recording and self.video_writer is not None:
#                     self.video_writer.release()
#                     self.video_writer = None
#                     if self.verbose > 0:
#                         print(f"📹 Saved training video: {self.current_video_path}")
#                     self.recording = False
#                     self.waypoint_drawn = False

#                     # NEW: remove any A* overlays from the figure
#                     for art in self._path_artists:
#                         try:
#                             art.remove()
#                         except Exception:
#                             pass
#                     self._path_artists = []

                
#                 # Start recording next episode if it's time
#                 if self.episode_count % self.record_freq == 0:
#                     self.recording = True
#                     self.waypoint_drawn = False
#                     self.current_video_path = self.video_dir / f"training_ep{self.episode_count:05d}.mp4"
#                     if self.verbose > 0:
#                         print(f"🎬 Recording episode {self.episode_count}...")
        
#         # Capture frame if recording
#         if self.recording:
#             # Get frame from first environment (index 0)
#             if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
#                 env = self.training_env.envs[0]
#                 # Unwrap Monitor
#                 if hasattr(env, 'env'):
#                     env = env.env
                    
#                 # # Draw waypoints on first frame of episode
#                 # if not self.waypoint_drawn and hasattr(env, 'waypoints') and hasattr(env, 'ax'):
#                 #     if env.waypoints is not None and env.ax is not None and len(env.waypoints) > 0:
#                 #         waypoints = np.array(env.waypoints)
#                 #         # Draw path as red dashed line
#                 #         env.ax.plot(waypoints[:, 0], waypoints[:, 1], 
#                 #                    'r--', linewidth=2.5, alpha=0.8, 
#                 #                    label='A* Path', zorder=2.5)
#                 #         # Draw waypoints as red dots
#                 #         env.ax.scatter(waypoints[:, 0], waypoints[:, 1], 
#                 #                       c='red', s=30, zorder=2.6, alpha=0.7)
#                 #         self.waypoint_drawn = True
#                                 # Draw waypoints on first frame of episode
#                 if not self.waypoint_drawn and hasattr(env, 'waypoints') and hasattr(env, 'ax'):
#                     if env.waypoints is not None and env.ax is not None and len(env.waypoints) > 0:
#                         waypoints = np.array(env.waypoints)

#                         # NEW: remove any leftover path artists (from previous recorded eps)
#                         for art in self._path_artists:
#                             try:
#                                 art.remove()
#                             except Exception:
#                                 pass
#                         self._path_artists = []

#                         # Draw path as red dashed line
#                         line, = env.ax.plot(
#                             waypoints[:, 0], waypoints[:, 1],
#                             'r--', linewidth=2.5, alpha=0.8,
#                             zorder=2.5,
#                         )
#                         # Draw waypoints as red dots
#                         pts = env.ax.scatter(
#                             waypoints[:, 0], waypoints[:, 1],
#                             c='red', s=30, zorder=2.6, alpha=0.7,
#                         )

#                         # NEW: remember these so we can remove them later
#                         self._path_artists = [line, pts]
#                         self.waypoint_drawn = True
                        
#                         # Update title with episode info for debugging
#                         spawn_x, spawn_y = env.state[0], env.state[1]
#                         bay_id = env.goal_bay.get('id', 'Unknown')
#                         lot = env.lot_name
#                         env.ax.set_title(
#                             f"{lot.upper()} | Goal: {bay_id} | "
#                             f"Spawn: ({spawn_x:.1f}, {spawn_y:.1f}) | "
#                             f"Ep {self.episode_count}",
#                             fontsize=10
#                         )

                
#                 if hasattr(env, 'render'):
#                     # Setup rendering if needed
#                     if env.fig is None:
#                         env._setup_render()
                    
#                     # Update the plot
#                     env.render()
                    
#                     # Capture frame from matplotlib figure
#                     if env.fig is not None:
#                         env.fig.canvas.draw()
#                         # Get buffer from canvas (macOS uses ARGB, not RGB)
#                         try:
#                             # Try RGB first (Linux/Windows)
#                             buf = np.frombuffer(env.fig.canvas.tostring_rgb(), dtype=np.uint8)
#                             w, h = env.fig.canvas.get_width_height()
#                             frame = buf.reshape(h, w, 3)
#                         except AttributeError:
#                             # macOS uses ARGB
#                             buf = np.frombuffer(env.fig.canvas.tostring_argb(), dtype=np.uint8)
#                             w, h = env.fig.canvas.get_width_height()
                            
#                             # Handle Retina display (2x scaling)
#                             if buf.size == w * h * 4 * 4:  # 2x width, 2x height
#                                 w, h = w * 2, h * 2
                                
#                             frame_argb = buf.reshape(h, w, 4)
#                             # Convert ARGB to RGB (drop alpha channel)
#                             frame = frame_argb[:, :, 1:]  # Skip A, keep RGB
                        
#                         if frame is not None and isinstance(frame, np.ndarray):
#                             # Initialize video writer on first frame
#                             if self.video_writer is None:
#                                 h, w = frame.shape[:2]
#                                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                                 self.video_writer = cv2.VideoWriter(
#                                     str(self.current_video_path), 
#                                     fourcc, 
#                                     20.0,  # 20 FPS
#                                     (w, h)
#                                 )
                            
#                             # Convert RGB to BGR for OpenCV
#                             if len(frame.shape) == 3 and frame.shape[2] == 3:
#                                 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                                 self.video_writer.write(frame_bgr)
        
#         return True
    
#     def _on_training_end(self) -> None:
#         """Release any open video writer."""
#         if self.video_writer is not None:
#             self.video_writer.release()


class VideoRecorderCallback(BaseCallback):
    """
    Callback to:
      - Record training episodes as video frames every N episodes (periodic).
      - Save short videos + .npz trajectories for SUCCESS episodes.

    Design:
      - Episode numbering is based on env index 0.
      - Periodic videos: full episode, same as before.
      - Success videos: last `success_buffer_len` frames of that episode.
      - Success .npz: full trajectory (obs/actions/rewards/infos) for that episode.
    """

    def __init__(
        self,
        video_dir: str,
        record_freq: int = 100,
        verbose: int = 0,
        success_video_dir: str | None = None,
        success_npz_dir: str | None = None,
        save_success_npz: bool = False,
        success_buffer_len: int = 200,
    ):
        super().__init__(verbose)
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.record_freq = max(1, record_freq)  # avoid 0

        # Success-specific outputs
        self.success_video_dir = Path(success_video_dir) if success_video_dir else None
        if self.success_video_dir is not None:
            self.success_video_dir.mkdir(parents=True, exist_ok=True)

        self.success_npz_dir = Path(success_npz_dir) if success_npz_dir else None
        if self.success_npz_dir is not None:
            self.success_npz_dir.mkdir(parents=True, exist_ok=True)

        self.save_success_npz = save_success_npz
        self.success_buffer_len = max(1, success_buffer_len)

        self.episode_count = 0
        self.recording = False      # periodic training video flag
        self.current_video_path = None
        self.video_writer = None
        self.waypoint_drawn = False
        self._path_artists = []     # matplotlib artists to remove

        # Buffers for SUCCESS episodes
        self._success_frames = deque(maxlen=self.success_buffer_len)
        self._ep_obs = []
        self._ep_actions = []
        self._ep_rewards = []
        self._ep_infos = []

    def _reset_episode_buffers(self):
        """Clear per-episode buffers."""
        self._success_frames.clear()
        self._ep_obs.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        self._ep_infos.clear()
        self.waypoint_drawn = False

    def _save_success_npz(self, episode_idx: int):
        """Save full trajectory of a success episode as .npz."""
        if not (self.save_success_npz and self.success_npz_dir is not None):
            return

        if len(self._ep_obs) == 0:
            return  # nothing to save

        npz_path = self.success_npz_dir / f"success_ep{episode_idx:05d}.npz"
        try:
            np.savez(
                npz_path,
                observations=np.array(self._ep_obs),
                actions=np.array(self._ep_actions),
                rewards=np.array(self._ep_rewards),
                infos=np.array(self._ep_infos, dtype=object),
            )
            if self.verbose > 0:
                print(f"💾 Saved success trajectory: {npz_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to save success npz ({npz_path}): {e}")

    def _save_success_video(self, episode_idx: int):
        """Save short success video from the buffered last frames."""
        if self.success_video_dir is None or cv2 is None:
            return
        if len(self._success_frames) == 0:
            return

        out_path = self.success_video_dir / f"success_ep{episode_idx:05d}.mp4"
        try:
            # All frames same shape
            h, w = self._success_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (w, h))

            for fr in self._success_frames:
                if fr is None:
                    continue
                if len(fr.shape) == 3 and fr.shape[2] == 3:
                    fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                else:
                    # fallback: grayscale -> BGR
                    fr_bgr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
                writer.write(fr_bgr)

            writer.release()
            if self.verbose > 0:
                print(f"🏁 Saved SUCCESS video: {out_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to save success video ({out_path}): {e}")

    def _on_step(self) -> bool:
        # ==== 1) Collect per-step trajectory data for env 0 ====
        try:
            obs = self.locals.get("new_obs", None)
            if obs is not None:
                self._ep_obs.append(obs[0].copy())

            if "actions" in self.locals:
                self._ep_actions.append(self.locals["actions"][0].copy())

            if "rewards" in self.locals:
                self._ep_rewards.append(float(self.locals["rewards"][0]))

            if "infos" in self.locals:
                self._ep_infos.append(self.locals["infos"][0])
        except Exception:
            # don't break training if logging fails
            pass

        # ==== 2) Handle episode end (env index 0) ====
        dones = self.locals.get("dones", [])
        if len(dones) > 0 and dones[0]:
            # Episode finished for env 0
            self.episode_count += 1

            # Extract info for success flag
            info0 = None
            try:
                info0 = self.locals.get("infos", [None])[0]
            except Exception:
                info0 = None

            # is_success = bool(info0.get("success", 0)) if isinstance(info0, dict) else False
            is_success = False
            if isinstance(info0, dict):
                is_success = bool(
                    info0.get("success", 0)
                    or info0.get("parking_success", 0)
                    or info0.get("final_success", 0)
                )
            # DEBUG PRINT FOR SUCCESS DETECTION
            print(f"[VideoCB] Episode {self.episode_count+1}: success={is_success}  "
                  f"(recording_next={self.episode_count % self.record_freq == 0})")


            # Close periodic training video if we were recording
            if self.recording and self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None
                if self.verbose > 0:
                    print(f"📹 Saved training video: {self.current_video_path}")
                self.recording = False

                # Remove any A* overlays from the figure
                for art in self._path_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                self._path_artists = []

            # Save success outputs (video + npz) for this episode
            if is_success:
                self._save_success_npz(self.episode_count)
                self._save_success_video(self.episode_count)
                print(f"[VideoCB] Saved SUCCESS video for ep {self.episode_count}")

            # Decide whether we record the *next* episode periodically
            if self.episode_count % self.record_freq == 0:
                self.recording = True
                self.current_video_path = self.video_dir / f"training_ep{self.episode_count:05d}.mp4"
                if self.verbose > 0:
                    print(f"🎬 Recording episode {self.episode_count} (periodic)...")

            # Reset per-episode buffers (for the *next* episode)
            self._reset_episode_buffers()

        # ==== 3) Capture frame for env 0 (for both periodic + success) ====
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            # Unwrap Monitor
            if hasattr(env, 'env'):
                env = env.env

            # Draw waypoints ONLY when we are recording a periodic video
            if self.recording and (not self.waypoint_drawn) and hasattr(env, 'waypoints') and hasattr(env, 'ax'):
                if env.waypoints is not None and env.ax is not None and len(env.waypoints) > 0:
                    waypoints = np.array(env.waypoints)

                    # Remove leftover path artists if any
                    for art in self._path_artists:
                        try:
                            art.remove()
                        except Exception:
                            pass
                    self._path_artists = []

                    # Draw path + waypoints
                    line, = env.ax.plot(
                        waypoints[:, 0], waypoints[:, 1],
                        'r--', linewidth=2.5, alpha=0.8, zorder=2.5,
                    )
                    pts = env.ax.scatter(
                        waypoints[:, 0], waypoints[:, 1],
                        c='red', s=30, zorder=2.6, alpha=0.7,
                    )
                    self._path_artists = [line, pts]
                    self.waypoint_drawn = True

                    # Optional: title with debug info
                    try:
                        spawn_x, spawn_y = env.state[0], env.state[1]
                        bay_id = env.goal_bay.get('id', 'Unknown') if hasattr(env, "goal_bay") else "Unknown"
                        lot = getattr(env, "lot_name", "lot")
                        env.ax.set_title(
                            f"{lot.upper()} | Goal: {bay_id} | "
                            f"Spawn: ({spawn_x:.1f}, {spawn_y:.1f}) | "
                            f"Ep {self.episode_count}",
                            fontsize=10
                        )
                    except Exception:
                        pass

            # Now render & grab frame for both periodic + success buffer
            if hasattr(env, 'render'):
                if getattr(env, "fig", None) is None:
                    try:
                        env._setup_render()
                    except Exception:
                        pass

                env.render()

                frame = None
                if getattr(env, "fig", None) is not None:
                    env.fig.canvas.draw()
                    try:
                        # Try RGB (Linux/Windows)
                        buf = np.frombuffer(env.fig.canvas.tostring_rgb(), dtype=np.uint8)
                        w, h = env.fig.canvas.get_width_height()
                        frame = buf.reshape(h, w, 3)
                    except Exception:
                        # macOS ARGB path
                        try:
                            buf = np.frombuffer(env.fig.canvas.tostring_argb(), dtype=np.uint8)
                            w, h = env.fig.canvas.get_width_height()

                            # Handle potential Retina scaling (2x)
                            expected = w * h * 4
                            if buf.size == expected * 4:
                                w, h = w * 2, h * 2

                            frame_argb = buf.reshape(h, w, 4)
                            frame = frame_argb[:, :, 1:]  # drop alpha
                        except Exception:
                            frame = None

                if frame is not None and isinstance(frame, np.ndarray):
                    # Always keep last N frames for potential SUCCESS video
                    self._success_frames.append(frame.copy())

                    # If this episode is a periodic training episode, also write full video
                    if self.recording and self.current_video_path is not None and cv2 is not None:
                        if self.video_writer is None:
                            h, w = frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            self.video_writer = cv2.VideoWriter(
                                str(self.current_video_path),
                                fourcc,
                                20.0,
                                (w, h),
                            )

                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        self.video_writer.write(frame_bgr)

        return True

    def _on_training_end(self) -> None:
        """Release any open video writer."""
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None


class CurriculumEarlyStopCallback(BaseCallback):
    """
    Early-stop + logging for curriculum training.

    - Logs curriculum stage + success rate to TensorBoard.
    - Optionally stops when final stage (Smax) is mastered.
    """

    def __init__(
        self,
        check_freq: int = 1000,
        min_total_steps: int = 0,
        smax_success_thresh: float = 0.7,
        smax_min_steps: int = 100_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.check_freq = max(1, check_freq)  # v41: Guard against 0 (n_envs > 1000)
        self.min_total_steps = min_total_steps
        self.smax_success_thresh = smax_success_thresh
        self.smax_min_steps = smax_min_steps

        self.base_env = None
        self.smax_first_step = None

    def _init_callback(self) -> None:
        """
        Grab underlying WaypointEnv from DummyVecEnv -> Monitor -> WaypointEnv.
        """
        if hasattr(self.training_env, "envs") and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            # Unwrap Monitor / other wrappers
            while hasattr(env, "env"):
                env = env.env
            self.base_env = env
            if self.verbose and getattr(env, "enable_curriculum", False):
                print(
                    f"[CurriculumEarlyStop] Attached to env with curriculum manager: "
                    f"{type(getattr(env, 'curriculum', None)).__name__}"
                )

    def _on_step(self) -> bool:
        # Only check every N calls
        if self.n_calls % self.check_freq != 0:
            return True

        # Always log total timesteps
        self.logger.record("train/total_timesteps", self.num_timesteps)

        env = self.base_env
        if env is None or not getattr(env, "enable_curriculum", False):
            return True

        cur = getattr(env, "curriculum", None)
        if cur is None:
            return True

        stage_idx = cur.current_stage_idx
        smax_idx = len(cur.stages) - 1

        # Recent success rate from WaypointEnv helper
        if hasattr(env, "get_recent_success_rate"):
            success_rate = env.get_recent_success_rate()
        else:
            success_rate = 0.0

        # Log to TensorBoard
        self.logger.record("curriculum/stage_idx", stage_idx + 1)
        self.logger.record("curriculum/success_rate", success_rate)

        # Don't early-stop too early
        if self.num_timesteps < self.min_total_steps:
            return True

        # Only early-stop logic when in final stage
        if stage_idx == smax_idx:
            if self.smax_first_step is None:
                self.smax_first_step = self.num_timesteps

            time_in_smax = self.num_timesteps - self.smax_first_step
            self.logger.record("curriculum/time_in_smax", time_in_smax)

            if (
                time_in_smax >= self.smax_min_steps
                and success_rate >= self.smax_success_thresh
            ):
                if self.verbose:
                    print("\n" + "=" * 60)
                    print("⏹ EARLY STOP: Final curriculum stage mastered")
                    print(f"  Total steps      : {self.num_timesteps:,}")
                    print(
                        f"  Stage            : {stage_idx + 1}/{len(cur.stages)} "
                        f"({cur.current_stage.name})"
                    )
                    print(f"  Success (recent) : {success_rate:.1%}")
                    print(f"  Time in Smax     : {time_in_smax:,} steps")
                    print("=" * 60 + "\n")
                # Returning False tells SB3 to stop training
                return False

        return True



def make_waypoint_env(
    rank: int, 
    max_episode_steps: int = 800, 
    multi_lot: bool = True, 
    enable_curriculum: bool = False, 
    verbose: bool = True,
    seed: int = 42,  # v41: Explicit seed parameter
    # v38.9: Reward tuning parameters
    align_w: float = 50.0,
    success_bonus: float = 50.0,
    bay_entry_bonus: float = 60.0,
    corridor_penalty: float = 0.05,
    vel_reward_w: float = 0.05,
    anti_freeze_penalty: float = 0.01,    # v42
    backward_penalty_weight: float = 2.0, # v42
):
    """
    Create a waypoint-following environment for hierarchical RL.
    
    Args:
        rank: Environment ID for seeding
        max_episode_steps: Maximum steps per episode
        multi_lot: If True, randomly select between lot_a and lot_b on each reset
        enable_curriculum: If True, use v15 CurriculumManager for progressive learning
        seed: Base seed for reproducibility (actual seed = seed + rank)
        align_w: Weight for parking alignment reward
        success_bonus: Bonus for successful parking
        bay_entry_bonus: Bonus for entering bay
        corridor_penalty: Penalty for corridor deviations
        vel_reward_w: Weight for velocity reward
    """
    def _init():
        env = WaypointEnv(
            lot_name="lot_a",  # Default lot
            multi_lot=multi_lot,  # Enable multi-lot training
            enable_curriculum=enable_curriculum,  # v15: Curriculum learning
            render_mode=None,
            max_steps=max_episode_steps,  # Pass max steps to environment
            verbose=verbose,
            # v38.9: Pass reward tuning params to env
            align_w=align_w,
            success_bonus=success_bonus,
            bay_entry_bonus=bay_entry_bonus,
            corridor_penalty=corridor_penalty,
            vel_reward_w=vel_reward_w,
            anti_freeze_penalty=anti_freeze_penalty,
            backward_penalty_weight=backward_penalty_weight,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)  # v41: Use explicit seed parameter
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Hierarchical RL Training")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=2000)  # Increased for final parking
    parser.add_argument("--run-name", type=str, default="production_run",
                        help="Name of the run (for Tensorboard/checkpoints)")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--record-video", action="store_true", help="Record training videos silently")
    parser.add_argument("--video-freq", type=int, default=25, help="Record every N episodes")
    parser.add_argument("--use-curriculum", action="store_true", help="Enable v15 micro-curriculum")  # v15
    parser.add_argument("--multi-lot", action="store_true", default=False, help="Train on multiple lots (curriculum manages lots internally)")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Eval frequency in timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Checkpoint save frequency in timesteps")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="SB3 log interval (in rollouts)")
    parser.add_argument("--quiet-env", action="store_true",
                        help="Disable per-episode/per-step env prints")
    
    # ===== v40: TUNED DEFAULTS (Full Pipeline: rand_005 + ppo_003) =====
    # Reward weights - from Stage 1 winner: rand_005_045940 (score: -150.0)
    parser.add_argument("--align-w", type=float, default=100.0,
                        help="Parking alignment reward weight (TUNED v40)")
    parser.add_argument("--success-bonus", type=float, default=50.0,
                        help="Success bonus reward (TUNED v40)")
    parser.add_argument("--bay-entry-bonus", type=float, default=90.0,
                        help="Bay entry bonus (TUNED v40)")
    parser.add_argument("--corridor-penalty", type=float, default=0.1,
                        help="Corridor penalty weight (TUNED v40)")
    parser.add_argument("--vel-reward-w", type=float, default=0.01,
                        help="Velocity reward weight (TUNED v40)")
    parser.add_argument("--anti-freeze-penalty", type=float, default=0.01,
                        help="Anti-freeze penalty (v42 tunable)")
    parser.add_argument("--backward-penalty-weight", type=float, default=2.0,
                        help="Backward motion penalty weight (v42 tunable)")
    
    # PPO hyperparameters - from Stage 2 winner: ppo_003_055433 (score: -443.7)
    parser.add_argument("--ent-coef", type=float, default=0.005,
                        help="PPO entropy coefficient (TUNED v40)")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="PPO learning rate (TUNED v40)")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range (TUNED v40)")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="PPO discount factor (TUNED v40)")
    parser.add_argument("--gae-lambda", type=float, default=0.98,
                        help="PPO GAE lambda (TUNED v40)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout steps per env (TUNED v40)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="PPO minibatch size (TUNED v40)")
    parser.add_argument("--vf-coef", type=float, default=0.7,
                        help="PPO value function coefficient (TUNED v40)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="PPO max gradient norm (TUNED v40)")
    parser.add_argument("--n-epochs", type=int, default=15,
                        help="PPO epochs per update (TUNED v40)")
    parser.add_argument("--seed", type=int, default=42,
                        help="v41: Global random seed for reproducibility")
    
    args = parser.parse_args()
    
    # v41: Global seeding for reproducibility
    np.random.seed(args.seed)
    try:
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(args.seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    
    # Results directory
    base_dir = os.path.join("results", "ppo_hierarchical")
    log_dir = os.path.join(base_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Absolute paths for nice printing (~/ style)
    abs_log_dir = Path(log_dir).expanduser().resolve()
    abs_tb_dir = Path(base_dir, "tb").expanduser().resolve()
    abs_video_dir = abs_log_dir / "training_videos"
    
    # print(f"\n{'='*60}")
    # print(f"🎯 HIERARCHICAL RL TRAINING - Hybrid Approach")
    # print(f"{'='*60}")
    # print(f"High-level      : A* path planning")
    # print(f"Low-level       : PPO waypoint following")
    # print(f"Total timesteps : {args.total_steps:,}")
    # print(f"Parallel envs   : {args.n_envs}")
    # print(f"Run name        : {args.run_name}")
    # print(f"Log directory   : {log_dir}")
    # print(f"Video recording : {'Enabled (every ' + str(args.video_freq) + ' eps)' if args.record_video else 'Disabled'}")
    # print(f"{'='*60}\n")
    
    # # Create vectorized training environment (single-process for stability)
    # print("Creating hierarchical environments...")
    # print(f"Curriculum learning: {'ENABLED' if args.use_curriculum else 'Disabled'}")
    # print(f"Multi-lot training: {'ENABLED' if args.multi_lot else 'Disabled'}\n")
    # def to_tilde_path(p: Path) -> str:
    #     """Convert an absolute Path to a string starting with ~ if inside home."""
    #     p = p.resolve()
    #     home = Path.home().resolve()
    #     try:
    #         return "~" + str(p).split(str(home), 1)[1]
    #     except Exception:
    #     return str(p)

    print(f"\n{'='*60}")
    print("🎯 HIERARCHICAL RL TRAINING - Hybrid Approach")
    print(f"{'='*60}")
    print(f"High-level        : A* path planning")
    print(f"Low-level         : PPO waypoint following")
    print(f"Total timesteps   : {args.total_steps:,}")
    print(f"Parallel envs     : {args.n_envs}")
    print(f"Run name          : {args.run_name}")
    print(f"Log directory     : {to_tilde_path(abs_log_dir)}")
    print(f"TensorBoard logs  : {to_tilde_path(abs_tb_dir)}")
    if args.record_video:
        print(f"Video directory   : {to_tilde_path(abs_video_dir)}")
    else:
        print("Video directory   : (recording disabled)")
    print(f"{'='*60}\n")

    
    # Create vectorized training environment (single-process for stability)
    print("Creating hierarchical environments...")
    print(f"Curriculum learning: {'ENABLED' if args.use_curriculum else 'Disabled'}")

    if args.use_curriculum:
        print("Lot selection      : CURRICULUM (v15 stages S1–S15)\n")
    else:
        if args.multi_lot:
            print("Lot selection      : RANDOM LOT (lot_a / lot_b)\n")
        else:
            print("Lot selection      : SINGLE LOT (lot_a)\n")


    train_env = DummyVecEnv([
        make_waypoint_env(
            i, 
            max_episode_steps=args.max_episode_steps,
            multi_lot=args.multi_lot,
            enable_curriculum=args.use_curriculum,
            verbose=not args.quiet_env,
            seed=args.seed,  # v41: Use global seed
            # v38.8: Pass tuning args
            align_w=args.align_w,
            success_bonus=args.success_bonus,
            bay_entry_bonus=args.bay_entry_bonus,
            corridor_penalty=args.corridor_penalty,
            vel_reward_w=args.vel_reward_w,
            anti_freeze_penalty=args.anti_freeze_penalty,
            backward_penalty_weight=args.backward_penalty_weight,
        ) for i in range(args.n_envs)
    ])
    
    # Create evaluation environment (single env, no curriculum for eval)
    eval_env = DummyVecEnv([
        make_waypoint_env(
            0, 
            max_episode_steps=args.max_episode_steps,
            multi_lot=False,  # v41: Fixed lot for deterministic eval
            enable_curriculum=False,  # No curriculum during evaluation
            verbose=False,
            seed=args.seed + 1000,  # v41: Different seed for eval
            # v38.8: Pass tuning args
            align_w=args.align_w,
            success_bonus=args.success_bonus,
            bay_entry_bonus=args.bay_entry_bonus,
            corridor_penalty=args.corridor_penalty,
            vel_reward_w=args.vel_reward_w,
            anti_freeze_penalty=args.anti_freeze_penalty,
            backward_penalty_weight=args.backward_penalty_weight,
        )
    ])
    
    # PPO model - all params from CLI args for tuning
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=os.path.join(base_dir, "tb"),
    )

    # tb_run_dir = getattr(model.logger, "dir", None)
    # if tb_run_dir is not None:
    #     tb_path = Path(tb_run_dir)
    #     tb_pretty = str(tb_path).replace(str(Path.home()), "~")
    #     print(f"TensorBoard run dir  : {tb_pretty}")
    # else:
    #     print("TensorBoard run dir  : <logger has no directory>")

        # Make sure TB base dir exists
    os.makedirs(abs_tb_dir, exist_ok=True)

    # Try to get the SB3 logger directory
    tb_run_dir = None
    try:
        # v41 FIX: Use public 'logger' attribute, not private '_logger'
        raw_logger = getattr(model, "logger", None) or getattr(model, "_logger", None)
        if raw_logger is not None:
            if hasattr(raw_logger, "get_dir"):
                tb_run_dir = raw_logger.get_dir()
            elif hasattr(raw_logger, "dir"):
                tb_run_dir = raw_logger.dir
    except Exception as e:
        print(f"[WARN] Could not read TensorBoard run dir: {e}")
        tb_run_dir = None

    if tb_run_dir:
        tb_path = Path(tb_run_dir)
        tb_pretty = str(tb_path).replace(str(Path.home()), "~")
        print(f"TensorBoard run dir  : {tb_pretty}")
    else:
        print("TensorBoard run dir  : <unknown / logger not initialized>")


    
    # # Callbacks
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=args.save_freq // args.n_envs,
    #     save_path=log_dir,
    #     name_prefix="ppo_hierarchical",
    # )
    
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=log_dir,
    #     log_path=log_dir,
    #     eval_freq=2500 // args.n_envs,
    #     n_eval_episodes=5,
    #     deterministic=True,
    # )
    
    # # Video recording callback (optional)
    # callbacks = [checkpoint_callback, eval_callback]
    # --- Directories for structured outputs ---
    best_model_dir = os.path.join(log_dir, "best_model")
    eval_log_dir = os.path.join(log_dir, "eval_logs")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    periodic_model_dir = os.path.join(log_dir, "models")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(periodic_model_dir, exist_ok=True)

    # 1) EvalCallback: save best_model.zip based on eval performance
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # 2) CheckpointCallback: robust recovery
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=checkpoint_dir,
        name_prefix="hier_v15",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    callbacks = [checkpoint_callback, eval_callback]

    # --- Optional video + success recording callback ---
    if args.record_video and cv2 is not None:
        success_video_dir = abs_video_dir / "success_episodes"
        success_npz_dir   = abs_log_dir   / "success_npz"

        video_cb = VideoRecorderCallback(
            video_dir=str(abs_video_dir),
            record_freq=args.video_freq,
            verbose=1,
            success_video_dir=str(success_video_dir),
            success_npz_dir=str(success_npz_dir),
            save_success_npz=True,
            success_buffer_len=200,  # last N frames stored
        )
        callbacks.append(video_cb)
    else:
        if args.record_video and cv2 is None:
            print("[WARN] OpenCV (cv2) not available; video recording disabled.")

    # 3) Periodic full-model save
    if args.save_freq > 0:
        class PeriodicSaveCallback(BaseCallback):
            def __init__(self, save_freq, save_path, verbose=0):
                super().__init__(verbose)
                self.save_freq = save_freq
                self.save_path = save_path
                os.makedirs(self.save_path, exist_ok=True)

            def _on_step(self) -> bool:
                if self.num_timesteps % self.save_freq == 0:
                    path = os.path.join(
                        self.save_path,
                        f"model_{self.num_timesteps}.zip"
                    )
                    self.model.save(path)
                    if self.verbose:
                        print(f"[PeriodicSave] Saved model -> {path}")
                return True

        periodic_save_cb = PeriodicSaveCallback(
            save_freq=max(1, args.save_freq // args.n_envs),
            save_path=periodic_model_dir,
            verbose=1,
        )
        callbacks.append(periodic_save_cb)


    # Curriculum-aware early stop + logging
    if args.use_curriculum:
        early_stop_callback = CurriculumEarlyStopCallback(
            check_freq=max(1, 1000 // args.n_envs),  # v41: Guard against n_envs > 1000
            min_total_steps=args.total_steps // 3,   # don't stop in first 1/3 of budget
            smax_success_thresh=0.7,
            smax_min_steps=100_000,
            verbose=1,
        )
        callbacks.append(early_stop_callback)
        print("🧠 CurriculumEarlyStopCallback enabled "
              "(will stop when Smax is stable and successful).")

    # if args.record_video:
    #     if cv2 is None:
    #         print("⚠️ --record-video requested, but OpenCV is not installed. Skipping video recording.")
    #     else:
    #         video_dir = os.path.join(log_dir, "training_videos")
    #         video_callback = VideoRecorderCallback(
    #             video_dir=video_dir,
    #             record_freq=args.video_freq,
    #             verbose=1
    #         )
    #         callbacks.append(video_callback)
    #         print(f"📹 Video recording enabled: {video_dir}")
    # if args.record_video:
    #     if cv2 is None:
    #         print("⚠️ --record-video requested, but OpenCV is not installed. Skipping video recording.")
    #     else:
    #         video_dir = os.path.join(log_dir, "training_videos")
    #         success_video_dir = os.path.join(log_dir, "success_videos")
    #         success_npz_dir = os.path.join(log_dir, "success_npz")

    #         video_callback = VideoRecorderCallback(
    #             video_dir=video_dir or log_dir,
    #             record_freq=args.video_freq,   # every N episodes
    #             verbose=1,
    #             success_video_dir=success_video_dir,
    #             success_npz_dir=success_npz_dir,
    #             save_success_npz=True,
    #             success_buffer_len=200,        # ~last 200 frames per success
    #         )
    #         callbacks.append(video_callback)
    #         print(f"📹 Video recording enabled: {video_dir}")
    #         print(f"🏁 Success videos dir     : {success_video_dir}")
    #         print(f"💾 Success npz dir        : {success_npz_dir}")
    # --- Video + success recording callback (always attach) ---
    video_dir = os.path.join(log_dir, "training_videos")
    success_video_dir = os.path.join(log_dir, "success_videos")
    success_npz_dir = os.path.join(log_dir, "success_npz")

    os.makedirs(success_video_dir, exist_ok=True)
    os.makedirs(success_npz_dir, exist_ok=True)

    if cv2 is None:
        if args.record_video:
            print("⚠️ --record-video requested, but OpenCV is not installed. "
                "Skipping periodic training videos (success npz still enabled).")

        # We still attach the callback so that SUCCESS .npz trajectories work.
        video_callback = VideoRecorderCallback(
            video_dir=video_dir or log_dir,
            record_freq=10**9,  # effectively disable periodic video
            verbose=1,
            success_video_dir=success_video_dir,
            success_npz_dir=success_npz_dir,
            save_success_npz=True,
            success_buffer_len=200,
        )
    else:
        # If --record-video: use user-specified frequency for periodic videos.
        # If not: still attach, but periodic videos effectively off.
        effective_record_freq = args.video_freq if args.record_video else 10**9

        video_callback = VideoRecorderCallback(
            video_dir=video_dir or log_dir,
            record_freq=effective_record_freq,
            verbose=1,
            success_video_dir=success_video_dir,
            success_npz_dir=success_npz_dir,
            save_success_npz=True,
            success_buffer_len=200,
        )

        if args.record_video:
            print(f"📹 Periodic training videos : {video_dir} (every {args.video_freq} episodes)")
    print(f"🏁 Success videos dir        : {success_video_dir}")
    print(f"💾 Success npz dir           : {success_npz_dir}")

    callbacks.append(video_callback)

    
    # Train
    print("\n🚀 Starting hierarchical training...")
    print("  Learning to follow A*-generated waypoints...\n")
    
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            log_interval=args.log_interval,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # # Save final model
    # model_path = os.path.join(log_dir, "ppo_parking_final.zip")
    # model.save(model_path)
    
    # print(f"\n{'='*60}")
    # print("✅ Hierarchical training complete!")
    # print(f"Final model saved at: {model_path}")
    # print(f"\nTo evaluate:")
    # print(f"  python -m autonomous_parking.sb3_eval_hierarchical --lot lot_a --model-dir {log_dir}")
    # print(f"{'='*60}\n")
    
    # Save final model with steps + (optional) curriculum stage in the filename
    final_steps = int(model.num_timesteps)
    stage_idx = None
    stage_name = None

    if args.use_curriculum:
        try:
            env0 = train_env.envs[0]
            # Unwrap Monitor -> WaypointEnv
            while hasattr(env0, "env"):
                env0 = env0.env
            if getattr(env0, "enable_curriculum", False) and env0.curriculum is not None:
                stage_idx = int(env0.curriculum.current_stage_idx)
                stage_name = env0.curriculum.current_stage.name
        except Exception as e:
            print(f"[WARN] Could not read curriculum stage for metadata: {e}")

    if stage_idx is not None:
        model_filename = f"ppo_parking_final_steps_{final_steps}_S{stage_idx + 1}.zip"
    else:
        model_filename = f"ppo_parking_final_steps_{final_steps}.zip"

    model_path = os.path.join(log_dir, model_filename)
    model.save(model_path)

    # Save metadata for reproducibility
    meta = {
        "run_name": args.run_name,
        "total_timesteps": final_steps,
        "use_curriculum": bool(args.use_curriculum),
        "curriculum_stage_index": stage_idx,
        "curriculum_stage_name": stage_name,
        "cli_args": vars(args),  # v41: Full CLI args for reproducibility
    }
    meta_path = model_path.replace(".zip", ".meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Metadata saved to: {meta_path}")
    except Exception as e:
        print(f"[WARN] Failed to write metadata JSON: {e}")

    print(f"\n{'='*60}")
    print("✅ Hierarchical training complete!")
    print(f"Final model saved at: {model_path}")
    print(f"\nTo evaluate:")
    print(f"  python -m autonomous_parking.sb3_eval_hierarchical --lot lot_a --model-dir {log_dir}")
    print(f"{'='*60}\n")

    
    try:
        train_env.close()
        eval_env.close()
    except Exception:
        pass  # Ignore errors during cleanup (e.g. broken pipe)


if __name__ == "__main__":
    main()
