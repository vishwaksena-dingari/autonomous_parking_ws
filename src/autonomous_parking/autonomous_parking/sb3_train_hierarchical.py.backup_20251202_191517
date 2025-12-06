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
import cv2
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


class VideoRecorderCallback(BaseCallback):
    """
    Callback to record training episodes as video frames (silently).
    Records every N episodes to avoid massive video files.
    """
    def __init__(self, video_dir: str, record_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.record_freq = record_freq  # Record every N episodes
        
        self.episode_count = 0
        self.recording = False
        self.current_video_path = None
        self.video_writer = None
        self.waypoint_drawn = False  # Track if waypoints have been drawn for current episode
        # NEW: keep track of path artists so we can remove them
        self._path_artists = []
    def _on_step(self) -> bool:
        # Check if we should start recording this episode
        if len(self.locals.get("dones", [])) > 0:
            if self.locals["dones"][0]:  # Only trigger if the RECORDED env (index 0) is done
                self.episode_count += 1
                
                # Save video if we were recording
                if self.recording and self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                    if self.verbose > 0:
                        print(f"📹 Saved training video: {self.current_video_path}")
                    self.recording = False
                    self.waypoint_drawn = False

                    # NEW: remove any A* overlays from the figure
                    for art in self._path_artists:
                        try:
                            art.remove()
                        except Exception:
                            pass
                    self._path_artists = []

                
                # Start recording next episode if it's time
                if self.episode_count % self.record_freq == 0:
                    self.recording = True
                    self.waypoint_drawn = False
                    self.current_video_path = self.video_dir / f"training_ep{self.episode_count:05d}.mp4"
                    if self.verbose > 0:
                        print(f"🎬 Recording episode {self.episode_count}...")
        
        # Capture frame if recording
        if self.recording:
            # Get frame from first environment (index 0)
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                env = self.training_env.envs[0]
                # Unwrap Monitor
                if hasattr(env, 'env'):
                    env = env.env
                    
                # # Draw waypoints on first frame of episode
                # if not self.waypoint_drawn and hasattr(env, 'waypoints') and hasattr(env, 'ax'):
                #     if env.waypoints is not None and env.ax is not None and len(env.waypoints) > 0:
                #         waypoints = np.array(env.waypoints)
                #         # Draw path as red dashed line
                #         env.ax.plot(waypoints[:, 0], waypoints[:, 1], 
                #                    'r--', linewidth=2.5, alpha=0.8, 
                #                    label='A* Path', zorder=2.5)
                #         # Draw waypoints as red dots
                #         env.ax.scatter(waypoints[:, 0], waypoints[:, 1], 
                #                       c='red', s=30, zorder=2.6, alpha=0.7)
                #         self.waypoint_drawn = True
                                # Draw waypoints on first frame of episode
                if not self.waypoint_drawn and hasattr(env, 'waypoints') and hasattr(env, 'ax'):
                    if env.waypoints is not None and env.ax is not None and len(env.waypoints) > 0:
                        waypoints = np.array(env.waypoints)

                        # NEW: remove any leftover path artists (from previous recorded eps)
                        for art in self._path_artists:
                            try:
                                art.remove()
                            except Exception:
                                pass
                        self._path_artists = []

                        # Draw path as red dashed line
                        line, = env.ax.plot(
                            waypoints[:, 0], waypoints[:, 1],
                            'r--', linewidth=2.5, alpha=0.8,
                            zorder=2.5,
                        )
                        # Draw waypoints as red dots
                        pts = env.ax.scatter(
                            waypoints[:, 0], waypoints[:, 1],
                            c='red', s=30, zorder=2.6, alpha=0.7,
                        )

                        # NEW: remember these so we can remove them later
                        self._path_artists = [line, pts]
                        self.waypoint_drawn = True
                        
                        # Update title with episode info for debugging
                        spawn_x, spawn_y = env.state[0], env.state[1]
                        bay_id = env.goal_bay.get('id', 'Unknown')
                        lot = env.lot_name
                        env.ax.set_title(
                            f"{lot.upper()} | Goal: {bay_id} | "
                            f"Spawn: ({spawn_x:.1f}, {spawn_y:.1f}) | "
                            f"Ep {self.episode_count}",
                            fontsize=10
                        )

                
                if hasattr(env, 'render'):
                    # Setup rendering if needed
                    if env.fig is None:
                        env._setup_render()
                    
                    # Update the plot
                    env.render()
                    
                    # Capture frame from matplotlib figure
                    if env.fig is not None:
                        env.fig.canvas.draw()
                        # Get buffer from canvas (macOS uses ARGB, not RGB)
                        try:
                            # Try RGB first (Linux/Windows)
                            buf = np.frombuffer(env.fig.canvas.tostring_rgb(), dtype=np.uint8)
                            w, h = env.fig.canvas.get_width_height()
                            frame = buf.reshape(h, w, 3)
                        except AttributeError:
                            # macOS uses ARGB
                            buf = np.frombuffer(env.fig.canvas.tostring_argb(), dtype=np.uint8)
                            w, h = env.fig.canvas.get_width_height()
                            
                            # Handle Retina display (2x scaling)
                            if buf.size == w * h * 4 * 4:  # 2x width, 2x height
                                w, h = w * 2, h * 2
                                
                            frame_argb = buf.reshape(h, w, 4)
                            # Convert ARGB to RGB (drop alpha channel)
                            frame = frame_argb[:, :, 1:]  # Skip A, keep RGB
                        
                        if frame is not None and isinstance(frame, np.ndarray):
                            # Initialize video writer on first frame
                            if self.video_writer is None:
                                h, w = frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                self.video_writer = cv2.VideoWriter(
                                    str(self.current_video_path), 
                                    fourcc, 
                                    20.0,  # 20 FPS
                                    (w, h)
                                )
                            
                            # Convert RGB to BGR for OpenCV
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                self.video_writer.write(frame_bgr)
        
        return True
    
    def _on_training_end(self) -> None:
        """Release any open video writer."""
        if self.video_writer is not None:
            self.video_writer.release()

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
        self.check_freq = check_freq
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



def make_waypoint_env(rank: int, max_episode_steps: int = 800, multi_lot: bool = True, enable_curriculum: bool = False, verbose: bool = True):
    """
    Create a waypoint-following environment for hierarchical RL.
    
    Args:
        rank: Environment ID for seeding
        max_episode_steps: Maximum steps per episode
        multi_lot: If True, randomly select between lot_a and lot_b on each reset
        enable_curriculum: If True, use v15 CurriculumManager for progressive learning
    """
    def _init():
        env = WaypointEnv(
            lot_name="lot_a",  # Default lot
            multi_lot=multi_lot,  # Enable multi-lot training
            enable_curriculum=enable_curriculum,  # v15: Curriculum learning
            render_mode=None,
            max_steps=max_episode_steps,  # Pass max steps to environment
            verbose=verbose,
        )
        env = Monitor(env)
        env.reset(seed=42 + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Hierarchical RL Training")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--max-episode-steps", type=int, default=2000)  # Increased for final parking
    parser.add_argument("--run-name", type=str, default="hierarchical_v14_20",
                        help="Name for this training run")
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
    args = parser.parse_args()
    
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
        ) for i in range(args.n_envs)
    ])
    
    # Create evaluation environment (single env, no curriculum for eval)
    eval_env = DummyVecEnv([
        make_waypoint_env(
            0, 
            max_episode_steps=args.max_episode_steps,
            multi_lot=args.multi_lot,
            enable_curriculum=False,  # No curriculum during evaluation
            verbose=False,   
        )
    ])
    
    # PPO model (simpler task = faster learning)
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        n_steps=2048 // args.n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,  # Lowered for stability (was 3e-4)
        clip_range=0.1,  # More conservative updates (was 0.2)
        ent_coef=0.05,  # Higher exploration (was 0.01)
        vf_coef=0.5,
        tensorboard_log=os.path.join(base_dir, "tb"),
    )

    # tb_run_dir = getattr(model.logger, "dir", None)
    # if tb_run_dir is not None:
    #     tb_path = Path(tb_run_dir)
    #     tb_pretty = str(tb_path).replace(str(Path.home()), "~")
    #     print(f"TensorBoard run dir  : {tb_pretty}")
    # else:
    #     print("TensorBoard run dir  : <logger has no directory>")

        # Make sure TB base dir exists (optional)
    os.makedirs(abs_tb_dir, exist_ok=True)

    # Make sure TB base dir exists (optional but nice)
    os.makedirs(abs_tb_dir, exist_ok=True)

    # Try to get the SB3 logger directory without depending on model.logger property
    tb_run_dir = None
    try:
        raw_logger = getattr(model, "_logger", None)
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
            check_freq=1000 // args.n_envs,
            min_total_steps=args.total_steps // 3,   # don't stop in first 1/3 of budget
            smax_success_thresh=0.7,
            smax_min_steps=100_000,
            verbose=1,
        )
        callbacks.append(early_stop_callback)
        print("🧠 CurriculumEarlyStopCallback enabled "
              "(will stop when Smax is stable and successful).")

    if args.record_video:
        video_dir = os.path.join(log_dir, "training_videos")
        video_callback = VideoRecorderCallback(
            video_dir=video_dir,
            record_freq=args.video_freq,
            verbose=1
        )
        callbacks.append(video_callback)
        print(f"📹 Video recording enabled: {video_dir}")
    
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
