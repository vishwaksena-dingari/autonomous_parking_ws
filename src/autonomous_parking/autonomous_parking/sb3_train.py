#!/usr/bin/env python3
"""
Train PPO agent on ParkingEnv using Stable-Baselines3.

Usage:
    python -m autonomous_parking.sb3_train --lot lot_a --total-steps 50000
"""

import os
import argparse

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from .sb3_env import ParkingSB3Env


def make_env(lot: str, max_episode_steps: int, log_dir: str | None = None) -> gym.Env:
    """Create a monitored ParkingSB3Env for SB3 training."""
    env = ParkingSB3Env(
        lot_name=lot,
        max_episode_steps=max_episode_steps,
        render_mode=None,
    )
    if log_dir is not None:
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on ParkingEnv using Stable-Baselines3"
    )
    parser.add_argument(
        "--lot",
        type=str,
        default="lot_a",
        choices=["lot_a", "lot_b"],
        help="Which parking lot configuration to train on",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=50_000,
        help="Total environment steps for training",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Max steps per episode for the wrapper (safety cap)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="run1",
        help="Name for this training run (used in log directory)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Save checkpoint every N environment steps",
    )

    args = parser.parse_args()

    # Results directory structure
    base_dir = os.path.join("results", f"ppo_{args.lot}")
    log_dir = os.path.join(base_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PPO Training - {args.lot.upper()}")
    print(f"{'='*60}")
    print(f"Total timesteps : {args.total_steps:,}")
    print(f"Run name        : {args.run_name}")
    print(f"Log directory   : {log_dir}")
    print(f"{'='*60}\n")

    # Training and eval envs
    print("Creating training environment...")
    train_env = make_env(args.lot, args.max_episode_steps, log_dir)

    print("Creating evaluation environment...")
    # Evaluation environment (separate from training)
    eval_env = ParkingSB3Env(
        lot_name=args.lot,
        max_episode_steps=args.max_episode_steps,
        render_mode=None,
    )
    # IMPORTANT: Wrap eval env with Monitor for proper metric tracking
    eval_env = Monitor(eval_env)

    # PPO configuration optimized for high-dimensional observations (69D)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        n_steps=4096,        # INCREASED from 2048 for better sampling
        batch_size=128,      # INCREASED from 64 for stable gradients  
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,  # REDUCED from 3e-4 for stability
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=os.path.join(base_dir, "tb"),
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=log_dir,
        name_prefix=f"ppo_{args.lot}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Train
    print("\n🚀 Starting PPO training...\n")
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

    # Save final model
    model_path = os.path.join(log_dir, "ppo_parking_final.zip")
    model.save(model_path)
    print(f"\n{'='*60}")
    print("✅ Training complete!")
    print(f"Final model saved at: {model_path}")
    print(f"\nTo evaluate, run:")
    print(
        f"  python -m autonomous_parking.sb3_eval "
        f"--lot {args.lot} --model-dir {log_dir}"
    )
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {base_dir}/tb")
    print(f"{'='*60}\n")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
