#!/usr/bin/env python3
"""
Multi-lot training script for autonomous parking.
Trains on lot_a + lot_b with all bays randomized.

Usage:
    python -m autonomous_parking.sb3_train_multi --total-steps 200000
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
from stable_baselines3.common.vec_env import SubprocVecEnv

from .sb3_env import ParkingSB3Env


def make_env(lot: str, rank: int, max_episode_steps: int):
    """Create environment for parallel training."""
    def _init():
        env = ParkingSB3Env(
            lot_name=lot,
            max_episode_steps=max_episode_steps,
            render_mode=None,
        )
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on multiple parking lots"
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=200_000,
        help="Total environment steps for training",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="multi_lot",
        help="Name for this training run",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )

    args = parser.parse_args()

    # Results directory
    base_dir = os.path.join("results", "ppo_multi")
    log_dir = os.path.join(base_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PPO Multi-Lot Training - LOT_A + LOT_B")
    print(f"{'='*60}")
    print(f"Total timesteps : {args.total_steps:,}")
    print(f"Parallel envs   : {args.n_envs}")
    print(f"Run name        : {args.run_name}")
    print(f"Log directory   : {log_dir}")
    print(f"{'='*60}\n")

    # Create vectorized environments
    # Alternate between lot_a and lot_b for diversity
    print("Creating parallel training environments...")
    lots = ["lot_a", "lot_b"]
    env_fns = [
        make_env(lots[i % len(lots)], i, args.max_episode_steps)
        for i in range(args.n_envs)
    ]
    train_env = SubprocVecEnv(env_fns)

    # Single eval environment on lot_a
    print("Creating evaluation environment...")
    eval_env = ParkingSB3Env(
        lot_name="lot_a",
        max_episode_steps=args.max_episode_steps,
        render_mode=None,
    )
    eval_env = Monitor(eval_env)

    # PPO configuration optimized for 37D observations
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        n_steps=2048 // args.n_envs,  # Adjusted for vectorized env
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=os.path.join(base_dir, "tb"),
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,  # Adjusted for vectorized env
        save_path=log_dir,
        name_prefix=f"ppo_multi",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5_000 // args.n_envs,  # Adjusted for vectorized env
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Train
    print("\n🚀 Starting multi-lot PPO training...\n")
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
        f"--lot lot_a --model-dir {log_dir}"
    )
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {base_dir}/tb")
    print(f"{'='*60}\n")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
