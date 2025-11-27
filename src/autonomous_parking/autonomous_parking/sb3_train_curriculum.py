#!/usr/bin/env python3
"""
Curriculum RL Training Script - Pure RL Approach

Trains PPO with progressive difficulty stages.
"""

import os
import argparse
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", message="Training and eval env are not of the same type")

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from autonomous_parking.env2d.curriculum_env import CurriculumParkingEnv
from autonomous_parking.sb3_env import ParkingSB3Env


class CurriculumCallback(BaseCallback):
    """Callback to manage curriculum progression."""
    
    def __init__(self, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.current_stage = 1
        
    def _on_step(self) -> bool:
        # Check every check_freq steps
        if self.n_calls % self.check_freq == 0:
            # Get success rate from first environment
            # For SubprocVecEnv, we can't directly access envs
            # Instead, we'll track via info dicts
            if hasattr(self, 'locals') and 'infos' in self.locals:
                infos = self.locals['infos']
                if len(infos) > 0 and 'curriculum_stage' in infos[0]:
                    stage = infos[0]['curriculum_stage']
                    success_rate = infos[0].get('success_rate', 0.0)
                    should_progress = infos[0].get('should_progress', False)
                    
                    if should_progress and stage < 5:
                        print(f"\n{'='*60}")
                        print(f"🎓 CURRICULUM PROGRESSING: Stage {stage} → {stage + 1}")
                        print(f"   Success rate: {success_rate:.1%}")
                        print(f"{'='*60}\n")
                        # Note: Can't directly set stage in SubprocVecEnv
                        # Would need a custom VecEnv method for this
        
        return True


def make_curriculum_env(rank: int, max_episode_steps: int):
    """Create curriculum environment."""
    def _init():
        # CurriculumParkingEnv is already a Gymnasium env (inherits from ParkingEnv)
        from autonomous_parking.env2d.curriculum_env import CurriculumParkingEnv
        env = CurriculumParkingEnv(lot_name="lot_a")
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Curriculum RL Training")
    parser.add_argument("--total-steps", type=int, default=350_000)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--run-name", type=str, default="curriculum_final")
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    
    args = parser.parse_args()
    
    # Results directory
    base_dir = os.path.join("results", "ppo_curriculum")
    log_dir = os.path.join(base_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎓 CURRICULUM RL TRAINING - Pure RL Approach")
    print(f"{'='*60}")
    print(f"Total timesteps : {args.total_steps:,}")
    print(f"Parallel envs   : {args.n_envs}")
    print(f"Run name        : {args.run_name}")
    print(f"Log directory   : {log_dir}")
    print(f"{'='*60}\n")
    
    # Create curriculum environments
    print("Creating curriculum environments...")
    env_fns = [make_curriculum_env(i, args.max_episode_steps) for i in range(args.n_envs)]
    train_env = SubprocVecEnv(env_fns)
    
    # PPO model
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        n_steps=2048 // args.n_envs,
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
        save_freq=args.save_freq // args.n_envs,
        save_path=log_dir,
        name_prefix="ppo_curriculum",
    )
    
    curriculum_callback = CurriculumCallback(
        check_freq=5000 // args.n_envs,
        verbose=1
    )
    
    # Train
    print("\n🚀 Starting curriculum training...\n")
    print("Curriculum stages will progress automatically based on success rate:")
    print("  Stage 1 (80%): Close start (2-3m)")
    print("  Stage 2 (70%): Medium start (5-7m)")
    print("  Stage 3 (60%): Far start (10-12m)")
    print("  Stage 4 (50%): Random in lot")
    print("  Stage 5: Final stage\n")
    
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_callback, curriculum_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    model_path = os.path.join(log_dir, "ppo_parking_final.zip")
    model.save(model_path)
    
    print(f"\n{'='*60}")
    print("✅ Curriculum training complete!")
    print(f"Final model saved at: {model_path}")
    print(f"\nTo evaluate:")
    print(f"  python -m autonomous_parking.sb3_eval --lot lot_a --model-dir {log_dir}")
    print(f"{'='*60}\n")
    
    try:
        train_env.close()
    except Exception:
        pass  # Ignore errors during cleanup


if __name__ == "__main__":
    main()
