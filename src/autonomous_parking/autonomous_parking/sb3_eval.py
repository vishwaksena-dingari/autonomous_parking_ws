#!/usr/bin/env python3
"""
Evaluate a trained PPO model on ParkingEnv with visualization.

Usage:
    python -m autonomous_parking.sb3_eval --lot lot_a --model-dir results/ppo_lot_a/run1
"""

import os
import time
import argparse
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO

from .env2d.parking_env import ParkingEnv


def run_eval(
    model_path: str,
    lot: str = "lot_a",
    episodes: int = 5,
    max_episode_steps: int = 600,
    dt: float = 0.05,
):
    """
    Evaluate trained PPO model with visualization.

    Args:
        model_path: Path to saved model (.zip file)
        lot: Parking lot name ('lot_a' or 'lot_b')
        episodes: Number of evaluation episodes
        max_episode_steps: Max steps per episode
        dt: Sleep time between steps for visualization (seconds)
    """
    # Create env with rendering enabled
    # Create environment (now Gymnasium-native)
    env = ParkingEnv(
        lot_name=lot,
        render_mode="human",
    )

    model = PPO.load(model_path)
    print(f"Loaded PPO model from: {model_path}")

    all_returns = []
    all_lengths = []
    successes = 0
    collisions = 0

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        step_count = 0

        print(f"\n=== Episode {ep} ({lot}) ===")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_count += 1

            dist = info.get("dist", None)
            yaw_err = info.get("yaw_err", None)
            success = info.get("success", False)
            collision = info.get("collision", False)

            if step_count % 20 == 0:
                print(
                    f"step={step_count:03d} "
                    f"reward={reward:7.3f} "
                    f"dist={dist:6.3f} "
                    f"yaw_err={yaw_err:6.3f} "
                    f"success={success} "
                    f"collision={collision}"
                )

            env.render()
            time.sleep(dt)

        all_returns.append(ep_reward)
        all_lengths.append(step_count)
        if info.get("success", False):
            successes += 1
        if info.get("collision", False):
            collisions += 1

        print(
            f"Episode {ep} finished: "
            f"total_reward={ep_reward:.3f}, "
            f"steps={step_count}, "
            f"done={done}, truncated={truncated}, "
            f"success={info.get('success', False)}, "
            f"collision={info.get('collision', False)}"
        )

    print("\n=== EVAL SUMMARY ===")
    print(f"Episodes       : {episodes}")
    print(f"Avg return     : {np.mean(all_returns):.3f}")
    print(f"Std return     : {np.std(all_returns):.3f}")
    print(f"Avg length     : {np.mean(all_lengths):.1f}")
    print(f"Successes      : {successes}/{episodes}")
    print(f"Collisions     : {collisions}/{episodes}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model on ParkingEnv in 2D"
    )
    parser.add_argument(
        "--lot",
        type=str,
        default="lot_a",
        choices=["lot_a", "lot_b"],
        help="Which parking lot configuration to evaluate on",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="results/ppo_lot_a/run1",
        help="Directory that contains the trained model zip",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_parking_final.zip",
        help="Model filename inside model-dir",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=600,
        help="Max steps per episode wrapper",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Sleep time between steps for visualization (seconds)",
    )

    # args = parser.parse_args()
    # model_path = os.path.join(args.model_dir, args.model_name)

    # run_eval(
    #     model_path=model_path,
    #     lot=args.lot,
    #     episodes=args.episodes,
    #     max_episode_steps=args.max_episode_steps,
    #     dt=args.dt,
    # )
    
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    #  AUTOMATIC MODEL PICKER + METADATA LOADER (2D PPO)
    # ----------------------------------------------------------------------
    import glob
    import json

    # 1) Direct path (user-specified filename, default: ppo_parking_final.zip)
    model_path = os.path.join(args.model_dir, args.model_name)

    candidate_paths = []

    if os.path.exists(model_path):
        candidate_paths.append(model_path)

    # 2) New-style final models with step count (if you adopt that later)
    step_models = glob.glob(os.path.join(args.model_dir, "ppo_parking_final_steps_*.zip"))
    candidate_paths.extend(step_models)

    # 3) Standard final model used by sb3_train / sb3_train_multi
    final_plain = os.path.join(args.model_dir, "ppo_parking_final.zip")
    if os.path.exists(final_plain):
        candidate_paths.append(final_plain)

    # 4) Best model from EvalCallback
    best_model = os.path.join(args.model_dir, "best_model.zip")
    if os.path.exists(best_model):
        candidate_paths.append(best_model)

    # 5) Generic PPO checkpoints (plain 2D + multi-lot)
    #    e.g., ppo_lot_a_*.zip, ppo_lot_b_*.zip, ppo_multi_*.zip, etc.
    generic_checkpoints = glob.glob(os.path.join(args.model_dir, "ppo_*.zip"))
    candidate_paths.extend(generic_checkpoints)

    if not candidate_paths:
        raise FileNotFoundError(f"No model found in directory: {args.model_dir}")

    # Pick the model with the highest step count if encoded, otherwise by name
    def extract_steps(path: str) -> int:
        import re
        m = re.search(r"steps_(\d+)", path)
        return int(m.group(1)) if m else -1

    candidate_paths = sorted(candidate_paths, key=lambda p: (extract_steps(p), p))
    model_path = candidate_paths[-1]

    print(f"\n📌 Selected model: {model_path}")

    # Try to load metadata if present (future-proof: if you add meta writing later)
    meta_path = model_path.replace(".zip", ".meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print("\n📄 Metadata loaded:")
            print(f"  Run: {meta.get('run_name')}")
            print(f"  Total Steps: {meta.get('total_timesteps')}")
            stage_idx = meta.get("curriculum_stage_index")
            stage_name = meta.get("curriculum_stage_name")
            if stage_idx is not None:
                print(f"  Curriculum Stage: {stage_idx + 1} ({stage_name})")
            print()
        except Exception as e:
            print(f"[WARN] Could not read metadata: {e}")

    # Now run evaluation with the chosen model
    run_eval(
        model_path=model_path,
        lot=args.lot,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        dt=args.dt,
    )



if __name__ == "__main__":
    main()
