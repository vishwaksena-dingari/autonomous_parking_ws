#!/usr/bin/env python3
"""
Evaluate a trained Hierarchical PPO model on WaypointEnv with visualization.

Usage:
    python -m autonomous_parking.sb3_eval_hierarchical --lot lot_a --model-dir results/ppo_hierarchical/hierarchical_final
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import PPO

from .env2d.waypoint_env import WaypointEnv


def run_eval(
    model_path: str,
    lot: str = "lot_a",
    episodes: int = 5,
    max_episode_steps: int = 800,
    dt: float = 0.05,
    deterministic: bool = True,
    render: bool = True,
):
    """Evaluate trained Hierarchical PPO model with visualization."""
    env = WaypointEnv(
        lot_name=lot,
        render_mode="human" if render else None,
        max_steps=max_episode_steps,
    )

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

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
        print(f"Waypoints generated: {len(env.waypoints)}")

        if hasattr(env, 'ax') and env.ax is not None:
            waypoints = np.array(env.waypoints)
            env.ax.plot(waypoints[:, 0], waypoints[:, 1], 'r--', linewidth=2, alpha=0.7, label='A* Path')
            env.ax.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=20, zorder=5)
            env.fig.canvas.draw()
            env.fig.canvas.flush_events()

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_count += 1

            if ep == 1 or (ep % 10 == 0):
                # Note: 't' and 'dist' are not defined in this scope.
                # Assuming 't' should be 'step_count' and 'dist' should be 'info.get("dist_to_goal", 0)'.
                # Also, 'env.current_waypoint_idx' and 'env.success' might not be directly accessible,
                # using 'info.get' for consistency.
                print(
                    f"step={step_count:03d} "
                    f"reward={reward:7.3f} "
                    f"wp={info.get('waypoint_idx', 0)}/{info.get('total_waypoints', 0)} "
                    f"dist_goal={info.get('dist_to_goal', 0):.2f} "
                    f"success={info.get('success', False)}"
                )

            if render:
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
            f"success={info.get('success', False)}, "
            f"collision={info.get('collision', False)}"
        )

    print("\n=== EVAL SUMMARY ===")
    print(f"Episodes       : {episodes}")
    print(f"Avg return     : {np.mean(all_returns):.3f}")
    print(f"Successes      : {successes}/{episodes}")
    print(f"Collisions     : {collisions}/{episodes}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical PPO model on WaypointEnv")
    parser.add_argument("--lot", type=str, default="lot_a", choices=["lot_a", "lot_b"], help="Parking lot configuration")
    parser.add_argument("--model-dir", type=str, default="results/ppo_hierarchical/hierarchical_final", help="Directory containing trained model")
    parser.add_argument("--model-name", type=str, default="ppo_parking_final.zip", help="Model filename")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--dt", type=float, default=0.05, help="Visualization sleep time")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--no-render", action="store_true", help="Disable visualization")
    parser.add_argument("--max-episode-steps", type=int, default=800, help="Maximum steps per episode")

    args = parser.parse_args()
    # model_path = os.path.join(args.model_dir, args.model_name)

    # if not os.path.exists(model_path):
    #     best_model = os.path.join(args.model_dir, "best_model.zip")
    #     if os.path.exists(best_model):
    #         print(f"Final model not found, using best_model: {best_model}")
    #         model_path = best_model
    #     else:
    #         import glob
    #         checkpoints = glob.glob(os.path.join(args.model_dir, "ppo_hierarchical_*.zip"))
    #         if checkpoints:
    #             model_path = sorted(checkpoints)[-1]
    #             print(f"Using latest checkpoint: {model_path}")
    # ----------------------------------------------------------------------
    #  AUTOMATIC MODEL PICKER + METADATA LOADER
    # ----------------------------------------------------------------------
    import glob
    import json

    # 1) Direct path (user specified exact filename)
    model_path = os.path.join(args.model_dir, args.model_name)

    candidate_paths = []

    if os.path.exists(model_path):
        candidate_paths.append(model_path)

    # 2) Collect final-step models (preferred)
    step_models = glob.glob(os.path.join(args.model_dir, "ppo_parking_final_steps_*.zip"))
    candidate_paths.extend(step_models)

    # 3) Best model from EvalCallback
    best_model = os.path.join(args.model_dir, "best_model.zip")
    if os.path.exists(best_model):
        candidate_paths.append(best_model)

    # 4) Fallback: checkpoints
    checkpoints = glob.glob(os.path.join(args.model_dir, "ppo_hierarchical_*.zip"))
    candidate_paths.extend(checkpoints)

    if not candidate_paths:
        raise FileNotFoundError(f"No model found in directory: {args.model_dir}")

    # Pick the model with the highest step count
    def extract_steps(path):
        import re
        m = re.search(r"steps_(\d+)", path)
        return int(m.group(1)) if m else -1

    model_path = sorted(candidate_paths, key=extract_steps)[-1]

    print(f"\n📌 Selected model: {model_path}")

    # Load metadata JSON if exists
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


    run_eval(
        model_path=model_path,
        lot=args.lot,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        dt=args.dt,
        deterministic=not args.stochastic,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
