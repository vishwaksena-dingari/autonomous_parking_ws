#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Parking Agent
v38.7 - Systematic parameter search

Usage:
    python hyper_tune.py --mode grid    # Full grid search
    python hyper_tune.py --mode random  # Random search (faster)
    python hyper_tune.py --mode single --config best  # Run single best config
"""

import itertools
import subprocess
import re
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import imageio
from stable_baselines3 import PPO
from autonomous_parking.env2d.waypoint_env import WaypointEnv

# =====================================================
# TUNING CONFIGURATION (ROUND 2 - Narrowed Search)
# =====================================================

# Based on Round 1 Winners:
#   🥇 rand_002: align=100, corridor=0.1, ent=0.007
#   🥈 rand_008: align=30, corridor=0.05, ent=0.02
#   🥉 rand_014: align=45, corridor=0.05, ent=0.01
# 
# Key insight: ALL TOP 3 had success_bonus=70.0!

# v40: New Physics Regime (Round 1 Reset)
# Collisions are now fatal (-500). Parameters must be robust.
PARAM_GRID = {
    'align_w': [50.0, 80.0, 100.0, 120.0, 150.0],
    'success_bonus': [50.0, 70.0, 100.0],
    'bay_entry_bonus': [50.0, 70.0, 90.0],
    'corridor_penalty': [0.05, 0.1, 0.2, 0.5],
    'vel_reward_w': [0.01, 0.02, 0.05],
    'ent_coef': [0.005, 0.01, 0.02, 0.05]
}

# Short training settings for tuning
TUNE_STEPS = 80_000  # Keep same duration
TUNE_N_ENVS = 2      # Fewer envs = less noise
TUNE_MAX_EP_STEPS = 800


@dataclass
class TuningResult:
    """Results from a single tuning run."""
    config: Dict
    run_name: str
    success_rate: float
    mean_reward: float
    path_completion: float
    collision_rate: float
    off_path_rate: float
    mean_final_dist: float
    mean_ep_length: float
    score: float  # Combined tuning score


def parse_log_file(log_path: Path) -> Dict:
    """Parse training log for key metrics."""
    if not log_path.exists():
        return {}
    
    text = log_path.read_text()
    
    # Extract metrics using regex
    metrics = {
        "success_rate": 0.0,
        "mean_reward": -999999.0,
        "mean_ep_length": 999.0,
        "path_completion": 0.0,
        "collision_rate": 0.0,
        "off_path_rate": 0.0,
        "mean_final_dist": 20.0,
    }
    
    # Success rate: |    success_rate         | 0.34         |
    m = re.findall(r"\|\s*success_rate\s*\|\s*([0-9.]+)", text)
    if m:
        metrics["success_rate"] = float(m[-1])
    
    # Mean reward: episode_reward=XXX or mean_reward=XXX
    m = re.findall(r"mean_reward\s*[=|]\s*(-?[0-9.e+\-]+)", text)
    if m:
        try:
            metrics["mean_reward"] = float(m[-1])
        except ValueError:
            pass
    
    # Episode length: mean_ep_length
    m = re.findall(r"mean_ep_length\s*[=|]\s*([0-9.]+)", text)
    if m:
        metrics["mean_ep_length"] = float(m[-1])
    
    # Parse EP_METRICS lines if present
    ep_metrics = re.findall(
        r"\[EP_METRICS\].*?path=([0-9.]+).*?dist=([0-9.]+).*?coll=([01]).*?offpath=([01])",
        text
    )
    if ep_metrics:
        path_completions = [float(m[0]) for m in ep_metrics]
        final_dists = [float(m[1]) for m in ep_metrics]
        collisions = [int(m[2]) for m in ep_metrics]
        off_paths = [int(m[3]) for m in ep_metrics]
        
        n = len(ep_metrics)
        metrics["path_completion"] = sum(path_completions) / n
        metrics["mean_final_dist"] = sum(final_dists) / n
        metrics["collision_rate"] = sum(collisions) / n
        metrics["off_path_rate"] = sum(off_paths) / n
    
    # Fallback: parse waypoint progress from log
    waypoint_matches = re.findall(r"✓ Waypoint (\d+)/(\d+)", text)
    if waypoint_matches and not ep_metrics:
        completions = [int(m[0]) / int(m[1]) for m in waypoint_matches[-50:]]
        if completions:
            metrics["path_completion"] = sum(completions) / len(completions)
    
    return metrics


def calculate_score(metrics: Dict) -> float:
    """
    Calculate combined tuning score.
    Higher is better.
    
    Prioritizes:
    1. Success rate (most important)
    2. Path completion (secondary)
    3. Low collisions/off-path (safety)
    4. Short final distance (precision)
    """
    score = (
        100.0 * metrics.get("success_rate", 0.0)
        + 30.0 * metrics.get("path_completion", 0.0)
        - 0.1 * metrics.get("mean_ep_length", 999.0)
        - 40.0 * metrics.get("collision_rate", 0.0)
        - 25.0 * metrics.get("off_path_rate", 0.0)
        - 2.0 * metrics.get("mean_final_dist", 20.0)
    )
    
    # Bonus for reasonable rewards (not exploding)
    mean_rew = metrics.get("mean_reward", -999999.0)
    if -1000 < mean_rew < 1000:
        score += 10.0  # Bonus for stable rewards
    elif -10000 < mean_rew < 10000:
        score += 5.0  # Small bonus
    
    return score


def run_training(config: Dict, run_name: str, log_dir: Path) -> Path:
    """Run a single training with given config."""
    log_path = log_dir / f"{run_name}.log"
    
    # Use absolute path to Python
    python_path = Path.home() / "autonomous_parking_ws" / ".venv" / "bin" / "python"
    script_dir = Path.home() / "autonomous_parking_ws" / "src" / "autonomous_parking"
    
    cmd = [
        str(python_path), "-m", "autonomous_parking.sb3_train_hierarchical",
        "--total-steps", str(TUNE_STEPS),
        "--n-envs", str(TUNE_N_ENVS),
        "--max-episode-steps", str(TUNE_MAX_EP_STEPS),
        "--use-curriculum",
        "--run-name", run_name,
        "--quiet-env",
        # Video params
        "--record-video",
        "--video-freq", "25",
        # Reward params
        "--align-w", str(config["align_w"]),
        "--success-bonus", str(config["success_bonus"]),
        "--bay-entry-bonus", str(config["bay_entry_bonus"]),
        "--corridor-penalty", str(config["corridor_penalty"]),
        "--vel-reward-w", str(config["vel_reward_w"]),
        # PPO params
        "--ent-coef", str(config["ent_coef"]),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Config: {config}")
    print(f"{'='*60}")
    
    with log_path.open("w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(script_dir),
        )
    
    if result.returncode != 0:
        print(f"⚠️  Run {run_name} failed with code {result.returncode}")
    
    return log_path


def grid_search(log_dir: Path) -> List[TuningResult]:
    """Run full grid search over all parameter combinations."""
    results = []
    
    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combinations = list(itertools.product(*values))
    
    print(f"\n🔍 GRID SEARCH: {len(combinations)} configurations")
    print(f"Parameters: {keys}")
    print(f"Estimated time: ~{len(combinations) * 5} minutes\n")
    
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        run_name = f"tune_{i:03d}_a{config['align_w']:.0f}_b{config['bay_entry_bonus']:.0f}_e{config['ent_coef']}"
        
        log_path = run_training(config, run_name, log_dir)
        metrics = parse_log_file(log_path)
        score = calculate_score(metrics)
        
        result = TuningResult(
            config=config,
            run_name=run_name,
            success_rate=metrics.get("success_rate", 0.0),
            mean_reward=metrics.get("mean_reward", -999999.0),
            path_completion=metrics.get("path_completion", 0.0),
            collision_rate=metrics.get("collision_rate", 0.0),
            off_path_rate=metrics.get("off_path_rate", 0.0),
            mean_final_dist=metrics.get("mean_final_dist", 20.0),
            mean_ep_length=metrics.get("mean_ep_length", 999.0),
            score=score,
        )
        results.append(result)
        
        print(f"[{i+1}/{len(combinations)}] {run_name}: score={score:.1f}, "
              f"success={metrics.get('success_rate', 0):.1%}, "
              f"path={metrics.get('path_completion', 0):.1%}")
    
    return results


def random_search(log_dir: Path, n_samples: int = 15, max_workers: int = 2) -> List[TuningResult]:
    """Run random search over parameter space in parallel."""
    import random
    results: List[TuningResult] = []
    
    print(f"\n🎲 RANDOM SEARCH (parallel): {n_samples} configurations")
    print(f"   Max workers: {max_workers}")
    print(f"   Each worker uses {TUNE_N_ENVS} envs → total: {max_workers * TUNE_N_ENVS} parallel envs")
    
    # 1) Pre-generate all configs and run_names
    jobs = []
    for i in range(n_samples):
        config = {k: random.choice(v) for k, v in PARAM_GRID.items()}
        run_name = f"rand_{i:03d}_{datetime.now().strftime('%H%M%S')}"
        jobs.append((i, config, run_name))
    
    # 2) Worker function: train + parse + score
    def worker(idx: int, config: Dict, run_name: str) -> TuningResult:
        log_path = run_training(config, run_name, log_dir)
        metrics = parse_log_file(log_path)
        score = calculate_score(metrics)
        
        return TuningResult(
            config=config,
            run_name=run_name,
            success_rate=metrics.get("success_rate", 0.0),
            mean_reward=metrics.get("mean_reward", -999999.0),
            path_completion=metrics.get("path_completion", 0.0),
            collision_rate=metrics.get("collision_rate", 0.0),
            off_path_rate=metrics.get("off_path_rate", 0.0),
            mean_final_dist=metrics.get("mean_final_dist", 20.0),
            mean_ep_length=metrics.get("mean_ep_length", 999.0),
            score=score,
        )
    
    # 3) Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(worker, idx, config, run_name): idx
            for idx, config, run_name in jobs
        }
        
        finished = 0
        total = len(jobs)
        
        for future in as_completed(future_to_idx):
            r = future.result()
            results.append(r)
            finished += 1
            
            print(f"[{finished}/{total}] {r.run_name}: "
                  f"score={r.score:.1f}, "
                  f"success={r.success_rate:.1%}, "
                  f"path={r.path_completion:.1%}")
    
    return results


def print_results(results: List[TuningResult], top_n: int = 10):
    """Print sorted results."""
    results.sort(key=lambda r: r.score, reverse=True)
    
    print(f"\n{'='*80}")
    print(f"🏆 TOP {min(top_n, len(results))} CONFIGURATIONS")
    print(f"{'='*80}")
    
    for i, r in enumerate(results[:top_n]):
        print(f"\n#{i+1}: {r.run_name}")
        print(f"   Score: {r.score:.1f}")
        print(f"   Success: {r.success_rate:.1%} | Path: {r.path_completion:.1%}")
        print(f"   Reward: {r.mean_reward:.0f} | Dist: {r.mean_final_dist:.1f}m")
        print(f"   Collision: {r.collision_rate:.1%} | Off-path: {r.off_path_rate:.1%}")
        print(f"   Config: {r.config}")
    
    # Print best config as command
    if results:
        best = results[0]
        print(f"\n{'='*80}")
        print(f"🚀 BEST CONFIG - Use this for full training:")
        print(f"{'='*80}")
        cmd = (
            f"python -m autonomous_parking.sb3_train_hierarchical \\\n"
            f"  --total-steps 1500000 \\\n"
            f"  --n-envs 6 \\\n"
            f"  --max-episode-steps 2000 \\\n"
            f"  --use-curriculum \\\n"
            f"  --run-name v38_TUNED \\\n"
            f"  --record-video \\\n"
            f"  --video-freq 25 \\\n"
            f"  --align-w {best.config['align_w']} \\\n"
            f"  --success-bonus {best.config['success_bonus']} \\\n"
            f"  --bay-entry-bonus {best.config['bay_entry_bonus']} \\\n"
            f"  --corridor-penalty {best.config['corridor_penalty']} \\\n"
            f"  --vel-reward-w {best.config['vel_reward_w']} \\\n"
            f"  --ent-coef {best.config['ent_coef']}"
        )
        print(cmd)


def save_results(results: List[TuningResult], output_path: Path):
    """Save results to JSON."""
    data = [
        {
            "run_name": r.run_name,
            "score": r.score,
            "success_rate": r.success_rate,
            "mean_reward": r.mean_reward,
            "path_completion": r.path_completion,
            "collision_rate": r.collision_rate,
            "off_path_rate": r.off_path_rate,
            "mean_final_dist": r.mean_final_dist,
            "mean_ep_length": r.mean_ep_length,
            "config": r.config,
        }
        for r in results
    ]
    
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_path}")


def generate_best_video(log_dir: Path, best_result: TuningResult):
    """Generate a video of the best agent."""
    print(f"\n🎥 Generating video for best run: {best_result.run_name}...")
    
    model_path = log_dir / best_result.run_name / "best_model.zip"
    if not model_path.exists():
        print(f"⚠️ Best model not found at {model_path}, skipping video.")
        return

    # Load environment with render mode
    # Assuming config is passed or defaults used. Ideally pass config params.
    # For now, use default env ID params, or minimal override.
    # If the tuned params affect env creation (like rewards), they are baked into the model's behavior,
    # but the environment instantiation here should match.
    # We mainly need the visual behavior, so defaults are okay-ish, 
    # BUT if specific rewards affect termination logic, it might matter.
    # Let's just use defaults for visualization.
    env = WaypointEnv(render_mode="rgb_array", verbose=False)
    
    # Load model
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    video_path = log_dir / "best_agent_video.mp4"
    frames = []
    
    print(f"   Recording 25 episodes (All starting at Level 1)...")
    
    # User Request: "ALL TURNS 25 EPS EVERY START WITH EPS 1"
    for i in range(25):
        # Force Level 1 Curriculum (Easy Mode)
        # ParkingEnv increments episode_count in reset(), so set to 0 here.
        if hasattr(env, 'unwrapped'):
            env.unwrapped.episode_count = 0
            # Also reset WaypointEnv curriculum if it exists
            if hasattr(env.unwrapped, 'curriculum') and env.unwrapped.curriculum:
                env.unwrapped.curriculum.current_stage_idx = 0
        
        obs, _ = env.reset(seed=42 + i) # Vary seed slightly for variety
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                 frames.append(frame)
            done = terminated or truncated
            
    env.close()
    
    # Save video
    if frames:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"✅ Video saved to: {video_path}")
    else:
        print("⚠️ No frames captured.")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--mode", choices=["grid", "random", "single"],
                        default="random", help="Search mode")
    parser.add_argument("--n-samples", type=int, default=15,
                        help="Number of random samples")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Max parallel training jobs (default: 2)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Show top N results")
    args = parser.parse_args()
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"tune_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🔧 HYPERPARAMETER TUNING for Parking Agent")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Log directory: {log_dir}")
    print(f"Tune steps: {TUNE_STEPS:,}")
    print(f"{'='*60}")
    
    # Run search
    if args.mode == "grid":
        results = grid_search(log_dir)
    elif args.mode == "random":
        results = random_search(
            log_dir,
            n_samples=args.n_samples,
            max_workers=args.max_workers,
        )
    else:
        print("Single mode not implemented yet")
        return
    
    # Print and save results
    print_results(results, top_n=args.top_n)
    save_results(results, log_dir / "tuning_results.json")

    # Generate video for the winner
    if results:
        best_result = results[0]  # sorted in print_results? 
        # Wait, print_results sorts them in place? Yes: results.sort(...)
        generate_best_video(log_dir, best_result)


if __name__ == "__main__":
    main()
