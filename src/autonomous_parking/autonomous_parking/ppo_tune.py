#!/usr/bin/env python3
"""
PPO Hyperparameter Tuning Script for Parking Agent
Stage 2 – assumes reward weights are already tuned (from hyper_tune.py)

Usage:
    python -m autonomous_parking.ppo_tune --mode random --n-samples 20
    python -m autonomous_parking.ppo_tune --mode random --n-samples 30 --max-workers 2
"""

import subprocess
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse parsing + scoring from reward tuner
from autonomous_parking.hyper_tune import (
    parse_log_file,
    calculate_score,
)

# =============================================================================
# BASE REWARD CONFIG – UPDATE AFTER ROUND 1 & 2 REWARD TUNING
# =============================================================================

BASE_REWARD_CONFIG = {
    # TODO: Fill with your best reward config after hyper_tune.py
    "align_w": 100.0,
    "success_bonus": 70.0,
    "bay_entry_bonus": 80.0,
    "corridor_penalty": 0.1,
    "vel_reward_w": 0.02,
    "ent_coef": 0.007,  # Can also be tuned in PPO grid if desired
}

# =============================================================================
# PPO PARAMETER GRID (Stage-2 search)
# =============================================================================

PPO_PARAM_GRID: Dict[str, List] = {
    # Learning rate
    "learning_rate": [5e-5, 1e-4, 3e-4, 5e-4],

    # Discount factor
    "gamma": [0.97, 0.99, 0.995],

    # GAE lambda
    "gae_lambda": [0.9, 0.95, 0.98],

    # Rollout length per env
    "n_steps": [512, 1024, 2048],

    # Minibatch size
    "batch_size": [64, 128, 256],

    # Clipping range
    "clip_range": [0.1, 0.2, 0.3],

    # Value function coefficient
    "vf_coef": [0.3, 0.5, 0.7],

    # Gradient clipping
    "max_grad_norm": [0.5, 1.0],

    # PPO epochs per update
    "n_epochs": [5, 10, 15],
}

# Short training settings
TUNE_STEPS = 80_000
TUNE_N_ENVS = 2
TUNE_MAX_EP_STEPS = 800


@dataclass
class PPOTuningResult:
    """Results from a single PPO tuning run."""
    config: Dict
    run_name: str
    success_rate: float
    mean_reward: float
    path_completion: float
    collision_rate: float
    off_path_rate: float
    mean_final_dist: float
    mean_ep_length: float
    score: float


def run_training(config: Dict, run_name: str, log_dir: Path) -> Path:
    """Run a single training with given PPO config (reward fixed)."""
    log_path = log_dir / f"{run_name}.log"

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

        # Fixed reward parameters (from best reward tuning)
        "--align-w", str(BASE_REWARD_CONFIG["align_w"]),
        "--success-bonus", str(BASE_REWARD_CONFIG["success_bonus"]),
        "--bay-entry-bonus", str(BASE_REWARD_CONFIG["bay_entry_bonus"]),
        "--corridor-penalty", str(BASE_REWARD_CONFIG["corridor_penalty"]),
        "--vel-reward-w", str(BASE_REWARD_CONFIG["vel_reward_w"]),
        "--ent-coef", str(BASE_REWARD_CONFIG["ent_coef"]),

        # PPO parameters (this run's config)
        "--learning-rate", str(config["learning_rate"]),
        "--gamma", str(config["gamma"]),
        "--gae-lambda", str(config["gae_lambda"]),
        "--n-steps", str(config["n_steps"]),
        "--batch-size", str(config["batch_size"]),
        "--clip-range", str(config["clip_range"]),
        "--vf-coef", str(config["vf_coef"]),
        "--max-grad-norm", str(config["max_grad_norm"]),
        "--n-epochs", str(config["n_epochs"]),
    ]

    print(f"\n{'='*60}")
    print(f"Running PPO tune: {run_name}")
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
        print(f"⚠️  PPO run {run_name} failed with code {result.returncode}")

    return log_path


def random_search(log_dir: Path, n_samples: int = 20, max_workers: int = 1) -> List[PPOTuningResult]:
    """Run random search over PPO hyperparameters."""
    results: List[PPOTuningResult] = []

    print(f"\n🎲 PPO RANDOM SEARCH: {n_samples} configs")
    print(f"   Max workers: {max_workers}")
    print(f"   Reward config (fixed): {BASE_REWARD_CONFIG}")

    # Pre-generate jobs
    jobs = []
    for i in range(n_samples):
        config = {k: random.choice(v) for k, v in PPO_PARAM_GRID.items()}
        run_name = f"ppo_{i:03d}_{datetime.now().strftime('%H%M%S')}"
        jobs.append((i, config, run_name))

    def worker(idx: int, cfg: Dict, name: str) -> PPOTuningResult:
        log_path = run_training(cfg, name, log_dir)
        metrics = parse_log_file(log_path)
        score = calculate_score(metrics)
        return PPOTuningResult(
            config=cfg,
            run_name=name,
            success_rate=metrics.get("success_rate", 0.0),
            mean_reward=metrics.get("mean_reward", -999999.0),
            path_completion=metrics.get("path_completion", 0.0),
            collision_rate=metrics.get("collision_rate", 0.0),
            off_path_rate=metrics.get("off_path_rate", 0.0),
            mean_final_dist=metrics.get("mean_final_dist", 20.0),
            mean_ep_length=metrics.get("mean_ep_length", 999.0),
            score=score,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(worker, idx, cfg, name): idx
            for idx, cfg, name in jobs
        }

        finished = 0
        total = len(jobs)
        for fut in as_completed(future_to_idx):
            r = fut.result()
            results.append(r)
            finished += 1
            print(f"[{finished}/{total}] {r.run_name}: "
                  f"score={r.score:.1f}, reward={r.mean_reward:.0f}")

    return results


def print_results(results: List[PPOTuningResult], top_n: int = 10) -> None:
    """Print sorted PPO tuning results."""
    results.sort(key=lambda r: r.score, reverse=True)

    print(f"\n{'='*80}")
    print(f"🏆 TOP {min(top_n, len(results))} PPO CONFIGS")
    print(f"{'='*80}")

    for i, r in enumerate(results[:top_n]):
        print(f"\n#{i+1}: {r.run_name}")
        print(f"   Score: {r.score:.1f}")
        print(f"   Reward: {r.mean_reward:.0f} | Ep Length: {r.mean_ep_length:.0f}")
        print(f"   Config: lr={r.config['learning_rate']}, gamma={r.config['gamma']}, "
              f"gae={r.config['gae_lambda']}, n_steps={r.config['n_steps']}")

    if results:
        best = results[0]
        print(f"\n{'='*80}")
        print(f"🚀 BEST PPO CONFIG – Full training command")
        print(f"{'='*80}")
        print(f"""
cd ~/autonomous_parking_ws/src/autonomous_parking && \\
nohup ../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \\
  --total-steps 1500000 \\
  --n-envs 6 \\
  --max-episode-steps 2000 \\
  --use-curriculum \\
  --run-name v40_PPO_TUNED \\
  --record-video \\
  --video-freq 25 \\
  --align-w {BASE_REWARD_CONFIG['align_w']} \\
  --success-bonus {BASE_REWARD_CONFIG['success_bonus']} \\
  --bay-entry-bonus {BASE_REWARD_CONFIG['bay_entry_bonus']} \\
  --corridor-penalty {BASE_REWARD_CONFIG['corridor_penalty']} \\
  --vel-reward-w {BASE_REWARD_CONFIG['vel_reward_w']} \\
  --ent-coef {BASE_REWARD_CONFIG['ent_coef']} \\
  --learning-rate {best.config['learning_rate']} \\
  --gamma {best.config['gamma']} \\
  --gae-lambda {best.config['gae_lambda']} \\
  --n-steps {best.config['n_steps']} \\
  --batch-size {best.config['batch_size']} \\
  --clip-range {best.config['clip_range']} \\
  --vf-coef {best.config['vf_coef']} \\
  --max-grad-norm {best.config['max_grad_norm']} \\
  --n-epochs {best.config['n_epochs']} \\
  > logs/v40_PPO_TUNED.log 2>&1 &
""")


def save_results(results: List[PPOTuningResult], output_path: Path) -> None:
    """Save PPO tuning results to JSON."""
    data = [
        {
            "run_name": r.run_name,
            "score": r.score,
            "success_rate": r.success_rate,
            "mean_reward": r.mean_reward,
            "path_completion": r.path_completion,
            "mean_ep_length": r.mean_ep_length,
            "config": r.config,
        }
        for r in results
    ]
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"\n📊 PPO tuning results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Hyperparameter Tuning")
    parser.add_argument("--mode", choices=["random"], default="random",
                        help="Search mode")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of random PPO configs")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max parallel training jobs")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Show top N results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"ppo_tune_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🔧 PPO HYPERPARAMETER TUNING for Parking Agent")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Log directory: {log_dir}")
    print(f"Tune steps: {TUNE_STEPS:,}")
    print(f"Fixed reward config: {BASE_REWARD_CONFIG}")
    print(f"{'='*60}")

    results = random_search(
        log_dir,
        n_samples=args.n_samples,
        max_workers=args.max_workers,
    )

    print_results(results, top_n=args.top_n)
    save_results(results, log_dir / "ppo_tuning_results.json")


if __name__ == "__main__":
    main()
