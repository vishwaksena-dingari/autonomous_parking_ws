#!/usr/bin/env python3
"""
Unified Tuning Pipeline for Autonomous Parking Agent

This single script replaces:
  - analyze_tuning_trends.py
  - hyper_tune.py
  - ppo_tune.py

Features:
  1) Reward-weight tuning (Stage 1)
  2) PPO hyperparameter tuning (Stage 2)
  3) Learning-curve plotting

Modes:
  --mode reward   : tune reward weights only
  --mode ppo      : tune PPO hyperparams only (uses latest reward JSON)
  --mode full     : reward tuning THEN PPO tuning (end-to-end)
  --mode plot     : plot learning curves from SB3 logs

Canonical reward-results JSON (used by PPO tuning):

    ~/autonomous_parking_ws/logs/reward_tuning/latest_reward_results.json
"""

import re
import json
import argparse
import itertools
import subprocess
import threading # Added by user instruction
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


# Global lock for printing so parallel workers don't interleave output
PRINT_LOCK = threading.Lock()


# =============================================================================
# SECTION 1: LEARNING CURVE ANALYSIS (from analyze_tuning_trends.py)
# =============================================================================

def parse_log_history(log_path: Path) -> Dict[str, List[float]]:
    """Extract time-series data (timesteps, mean_reward) from an SB3 training log."""
    text = log_path.read_text()

    timesteps: List[float] = []
    rewards: List[float] = []

    # Blocks are separated by SB3's "----------------------------------------" lines
    blocks = text.split("------------------------------------------")

    for block in blocks:
        ts_match = re.search(r"\|\s*total_timesteps\s*\|\s*(\d+)", block)
        rew_match = re.search(r"\|\s*mean_reward\s*\|\s*(-?[\d\.e\+]+)", block)

        if ts_match and rew_match:
            timesteps.append(int(ts_match.group(1)))
            rewards.append(float(rew_match.group(1)))

    return {"timesteps": timesteps, "rewards": rewards}


def analyze_stability(rewards: List[float]) -> float:
    """Calculate a simple stability score (std-dev of rewards). Lower is better."""
    if len(rewards) < 2:
        return 9999.0
    return float(np.std(rewards))


def plot_learning_curves(log_dir: Path) -> None:
    """Plot learning curves for all *.log files inside log_dir."""
    log_files = sorted(list(log_dir.glob("*.log")))
    if not log_files:
        print(f"⚠️ No .log files found in {log_dir}")
        return

    print(f"\n📊 ANALYZING {len(log_files)} RUNS IN: {log_dir}\n")
    print(f"{'Run Name':<30} | {'Final Rew':<10} | {'Stability (StdDev)':<18} | {'Trend'}")
    print("-" * 90)

    plt.figure(figsize=(12, 8))
    for lf in log_files:
        data = parse_log_history(lf)
        if not data["rewards"]:
            continue

        timesteps = data["timesteps"]
        rewards = data["rewards"]
        stability = analyze_stability(rewards)
        final_rew = rewards[-1]
        improvement = final_rew - rewards[0]
        trend = "⬆️" if improvement > 0 else "⬇️"

        print(f"{lf.stem:<30} | {final_rew:<10.0f} | {stability:<18.0f} | {trend} ({improvement:+.0f})")

        plt.plot(timesteps, rewards, label=lf.stem, alpha=0.7)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title(f"Learning Curves: {log_dir}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = log_dir / "learning_curves.png"
    plt.savefig(output)
    print(f"\n📈 Learning curves saved to: {output}")
    print("💡 Look for runs with HIGH final reward and LOW stability (smooth learning).")


# =============================================================================
# SECTION 2: SHARED PARSING + SCORING (used by both reward + PPO tuning)
# =============================================================================

def parse_log_file(log_path: Path) -> Dict:
    """Parse training log for key metrics used in scoring."""
    if not log_path.exists():
        return {}

    text = log_path.read_text()

    metrics = {
        "success_rate": 0.0,
        "mean_reward": -999999.0,
        "mean_ep_length": 999.0,
        "path_completion": 0.0,
        "collision_rate": 0.0,
        "off_path_rate": 0.0,
        "mean_final_dist": 20.0,
    }

    # Success rate
    m = re.findall(r"\|\s*success_rate\s*\|\s*([0-9.]+)", text)
    if m:
        metrics["success_rate"] = float(m[-1])

    # Mean reward
    m = re.findall(r"mean_reward\s*[=|]\s*(-?[0-9.e+\-]+)", text)
    if m:
        try:
            metrics["mean_reward"] = float(m[-1])
        except ValueError:
            pass

    # Mean episode length
    m = re.findall(r"mean_ep_length\s*[=|]\s*([0-9.]+)", text)
    if m:
        metrics["mean_ep_length"] = float(m[-1])

    # Episode-level metrics (if logged)
    ep_metrics = re.findall(
        r"\[EP_METRICS\].*?path=([0-9.]+).*?dist=([0-9.]+).*?coll=([01]).*?offpath=([01])",
        text,
    )
    if ep_metrics:
        path_completions = [float(t[0]) for t in ep_metrics]
        final_dists = [float(t[1]) for t in ep_metrics]
        collisions = [int(t[2]) for t in ep_metrics]
        off_paths = [int(t[3]) for t in ep_metrics]

        n = len(ep_metrics)
        metrics["path_completion"] = sum(path_completions) / n
        metrics["mean_final_dist"] = sum(final_dists) / n
        metrics["collision_rate"] = sum(collisions) / n
        metrics["off_path_rate"] = sum(off_paths) / n

    # Fallback: waypoint-based completion ratio
    if ep_metrics == []:
        waypoint_matches = re.findall(r"✓ Waypoint (\d+)/(\d+)", text)
        if waypoint_matches:
            completions = [int(x) / int(y) for x, y in waypoint_matches[-50:]]
            if completions:
                metrics["path_completion"] = sum(completions) / len(completions)

    return metrics


def calculate_score(metrics: Dict) -> float:
    """
    Combined tuning score: higher is better.

    Prioritizes:
      1. Success rate & path completion
      2. Safety (collisions, off-path)
      3. Precision (final distance)
      4. Efficiency (episode length)
      5. Curve quality (stability + final reward)
    """
    success_rate    = metrics.get("success_rate", 0.0)
    path_completion = metrics.get("path_completion", 0.0)
    mean_ep_length  = metrics.get("mean_ep_length", 999.0)
    collision_rate  = metrics.get("collision_rate", 0.0)
    off_path_rate   = metrics.get("off_path_rate", 0.0)
    mean_final_dist = metrics.get("mean_final_dist", 20.0)
    mean_rew        = metrics.get("mean_reward", -999999.0)
    stability       = metrics.get("stability", 9999.0)   # std of rewards over time
    final_rew       = metrics.get("final_rew", mean_rew) # last reward point

    # 1) Core behavior score (same spirit as before)
    score = (
          100.0 * success_rate
        + 30.0  * path_completion
        -  0.1  * mean_ep_length
        - 40.0  * collision_rate
        - 25.0  * off_path_rate
        -  2.0  * mean_final_dist
    )

    # 2) Learning-curve quality (your trend logic)
    # Penalize very noisy runs
    if stability < 9999.0:
        score -= 0.5 * stability      # stronger = more penalty for noisy curves

    # Gentle nudge toward better final reward, but not dominating behavior metrics
    if -10000 < final_rew < 10000:
        score += 0.01 * final_rew

    # Tiny tie-breaker using overall mean reward
    if -10000 < mean_rew < 10000:
        score += 0.001 * mean_rew

    return float(score)


# =============================================================================
# SECTION 3: REWARD TUNING (Stage 1 – from hyper_tune.py)
# =============================================================================

# Reward-parameter grid (Round 2 narrowed search)
PARAM_GRID: Dict[str, List] = {
    "align_w": [50.0, 80.0, 100.0, 120.0, 150.0],
    "success_bonus": [50.0, 70.0, 100.0],
    "bay_entry_bonus": [50.0, 70.0, 90.0],
    "corridor_penalty": [0.05, 0.1, 0.2, 0.5],
    "vel_reward_w": [0.01, 0.02, 0.05],
    "ent_coef": [0.005, 0.01, 0.02, 0.05],
}

# Short training regime for tuning
TUNE_STEPS_REWARD = 80_000
TUNE_N_ENVS_REWARD = 2
TUNE_MAX_EP_STEPS = 800


@dataclass
class RewardTuningResult:
    """Results from a single reward-tuning run."""
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


def _reward_python_and_cwd() -> Tuple[Path, Path]:
    """Return (python_path, script_dir) for sb3_train_hierarchical."""
    python_path = Path.home() / "autonomous_parking_ws" / ".venv" / "bin" / "python"
    script_dir = Path.home() / "autonomous_parking_ws" / "src" / "autonomous_parking"
    return python_path, script_dir


def run_reward_training(config: Dict, run_name: str, log_dir: Path) -> Path:
    """Launch a reward-tuning training run."""
    log_path = log_dir / f"{run_name}.log"
    python_path, script_dir = _reward_python_and_cwd()

    cmd = [
        str(python_path), "-m", "autonomous_parking.sb3_train_hierarchical",
        "--total-steps", str(TUNE_STEPS_REWARD),
        "--n-envs", str(TUNE_N_ENVS_REWARD),
        "--max-episode-steps", str(TUNE_MAX_EP_STEPS),
        "--use-curriculum",
        "--run-name", run_name,
        "--quiet-env",
        "--record-video",
        "--video-freq", "25",
        "--align-w", str(config["align_w"]),
        "--success-bonus", str(config["success_bonus"]),
        "--bay-entry-bonus", str(config["bay_entry_bonus"]),
        "--corridor-penalty", str(config["corridor_penalty"]),
        "--vel-reward-w", str(config["vel_reward_w"]),
        "--ent-coef", str(config["ent_coef"]),
    ]

    with PRINT_LOCK:
        print(f"\n{'='*60}")
        print(f"Running REWARD tune: {run_name}")
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
        with PRINT_LOCK:
            print(f"⚠️  Reward run {run_name} failed with code {result.returncode}")

    return log_path


def grid_search_reward(log_dir: Path) -> List[RewardTuningResult]:
    """Full grid search over reward parameters (slow)."""
    results: List[RewardTuningResult] = []

    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    combinations = list(itertools.product(*values))

    print(f"\n🔍 REWARD GRID SEARCH: {len(combinations)} configurations")
    print(f"Parameters: {keys}\n")

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        run_name = (
            f"tune_{i:03d}_a{config['align_w']:.0f}_b{config['bay_entry_bonus']:.0f}_e{config['ent_coef']}"
        )

        log_path = run_reward_training(config, run_name, log_dir)
        metrics = parse_log_file(log_path)
        
        # Add trend-based info
        hist = parse_log_history(log_path)
        if hist["rewards"]:
            metrics["stability"] = analyze_stability(hist["rewards"])
            metrics["final_rew"] = hist["rewards"][-1]
        else:
            metrics["stability"] = 9999.0
            metrics["final_rew"] = metrics.get("mean_reward", -999999.0)
            
        score = calculate_score(metrics)

        r = RewardTuningResult(
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
        results.append(r)

        print(
            f"[{i+1}/{len(combinations)}] {run_name}: "
            f"score={score:.1f}, success={r.success_rate:.1%}, path={r.path_completion:.1%}"
        )

    return results


def random_search_reward(
    log_dir: Path,
    n_samples: int = 15,
    max_workers: int = 2,
) -> List[RewardTuningResult]:
    """Random search over reward parameter space (parallel)."""
    import random

    results: List[RewardTuningResult] = []

    print(f"\n🎲 REWARD RANDOM SEARCH: {n_samples} configs")
    print(f"   Max workers: {max_workers}")
    print(f"   Each worker uses {TUNE_N_ENVS_REWARD} envs")

    jobs = []
    for i in range(n_samples):
        config = {k: random.choice(v) for k, v in PARAM_GRID.items()}
        run_name = f"rand_{i:03d}_{datetime.now().strftime('%H%M%S')}"
        jobs.append((i, config, run_name))

    def worker(idx: int, cfg: Dict, name: str) -> RewardTuningResult:
        log_path = run_reward_training(cfg, name, log_dir)
        metrics = parse_log_file(log_path)
        
        # Add trend-based info
        hist = parse_log_history(log_path)
        if hist["rewards"]:
            metrics["stability"] = analyze_stability(hist["rewards"])
            metrics["final_rew"] = hist["rewards"][-1]
        else:
            metrics["stability"] = 9999.0
            metrics["final_rew"] = metrics.get("mean_reward", -999999.0)
            
        score = calculate_score(metrics)

        return RewardTuningResult(
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
            with PRINT_LOCK:
                print(
                    f"[{finished}/{total}] {r.run_name}: "
                    f"score={r.score:.1f}, success={r.success_rate:.1%}, path={r.path_completion:.1%}"
                )

    return results




def print_reward_results(results: List[RewardTuningResult], top_n: int = 10) -> None:
    """Print top reward-tuning configurations."""
    results.sort(key=lambda r: r.score, reverse=True)

    print(f"\n{'='*80}")
    print(f"🏆 TOP {min(top_n, len(results))} REWARD CONFIGURATIONS")
    print(f"{'='*80}")

    for i, r in enumerate(results[:top_n]):
        print(f"\n#{i+1}: {r.run_name}")
        print(f"   Score       : {r.score:.1f}")
        print(f"   Success     : {r.success_rate:.1%} | Path: {r.path_completion:.1%}")
        print(f"   Reward      : {r.mean_reward:.0f} | Dist: {r.mean_final_dist:.1f}m")
        print(f"   Collision   : {r.collision_rate:.1%} | Off-path: {r.off_path_rate:.1%}")
        print(f"   Config      : {r.config}")


def save_reward_results(results: List[RewardTuningResult], output_path: Path) -> None:
    """Save reward-tuning results to JSON."""
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"\n📊 Reward tuning results saved to: {output_path}")


def best_reward_config_from_results(results: List[RewardTuningResult]) -> Dict:
    """Return the config of the best-scoring reward run."""
    if not results:
        raise ValueError("No reward-tuning results to choose from.")
    best = max(results, key=lambda r: r.score)
    return best.config


def load_best_reward_config_from_json(json_path: Path) -> Dict:
    """Load best reward config from canonical JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Reward results JSON not found: {json_path}")

    data = json.loads(json_path.read_text())
    if not data:
        raise ValueError(f"No entries in reward results JSON: {json_path}")

    best = max(data, key=lambda d: d.get("score", -1e9))
    cfg = best.get("config", {})
    required = ["align_w", "success_bonus", "bay_entry_bonus", "corridor_penalty", "vel_reward_w", "ent_coef"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing '{k}' in best reward config from {json_path}")

    print(f"\n✅ Loaded BEST reward config from {json_path}: {cfg}")
    return cfg


# =============================================================================
# SECTION 4: PPO HYPERPARAMETER TUNING (Stage 2 – from ppo_tune.py)
# =============================================================================

PPO_PARAM_GRID: Dict[str, List] = {
    "learning_rate": [5e-5, 1e-4, 3e-4, 5e-4],
    "gamma": [0.97, 0.99, 0.995],
    "gae_lambda": [0.9, 0.95, 0.98],
    "n_steps": [512, 1024, 2048],
    "batch_size": [64, 128, 256],
    "clip_range": [0.1, 0.2, 0.3],
    "vf_coef": [0.3, 0.5, 0.7],
    "max_grad_norm": [0.5, 1.0],
    "n_epochs": [5, 10, 15],
}

TUNE_STEPS_PPO = 80_000
TUNE_N_ENVS_PPO = 2
TUNE_MAX_EP_STEPS_PPO = 800


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


def run_ppo_training(
    reward_cfg: Dict,
    ppo_cfg: Dict,
    run_name: str,
    log_dir: Path,
) -> Path:
    """Run a single PPO hyperparameter tuning run (reward fixed)."""
    log_path = log_dir / f"{run_name}.log"
    python_path, script_dir = _reward_python_and_cwd()

    cmd = [
        str(python_path), "-m", "autonomous_parking.sb3_train_hierarchical",
        "--total-steps", str(TUNE_STEPS_PPO),
        "--n-envs", str(TUNE_N_ENVS_PPO),
        "--max-episode-steps", str(TUNE_MAX_EP_STEPS_PPO),
        "--use-curriculum",
        "--run-name", run_name,
        "--quiet-env",
        # Fixed reward params
        "--align-w", str(reward_cfg["align_w"]),
        "--success-bonus", str(reward_cfg["success_bonus"]),
        "--bay-entry-bonus", str(reward_cfg["bay_entry_bonus"]),
        "--corridor-penalty", str(reward_cfg["corridor_penalty"]),
        "--vel-reward-w", str(reward_cfg["vel_reward_w"]),
        "--ent-coef", str(reward_cfg["ent_coef"]),
        # PPO params
        "--learning-rate", str(ppo_cfg["learning_rate"]),
        "--gamma", str(ppo_cfg["gamma"]),
        "--gae-lambda", str(ppo_cfg["gae_lambda"]),
        "--n-steps", str(ppo_cfg["n_steps"]),
        "--batch-size", str(ppo_cfg["batch_size"]),
        "--clip-range", str(ppo_cfg["clip_range"]),
        "--vf-coef", str(ppo_cfg["vf_coef"]),
        "--max-grad-norm", str(ppo_cfg["max_grad_norm"]),
        "--n-epochs", str(ppo_cfg["n_epochs"]),
    ]

    with PRINT_LOCK:
        print(f"\n{'='*60}")
        print(f"Running PPO tune: {run_name}")
        print(f"Reward config: {reward_cfg}")
        print(f"PPO config   : {ppo_cfg}")
        print(f"{'='*60}")

    with log_path.open("w") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(script_dir),
        )

    if result.returncode != 0:
        with PRINT_LOCK:
            print(f"⚠️  PPO run {run_name} failed with code {result.returncode}")

    return log_path


def random_search_ppo(
    reward_cfg: Dict,
    log_dir: Path,
    n_samples: int = 20,
    max_workers: int = 1,
) -> List[PPOTuningResult]:
    """Random search over PPO hyperparameters (reward fixed)."""
    import random

    results: List[PPOTuningResult] = []

    print(f"\n🎲 PPO RANDOM SEARCH: {n_samples} configs")
    print(f"   Max workers   : {max_workers}")
    print(f"   Reward config : {reward_cfg}")

    jobs = []
    for i in range(n_samples):
        ppo_cfg = {k: random.choice(v) for k, v in PPO_PARAM_GRID.items()}
        run_name = f"ppo_{i:03d}_{datetime.now().strftime('%H%M%S')}"
        jobs.append((i, ppo_cfg, run_name))

    def worker(idx: int, cfg: Dict, name: str) -> PPOTuningResult:
        log_path = run_ppo_training(reward_cfg, cfg, name, log_dir)
        
        # 1) Core episode metrics
        metrics = parse_log_file(log_path)
        
        # 2) Learning-curve metrics (stability + final reward)
        hist = parse_log_history(log_path)
        if hist["rewards"]:
            metrics["stability"] = analyze_stability(hist["rewards"])
            metrics["final_rew"] = hist["rewards"][-1]
        else:
            metrics["stability"] = 9999.0
            metrics["final_rew"] = metrics.get("mean_reward", -999999.0)
        
        # 3) Unified score
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
            with PRINT_LOCK:
                print(
                    f"[{finished}/{total}] {r.run_name}: "
                    f"score={r.score:.1f}, reward={r.mean_reward:.0f}"
                )

    return results


def print_ppo_results(results: List[PPOTuningResult], top_n: int = 10) -> None:
    """Print top PPO configs."""
    results.sort(key=lambda r: r.score, reverse=True)

    print(f"\n{'='*80}")
    print(f"🏆 TOP {min(top_n, len(results))} PPO CONFIGS")
    print(f"{'='*80}")

    for i, r in enumerate(results[:top_n]):
        print(f"\n#{i+1}: {r.run_name}")
        print(f"   Score    : {r.score:.1f}")
        print(f"   Reward   : {r.mean_reward:.0f} | Ep Len: {r.mean_ep_length:.0f}")
        print(f"   Config   : {r.config}")


def save_ppo_results(results: List[PPOTuningResult], output_path: Path) -> None:
    """Save PPO tuning results to JSON."""
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"\n📊 PPO tuning results saved to: {output_path}")


def best_ppo_config_from_results(results: List[PPOTuningResult]) -> Dict:
    """Return the config of the best-scoring PPO run."""
    if not results:
        raise ValueError("No PPO-tuning results to choose from.")
    best = max(results, key=lambda r: r.score)
    return best.config


def print_final_training_command(reward_cfg: Dict, ppo_cfg: Dict) -> None:
    """Print a ready-to-copy command for long training with best params."""
    print(f"\n{'='*80}")
    print("🚀 READY-MADE TRAINING COMMAND (1.5M Steps)")
    print(f"{'='*80}")

    python_path, script_dir = _reward_python_and_cwd()
    python_cmd = str(python_path)

    cmd_parts = [
        f"{python_cmd} -m autonomous_parking.sb3_train_hierarchical",
        "--total-steps 1500000",
        "--n-envs 4",
        "--max-episode-steps 1000",
        "--use-curriculum",
        "--run-name PROD_RUN_001",
        "--record-video",
        "--video-freq 50",
    ]

    # Reward params
    for k, v in reward_cfg.items():
        cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    # PPO params
    for k, v in ppo_cfg.items():
        cmd_parts.append(f"--{k.replace('_', '-')} {v}")

    print(" \\\n    ".join(cmd_parts))
    print(f"\n👉 Run this from: {script_dir.parent}")
    print(f"{'='*80}\n")


# =============================================================================
# SECTION 5: MAIN ENTRYPOINT – MODES
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified tuning pipeline for parking agent")
    parser.add_argument(
        "--mode",
        choices=["reward", "ppo", "full", "plot"],
        default="full",
        help="reward=Stage1, ppo=Stage2, full=reward+ppo, plot=learning curves",
    )
    parser.add_argument(
        "--search",
        choices=["random", "grid"],
        default="random",
        help="Search type (grid only used for reward, ignored by ppo which is always random)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=15,
        help="Number of random configs (reward or PPO)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Max parallel training jobs for random search (Note: total envs = workers * n_envs)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Show top N results when printing",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="(plot mode) directory containing .log files. If empty, will try to use the latest reward tuning log directory.",
    )

    args = parser.parse_args()

    # Canonical path for "latest reward results"
    canonical_reward_json = (
        Path.home()
        / "autonomous_parking_ws"
        / "logs"
        / "reward_tuning"
        / "latest_reward_results.json"
    )

    if args.mode == "plot":
        # Plot learning curves
        if args.log_dir:
            log_dir = Path(args.log_dir)
        else:
            # Try to infer latest reward_tune_* directory under ./logs
            root_logs = Path("logs")
            candidates = sorted(root_logs.glob("reward_tune_*"))
            if not candidates:
                print("⚠️ No --log-dir provided and no reward_tune_* directories found in ./logs")
                return
            log_dir = candidates[-1]
            print(f"ℹ️ No --log-dir given, using latest reward tuning dir: {log_dir}")
            
        plot_learning_curves(log_dir)
        return

    # Root logs directory for this script
    root_logs = Path("logs")
    root_logs.mkdir(exist_ok=True)

    if args.mode in ("reward", "full"):
        # Reward tuning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reward_log_dir = root_logs / f"reward_tune_{timestamp}"
        reward_log_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("🔧 STAGE 1: REWARD TUNING")
        print(f"{'='*60}")
        print(f"Mode       : {args.search}")
        print(f"Log dir    : {reward_log_dir}")
        print(f"Steps/run  : {TUNE_STEPS_REWARD:,}")
        print(f"{'='*60}")

        if args.search == "grid":
            reward_results = grid_search_reward(reward_log_dir)
        else:
            reward_results = random_search_reward(
                reward_log_dir,
                n_samples=args.n_samples,
                max_workers=args.max_workers,
            )

        print_reward_results(reward_results, top_n=args.top_n)

        # Save detailed results in run-specific folder
        run_json = reward_log_dir / "reward_tuning_results.json"
        save_reward_results(reward_results, run_json)

        # Also save/overwrite canonical JSON for PPO stage to find
        canon_data_path = canonical_reward_json
        save_reward_results(reward_results, canon_data_path)

        # Best reward config (in-memory)
        best_reward_cfg = best_reward_config_from_results(reward_results)
    else:
        best_reward_cfg = None

    if args.mode == "reward":
        # Only Stage 1
        print("\n✅ Completed reward tuning (Stage 1 only).")
        return

    # === PPO tuning (Stage 2) ===
    if args.mode == "ppo":
        # Load reward config from canonical JSON
        best_reward_cfg = load_best_reward_config_from_json(canonical_reward_json)

    # args.mode in ("ppo", "full") now has best_reward_cfg
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ppo_log_dir = root_logs / f"ppo_tune_{timestamp}"
    ppo_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("🔧 STAGE 2: PPO TUNING")
    print(f"{'='*60}")
    print(f"Reward cfg : {best_reward_cfg}")
    print(f"Log dir    : {ppo_log_dir}")
    print(f"Steps/run  : {TUNE_STEPS_PPO:,}")
    print(f"{'='*60}")

    # Only random search for PPO (grid would be huge)
    ppo_results = random_search_ppo(
        best_reward_cfg,
        ppo_log_dir,
        n_samples=args.n_samples,
        max_workers=max(1, args.max_workers),
    )

    print_ppo_results(ppo_results, top_n=args.top_n)
    save_ppo_results(ppo_results, ppo_log_dir / "ppo_tuning_results.json")

    # Get best PPO config
    best_ppo_cfg = best_ppo_config_from_results(ppo_results)

    # PRINT FINAL COMMAND
    print_final_training_command(best_reward_cfg, best_ppo_cfg)

    print("\n✅ Tuning pipeline complete.")


if __name__ == "__main__":
    main()
