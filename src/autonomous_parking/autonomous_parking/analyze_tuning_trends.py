#!/usr/bin/env python3
"""
Analyze Tuning Trends
Parses training logs to visualize learning curves and stability.
"""

import re
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List

def parse_log_history(log_path: Path) -> Dict[str, List[float]]:
    """Extract time-series data from log."""
    text = log_path.read_text()
    
    # Extract timesteps and rewards
    # Format: |    total_timesteps      | 24000        |
    #         |    mean_reward          | -4.25e+03    |
    
    timesteps = []
    rewards = []
    
    # Split by evaluation blocks
    blocks = text.split("------------------------------------------")
    
    for block in blocks:
        ts_match = re.search(r"\|\s*total_timesteps\s*\|\s*(\d+)", block)
        rew_match = re.search(r"\|\s*mean_reward\s*\|\s*(-?[\d\.e\+]+)", block)
        
        if ts_match and rew_match:
            timesteps.append(int(ts_match.group(1)))
            rewards.append(float(rew_match.group(1)))
            
    return {"timesteps": timesteps, "rewards": rewards}

def analyze_stability(rewards: List[float]) -> float:
    """Calculate stability score (lower std dev is better)."""
    if len(rewards) < 2:
        return 9999.0
    
    # Calculate variation relative to mean magnitude
    # We want consistent improvement, not wild swings
    return np.std(rewards)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True, help="Directory containing tuning logs")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    log_files = sorted(list(log_dir.glob("*.log")))
    
    print(f"\n📊 ANALYZING {len(log_files)} TUNING RUNS\n")
    print(f"{'Run Name':<25} | {'Final Rew':<10} | {'Stability (StdDev)':<18} | {'Trend'}")
    print("-" * 80)
    
    results = []
    
    for log_file in log_files:
        data = parse_log_history(log_file)
        if not data["rewards"]:
            continue
            
        final_rew = data["rewards"][-1]
        stability = analyze_stability(data["rewards"])
        
        # Simple trend: Final - Initial
        improvement = final_rew - data["rewards"][0]
        trend = "⬆️" if improvement > 0 else "⬇️"
        
        print(f"{log_file.stem:<25} | {final_rew:<10.0f} | {stability:<18.0f} | {trend} ({improvement:+.0f})")
        
        results.append({
            "name": log_file.stem,
            "data": data,
            "final_rew": final_rew
        })
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for res in results:
        plt.plot(res["data"]["timesteps"], res["data"]["rewards"], label=res["name"], alpha=0.7)
        
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Learning Curves: Hyperparameter Tuning")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_plot = log_dir / "learning_curves.png"
    plt.savefig(output_plot)
    print(f"\n📈 Learning curves saved to: {output_plot}")
    print(f"💡 Recommendation: Look for runs with HIGH final reward and LOW stability score (smooth learning).")

if __name__ == "__main__":
    main()
