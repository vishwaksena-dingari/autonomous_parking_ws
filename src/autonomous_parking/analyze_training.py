#!/usr/bin/env python3
"""
Training Analysis Script for v17.3

Run this after training completes to get a comprehensive analysis.
Usage:
    python analyze_training.py hier_v17_3_fixed
"""

import sys
import os
import re
from pathlib import Path
import json

def analyze_training(run_name):
    """Analyze training results and generate report."""
    
    print("="*70)
    print(f"📊 TRAINING ANALYSIS: {run_name}")
    print("="*70)
    print()
    
    base_dir = Path("results/ppo_hierarchical") / run_name
    log_pattern = f"logs/train_{run_name}_*.log"
    
    # ========== 1. Check if training completed ==========
    print("1️⃣  TRAINING COMPLETION CHECK")
    print("-" * 70)
    
    log_files = list(Path("logs").glob(f"train_{run_name}_*.log"))
    if not log_files:
        print(f"❌ No log file found matching: {log_pattern}")
        return
    
    log_file = log_files[0]
    print(f"✅ Log file: {log_file}")
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Check completion
    if "Hierarchical training complete" in log_content:
        print("✅ Training completed successfully")
    else:
        print("⚠️  Training may have been interrupted")
    
    # Extract final timesteps
    timestep_matches = re.findall(r'total_timesteps\s+\|\s+(\d+)', log_content)
    if timestep_matches:
        final_steps = int(timestep_matches[-1])
        print(f"✅ Final timesteps: {final_steps:,}")
    else:
        print("❌ Could not find timestep information")
        final_steps = 0
    
    print()
    
    # ========== 2. Extract final metrics ==========
    print("2️⃣  FINAL TRAINING METRICS")
    print("-" * 70)
    
    # Find last eval block
    eval_pattern = r'Eval num_timesteps=(\d+), episode_reward=([-\d.]+) \+/- ([\d.]+)'
    eval_matches = re.findall(eval_pattern, log_content)
    
    if eval_matches:
        last_eval = eval_matches[-1]
        eval_steps, mean_reward, std_reward = last_eval
        print(f"Evaluation at step: {int(eval_steps):,}")
        print(f"Mean reward: {float(mean_reward):.2f} ± {float(std_reward):.2f}")
    else:
        print("⚠️  No evaluation data found")
        mean_reward = "N/A"
    
    # Extract curriculum stage
    stage_matches = re.findall(r'stage_idx\s+\|\s+(\d+)', log_content)
    if stage_matches:
        final_stage = int(stage_matches[-1])
        print(f"Final curriculum stage: S{final_stage} / 15")
    else:
        print("⚠️  No curriculum stage data found")
        final_stage = 0
    
    # Extract success rate from log
    success_matches = re.findall(r'success_rate\s+\|\s+([\d.]+)', log_content)
    if success_matches:
        final_success = float(success_matches[-1])
        print(f"Training success rate: {final_success*100:.1f}%")
    else:
        print("⚠️  No success rate data found")
    
    print()
    
    # ========== 3. Check saved models ==========
    print("3️⃣  SAVED MODELS")
    print("-" * 70)
    
    best_model = base_dir / "best_model" / "best_model.zip"
    final_model_pattern = base_dir / "ppo_parking_final_*.zip"
    
    if best_model.exists():
        size_mb = best_model.stat().st_size / (1024*1024)
        print(f"✅ Best model: {best_model} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Best model not found: {best_model}")
    
    final_models = list(base_dir.glob("ppo_parking_final_*.zip"))
    if final_models:
        for model in final_models:
            size_mb = model.stat().st_size / (1024*1024)
            print(f"✅ Final model: {model.name} ({size_mb:.1f} MB)")
    else:
        print("❌ No final model found")
    
    print()
    
    # ========== 4. Video analysis ==========
    print("4️⃣  TRAINING VIDEOS")
    print("-" * 70)
    
    video_dir = base_dir / "training_videos"
    if video_dir.exists():
        videos = sorted(video_dir.glob("*.mp4"))
        print(f"✅ Total videos: {len(videos)}")
        if videos:
            print(f"   First: {videos[0].name}")
            print(f"   Last:  {videos[-1].name}")
            
            # Extract episode numbers
            ep_nums = []
            for v in videos:
                match = re.search(r'ep(\d+)', v.name)
                if match:
                    ep_nums.append(int(match.group(1)))
            
            if ep_nums:
                print(f"   Episode range: {min(ep_nums)} - {max(ep_nums)}")
    else:
        print(f"❌ Video directory not found: {video_dir}")
    
    print()
    
    # ========== 5. Comparison with v17.2 ==========
    print("5️⃣  COMPARISON WITH v17.2")
    print("-" * 70)
    
    v172_log = list(Path("logs").glob("train_hier_v17_2_full_*.log"))
    if v172_log:
        with open(v172_log[0], 'r') as f:
            v172_content = f.read()
        
        # Extract v17.2 metrics
        v172_eval = re.findall(eval_pattern, v172_content)
        if v172_eval:
            _, v172_reward, _ = v172_eval[-1]
            v172_reward = float(v172_reward)
            
            if mean_reward != "N/A":
                current_reward = float(mean_reward)
                improvement = current_reward - v172_reward
                print(f"v17.2 mean reward: {v172_reward:.2f}")
                print(f"v17.3 mean reward: {current_reward:.2f}")
                print(f"Improvement: {improvement:+.2f} ({improvement/abs(v172_reward)*100:+.1f}%)")
            else:
                print(f"v17.2 mean reward: {v172_reward:.2f}")
                print("v17.3 mean reward: N/A")
        
        v172_stage = re.findall(r'stage_idx\s+\|\s+(\d+)', v172_content)
        if v172_stage:
            v172_final_stage = int(v172_stage[-1])
            print(f"v17.2 final stage: S{v172_final_stage}")
            print(f"v17.3 final stage: S{final_stage}")
            if final_stage > v172_final_stage:
                print(f"✅ Advanced {final_stage - v172_final_stage} more stages!")
    else:
        print("⚠️  v17.2 log not found for comparison")
    
    print()
    
    # ========== 6. Generate summary report ==========
    print("6️⃣  SUMMARY & RECOMMENDATIONS")
    print("-" * 70)
    
    summary = {
        "completed": "Hierarchical training complete" in log_content,
        "final_steps": final_steps,
        "mean_reward": mean_reward,
        "final_stage": final_stage,
        "best_model_exists": best_model.exists(),
    }
    
    # Recommendations
    if summary["completed"] and summary["best_model_exists"]:
        print("✅ Training completed successfully!")
        print()
        print("📋 NEXT STEPS:")
        print()
        print("1. Run evaluation on best model:")
        print(f"   ../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \\")
        print(f"     --lot lot_a \\")
        print(f"     --model-dir results/ppo_hierarchical/{run_name}/best_model \\")
        print(f"     --episodes 50")
        print()
        print("2. Check latest training videos:")
        print(f"   ls -lt results/ppo_hierarchical/{run_name}/training_videos/ | head -10")
        print()
        print("3. View TensorBoard:")
        print(f"   ../../.venv/bin/python -m tensorboard.main --logdir results/ppo_hierarchical/tb")
        print()
        
        if mean_reward != "N/A":
            reward_val = float(mean_reward)
            if reward_val > 0:
                print("🎉 POSITIVE REWARD! Agent is learning!")
            elif reward_val > -50:
                print("📈 Reward improved from v17.2 (-72.2)")
            else:
                print("⚠️  Reward still negative. May need further tuning.")
        
        if final_stage >= 12:
            print("🎯 Reached advanced curriculum stages (S12+)")
        elif final_stage > 10:
            print("📈 Advanced past v17.2 (S10)")
        else:
            print("⚠️  Did not advance past v17.2 curriculum stage")
    else:
        print("❌ Training incomplete or models missing")
        print("   Check the log file for errors:")
        print(f"   tail -100 {log_file}")
    
    print()
    print("="*70)
    print("Analysis complete!")
    print("="*70)
    
    # Save summary to JSON
    summary_file = base_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_training.py <run_name>")
        print("Example: python analyze_training.py hier_v17_3_fixed")
        sys.exit(1)
    
    run_name = sys.argv[1]
    analyze_training(run_name)
