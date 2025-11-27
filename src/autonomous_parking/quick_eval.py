#!/usr/bin/env python3
"""
Quick Evaluation Script for v17.3

Runs evaluation and generates a clean summary report.
Usage:
    python quick_eval.py hier_v17_3_fixed
"""

import sys
import subprocess
from pathlib import Path

def run_evaluation(run_name, num_episodes=50):
    """Run evaluation and parse results."""
    
    print("="*70)
    print(f"🎯 EVALUATING: {run_name}")
    print("="*70)
    print()
    
    model_dir = f"results/ppo_hierarchical/{run_name}/best_model"
    
    # Check if model exists
    model_path = Path(model_dir) / "best_model.zip"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nTrying final model instead...")
        
        # Try final model
        final_models = list(Path(f"results/ppo_hierarchical/{run_name}").glob("ppo_parking_final_*.zip"))
        if final_models:
            model_dir = f"results/ppo_hierarchical/{run_name}"
            print(f"✅ Using final model: {final_models[0].name}")
        else:
            print("❌ No models found!")
            return
    
    print(f"📂 Model directory: {model_dir}")
    print(f"📊 Episodes: {num_episodes}")
    print()
    print("Running evaluation (this may take a few minutes)...")
    print("-" * 70)
    print()
    
    # Run evaluation
    cmd = [
        "../../.venv/bin/python", "-m", "autonomous_parking.sb3_eval_hierarchical",
        "--lot", "lot_a",
        "--model-dir", model_dir,
        "--episodes", str(num_episodes)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Print full output
        print(output)
        
        # Parse summary
        print()
        print("="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        
        # Extract key metrics
        lines = output.split('\n')
        for line in lines:
            if "EVAL SUMMARY" in line:
                # Start of summary section
                summary_started = True
            elif "Episodes" in line or "Avg return" in line or "Successes" in line or "Collisions" in line:
                print(line)
        
        # Extract success rate
        for line in lines:
            if "Successes" in line:
                # Parse "Successes : X/Y"
                parts = line.split(':')
                if len(parts) == 2:
                    success_str = parts[1].strip()
                    if '/' in success_str:
                        success, total = success_str.split('/')
                        success_rate = int(success) / int(total) * 100
                        
                        print()
                        print(f"🎯 SUCCESS RATE: {success_rate:.1f}% ({success}/{total})")
                        print()
                        
                        # Comparison with v17.2
                        print("📈 COMPARISON WITH v17.2:")
                        print(f"   v17.2: 0% (0/20)")
                        print(f"   v17.3: {success_rate:.1f}% ({success}/{total})")
                        
                        if success_rate > 0:
                            print(f"   ✅ IMPROVEMENT: +{success_rate:.1f}%")
                        else:
                            print(f"   ⚠️  No improvement yet")
                        
                        print()
                        
                        # Recommendations
                        print("💡 RECOMMENDATIONS:")
                        if success_rate >= 30:
                            print("   ✅ EXCELLENT! Ready for Phase 3 submission")
                            print("   📝 Document this improvement in your report")
                        elif success_rate >= 15:
                            print("   ✅ GOOD! Significant improvement over v17.2")
                            print("   💡 Consider one more round of tuning for higher success")
                        elif success_rate > 0:
                            print("   📈 PROGRESS! Agent is learning to park")
                            print("   💡 Recommend: Boost final approach rewards (v17.4)")
                        else:
                            print("   ⚠️  No successes yet")
                            print("   💡 Recommend: Apply additional fixes (boost rewards, relax thresholds)")
        
        print()
        print("="*70)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed!")
        print(f"Error: {e.stderr}")
    except FileNotFoundError:
        print("❌ Python environment not found!")
        print("Make sure you're in the correct directory:")
        print("  cd ~/autonomous_parking_ws/src/autonomous_parking")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_eval.py <run_name> [num_episodes]")
        print("Example: python quick_eval.py hier_v17_3_fixed 50")
        sys.exit(1)
    
    run_name = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    run_evaluation(run_name, num_episodes)
