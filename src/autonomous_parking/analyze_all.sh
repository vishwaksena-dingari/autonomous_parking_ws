#!/bin/bash
# Complete Analysis Pipeline for v17.3
# Run this after training completes

echo "======================================================================"
echo "🔬 COMPLETE TRAINING ANALYSIS PIPELINE"
echo "======================================================================"
echo ""

RUN_NAME="hier_v17_3_fixed"

# Step 1: Analyze training logs
echo "Step 1/3: Analyzing training logs..."
echo "----------------------------------------------------------------------"
python analyze_training.py $RUN_NAME
echo ""
echo ""

# Step 2: Run evaluation
echo "Step 2/3: Running evaluation (50 episodes)..."
echo "----------------------------------------------------------------------"
python quick_eval.py $RUN_NAME 50
echo ""
echo ""

# Step 3: Generate comparison report
echo "Step 3/3: Generating comparison report..."
echo "----------------------------------------------------------------------"

# Extract key metrics
LOG_FILE=$(ls logs/train_${RUN_NAME}_*.log 2>/dev/null | head -1)

if [ -f "$LOG_FILE" ]; then
    echo "📊 QUICK METRICS COMPARISON"
    echo ""
    echo "Metric                  | v17.2      | v17.3      | Change"
    echo "------------------------|------------|------------|------------"
    
    # Mean reward
    V172_REWARD=$(grep "mean_reward" logs/train_hier_v17_2_full_*.log 2>/dev/null | tail -1 | awk '{print $4}')
    V173_REWARD=$(grep "mean_reward" "$LOG_FILE" | tail -1 | awk '{print $4}')
    echo "Mean Reward             | ${V172_REWARD:-N/A}     | ${V173_REWARD:-N/A}     | TBD"
    
    # Curriculum stage
    V172_STAGE=$(grep "stage_idx" logs/train_hier_v17_2_full_*.log 2>/dev/null | tail -1 | awk '{print $4}')
    V173_STAGE=$(grep "stage_idx" "$LOG_FILE" | tail -1 | awk '{print $4}')
    echo "Curriculum Stage        | S${V172_STAGE:-?}        | S${V173_STAGE:-?}        | TBD"
    
    # Success rate (from eval)
    echo "Success Rate (eval)     | 0%         | TBD        | TBD"
    
    echo ""
else
    echo "⚠️  Log file not found: $LOG_FILE"
fi

echo ""
echo "======================================================================"
echo "✅ ANALYSIS COMPLETE"
echo "======================================================================"
echo ""
echo "📋 NEXT ACTIONS:"
echo ""
echo "1. Review the evaluation results above"
echo "2. Check training videos:"
echo "   ls -lt results/ppo_hierarchical/${RUN_NAME}/training_videos/ | head -10"
echo ""
echo "3. View TensorBoard for detailed metrics:"
echo "   ../../.venv/bin/python -m tensorboard.main --logdir results/ppo_hierarchical/tb"
echo ""
echo "4. If success rate < 30%, consider v17.4 improvements"
echo "   (I can help you with this)"
echo ""
