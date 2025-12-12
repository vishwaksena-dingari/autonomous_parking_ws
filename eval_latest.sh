#!/bin/bash
# Evaluates the pre-trained PROD_TUNED_2M model
echo "🔎 Evaluating Best Model: PROD_TUNED_2M"

python3 -m autonomous_parking.sb3_eval_hierarchical \
    --model-dir src/results/ppo_hierarchical/PROD_TUNED_2M \
    --lot lot_a \
    --episodes 5
