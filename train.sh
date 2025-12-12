# Run in foreground (for Docker/Menu) BUT save logs to file using 'tee'
echo "🚀 Starting Full Production Training..."

python3 -m autonomous_parking.sb3_train_hierarchical \
    --total-steps 2000000 \
    --n-envs 4 \
    --max-episode-steps 1000 \
    --use-curriculum \
    --multi-lot \
    --run-name production_curriculum_final \
    \
    --record-video \
    --video-freq 25 \
    \
    --align-w 150.0 \
    --success-bonus 50.0 \
    --bay-entry-bonus 50.0 \
    --corridor-penalty 0.05 \
    --vel-reward-w 0.01 \
    --backward-penalty-weight 2.0 \
    --anti-freeze-penalty 0.01 \
    \
    --ent-coef 0.005 \
    --learning-rate 0.0005 \
    --gamma 0.97 \
    --gae-lambda 0.98 \
    --n-steps 512 \
    --batch-size 128 \
    --clip-range 0.3 \
    --vf-coef 0.3 \
    --max-grad-norm 1.0 \
    --n-epochs 15 \
    # 2>&1 | tee production_curriculum_final.log
