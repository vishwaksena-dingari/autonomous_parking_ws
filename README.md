# Autonomous Parking Project

## 🎯 Current Status: v14.0

**Latest Training Run**: `hierarchical_v14_0_parking_fixes`  
**Status**: In Progress  
**Key Improvements**: Enhanced parking success, road compliance, heading alignment

---

## 📁 Project Structure

```
autonomous_parking_ws/
├── src/autonomous_parking/          # Main source code
│   ├── env2d/parking_env.py        # Core environment (v14.0)
│   ├── sb3_train_hierarchical.py   # Training script
│   └── sb3_eval_hierarchical.py    # Evaluation script
├── config/                          # Parking lot configurations
│   └── bays.yaml                   # Bay positions for lot_a & lot_b
├── results/                         # Training outputs
│   └── ppo_hierarchical/           # Saved models
├── spawn_verification/              # Spawn system test images
├── verify_spawn_logic.py           # Spawn verification script
├── TRAINING_ANALYSIS_v13_2.md      # v13.2 analysis
├── V14_0_UPDATE_SUMMARY.md         # v14.0 changes
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Training
```bash
cd src/autonomous_parking
../../.venv/bin/python -m autonomous_parking.sb3_train_hierarchical \
  --total-steps 50000 \
  --run-name my_training_run \
  --n-envs 4
```

### Evaluation
```bash
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_a \
  --model-dir results/ppo_hierarchical/my_training_run
```

### Spawn Verification
```bash
python verify_spawn_logic.py
# Check spawn_verification/ folder for images
```

---

## 📊 Version History

### v14.0 (Current) - Parking Success Improvements
**Date**: 2025-11-20  
**Focus**: Fix parking success rate and road compliance

**Key Changes**:
- Relaxed Level 1 tolerances (0.5m→0.8m, 28°→46°)
- Enhanced road compliance (continuous reward + stronger penalty)
- Improved heading alignment (4x stronger when close)
- Added bay orientation alignment reward

**Expected Results**:
- Success rate: 0% → >50%
- Min distance: 5.34m → <1.5m
- Better road compliance

### v13.2 - Final Spawn System
**Date**: 2025-11-20  
**Focus**: Perfect spawn system with 100% in-bounds guarantee

**Achievements**:
- 100% spawn success rate (60/60 tests)
- Red front stripe for orientation
- Dynamic offset clamping
- Correct road coordinates

**Issues Found**:
- Agent drives away from goal after reaching waypoints
- Yaw alignment failure (~2.0 rad error)
- No parking successes in evaluation

### v13.0 - Auto-Curriculum
**Date**: 2025-11-18  
**Focus**: 3-level curriculum learning

**Features**:
- Level 1: Easy (close spawns, aligned)
- Level 2: Medium (farther, random yaw)
- Level 3: Hard (entrance spawns)

---

## 🔧 Key Features

### Environment (parking_env.py)
- **Spawn System**: Geometry-aware, 100% in-bounds
- **Curriculum**: 3-level progressive difficulty
- **Reward Function**: Exponential distance, alignment, road compliance
- **Sensors**: 64-ray lidar, position, velocity
- **Visualization**: Red front stripe, green goal bay

### Training (sb3_train_hierarchical.py)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Parallel Envs**: 4-8 simultaneous environments
- **Evaluation**: Every 2,500 steps
- **Checkpointing**: Best model auto-saved

---

## 📈 Performance Metrics

### Training Metrics
- **Mean Reward**: Target < -1,000 (lower is better)
- **Min Distance**: Target < 1.0m
- **Success Rate**: Target > 50%
- **Episode Length**: 300 steps max

### Success Criteria
- **Level 1**: Position < 0.8m, Yaw < 0.8 rad (46°)
- **Level 2**: Position < 0.5m, Yaw < 0.5 rad (28°)
- **Level 3**: Position < 0.3m, Yaw < 0.3 rad (17°)

---

## 🐛 Known Issues & Solutions

### Issue: Agent drives away from goal
**Status**: Fixed in v14.0  
**Solution**: Enhanced heading alignment, bay orientation reward

### Issue: Yaw alignment failure
**Status**: Fixed in v14.0  
**Solution**: 4x stronger heading weight when close, bay alignment reward

### Issue: Poor road compliance
**Status**: Fixed in v14.0  
**Solution**: Continuous on-road reward, 2x stronger off-road penalty

---

## 📝 Configuration

### Parking Lots
- **lot_a**: Simple horizontal road, 12 bays (A1-A6, B1-B6)
- **lot_b**: T-shaped road, 10 bays (H1-H5, V1-V5)

### Car Dimensions
- Length: 4.2m
- Width: 2.0m
- Wheelbase: 2.7m

### Bay Dimensions
- Width: 2.7m
- Length: 5.0m

---

## 🔍 Debugging

### Verify Spawn System
```bash
python verify_spawn_logic.py
# Generates 60 test images in spawn_verification/
```

### Check Training Progress
```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

### Evaluate Specific Model
```bash
../../.venv/bin/python -m autonomous_parking.sb3_eval_hierarchical \
  --lot lot_b \
  --model-dir results/ppo_hierarchical/hierarchical_v14_0_parking_fixes
```

---

## 📚 Documentation

- **TRAINING_ANALYSIS_v13_2.md**: Detailed analysis of v13.2 issues
- **V14_0_UPDATE_SUMMARY.md**: v14.0 changes and recommendations
- **archive/**: Old documentation and test files

---

## 🎓 Training Tips

1. **Start Small**: 50k steps for quick testing
2. **Monitor Metrics**: Watch min_distance and success_rate
3. **Check Spawns**: Verify spawn_verification/ images look correct
4. **Use TensorBoard**: Track reward trends over time
5. **Evaluate Often**: Test model every 10-20k steps

---

## 🤝 Contributing

When making changes:
1. Update version number (e.g., v14.1)
2. Document changes in this README
3. Create analysis document if major changes
4. Test with verify_spawn_logic.py
5. Train for at least 50k steps

---

## 📞 Support

For issues or questions:
1. Check TRAINING_ANALYSIS_v13_2.md for common problems
2. Verify spawn system with verify_spawn_logic.py
3. Review TensorBoard logs
4. Check evaluation results

---

**Last Updated**: 2025-11-20  
**Current Version**: v14.0  
**Status**: Training in progress
