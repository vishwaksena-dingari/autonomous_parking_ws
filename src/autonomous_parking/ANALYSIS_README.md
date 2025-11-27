# Training Analysis Tools - v17.3

## 📋 Quick Start

After training completes, run this **ONE COMMAND**:

```bash
cd ~/autonomous_parking_ws/src/autonomous_parking
./analyze_all.sh
```

This will:
1. ✅ Analyze training logs
2. ✅ Run 50-episode evaluation
3. ✅ Generate comparison report with v17.2
4. ✅ Give you recommendations

---

## 🔧 Individual Scripts

### 1. Training Log Analysis
```bash
python analyze_training.py hier_v17_3_fixed
```

**What it shows:**
- Training completion status
- Final metrics (reward, curriculum stage)
- Model file locations
- Video count
- Comparison with v17.2

### 2. Quick Evaluation
```bash
python quick_eval.py hier_v17_3_fixed 50
```

**What it shows:**
- Success rate (X/50)
- Average return
- Collision rate
- Comparison with v17.2 (0%)
- Recommendations for next steps

### 3. Complete Analysis (All-in-One)
```bash
./analyze_all.sh
```

Runs both scripts above + generates comparison table.

---

## 📊 What to Share With Me

After running `./analyze_all.sh`, paste this:

```bash
# Copy the entire output from analyze_all.sh
# OR just these key lines:

grep "SUCCESS RATE" <output>
grep "Mean Reward" <output>
grep "Curriculum Stage" <output>
```

I'll tell you:
- ✅ If v17.3 is good enough for Phase 3
- 🔧 What to fix if success rate is still low
- 📈 Expected improvement from additional tweaks

---

## 🎯 Success Criteria

| Success Rate | Verdict | Action |
|--------------|---------|--------|
| **≥ 30%** | ✅ Excellent! | Ship for Phase 3 |
| **15-30%** | ✅ Good! | Optional: one more tuning round |
| **1-15%** | 📈 Progress | Apply v17.4 fixes (boost rewards) |
| **0%** | ⚠️ Issue | Debug + apply multiple fixes |

---

## 📹 Video Check (Optional)

```bash
# List latest videos
ls -lt results/ppo_hierarchical/hier_v17_3_fixed/training_videos/ | head -10

# Watch a specific episode (replace XXXX with episode number)
open results/ppo_hierarchical/hier_v17_3_fixed/training_videos/training_epXXXX.mp4
```

**What to look for:**
- Does the car reach the last waypoint? (Should: YES)
- Does it attempt to enter the bay? (Should: YES in v17.3)
- Does it get stuck oscillating? (Should: LESS in v17.3)

---

## 🐛 Troubleshooting

**"No log file found"**
- Check if training finished: `ls logs/train_hier_v17_3_fixed_*.log`
- If empty, training didn't start or crashed

**"Model not found"**
- Check: `ls results/ppo_hierarchical/hier_v17_3_fixed/`
- Should see `best_model/` directory

**"Evaluation fails"**
- Make sure you're in the right directory:
  ```bash
  cd ~/autonomous_parking_ws/src/autonomous_parking
  ```

---

## 📝 For Your Report

After analysis, you'll have:

1. **Quantitative Results:**
   - v17.2 success rate: 0%
   - v17.3 success rate: X%
   - Improvement: +X%

2. **Qualitative Analysis:**
   - Root cause: Stuck penalty too harsh (-200)
   - Fix: Reduced to -50
   - Result: Agent explores final parking zone

3. **Evidence:**
   - Training logs
   - Evaluation results
   - Video recordings

---

**Ready to analyze!** 🚀
