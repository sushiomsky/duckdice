# Quick Start: RNG Analysis

Get started analyzing the DuckDice RNG in 5 minutes!

## 1. Install Dependencies (2 minutes)

```bash
cd rng_analysis
pip install -r requirements_analysis.txt
```

## 2. Verify Your Data (30 seconds)

Make sure you have CSV files in `bet_history/`:

```bash
# Windows
dir ..\bet_history\*.csv

# Linux/Mac
ls -la ../bet_history/*.csv
```

**Files should contain columns:** Date, Nonce, Number, Result, Verification link

## 3. Run Analysis (2 minutes)

### Option A: Full Analysis (Recommended)

```bash
python main_analysis.py
```

This runs everything:
- ✅ Statistical tests
- ✅ Machine learning models
- ✅ Deep learning (LSTM, GRU, etc.)
- ✅ Visualizations

**Time:** ~5-10 minutes depending on data size

### Option B: Quick Analysis (Fast)

```bash
python main_analysis.py --skip-dl --skip-viz
```

This skips deep learning and visualizations.

**Time:** ~2-3 minutes

### Option C: Statistical Only (Fastest)

```bash
python pattern_analyzer.py
```

**Time:** ~30 seconds

## 4. Check Results

### Console Output

Look for these key sections:

**✅ PASS Results (Expected for secure RNG):**
```
Distribution Test: PASS ✅
Autocorrelation: PASS ✅
Runs Test: PASS ✅
```

**⚠️ Model Performance:**
```
Best ML Model: XGBoost
Improvement: 2.5% (expected: <5%)
```

**📊 Conclusion:**
```
✅ Models show minimal improvement
   The RNG appears cryptographically sound
```

### Visualizations

If generated (not skipped), check `visualizations/` folder:
- `distribution.png` - Should look uniform
- `autocorrelation.png` - Should be near zero
- `time_series.png` - Should look random
- `pattern_heatmap.png` - Should be uniform

## 5. Interpret Results

### What Good Results Look Like (Secure RNG)

✅ **Statistical Tests:**
- Distribution: PASS
- Autocorrelation: No significant correlations
- Runs Test: PASS

✅ **Machine Learning:**
- Improvement: < 5%
- R²: Close to 0
- MAE: ~2887 (baseline)

✅ **Deep Learning:**
- Cannot beat baseline
- Similar train/val loss
- No predictive power

### What Bad Results Look Like

⚠️ **But still NOT exploitable!**

Even if you see:
- Some test failures
- ML improvement > 10%
- Low R² but some correlation

This does NOT mean:
- ❌ You can predict future bets
- ❌ The RNG is broken
- ❌ You should bet based on predictions

It likely means:
- Overfitting to training data
- Random noise interpreted as patterns
- Temporal artifacts (time of day)
- None of these generalize!

## Common Questions

### Q: My model shows 15% improvement! Can I win?

**A: No.** This is overfitting. Performance on training data ≠ real predictive power.

### Q: I see patterns in the visualization!

**A: Normal.** Humans see patterns in randomness. These are statistical fluctuations.

### Q: Should I bet more after seeing these results?

**A: NEVER.** This analysis is educational. Do not use it for actual betting.

### Q: How do I verify DuckDice isn't cheating?

**A:** Use the verification links in your bet history:
```
https://codepen.io/DuckDice/pen/abdNzQE?serverSeed=xxx&clientSeed=yyy&nonce=nnn
```

This lets you cryptographically verify each bet was fair.

## Troubleshooting

### Error: "No CSV files found"

```bash
# Check your path
python main_analysis.py --data-dir ../bet_history
```

### Error: "ModuleNotFoundError"

```bash
# Install dependencies
pip install -r requirements_analysis.txt
```

### Out of Memory Error

```bash
# Skip deep learning (memory intensive)
python main_analysis.py --skip-dl
```

### Too Slow

```bash
# Reduce epochs
python main_analysis.py --dl-epochs 10

# Or skip DL entirely
python main_analysis.py --skip-dl
```

## Next Steps

1. **Read the full README.md** for detailed explanations
2. **Check visualizations** to understand patterns
3. **Experiment** with different parameters
4. **Learn** about cryptographic RNG
5. **DON'T** use results for actual betting!

## Output Example

```
====================================================================
                 COMPREHENSIVE RNG ANALYSIS REPORT
====================================================================

DATASET OVERVIEW:
  Total Bets: 10,000
  Win Rate: 12.05%
  Number Range: 0 - 9999
  Mean: 5001.23 (Expected: 5000)

STATISTICAL TESTS:
  Distribution Test: PASS ✅
  KS p-value: 0.234567
  Autocorrelation: PASS ✅
  Runs Test: PASS ✅

MACHINE LEARNING RESULTS:
  Best Model: XGBoost
  Improvement: 3.45%
  MAE: 2845.67

DEEP LEARNING RESULTS:
  Best Model: LSTM
  Improvement: 2.89%
  MAE: 2867.23

====================================================================
FINAL CONCLUSIONS:
====================================================================

✅ Models show minimal improvement (<5%)

This is the expected result for a secure RNG:
  - No exploitable patterns detected
  - Historical data provides no predictive power
  - The RNG appears cryptographically sound

====================================================================
RECOMMENDATIONS:
====================================================================

1. DO NOT attempt to exploit perceived patterns
   → They will not persist in real betting

2. DO verify your bets using verification links
   → This ensures the casino isn't cheating

3. DO gamble responsibly
   → Only bet what you can afford to lose

====================================================================
```

## Remember

🎲 **The house always wins in the long run**

Even if you find "patterns":
- They won't work in practice
- Cryptographic RNG is secure
- House edge guarantees casino profit
- Gamble for entertainment, not profit

**Educational purposes only. Gamble responsibly!**
