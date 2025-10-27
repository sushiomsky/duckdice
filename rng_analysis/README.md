## DuckDice RNG Analysis & Attack Attempt

A comprehensive analysis toolkit attempting to find patterns and exploit the DuckDice Random Number Generator using machine learning and deep learning techniques.

### ⚠️ IMPORTANT DISCLAIMER

**This tool is for EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- Cryptographic RNG systems are designed to be unpredictable
- Patterns found in historical data DO NOT predict future outcomes
- Attempting to exploit casino RNG will likely fail
- May violate terms of service
- **Gamble responsibly - only bet what you can afford to lose**

---

## What This Does

This analysis toolkit attempts to "attack" the DuckDice RNG using multiple approaches:

### 1. **Statistical Analysis**
- Distribution uniformity tests (Chi-square, Kolmogorov-Smirnov)
- Autocorrelation analysis
- Runs test for randomness
- Sequential pattern detection
- Fourier analysis for periodic patterns
- Seed correlation analysis
- Nonce pattern analysis

### 2. **Machine Learning**
- Random Forest Regression
- Gradient Boosting
- XGBoost
- LightGBM
- Neural Networks (MLP)
- Ridge/Lasso Regression
- Time series cross-validation
- Win/Loss classification

### 3. **Deep Learning**
- LSTM (Long Short-Term Memory) networks
- GRU (Gated Recurrent Unit) networks
- CNN-LSTM hybrid models
- Attention-based models
- Sequence-to-sequence prediction
- Overfitting analysis

### 4. **Visualization**
- Distribution plots
- Time series analysis
- Autocorrelation plots
- Pattern heatmaps
- Feature importance charts
- Prediction vs actual comparisons

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Navigate to rng_analysis directory
cd rng_analysis

# Install dependencies
pip install -r requirements_analysis.txt
```

Dependencies include:
- pandas, numpy
- scikit-learn
- tensorflow
- xgboost, lightgbm
- matplotlib, seaborn
- scipy

---

## Data Format

Place your bet history CSV files in the `bet_history/` directory.

**Required CSV columns:**
- `Date` - Timestamp of bet
- `Nonce` - Incremental counter
- `Number` - Result number (0-9999)
- `Result` - Win/Lose
- `Verification link` - Contains server seed and client seed

**Example:**
```csv
Date,Nonce,Number,Result,Verification link
2025-10-20 22:15:24 UTC,92256,2086,Lose,https://...?serverSeed=abc123&clientSeed=xyz789&nonce=92256
```

---

## Usage

### Quick Start

Run complete analysis:
```bash
python main_analysis.py
```

### Custom Options

```bash
# Specify data directory
python main_analysis.py --data-dir /path/to/bet_history

# Skip deep learning (faster)
python main_analysis.py --skip-dl

# Skip visualizations
python main_analysis.py --skip-viz

# Reduce deep learning epochs (faster)
python main_analysis.py --dl-epochs 10

# Full custom run
python main_analysis.py --data-dir ../bet_history --dl-epochs 50 --viz-dir my_viz
```

### Individual Analyses

Run components separately:

```bash
# Statistical analysis only
python pattern_analyzer.py

# Machine learning only
python ml_predictor.py

# Deep learning only
python deep_learning_predictor.py

# Visualizations only
python visualizer.py
```

---

## What to Expect

### If RNG is Secure (Expected)

✅ **Statistical tests PASS**
- Distribution appears uniform
- No significant autocorrelation
- Runs test passes
- No periodic patterns

✅ **ML models show <5% improvement**
- Cannot beat baseline predictions
- Low R² scores
- No feature importance
- Random performance

✅ **Deep learning fails**
- No better than random guessing
- High validation loss
- Overfitting without generalization

### If Patterns Are Found (Unlikely)

⚠️ **This does NOT mean the RNG is exploitable!**

Even if models show improvement:
1. **Overfitting** - Models memorize training data
2. **Spurious correlations** - Random noise looks like patterns
3. **Temporal artifacts** - Time-of-day effects that don't matter
4. **Sample bias** - Non-representative dataset

**Patterns in historical data DO NOT predict future outcomes.**

---

## Understanding the Results

### Distribution Analysis

**What it tests:** Whether numbers are uniformly distributed (0-9999)

**Good results:**
- Mean ≈ 5000
- KS test p-value > 0.05
- Chi-square test passes

**Bad results:**
- Skewed distribution
- Clustering in ranges
- P-values < 0.05

### Autocorrelation

**What it tests:** Whether current number predicts next number

**Good results:**
- All correlations near zero
- No significant lags

**Bad results:**
- Strong correlations at specific lags
- Periodic patterns

### Machine Learning

**What it tests:** Whether features predict outcomes

**Metrics:**
- **MAE** (Mean Absolute Error): Lower is better
- **R²**: Closer to 1 is better (0 = no predictive power)
- **Improvement**: % better than baseline

**Good results (secure RNG):**
- MAE close to baseline (~2887 for uniform)
- R² near 0
- <5% improvement

**Bad results:**
- R² > 0.1
- >10% improvement
- (But still likely not exploitable!)

### Deep Learning

**What it tests:** Whether sequences contain learnable patterns

**Good results (secure RNG):**
- Training loss ≈ validation loss
- No improvement over baseline
- Cannot predict next numbers

**Bad results:**
- Low training loss, high validation loss (overfitting)
- Some improvement (but likely not exploitable)

---

## Output Files

### Visualizations (if not skipped)

`visualizations/` directory contains:
- `distribution.png` - Number distribution analysis
- `time_series.png` - Numbers and win rates over time
- `autocorrelation.png` - Autocorrelation function
- `pattern_heatmap.png` - Transition probability matrix

### Console Output

Detailed analysis including:
- Statistical test results
- ML model performance
- Deep learning training logs
- Feature importance rankings
- Final comprehensive report

---

## Technical Background

### How DuckDice RNG Works

DuckDice uses a **Provably Fair** system:

```
Result = hash(ServerSeed + ClientSeed + Nonce) mod 10000
```

**Components:**
1. **Server Seed** - Secret, revealed after use
2. **Client Seed** - User-chosen or provided
3. **Nonce** - Incremental counter
4. **Hash Function** - SHA-256 (cryptographic)

**Properties:**
- **Unpredictable**: Cannot predict future results
- **Verifiable**: Can verify past results
- **Fair**: No party can manipulate outcomes

### Why It's Secure

**SHA-256 Properties:**
- **Avalanche Effect**: Small input change → completely different output
- **One-Way**: Cannot reverse engineer
- **Collision Resistant**: Infeasible to find collisions

**Security Principles:**
- Each bet is **cryptographically independent**
- Past outcomes provide **zero information** about future outcomes
- The server seed changes, making past patterns irrelevant
- Even with quantum computers, SHA-256 remains secure

### Why ML/DL Won't Work

1. **Cryptographic strength** - SHA-256 is designed to resist pattern analysis
2. **Independence** - Each outcome is independent
3. **Seed changes** - Server seed rotates, invalidating any learned patterns
4. **Overfitting** - Models memorize noise, not signal

---

## Common Misconceptions

### ❌ "I found a pattern!"

- Humans see patterns in randomness (pareidolia)
- Statistical noise looks like patterns
- Overfitting creates illusion of prediction

### ❌ "The model shows 10% improvement!"

- Improvement on training data ≠ predictive power
- Temporal features (time of day) don't affect cryptographic RNG
- Performance won't generalize to future bets

### ❌ "After 10 losses, a win is due!"

- **Gambler's Fallacy**
- Each bet is independent
- Past outcomes don't affect future probabilities

### ❌ "I'll use Martingale strategy!"

- Requires infinite bankroll
- Table limits prevent it
- House edge ensures long-term loss

---

## Responsible Gambling

### ⚠️ Important Reminders

1. **The house always has an edge**
   - 12% chance means 88% house edge on that specific bet
   - You WILL lose money over time

2. **No system beats the math**
   - All betting systems fail long-term
   - Mathematical expectation is negative

3. **Gamble responsibly**
   - Only bet entertainment money
   - Never chase losses
   - Set strict limits
   - Seek help if needed

### Help Resources

- **National Council on Problem Gambling**: 1-800-522-4700
- **Gamblers Anonymous**: https://www.gamblersanonymous.org/
- **UK GamCare**: https://www.gamcare.org.uk/

---

## FAQ

### Q: Can I actually predict the next number?

**A: No.** Cryptographic RNG is specifically designed to be unpredictable. Any perceived patterns are statistical noise.

### Q: What if my model shows good results?

**A: It's overfitting.** Good performance on training data doesn't translate to predictive power on future outcomes.

### Q: Should I bet based on these predictions?

**A: Absolutely not.** The predictions have no real predictive power and will lose you money.

### Q: Is DuckDice cheating?

**A: Unlikely.** Their provably fair system allows you to verify every bet. Use the verification links to check.

### Q: Can I verify the RNG is fair?

**A: Yes!** Use the verification link from each bet:
```
https://codepen.io/DuckDice/pen/abdNzQE?serverSeed=xxx&clientSeed=yyy&nonce=nnn
```

### Q: What's the best strategy?

**A: Don't play or play for entertainment only.** The house edge means you'll lose over time. If you play, set strict limits.

---

## Advanced Topics

### Modifying the Analysis

**Add new features:**
Edit `data_loader.py` → `preprocess_data()` method

**Add new ML models:**
Edit `ml_predictor.py` → `train_models()` method

**Add new DL architectures:**
Edit `deep_learning_predictor.py` → Add build method

**Add new visualizations:**
Edit `visualizer.py` → Add plot method

### Analyzing Different Games

The toolkit works with any game that provides:
- Sequential results
- Timestamps
- Win/Loss outcomes
- Verification data

Modify `data_loader.py` to parse different CSV formats.

### Performance Optimization

**Speed up analysis:**
```bash
# Skip deep learning
python main_analysis.py --skip-dl

# Reduce epochs
python main_analysis.py --dl-epochs 10

# Skip visualizations
python main_analysis.py --skip-viz
```

**For large datasets:**
- Use sample of data for initial exploration
- Increase batch size for deep learning
- Use fewer trees for Random Forest

---

## Limitations

### What This Cannot Do

❌ Predict future outcomes with certainty
❌ Break cryptographic hash functions
❌ Guarantee winning strategies
❌ Overcome house edge
❌ Replace responsible gambling

### Known Issues

1. **Large datasets** - May require significant RAM
2. **Deep learning** - Can be slow without GPU
3. **Visualization** - May fail with very large datasets
4. **Seed rotation** - When server seed changes, all patterns reset

---

## Contributing

Improvements welcome:
- Better statistical tests
- New ML architectures
- Performance optimizations
- Additional visualizations
- Documentation improvements

---

## License

MIT License - See main project LICENSE file

---

## Conclusion

This toolkit demonstrates that even with:
- ✅ Thousands of historical bets
- ✅ Advanced machine learning
- ✅ Deep neural networks
- ✅ Comprehensive statistical analysis

**Cryptographic RNG remains UNPREDICTABLE and SECURE.**

The house edge, combined with cryptographic security, ensures that:
- **No pattern analysis works**
- **No ML model works**
- **No betting system works**
- **The casino always wins long-term**

**Use this tool for education, not exploitation.**

---

**Remember: If it seems too good to be true, it is. Gamble responsibly. 🎲**
