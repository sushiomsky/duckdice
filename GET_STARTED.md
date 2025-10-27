# Get Started with DuckDice Tools

## Overview

This repository contains two main tools:

1. **DuckDice CLI** - Command-line tool for DuckDice API
2. **RNG Analysis** - Machine learning analysis of the RNG

---

## Part 1: DuckDice CLI Tool

### What It Does
- Place bets via command line
- Check account stats
- Get user information
- Automate betting strategies

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Get user info
python duckdice.py --api-key YOUR_API_KEY user-info

# Place a bet
python duckdice.py --api-key YOUR_API_KEY dice \
  --symbol XLM --amount 0.1 --chance 50 --high --faucet
```

### Documentation
- Full guide: [README.md](README.md)
- Quick start: [QUICK_START.md](QUICK_START.md)

---

## Part 2: RNG Analysis Tool

### What It Does
Attempts to "attack" the DuckDice RNG using:
- Statistical analysis
- Machine learning (7 models)
- Deep learning (LSTM, GRU, etc.)
- Pattern detection
- Visualizations

### Quick Start

```bash
# Install dependencies
cd rng_analysis
pip install -r requirements_analysis.txt

# Run full analysis
python main_analysis.py

# Or quick analysis (faster)
python main_analysis.py --skip-dl --skip-viz
```

### Expected Results

✅ **The RNG will prove to be secure:**
- Statistical tests pass
- ML models show minimal improvement
- No exploitable patterns
- Cryptographic security intact

### Documentation
- Full guide: [rng_analysis/README.md](rng_analysis/README.md)
- Quick start: [rng_analysis/QUICK_START_ANALYSIS.md](rng_analysis/QUICK_START_ANALYSIS.md)

---

## ⚠️ IMPORTANT WARNINGS

### About the RNG Analysis

**This tool is EDUCATIONAL ONLY. It will NOT help you win.**

Why the analysis will fail:
1. **SHA-256 is cryptographically secure**
2. **Each bet is independent**
3. **Server seed rotates**
4. **Provably fair = verifiable but unpredictable**

Even if patterns appear:
- ❌ They're overfitting
- ❌ They won't work in real betting
- ❌ They're statistical noise
- ❌ The house edge ensures casino profit

### About Gambling

🎲 **Gamble Responsibly**
- Only bet what you can afford to lose
- Understand the house edge
- No system beats the math
- Seek help if needed: 1-800-522-4700

---

## Project Structure

```
duckdice-cli/
├── duckdice.py              # Main CLI tool
├── requirements.txt         # CLI dependencies
├── README.md               # Main documentation
├── QUICK_START.md          # CLI quick start
│
├── examples/               # Example scripts
│   ├── auto_bet.py
│   ├── balance_tracker.py
│   └── stats_monitor.sh
│
├── rng_analysis/           # RNG Analysis toolkit
│   ├── main_analysis.py    # Run all analyses
│   ├── pattern_analyzer.py # Statistical tests
│   ├── ml_predictor.py     # Machine learning
│   ├── deep_learning_predictor.py  # Neural networks
│   ├── visualizer.py       # Visualizations
│   ├── README.md          # Full documentation
│   └── QUICK_START_ANALYSIS.md  # Quick start
│
└── bet_history/            # Your CSV files go here
    ├── bets_1.csv
    └── ...
```

---

## Common Tasks

### 1. Check Your Balance

```bash
python duckdice.py --api-key YOUR_KEY user-info
```

### 2. Get Currency Stats

```bash
python duckdice.py --api-key YOUR_KEY stats --symbol BTC
```

### 3. Place Automatic Bets

```bash
# See examples/auto_bet.py
python examples/auto_bet.py
```

### 4. Analyze Your Bet History

```bash
cd rng_analysis
python main_analysis.py
```

### 5. Create Visualizations

```bash
cd rng_analysis
python visualizer.py
```

---

## What You'll Learn

### From the CLI Tool
- ✅ How to use DuckDice API
- ✅ Automation and scripting
- ✅ JSON data handling

### From the RNG Analysis
- ✅ How provably fair RNG works
- ✅ Statistical analysis techniques
- ✅ Machine learning for sequences
- ✅ Why cryptographic systems are secure
- ✅ Why gambling systems fail
- ✅ The mathematics of house edge

---

## Next Steps

### For CLI Usage
1. Get your API key from DuckDice.io
2. Read [QUICK_START.md](QUICK_START.md)
3. Try basic commands
4. Explore example scripts

### For RNG Analysis
1. Export your bet history CSVs
2. Place them in `bet_history/` folder
3. Read [rng_analysis/QUICK_START_ANALYSIS.md](rng_analysis/QUICK_START_ANALYSIS.md)
4. Run the analysis
5. Study the results

---

## FAQ

### Can I predict the RNG?
**No.** It uses SHA-256 which is cryptographically secure.

### Will the ML models help me win?
**No.** Any patterns are overfitting or noise.

### Is DuckDice fair?
**Yes.** Use verification links to check each bet.

### Should I bet based on analysis results?
**Absolutely not.** This is educational only.

### What's the best betting strategy?
**Don't play, or play for entertainment with strict limits.**

---

## Support

- **CLI Issues**: See [README.md](README.md) troubleshooting
- **Analysis Issues**: See [rng_analysis/README.md](rng_analysis/README.md)
- **DuckDice API**: https://duckdice.io/bot-api
- **Gambling Help**: 1-800-522-4700 (US)

---

## License

MIT License - See [LICENSE](LICENSE) file.

**Remember:**
- 🎓 Educational purposes only
- 🚫 Cannot exploit cryptographic RNG
- 🎲 Gamble responsibly
- ⚠️ House always wins long-term

---

**Good luck with your learning! 📚**
