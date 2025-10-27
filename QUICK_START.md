# Quick Start Guide

Get started with the DuckDice CLI in 5 minutes!

## 1. Installation (1 minute)

```bash
# Clone or download the repository
git clone <repository-url>
cd duckdice-cli

# Install dependencies
pip install -r requirements.txt

# Make scripts executable (Unix/Linux/Mac)
chmod +x duckdice.py duckdice.sh
```

## 2. Get Your API Key (2 minutes)

1. Go to [https://duckdice.io](https://duckdice.io)
2. Log in to your account
3. Navigate to **Account Settings** → **Bot API**
4. Generate or copy your API key

## 3. Configure (30 seconds)

**Option A: Environment Variable (Recommended)**
```bash
export DUCKDICE_API_KEY="your-api-key-here"
```

**Option B: Use .env file**
```bash
cp .env.example .env
# Edit .env and add your API key
nano .env
```

## 4. Test It! (30 seconds)

```bash
# Check your account info
python duckdice.py --api-key "$DUCKDICE_API_KEY" user-info

# Or if using .env file with wrapper script
./duckdice.sh user-info
```

## 5. Place Your First Bet (1 minute)

```bash
# Play original dice: 0.1 XLM, 50% chance, bet high, faucet mode
python duckdice.py --api-key "$DUCKDICE_API_KEY" dice \
  --symbol XLM --amount 0.1 --chance 50 --high --faucet

# Play range dice: 0.05 XRP, in range 1000-5000
python duckdice.py --api-key "$DUCKDICE_API_KEY" range-dice \
  --symbol XRP --amount 0.05 --range 1000 5000 --in --faucet
```

## Common Commands Cheat Sheet

```bash
# Get user info
python duckdice.py --api-key $DUCKDICE_API_KEY user-info

# Get currency stats
python duckdice.py --api-key $DUCKDICE_API_KEY stats --symbol BTC

# Play dice (high)
python duckdice.py --api-key $DUCKDICE_API_KEY dice \
  --symbol XLM --amount 0.1 --chance 77.77 --high --faucet

# Play dice (low)
python duckdice.py --api-key $DUCKDICE_API_KEY dice \
  --symbol XLM --amount 0.1 --chance 50 --low --faucet

# Play range dice (in range)
python duckdice.py --api-key $DUCKDICE_API_KEY range-dice \
  --symbol BTC --amount 0.01 --range 7777 7777 --in

# Play range dice (out of range)
python duckdice.py --api-key $DUCKDICE_API_KEY range-dice \
  --symbol XRP --amount 0.5 --range 0 9998 --out --faucet

# Get JSON output (for scripting)
python duckdice.py --api-key $DUCKDICE_API_KEY --json user-info
```

## Using the Wrapper Scripts

If you created a `.env` file:

**Linux/Mac:**
```bash
./duckdice.sh user-info
./duckdice.sh stats --symbol BTC
./duckdice.sh dice --symbol XLM --amount 0.1 --chance 50 --high --faucet
```

**Windows:**
```cmd
duckdice.bat user-info
duckdice.bat stats --symbol BTC
duckdice.bat dice --symbol XLM --amount 0.1 --chance 50 --high --faucet
```

## Example Scripts

Check out the `examples/` directory for more advanced usage:

- **auto_bet.py** - Automated betting with strategies
- **balance_tracker.py** - Track your balances over time
- **stats_monitor.sh** - Real-time stats monitoring

## Need Help?

- Read the full [README.md](README.md) for detailed documentation
- Check the [DuckDice API docs](https://duckdice.io/bot-api)
- Run `python duckdice.py --help` for all options

## Safety Tips

⚠️ **Important Reminders:**
- Start with faucet mode (`--faucet`) to test without risking real funds
- Never share your API key
- Add `.env` to `.gitignore` (already done)
- Gamble responsibly - only bet what you can afford to lose

---

**Ready to dice? Good luck! 🎲**
