# Project Structure

Complete overview of the DuckDice CLI tool project structure.

## File Organization

```
duckdice-cli/
├── duckdice.py              # Main CLI application
├── requirements.txt         # Python dependencies
├── README.md               # Complete documentation
├── QUICK_START.md          # Quick start guide
├── PROJECT_STRUCTURE.md    # This file
├── LICENSE                 # MIT license
├── .gitignore             # Git ignore patterns
├── .env.example           # Environment variable template
├── test_basic.py          # Unit tests
│
├── duckdice.sh            # Unix/Linux/Mac wrapper script
├── duckdice.bat           # Windows wrapper script
│
└── examples/              # Example scripts and use cases
    ├── auto_bet.py        # Automated betting strategies
    ├── balance_tracker.py # Balance tracking utility
    └── stats_monitor.sh   # Real-time stats monitoring
```

## Core Files

### duckdice.py
**Main application file** - Contains all the logic for interacting with DuckDice API.

**Classes:**
- `GameType(Enum)` - Supported game types
- `DuckDiceConfig` - Configuration dataclass
- `DuckDiceAPI` - Main API client

**Functions:**
- `format_bet_result()` - Format bet results for display
- `format_currency_stats()` - Format currency statistics
- `format_user_info()` - Format user information
- `create_parser()` - Create argument parser
- `main()` - Main entry point

**Features:**
- ✅ Original Dice game
- ✅ Range Dice game
- ✅ Currency statistics
- ✅ User information
- ✅ Faucet mode support
- ✅ Wagering bonus support
- ✅ TLE (Time Limited Events) support
- ✅ JSON output format
- ✅ Human-readable output format
- ✅ Comprehensive error handling

### requirements.txt
Python dependencies:
- `requests>=2.31.0` - HTTP library for API calls

## Documentation Files

### README.md
Complete documentation including:
- Feature list
- Installation instructions
- Usage guide for all commands
- Examples for each feature
- API key management
- Error handling
- Advanced usage and scripting
- Troubleshooting guide

### QUICK_START.md
5-minute quick start guide:
- Installation steps
- API key setup
- First bet examples
- Command cheat sheet
- Safety tips

### PROJECT_STRUCTURE.md
This file - complete project overview.

## Configuration Files

### .env.example
Template for environment variables:
```bash
DUCKDICE_API_KEY=your-api-key-here
DUCKDICE_BASE_URL=https://duckdice.io/api  # optional
DUCKDICE_TIMEOUT=30                          # optional
```

Copy to `.env` and fill in your API key.

### .gitignore
Prevents committing:
- Environment files (`.env`, `*.key`)
- Python cache files
- Virtual environments
- IDE files
- Log files
- Balance history files

## Wrapper Scripts

### duckdice.sh (Unix/Linux/Mac)
Bash wrapper that:
- Loads environment variables from `.env`
- Passes API key automatically
- Simplifies command execution

Usage:
```bash
./duckdice.sh user-info
./duckdice.sh dice --symbol BTC --amount 0.1 --chance 50 --high
```

### duckdice.bat (Windows)
Windows batch wrapper with same functionality as shell script.

Usage:
```cmd
duckdice.bat user-info
duckdice.bat stats --symbol BTC
```

## Testing

### test_basic.py
Comprehensive unit tests covering:
- Configuration creation
- API client initialization
- API method calls (mocked)
- Output formatters
- Argument parser
- All commands

Run tests:
```bash
python test_basic.py
```

## Example Scripts

### examples/auto_bet.py
Automated betting with strategies:
- **Fixed Strategy** - Same amount each bet (safer)
- **Martingale Strategy** - Double after loss (risky)
- Session tracking and statistics
- Profit/loss calculation

Usage:
```bash
python examples/auto_bet.py
```

**⚠️ Warning:** Edit the script to set your API key and configure strategy before running.

### examples/balance_tracker.py
Balance tracking and monitoring:
- Display all currency balances
- Show wagered amounts
- Display active bonuses
- Show TLE participation
- Save historical snapshots

Usage:
```bash
python examples/balance_tracker.py YOUR_API_KEY
python examples/balance_tracker.py YOUR_API_KEY --save
python examples/balance_tracker.py YOUR_API_KEY --save my_balance.json
```

### examples/stats_monitor.sh
Real-time statistics dashboard:
- Monitor multiple currencies
- Auto-refresh display
- Color-coded output
- Win rate calculation
- Profit tracking

Requirements:
- `jq` - JSON processor
- `bc` - Calculator

Usage:
```bash
./examples/stats_monitor.sh
```

**Configuration:** Edit script to set currencies and refresh interval.

## API Implementation

### Implemented Endpoints

| Endpoint | Method | Command | Status |
|----------|--------|---------|--------|
| `/api/dice/play` | POST | `dice` | ✅ Complete |
| `/api/range-dice/play` | POST | `range-dice` | ✅ Complete |
| `/api/bot/stats/<symbol>` | GET | `stats` | ✅ Complete |
| `/api/bot/user-info` | GET | `user-info` | ✅ Complete |

### Request Parameters Supported

**Original Dice (`dice` command):**
- ✅ symbol - Currency symbol
- ✅ amount - Bet amount
- ✅ chance - Win chance percentage
- ✅ isHigh - High/Low betting
- ✅ faucet - Faucet mode
- ✅ userWageringBonusHash - Wagering bonus
- ✅ tleHash - Time Limited Event

**Range Dice (`range-dice` command):**
- ✅ symbol - Currency symbol
- ✅ amount - Bet amount
- ✅ range - Range boundaries [min, max]
- ✅ isIn - In/Out betting
- ✅ faucet - Faucet mode
- ✅ userWageringBonusHash - Wagering bonus
- ✅ tleHash - Time Limited Event

**Currency Stats (`stats` command):**
- ✅ symbol - Currency symbol

**User Info (`user-info` command):**
- No parameters required

## Output Formats

### Human-Readable Format
Default output with:
- Section headers with emojis
- Formatted tables and lists
- Color indicators (via emojis)
- Organized information hierarchy

### JSON Format
Raw API response with `--json` flag:
- Machine-readable
- Perfect for scripting
- Can be piped to `jq` or other tools
- Complete API response data

## Error Handling

The tool handles:
- ✅ HTTP errors (401, 404, 500, etc.)
- ✅ Network errors (timeout, connection)
- ✅ JSON parsing errors
- ✅ Invalid arguments
- ✅ Missing required parameters
- ✅ Keyboard interrupt (Ctrl+C)

Error messages include:
- Error type
- HTTP status code (if applicable)
- API response message
- Helpful suggestions

## Dependencies

### Required
- Python 3.7+
- `requests` library

### Optional (for examples)
- `jq` - JSON processor (for stats_monitor.sh)
- `bc` - Calculator (for stats_monitor.sh)

## Security

### Best Practices Implemented
- ✅ API key via environment variable
- ✅ `.env` file support (gitignored)
- ✅ No hardcoded credentials
- ✅ HTTPS only (API default)
- ✅ Secure session handling

### What's Protected
- API keys not committed to git
- Environment files ignored
- Secure HTTP headers
- Request timeout protection

## Supported Currencies

The tool supports all currencies available on DuckDice, including:
- BTC (Bitcoin)
- XLM (Stellar)
- XRP (Ripple)
- ETH (Ethereum)
- LTC (Litecoin)
- And many more...

Check DuckDice.io for the complete list.

## Future Enhancements

Possible additions:
- Configuration file support (YAML/JSON)
- Logging to file
- More betting strategies
- Historical analysis
- Statistics visualization
- Webhook notifications
- Multi-account support

## Development

### Adding New Features

1. **New API Endpoint:**
   - Add method to `DuckDiceAPI` class
   - Add subcommand to argument parser
   - Add formatter function if needed
   - Update command handler in `main()`
   - Add tests
   - Update documentation

2. **New Output Format:**
   - Add formatter function
   - Add flag to argument parser
   - Update command handlers
   - Update documentation

3. **New Example Script:**
   - Create in `examples/` directory
   - Add documentation to README
   - List in PROJECT_STRUCTURE
   - Add usage examples

### Code Style
- Python 3.7+ features
- Type hints where appropriate
- Docstrings for classes and functions
- Clear variable names
- Comments for complex logic

### Testing
Run tests before committing:
```bash
python test_basic.py
```

## License

MIT License - See LICENSE file for details.

## Support

- **DuckDice API Issues:** https://duckdice.io
- **Tool Issues:** GitHub Issues
- **API Documentation:** https://duckdice.io/bot-api

---

**Last Updated:** 2024
**Version:** 1.0.0
