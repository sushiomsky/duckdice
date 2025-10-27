# Changelog

All notable changes to the DuckDice CLI tool will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024

### Added
- Initial release of DuckDice CLI tool
- Complete implementation of DuckDice Bot API
- Support for Original Dice game with high/low betting
- Support for Range Dice game with in/out betting
- Currency statistics retrieval
- User information and account details
- Faucet mode support
- Wagering bonus hash support
- Time Limited Events (TLE) support
- Human-readable output format with emojis and formatting
- JSON output format for scripting
- Comprehensive error handling
- Command-line argument parsing
- Environment variable support via .env file
- Wrapper scripts for Unix/Linux/Mac (duckdice.sh)
- Wrapper scripts for Windows (duckdice.bat)
- Example script: auto_bet.py with betting strategies
- Example script: balance_tracker.py for balance monitoring
- Example script: stats_monitor.sh for real-time stats
- Unit tests for core functionality
- Complete documentation (README.md)
- Quick start guide (QUICK_START.md)
- Project structure documentation
- MIT License
- .gitignore for security

### Features
- Play Original Dice with customizable parameters
- Play Range Dice with range selection
- Get detailed currency statistics
- Get comprehensive user information
- Support for all DuckDice currencies
- Session management with requests library
- Timeout configuration
- Custom base URL support
- Multiple output formats

### Documentation
- Comprehensive README with examples
- Quick start guide for new users
- API reference documentation
- Example scripts with comments
- Troubleshooting guide
- Security best practices
- Project structure overview

### Examples
- Fixed betting strategy example
- Martingale betting strategy example
- Balance tracking with history
- Real-time statistics monitoring
- Bash scripting examples
- Python scripting examples

## [Unreleased]

### Planned Features
- Configuration file support (YAML/JSON)
- Logging to file with rotation
- More betting strategies (D'Alembert, Fibonacci, etc.)
- Historical bet analysis
- Statistics visualization (charts/graphs)
- Webhook notifications for wins/losses
- Multi-account support
- Batch betting mode
- Strategy backtesting
- Risk management tools
- Auto-stop loss limits
- Profit targets
- Session time limits

### Possible Enhancements
- Interactive mode (TUI)
- Database storage for bet history
- Export to CSV/Excel
- Email notifications
- Telegram bot integration
- API rate limiting awareness
- Retry logic for failed requests
- Caching for user info
- Performance optimizations

---

## Version History

- **1.0.0** - Initial release with full API support

## How to Update

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt
```

## Breaking Changes

None yet - initial release.

## Migration Guide

Not applicable for initial release.

---

For detailed changes and commits, see the [GitHub repository](https://github.com/yourusername/duckdice-cli).
