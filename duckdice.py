"""
Compatibility shim module `duckdice` for tests and simple usage.

Exposes:
- DuckDiceConfig, DuckDiceAPI (from duckdice_api.api)
- create_parser (from duckdice_cli.__main__)
- format_bet_result, format_currency_stats, format_user_info (lightweight formatters)

The project uses a src/ layout; this shim ensures imports work when only the
project root is on sys.path (as in some test environments).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

# Ensure the src/ directory is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_PATH = os.path.join(_PROJECT_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Re-export API classes
try:
    from duckdice_api.api import DuckDiceAPI, DuckDiceConfig  # type: ignore
except Exception as e:  # pragma: no cover - very early import failure guard
    # Re-raise with clearer context
    raise ImportError("Failed to import duckdice_api.api; ensure dependencies are installed.") from e

# Try to import CLI helpers for parser and internal formatters
try:
    from duckdice_cli.__main__ import create_parser as _create_parser  # type: ignore
    from duckdice_cli.__main__ import _format_stats as _cli_format_stats  # type: ignore
    from duckdice_cli.__main__ import _format_user_info as _cli_format_user_info  # type: ignore
except Exception:  # Fallbacks defined below if CLI internals unavailable
    _create_parser = None  # type: ignore
    _cli_format_stats = None  # type: ignore
    _cli_format_user_info = None  # type: ignore

# Public API: create_parser

def create_parser():
    """Expose the CLI argument parser factory used by tests.

    Returns the argparse.ArgumentParser from duckdice_cli.__main__.
    """
    if _create_parser is None:
        raise ImportError("duckdice_cli is unavailable; cannot create parser")
    return _create_parser()


# Simple, human-friendly formatters expected by tests

def format_bet_result(response: Dict[str, Any]) -> str:
    """Render a concise textual summary of a bet result.

    The tests only assert presence of several key fields; include those.
    """
    bet = response.get("bet", {}) or {}
    user = response.get("user", {}) or {}
    is_win = bet.get("result", False)
    status = "WIN" if is_win else "LOSE"
    parts = [
        f"{status}",
        f"hash={bet.get('hash','')}",
        f"number={bet.get('number','')}",
        f"user={user.get('username','')}",
        f"symbol={bet.get('symbol','')}",
    ]
    return " | ".join(str(p) for p in parts)


def format_currency_stats(response: Dict[str, Any], symbol: str) -> str:
    """Render currency stats; include symbol and key numeric fields."""
    if _cli_format_stats is not None:
        # Reuse CLI JSON pretty-printer
        base = _cli_format_stats(response)
        # Ensure symbol tag appears for test expectations
        return f"Currency: {symbol}\n{base}"
    # Fallback lightweight formatting
    return (
        f"Currency: {symbol}\n"
        f"Bets: {response.get('bets','')}\n"
        f"Wins: {response.get('wins','')}\n"
        f"Profit: {response.get('profit','')}\n"
    )


def format_user_info(response: Dict[str, Any]) -> str:
    """Render user info in a readable manner.

    Try to reuse CLI helper if available, else provide a simple fallback that
    includes key identifiers used by tests.
    """
    if _cli_format_user_info is not None:
        return _cli_format_user_info(response)
    username = response.get("username", "")
    user_hash = response.get("hash", "")
    balances = response.get("balances", []) or []
    currencies = ", ".join(str((b or {}).get("currency", "")) for b in balances)
    return (
        "User Information\n"
        f"Username: {username}\n"
        f"Hash: {user_hash}\n"
        f"Currencies: {currencies}\n"
    )


__all__ = [
    "DuckDiceAPI",
    "DuckDiceConfig",
    "create_parser",
    "format_bet_result",
    "format_currency_stats",
    "format_user_info",
]
