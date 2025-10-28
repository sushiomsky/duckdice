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

    Always include username and user hash, and list balances by currency with
    main/faucet amounts. This satisfies tests and is human-friendly regardless
    of CLI helper availability.
    """
    username = response.get("username", "")
    user_hash = response.get("hash", "")
    level = response.get("level", "")
    created_at = response.get("createdAt")

    lines = []
    lines.append("=" * 60)
    lines.append("User Information")
    lines.append("-" * 60)
    if username:
        lines.append(f"Username: {username}")
    if user_hash:
        lines.append(f"Hash: {user_hash}")
    if level != "":
        lines.append(f"Level: {level}")
    if created_at:
        lines.append(f"CreatedAt: {created_at}")

    balances = response.get("balances", []) or []
    for b in balances:
        b = b or {}
        cur = b.get("currency", "?")
        main = b.get("main", "0")
        faucet = b.get("faucet", "0")
        lines.append(f"Currency: {cur}")
        lines.append(f"  Main balance: {main}")
        lines.append(f"  Faucet: {faucet}")

    # Show wagered per currency if available
    wagered = response.get("wagered", []) or []
    if wagered:
        lines.append("Wagered:")
        for w in wagered:
            w = w or {}
            lines.append(f"  {w.get('currency','')}: {w.get('amount','')}")

    # Time-limited events
    tles = response.get("tle", []) or []
    if tles:
        lines.append("Time Limited Events:")
        for tle in tles:
            tle = tle or {}
            lines.append(
                f"  Name: {tle.get('name','')} Status: {tle.get('status','')} Hash: {tle.get('hash','')}"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = [
    "DuckDiceAPI",
    "DuckDiceConfig",
    "create_parser",
    "format_bet_result",
    "format_currency_stats",
    "format_user_info",
]
