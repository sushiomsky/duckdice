from __future__ import annotations
"""
DuckDice CLI entrypoint.

Usage after installation:
  duckdice --api-key KEY dice --symbol BTC --amount 0.1 --chance 77.77 --high
  duckdice --api-key KEY stats --symbol BTC
  duckdice --api-key KEY auto-bet --symbol XLM --list-strategies
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from duckdice_api.api import DuckDiceAPI, DuckDiceConfig
from betbot_engine import EngineConfig, run_auto_bet
from betbot_strategies import list_strategies, get_strategy


def _format_user_info(response: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("=" * 60)
    out.append("User Information")
    out.append("-" * 60)
    balances = response.get("balances", []) or []
    for b in balances:
        cur = (b or {}).get("currency", "?")
        main = (b or {}).get("main", "0")
        faucet = (b or {}).get("faucet", "0")
        bonus = (b or {}).get("wageringBonus", {}) or {}
        out.append(f"Currency: {cur}")
        out.append(f"  Main balance: {main}")
        out.append(f"  Faucet: {faucet}")
        if bonus:
            out.append(
                f"  WageringBonus: value={bonus.get('value','')} wagered={bonus.get('wagered','')} left={bonus.get('left','')}"
            )
    tles = response.get("tle", []) or []
    if tles:
        out.append("\nTime Limited Events:")
        for tle in tles:
            out.append(
                f"  Name: {tle.get('name','')} Status: {tle.get('status','')} Hash: {tle.get('hash','')}"
            )
    out.append("=" * 60)
    return "\n".join(out)


def _format_stats(response: Dict[str, Any]) -> str:
    return json.dumps(response, indent=2, ensure_ascii=False)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DuckDice API CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  duckdice --api-key KEY dice --symbol BTC --amount 0.1 --chance 77.77 --high
  duckdice --api-key KEY range-dice --symbol XRP --amount 0.01 --range 7777 7777 --in
  duckdice --api-key KEY stats --symbol BTC
  duckdice --api-key KEY user-info
  duckdice --api-key KEY auto-bet --symbol XLM --strategy kelly-capped --params chance=50 min_amount=0.000001
        """,
    )
    # Global
    p.add_argument("--api-key", required=True, help="DuckDice API key")
    p.add_argument("--base-url", default="https://duckdice.io/api")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--json", action="store_true", help="Output raw JSON where applicable")

    sub = p.add_subparsers(dest="command", required=True)

    # dice
    pd = sub.add_parser("dice", help="Play Original Dice")
    pd.add_argument("--symbol", required=True)
    pd.add_argument("--amount", required=True)
    pd.add_argument("--chance", required=True)
    g = pd.add_mutually_exclusive_group(required=True)
    g.add_argument("--high", action="store_true")
    g.add_argument("--low", action="store_true")
    pd.add_argument("--faucet", action="store_true")
    pd.add_argument("--wagering-bonus-hash")
    pd.add_argument("--tle-hash")

    # range-dice
    pr = sub.add_parser("range-dice", help="Play Range Dice")
    pr.add_argument("--symbol", required=True)
    pr.add_argument("--amount", required=True)
    pr.add_argument("--range", nargs=2, type=int, required=True, metavar=("MIN", "MAX"))
    g2 = pr.add_mutually_exclusive_group(required=True)
    g2.add_argument("--in", dest="in_range", action="store_true")
    g2.add_argument("--out", dest="out_range", action="store_true")
    pr.add_argument("--faucet", action="store_true")
    pr.add_argument("--wagering-bonus-hash")
    pr.add_argument("--tle-hash")

    # stats
    ps = sub.add_parser("stats", help="Currency stats")
    ps.add_argument("--symbol", required=True)

    # user-info
    sub.add_parser("user-info", help="User information")

    # auto-bet
    pa = sub.add_parser("auto-bet", help="Run auto-betting session")
    pa.add_argument("--symbol", required=True)
    pa.add_argument("--strategy", required=False)
    pa.add_argument("--list-strategies", action="store_true")
    pa.add_argument("--params", nargs="*", default=[], metavar="K=V")
    pa.add_argument("--params-json")
    pa.add_argument("--dry-run", action="store_true")
    pa.add_argument("--faucet", action="store_true")
    pa.add_argument("--delay-ms", type=int, default=750)
    pa.add_argument("--jitter-ms", type=int, default=500)
    pa.add_argument("--stop-loss", type=float, default=-0.02)
    pa.add_argument("--take-profit", type=float, default=0.02)
    pa.add_argument("--max-bet")
    pa.add_argument("--max-bets", type=int)
    pa.add_argument("--max-losses", type=int)
    pa.add_argument("--max-duration", type=int, dest="max_duration_sec")
    pa.add_argument("--seed", type=int)

    return p


def _coerce_value(v: str):
    s = (v or "").strip()
    # booleans
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    # integers
    try:
        if s.isdigit() or (s.startswith(("-", "+")) and s[1:].isdigit()):
            return int(s)
        # floats
        return float(s)
    except ValueError:
        return s


def _parse_kv_pairs(pairs: List[str]) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        d[k] = _coerce_value(v)
    return d


def _load_params_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    args = create_parser().parse_args(argv)

    api = DuckDiceAPI(DuckDiceConfig(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout))

    if args.command == "dice":
        res = api.play_dice(
            symbol=args.symbol,
            amount=args.amount,
            chance=args.chance,
            is_high=bool(args.high),
            faucet=bool(args.faucet),
            wagering_bonus_hash=getattr(args, "wagering_bonus_hash", None),
            tle_hash=getattr(args, "tle_hash", None),
        )
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.command == "range-dice":
        rng = [int(args.range[0]), int(args.range[1])]
        res = api.play_range_dice(
            symbol=args.symbol,
            amount=args.amount,
            range_values=rng,
            is_in=bool(args.in_range),
            faucet=bool(args.faucet),
            wagering_bonus_hash=getattr(args, "wagering_bonus_hash", None),
            tle_hash=getattr(args, "tle_hash", None),
        )
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.command == "stats":
        res = api.get_currency_stats(args.symbol)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(_format_stats(res))
        return 0

    if args.command == "user-info":
        res = api.get_user_info()
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(_format_user_info(res))
        return 0

    if args.command == "auto-bet":
        if args.list_strategies:
            items = list_strategies()
            if args.json:
                print(json.dumps(items, indent=2, ensure_ascii=False))
            else:
                print("Available strategies:")
                for it in items:
                    nm = it.get("name", "")
                    desc = it.get("description", "")
                    print(f"  - {nm}: {desc}")
            return 0

        if not args.strategy:
            print("--strategy is required unless --list-strategies is used", file=sys.stderr)
            return 2

        params: Dict[str, Any] = {}
        params.update(_parse_kv_pairs(args.params or []))
        if args.params_json:
            try:
                params.update(_load_params_json(args.params_json))
            except Exception as e:
                print(f"Failed to load params JSON: {e}", file=sys.stderr)
                return 2

        cfg = EngineConfig(
            symbol=args.symbol,
            delay_ms=args.delay_ms,
            jitter_ms=args.jitter_ms,
            dry_run=bool(args.dry_run),
            faucet=bool(args.faucet),
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            max_bet=args.max_bet,
            max_bets=args.max_bets,
            max_losses=args.max_losses,
            max_duration_sec=getattr(args, "max_duration_sec", None),
            seed=args.seed,
        )

        def printer(s: str) -> None:
            print(s)

        try:
            summary = run_auto_bet(api, args.strategy, params, cfg, printer=printer)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    print("Unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
