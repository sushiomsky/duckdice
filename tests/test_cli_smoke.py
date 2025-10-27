from __future__ import annotations

from duckdice_cli.__main__ import create_parser
from betbot_strategies import list_strategies


def test_cli_parser_builds():
    p = create_parser()
    # basic help should exist
    assert p is not None


def test_strategies_exist_via_registry():
    # duckdice_cli imports betbot_engine which imports engine, which imports built-in strategies
    items = list_strategies()
    # We expect at least one built-in strategy to be registered
    assert isinstance(items, list)
    assert any(it.get("name") for it in items)
