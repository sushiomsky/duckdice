"""
Strategy plugin registry for auto-betting.

Usage:
  from betbot_strategies import register, get_strategy, list_strategies

A strategy module should import `register` and decorate a concrete class
implementing the `AutoBetStrategy` protocol in `base.py`.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Type

_registry: Dict[str, Type] = {}


def register(name: str) -> Callable[[Type], Type]:
    """Class decorator to register a strategy under a name.

    Example:
        @register("anti-martingale-streak")
        class AntiMartingaleStreak(AutoBetStrategy):
            ...
    """
    def deco(cls: Type) -> Type:
        key = name.strip().lower()
        if key in _registry:
            # Allow re-registration of the same class in hot-reload scenarios
            if _registry[key] is not cls:
                raise ValueError(f"Strategy name already registered: {name}")
        _registry[key] = cls
        setattr(cls, "_strategy_name", key)
        return cls

    return deco


def get_strategy(name: str) -> Type:
    key = (name or "").strip().lower()
    if key not in _registry:
        raise KeyError(f"Unknown strategy: {name}. Available: {', '.join(sorted(_registry))}")
    return _registry[key]


def list_strategies() -> List[Dict[str, str]]:
    """Return list of strategies with names and optional descriptions.
    Each class may provide `describe()` classmethod.
    """
    items: List[Dict[str, str]] = []
    for name, cls in sorted(_registry.items()):
        desc = ""
        if hasattr(cls, "describe") and callable(getattr(cls, "describe")):
            try:
                desc = cls.describe()  # type: ignore[attr-defined]
            except Exception:
                desc = ""
        items.append({"name": name, "description": desc or getattr(cls, "__doc__", "").strip()})
    return items
