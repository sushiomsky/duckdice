from __future__ import annotations

def main() -> None:
    # Delegate to packaged GUI entry
    from .gui import main as _main
    _main()
