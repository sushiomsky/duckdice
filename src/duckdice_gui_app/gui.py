from __future__ import annotations
"""
Console entry for the packaged GUI.
Delegates to the existing root-level GUI implementation in `duckdice_gui.py`.

This keeps the actual UI code in one place while providing an installable
entry point `duckdice-gui` via `pyproject.toml`.
"""


def main() -> None:
    # Import the existing GUI application module and run its main()
    # The root file `duckdice_gui.py` contains the full Tk app.
    from duckdice_gui import main as _main  # type: ignore

    _main()
