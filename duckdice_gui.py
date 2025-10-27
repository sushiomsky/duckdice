#!/usr/bin/env python3
"""
DuckDice GUI
A simple graphical interface for the DuckDice command line tool using Tkinter.

Usage:
  python duckdice_gui.py

This GUI reuses the API client and formatting helpers defined in duckdice.py.
It provides tabs for:
  - Dice
  - Range Dice
  - Stats
  - User Info

Notes:
  - Network or API errors from the underlying client raise SystemExit in the
    same thread; the GUI traps these and shows a message instead of quitting.
  - No additional dependencies required beyond the project's requirements.
"""

import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List

# Use packaged modules (no legacy imports)
from duckdice_api.api import DuckDiceAPI, DuckDiceConfig
from betbot_engine.engine import run_auto_bet, EngineConfig
from betbot_strategies import list_strategies


def _fmt_bet_result(resp: dict) -> str:
    bet = (resp or {}).get("bet", {})
    user = (resp or {}).get("user", {})
    lines = []
    lines.append("=" * 60)
    lines.append("BET RESULT")
    lines.append("=" * 60)
    lines.append(f"Win: {'YES' if bet.get('result') else 'NO'}")
    lines.append(f"Profit: {bet.get('profit', '0')}")
    lines.append(f"Number: {bet.get('number', '')}")
    lines.append(f"Payout: {bet.get('payout', '')}")
    bal = user.get('balance')
    if bal is not None:
        lines.append(f"Balance: {bal}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _fmt_currency_stats(resp: dict, symbol: str) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"CURRENCY STATISTICS - {symbol}")
    lines.append("=" * 60)
    lines.append(f"Bets: {resp.get('bets', 'N/A')}")
    lines.append(f"Wins: {resp.get('wins', 'N/A')}")
    lines.append(f"Profit: {resp.get('profit', 'N/A')} {symbol}")
    lines.append(f"Volume: {resp.get('volume', 'N/A')} {symbol}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _fmt_user_info(resp: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("USER INFORMATION")
    lines.append("=" * 60)
    user = resp or {}
    balances = user.get('balances') or []
    if balances:
        lines.append("Balances:")
        for b in balances:
            cur = (b or {}).get('currency', '')
            main = (b or {}).get('main', '')
            faucet = (b or {}).get('faucet', '')
            lines.append(f"  {cur}: main={main} faucet={faucet}")
    wb = user.get('wageringBonus') or []
    if wb:
        lines.append("Wagering Bonuses:")
        for w in wb:
            lines.append(f"  {w.get('hash','')}: {w.get('name','')} status={w.get('status','')}")
    tles = user.get('tle') or []
    if tles:
        lines.append("TLE:")
        for t in tles:
            lines.append(f"  {t.get('hash','')}: {t.get('name','')} status={t.get('status','')}")
    lines.append("=" * 60)
    return "\n".join(lines)


class DuckDiceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DuckDice GUI")
        self.geometry("900x700")
        self.minsize(800, 600)

        self._busy = tk.BooleanVar(value=False)
        self._json_output = tk.BooleanVar(value=False)

        self._build_widgets()

    # ---------- UI construction ----------
    def _build_widgets(self) -> None:
        # Settings frame
        settings = ttk.LabelFrame(self, text="Settings")
        settings.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

        # Grid config
        for i in range(0, 8):
            settings.columnconfigure(i, weight=1)

        ttk.Label(settings, text="API Key:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.api_key_var = tk.StringVar()
        ttk.Entry(settings, textvariable=self.api_key_var, show="*").grid(row=0, column=1, columnspan=5, sticky="ew", padx=5, pady=5)

        ttk.Label(settings, text="Base URL:").grid(row=0, column=6, sticky="e", padx=5, pady=5)
        self.base_url_var = tk.StringVar(value="https://duckdice.io/api")
        ttk.Entry(settings, textvariable=self.base_url_var).grid(row=0, column=7, sticky="ew", padx=5, pady=5)

        ttk.Label(settings, text="Timeout (s):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.timeout_var = tk.StringVar(value="30")
        ttk.Entry(settings, textvariable=self.timeout_var, width=8).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Checkbutton(settings, text="JSON output", variable=self._json_output).grid(row=1, column=2, sticky="w", padx=5, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(settings, textvariable=self.status_var, foreground="#555").grid(row=1, column=7, sticky="e", padx=5, pady=5)

        # Tabs
        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self._dice_tab = self._build_dice_tab(notebook)
        self._range_tab = self._build_range_tab(notebook)
        self._stats_tab = self._build_stats_tab(notebook)
        self._user_tab = self._build_user_tab(notebook)
        self._auto_tab = self._build_auto_tab(notebook)

        notebook.add(self._dice_tab, text="Dice")
        notebook.add(self._range_tab, text="Range Dice")
        notebook.add(self._stats_tab, text="Stats")
        notebook.add(self._user_tab, text="User Info")
        notebook.add(self._auto_tab, text="Auto Bet")

        # Output
        output_frame = ttk.LabelFrame(self, text="Output")
        output_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        toolbar = ttk.Frame(output_frame)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Copy", command=self.copy_output).pack(side=tk.LEFT, padx=(5, 2), pady=5)
        ttk.Button(toolbar, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=2, pady=5)

        self.output = tk.Text(output_frame, wrap=tk.WORD, height=16)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output.configure(yscrollcommand=yscroll.set)

    def _build_dice_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)

        # Layout grid
        for i in range(0, 6):
            frame.columnconfigure(i, weight=1)

        # Row 0
        ttk.Label(frame, text="Symbol:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.dice_symbol = tk.StringVar()
        ttk.Entry(frame, textvariable=self.dice_symbol).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="Amount:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.dice_amount = tk.StringVar()
        ttk.Entry(frame, textvariable=self.dice_amount).grid(row=0, column=3, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="Chance (%):").grid(row=0, column=4, sticky="e", padx=5, pady=5)
        self.dice_chance = tk.StringVar()
        ttk.Entry(frame, textvariable=self.dice_chance).grid(row=0, column=5, sticky="ew", padx=5, pady=5)

        # Row 1 - High/Low
        self.dice_is_high = tk.BooleanVar(value=True)
        ttk.Radiobutton(frame, text="High", variable=self.dice_is_high, value=True).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(frame, text="Low", variable=self.dice_is_high, value=False).grid(row=1, column=2, sticky="w", padx=5, pady=5)

        self.dice_faucet = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Faucet", variable=self.dice_faucet).grid(row=1, column=3, sticky="w", padx=5, pady=5)

        # Row 2 - Optional hashes
        ttk.Label(frame, text="Wagering Bonus Hash:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.dice_bonus = tk.StringVar()
        ttk.Entry(frame, textvariable=self.dice_bonus).grid(row=2, column=1, columnspan=2, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="TLE Hash:").grid(row=2, column=3, sticky="e", padx=5, pady=5)
        self.dice_tle = tk.StringVar()
        ttk.Entry(frame, textvariable=self.dice_tle).grid(row=2, column=4, columnspan=2, sticky="ew", padx=5, pady=5)

        # Action
        self.dice_btn = ttk.Button(frame, text="Place Bet", command=self.on_dice_play)
        self.dice_btn.grid(row=3, column=5, sticky="e", padx=5, pady=10)

        return frame

    def _build_range_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)
        for i in range(0, 8):
            frame.columnconfigure(i, weight=1)

        ttk.Label(frame, text="Symbol:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.range_symbol = tk.StringVar()
        ttk.Entry(frame, textvariable=self.range_symbol).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="Amount:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.range_amount = tk.StringVar()
        ttk.Entry(frame, textvariable=self.range_amount).grid(row=0, column=3, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="Range Min:").grid(row=0, column=4, sticky="e", padx=5, pady=5)
        self.range_min = tk.StringVar()
        ttk.Entry(frame, textvariable=self.range_min).grid(row=0, column=5, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="Range Max:").grid(row=0, column=6, sticky="e", padx=5, pady=5)
        self.range_max = tk.StringVar()
        ttk.Entry(frame, textvariable=self.range_max).grid(row=0, column=7, sticky="ew", padx=5, pady=5)

        self.range_in = tk.BooleanVar(value=True)
        ttk.Radiobutton(frame, text="In", variable=self.range_in, value=True).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(frame, text="Out", variable=self.range_in, value=False).grid(row=1, column=2, sticky="w", padx=5, pady=5)

        self.range_faucet = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Faucet", variable=self.range_faucet).grid(row=1, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Wagering Bonus Hash:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.range_bonus = tk.StringVar()
        ttk.Entry(frame, textvariable=self.range_bonus).grid(row=2, column=1, columnspan=3, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="TLE Hash:").grid(row=2, column=4, sticky="e", padx=5, pady=5)
        self.range_tle = tk.StringVar()
        ttk.Entry(frame, textvariable=self.range_tle).grid(row=2, column=5, columnspan=3, sticky="ew", padx=5, pady=5)

        self.range_btn = ttk.Button(frame, text="Place Range Bet", command=self.on_range_play)
        self.range_btn.grid(row=3, column=7, sticky="e", padx=5, pady=10)

        return frame

    def _build_stats_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)

        for i in range(0, 4):
            frame.columnconfigure(i, weight=1)

        ttk.Label(frame, text="Symbol:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.stats_symbol = tk.StringVar()
        ttk.Entry(frame, textvariable=self.stats_symbol).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.stats_btn = ttk.Button(frame, text="Fetch Stats", command=self.on_stats)
        self.stats_btn.grid(row=0, column=3, sticky="e", padx=5, pady=5)

        return frame

    def _build_user_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)

        self.user_btn = ttk.Button(frame, text="Fetch User Info", command=self.on_user_info)
        self.user_btn.pack(side=tk.TOP, anchor="e", padx=5, pady=5)

        return frame

    # ---------- Helpers ----------
    def set_busy(self, busy: bool, status: str = "") -> None:
        self._busy.set(busy)
        if status:
            self.status_var.set(status)
        else:
            self.status_var.set("Working..." if busy else "Ready")

        state = tk.DISABLED if busy else tk.NORMAL
        # Disable all action buttons when busy
        for btn in (getattr(self, 'dice_btn', None), getattr(self, 'range_btn', None), getattr(self, 'stats_btn', None), getattr(self, 'user_btn', None)):
            if btn is not None:
                btn.configure(state=state)

    def append_output(self, text: str) -> None:
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def clear_output(self) -> None:
        self.output.delete("1.0", tk.END)

    def copy_output(self) -> None:
        try:
            content = self.output.get("1.0", tk.END)
            self.clipboard_clear()
            self.clipboard_append(content)
            messagebox.showinfo("Copied", "Output copied to clipboard")
        except Exception as e:
            messagebox.showerror("Copy Failed", str(e))

    def _build_api(self) -> Optional[DuckDiceAPI]:
        api_key = (self.api_key_var.get() or "").strip()
        base_url = (self.base_url_var.get() or "").strip() or "https://duckdice.io/api"
        timeout_str = (self.timeout_var.get() or "30").strip()

        if not api_key:
            messagebox.showwarning("Missing API Key", "Please enter your DuckDice API key in Settings.")
            return None
        try:
            timeout = int(timeout_str)
        except ValueError:
            messagebox.showwarning("Invalid Timeout", "Timeout must be an integer number of seconds.")
            return None

        config = DuckDiceConfig(api_key=api_key, base_url=base_url, timeout=timeout)
        return DuckDiceAPI(config)

    def _run_threaded(self, target, *args, success_message: Optional[str] = None):
        def worker():
            try:
                result = target(*args)
            except SystemExit as se:
                # The CLI code calls sys.exit on errors; show message instead
                self.after(0, lambda: self.append_output(f"Error: operation aborted (exit code {getattr(se, 'code', 1)})"))
                self.after(0, lambda: self.set_busy(False))
                return
            except Exception as e:
                self.after(0, lambda: self.append_output(f"Error: {e}"))
                self.after(0, lambda: self.set_busy(False))
                return

            def on_ok():
                if self._json_output.get():
                    try:
                        self.append_output(json.dumps(result, indent=2))
                    except Exception as e:
                        self.append_output(f"(JSON format error) {e}\n{result}")
                else:
                    # Formatter is decided by caller; here result may already be a string
                    if isinstance(result, str):
                        self.append_output(result)
                    else:
                        self.append_output(str(result))
                if success_message:
                    self.status_var.set(success_message)
                self.set_busy(False)

            self.after(0, on_ok)

        self.set_busy(True)
        threading.Thread(target=worker, daemon=True).start()

    # ---------- Actions ----------
    def on_dice_play(self) -> None:
        api = self._build_api()
        if not api:
            return

        symbol = (self.dice_symbol.get() or "").strip()
        amount = (self.dice_amount.get() or "").strip()
        chance = (self.dice_chance.get() or "").strip()
        is_high = bool(self.dice_is_high.get())
        faucet = bool(self.dice_faucet.get())
        bonus = (self.dice_bonus.get() or None) or None
        tle = (self.dice_tle.get() or None) or None

        if not symbol or not amount or not chance:
            messagebox.showwarning("Missing Fields", "Please enter Symbol, Amount, and Chance.")
            return

        def call_api():
            resp = api.play_dice(symbol=symbol, amount=amount, chance=chance, is_high=is_high, faucet=faucet, wagering_bonus_hash=bonus, tle_hash=tle)
            return json.dumps(resp, indent=2) if self._json_output.get() else _fmt_bet_result(resp)

        self._run_threaded(call_api, success_message="Dice bet completed")

    def on_range_play(self) -> None:
        api = self._build_api()
        if not api:
            return

        symbol = (self.range_symbol.get() or "").strip()
        amount = (self.range_amount.get() or "").strip()
        min_val = (self.range_min.get() or "").strip()
        max_val = (self.range_max.get() or "").strip()
        is_in = bool(self.range_in.get())
        faucet = bool(self.range_faucet.get())
        bonus = (self.range_bonus.get() or None) or None
        tle = (self.range_tle.get() or None) or None

        if not symbol or not amount or not min_val or not max_val:
            messagebox.showwarning("Missing Fields", "Please enter Symbol, Amount, Range Min and Range Max.")
            return
        try:
            min_int = int(min_val)
            max_int = int(max_val)
        except ValueError:
            messagebox.showwarning("Invalid Range", "Range Min and Max must be integers.")
            return

        def call_api():
            resp = api.play_range_dice(symbol=symbol, amount=amount, range_values=[min_int, max_int], is_in=is_in, faucet=faucet, wagering_bonus_hash=bonus, tle_hash=tle)
            return json.dumps(resp, indent=2) if self._json_output.get() else _fmt_bet_result(resp)

        self._run_threaded(call_api, success_message="Range bet completed")

    def on_stats(self) -> None:
        api = self._build_api()
        if not api:
            return

        symbol = (self.stats_symbol.get() or "").strip()
        if not symbol:
            messagebox.showwarning("Missing Symbol", "Please enter a currency Symbol (e.g., BTC, XLM, XRP).")
            return

        def call_api():
            resp = api.get_currency_stats(symbol)
            return json.dumps(resp, indent=2) if self._json_output.get() else _fmt_currency_stats(resp, symbol)

        self._run_threaded(call_api, success_message="Stats fetched")

    def on_user_info(self) -> None:
        api = self._build_api()
        if not api:
            return

        def call_api():
            resp = api.get_user_info()
            return json.dumps(resp, indent=2) if self._json_output.get() else _fmt_user_info(resp)

        self._run_threaded(call_api, success_message="User info fetched")

    def _build_auto_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)
        for i in range(0, 6):
            frame.columnconfigure(i, weight=1)

        # Row 0: Symbol and Strategy
        ttk.Label(frame, text="Symbol:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.auto_symbol = tk.StringVar()
        ttk.Entry(frame, textvariable=self.auto_symbol).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="Strategy:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.auto_strategy = tk.StringVar()
        names = [it.get('name') for it in (list_strategies() or [])]
        self.auto_strategy_cb = ttk.Combobox(frame, textvariable=self.auto_strategy, values=names, state="readonly")
        if names:
            self.auto_strategy_cb.current(0)
        self.auto_strategy_cb.grid(row=0, column=3, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Refresh", command=self._refresh_strategy_list).grid(row=0, column=5, sticky="e", padx=5, pady=5)

        # Row 1: Safety/delay
        ttk.Label(frame, text="Delay (ms):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.auto_delay = tk.StringVar(value="750")
        ttk.Entry(frame, textvariable=self.auto_delay, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Jitter (ms):").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.auto_jitter = tk.StringVar(value="500")
        ttk.Entry(frame, textvariable=self.auto_jitter, width=10).grid(row=1, column=3, sticky="w", padx=5, pady=5)

        self.auto_dry_run = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Dry-run", variable=self.auto_dry_run).grid(row=1, column=4, sticky="w", padx=5, pady=5)
        self.auto_faucet = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Faucet", variable=self.auto_faucet).grid(row=1, column=5, sticky="w", padx=5, pady=5)

        # Row 2: Risk controls
        ttk.Label(frame, text="Stop Loss (%):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.auto_stop_loss = tk.StringVar(value="-2.0")
        ttk.Entry(frame, textvariable=self.auto_stop_loss, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Take Profit (%):").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        self.auto_take_profit = tk.StringVar(value="2.0")
        ttk.Entry(frame, textvariable=self.auto_take_profit, width=10).grid(row=2, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Max Bet (amt):").grid(row=2, column=4, sticky="e", padx=5, pady=5)
        self.auto_max_bet = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_bet, width=14).grid(row=2, column=5, sticky="w", padx=5, pady=5)

        # Row 3: Limits
        ttk.Label(frame, text="Max Bets:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.auto_max_bets = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_bets, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Max Losses:").grid(row=3, column=2, sticky="e", padx=5, pady=5)
        self.auto_max_losses = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_losses, width=10).grid(row=3, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Max Duration (s):").grid(row=3, column=4, sticky="e", padx=5, pady=5)
        self.auto_max_duration = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_duration, width=12).grid(row=3, column=5, sticky="w", padx=5, pady=5)

        # Row 4: Params text
        ttk.Label(frame, text="Params (key=value per line or JSON)").grid(row=4, column=0, columnspan=6, sticky="w", padx=5, pady=(10, 2))
        self.auto_params_text = tk.Text(frame, height=6, wrap=tk.WORD)
        self.auto_params_text.grid(row=5, column=0, columnspan=6, sticky="nsew", padx=5, pady=5)
        frame.rowconfigure(5, weight=1)

        # Row 6: Actions
        actions = ttk.Frame(frame)
        actions.grid(row=6, column=0, columnspan=6, sticky="ew", padx=5, pady=8)
        actions.columnconfigure(0, weight=1)
        self.auto_start_btn = ttk.Button(actions, text="Start", command=self.on_auto_start)
        self.auto_start_btn.pack(side=tk.RIGHT, padx=5)
        self.auto_stop_btn = ttk.Button(actions, text="Stop", command=self.on_auto_stop, state=tk.DISABLED)
        self.auto_stop_btn.pack(side=tk.RIGHT, padx=5)

        # State for run control
        self._auto_stop_event = threading.Event()
        self._auto_thread = None

        return frame

    def _refresh_strategy_list(self) -> None:
        try:
            names = [it.get('name') for it in (list_strategies() or [])]
            self.auto_strategy_cb.configure(values=names)
            if names and not self.auto_strategy.get():
                self.auto_strategy_cb.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh strategies: {e}")

    def _set_auto_running(self, running: bool) -> None:
        self.auto_start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.auto_stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        if running:
            self.status_var.set("Auto bet running...")
        else:
            self.status_var.set("Ready")

    def _parse_params_text(self) -> dict:
        text = self.auto_params_text.get("1.0", tk.END).strip()
        if not text:
            return {}
        # Try JSON first
        try:
            if text.startswith("{"):
                return json.loads(text)
        except Exception:
            pass
        params = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            vl = v.strip()
            if vl.lower() in ("true", "false"):
                val = vl.lower() == "true"
            else:
                try:
                    if "." in vl:
                        val = float(vl)
                    else:
                        val = int(vl)
                except ValueError:
                    val = vl
            params[k.strip()] = val
        return params

    def on_auto_start(self) -> None:
        api = self._build_api()
        if not api:
            return
        symbol = (self.auto_symbol.get() or "").strip()
        if not symbol:
            messagebox.showwarning("Missing Symbol", "Please enter a currency Symbol (e.g., BTC, XLM, XRP).")
            return
        strategy = (self.auto_strategy.get() or "").strip()
        if not strategy:
            messagebox.showwarning("Missing Strategy", "Please select a strategy.")
            return
        # Engine config
        try:
            delay_ms = int((self.auto_delay.get() or "750").strip())
            jitter_ms = int((self.auto_jitter.get() or "500").strip())
            sl = float((self.auto_stop_loss.get() or "-2.0").strip()) / 100.0
            tp = float((self.auto_take_profit.get() or "2.0").strip()) / 100.0
        except ValueError:
            messagebox.showwarning("Invalid Numbers", "Please check delay/jitter/stop-loss/take-profit values.")
            return
        max_bet = (self.auto_max_bet.get() or "").strip() or None
        max_bets = (self.auto_max_bets.get() or "").strip()
        max_bets = int(max_bets) if max_bets else None
        max_losses = (self.auto_max_losses.get() or "").strip()
        max_losses = int(max_losses) if max_losses else None
        max_duration = (self.auto_max_duration.get() or "").strip()
        max_duration = int(max_duration) if max_duration else None
        params = self._parse_params_text()

        eng_conf = EngineConfig(
            symbol=symbol,
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            dry_run=bool(self.auto_dry_run.get()),
            faucet=bool(self.auto_faucet.get()),
            stop_loss=sl,
            take_profit=tp,
            max_bet=max_bet,
            max_bets=max_bets,
            max_losses=max_losses,
            max_duration_sec=max_duration,
        )

        # Stop event
        self._auto_stop_event = threading.Event()

        def printer(msg: str):
            self.after(0, lambda m=msg: self.append_output(m))

        def json_sink(rec: dict):
            if self._json_output.get():
                try:
                    pretty = json.dumps(rec, indent=2)
                    self.after(0, lambda p=pretty: self.append_output(p))
                except Exception:
                    self.after(0, lambda: self.append_output(str(rec)))
            else:
                # compact one-liner for bet events
                if rec.get('event') == 'bet':
                    betno = rec.get('bets_done')
                    win = rec.get('result', {}).get('win')
                    profit = rec.get('result', {}).get('profit')
                    bal = rec.get('balance')
                    self.after(0, lambda: self.append_output(f"bet#{betno} win={'Y' if win else 'N'} profit={profit} bal={bal}"))

        def stop_checker():
            return self._auto_stop_event.is_set()

        def worker():
            try:
                run_auto_bet(api, strategy, params, eng_conf, printer=printer, json_sink=json_sink, stop_checker=stop_checker)
            except SystemExit as se:
                self.after(0, lambda: self.append_output(f"AutoBet Error: exit {getattr(se, 'code', 1)}"))
            except Exception as e:
                self.after(0, lambda: self.append_output(f"AutoBet Error: {e}"))
            finally:
                self.after(0, lambda: self._set_auto_running(False))

        self._set_auto_running(True)
        t = threading.Thread(target=worker, daemon=True)
        self._auto_thread = t
        t.start()

    def on_auto_stop(self) -> None:
        if hasattr(self, '_auto_stop_event') and self._auto_stop_event:
            self._auto_stop_event.set()
        self._set_auto_running(False)



if __name__ == "__main__":
    app = DuckDiceGUI()
    app.mainloop()


    def _build_auto_tab(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent)
        for i in range(0, 6):
            frame.columnconfigure(i, weight=1)

        # Row 0: Symbol and Strategy
        ttk.Label(frame, text="Symbol:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.auto_symbol = tk.StringVar()
        ttk.Entry(frame, textvariable=self.auto_symbol).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(frame, text="Strategy:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.auto_strategy = tk.StringVar()
        names = [it.get('name') for it in (list_strategies() or [])]
        self.auto_strategy_cb = ttk.Combobox(frame, textvariable=self.auto_strategy, values=names, state="readonly")
        if names:
            self.auto_strategy_cb.current(0)
        self.auto_strategy_cb.grid(row=0, column=3, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Refresh", command=self._refresh_strategy_list).grid(row=0, column=5, sticky="e", padx=5, pady=5)

        # Row 1: Safety/delay
        ttk.Label(frame, text="Delay (ms):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.auto_delay = tk.StringVar(value="750")
        ttk.Entry(frame, textvariable=self.auto_delay, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Jitter (ms):").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.auto_jitter = tk.StringVar(value="500")
        ttk.Entry(frame, textvariable=self.auto_jitter, width=10).grid(row=1, column=3, sticky="w", padx=5, pady=5)

        self.auto_dry_run = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Dry-run", variable=self.auto_dry_run).grid(row=1, column=4, sticky="w", padx=5, pady=5)
        self.auto_faucet = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Faucet", variable=self.auto_faucet).grid(row=1, column=5, sticky="w", padx=5, pady=5)

        # Row 2: Risk controls
        ttk.Label(frame, text="Stop Loss (%):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.auto_stop_loss = tk.StringVar(value="-2.0")
        ttk.Entry(frame, textvariable=self.auto_stop_loss, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Take Profit (%):").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        self.auto_take_profit = tk.StringVar(value="2.0")
        ttk.Entry(frame, textvariable=self.auto_take_profit, width=10).grid(row=2, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Max Bet (amt):").grid(row=2, column=4, sticky="e", padx=5, pady=5)
        self.auto_max_bet = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_bet, width=14).grid(row=2, column=5, sticky="w", padx=5, pady=5)

        # Row 3: Limits
        ttk.Label(frame, text="Max Bets:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.auto_max_bets = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_bets, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Max Losses:").grid(row=3, column=2, sticky="e", padx=5, pady=5)
        self.auto_max_losses = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_losses, width=10).grid(row=3, column=3, sticky="w", padx=5, pady=5)

        ttk.Label(frame, text="Max Duration (s):").grid(row=3, column=4, sticky="e", padx=5, pady=5)
        self.auto_max_duration = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.auto_max_duration, width=12).grid(row=3, column=5, sticky="w", padx=5, pady=5)

        # Row 4: Params text
        ttk.Label(frame, text="Params (key=value per line or JSON)").grid(row=4, column=0, columnspan=6, sticky="w", padx=5, pady=(10, 2))
        self.auto_params_text = tk.Text(frame, height=6, wrap=tk.WORD)
        self.auto_params_text.grid(row=5, column=0, columnspan=6, sticky="nsew", padx=5, pady=5)
        frame.rowconfigure(5, weight=1)

        # Row 6: Actions
        actions = ttk.Frame(frame)
        actions.grid(row=6, column=0, columnspan=6, sticky="ew", padx=5, pady=8)
        actions.columnconfigure(0, weight=1)
        self.auto_start_btn = ttk.Button(actions, text="Start", command=self.on_auto_start)
        self.auto_start_btn.pack(side=tk.RIGHT, padx=5)
        self.auto_stop_btn = ttk.Button(actions, text="Stop", command=self.on_auto_stop, state=tk.DISABLED)
        self.auto_stop_btn.pack(side=tk.RIGHT, padx=5)

        # State for run control
        self._auto_stop_event = threading.Event()
        self._auto_thread = None

        return frame

    def _refresh_strategy_list(self) -> None:
        try:
            names = [it.get('name') for it in (list_strategies() or [])]
            self.auto_strategy_cb.configure(values=names)
            if names and not self.auto_strategy.get():
                self.auto_strategy_cb.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh strategies: {e}")

    def _set_auto_running(self, running: bool) -> None:
        self.auto_start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.auto_stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        if running:
            self.status_var.set("Auto bet running...")
        else:
            self.status_var.set("Ready")

    def _parse_params_text(self) -> dict:
        text = self.auto_params_text.get("1.0", tk.END).strip()
        if not text:
            return {}
        # Try JSON first
        try:
            if text.startswith("{"):
                return json.loads(text)
        except Exception:
            pass
        params = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            vl = v.strip()
            if vl.lower() in ("true", "false"):
                val = vl.lower() == "true"
            else:
                try:
                    if "." in vl:
                        val = float(vl)
                    else:
                        val = int(vl)
                except ValueError:
                    val = vl
            params[k.strip()] = val
        return params

    def on_auto_start(self) -> None:
        api = self._build_api()
        if not api:
            return
        symbol = (self.auto_symbol.get() or "").strip()
        if not symbol:
            messagebox.showwarning("Missing Symbol", "Please enter a currency Symbol (e.g., BTC, XLM, XRP).")
            return
        strategy = (self.auto_strategy.get() or "").strip()
        if not strategy:
            messagebox.showwarning("Missing Strategy", "Please select a strategy.")
            return
        # Engine config
        try:
            delay_ms = int((self.auto_delay.get() or "750").strip())
            jitter_ms = int((self.auto_jitter.get() or "500").strip())
            sl = float((self.auto_stop_loss.get() or "-2.0").strip()) / 100.0
            tp = float((self.auto_take_profit.get() or "2.0").strip()) / 100.0
        except ValueError:
            messagebox.showwarning("Invalid Numbers", "Please check delay/jitter/stop-loss/take-profit values.")
            return
        max_bet = (self.auto_max_bet.get() or "").strip() or None
        max_bets = (self.auto_max_bets.get() or "").strip()
        max_bets = int(max_bets) if max_bets else None
        max_losses = (self.auto_max_losses.get() or "").strip()
        max_losses = int(max_losses) if max_losses else None
        max_duration = (self.auto_max_duration.get() or "").strip()
        max_duration = int(max_duration) if max_duration else None
        params = self._parse_params_text()

        eng_conf = EngineConfig(
            symbol=symbol,
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            dry_run=bool(self.auto_dry_run.get()),
            faucet=bool(self.auto_faucet.get()),
            stop_loss=sl,
            take_profit=tp,
            max_bet=max_bet,
            max_bets=max_bets,
            max_losses=max_losses,
            max_duration_sec=max_duration,
        )

        # Stop event
        self._auto_stop_event = threading.Event()

        def printer(msg: str):
            self.after(0, lambda m=msg: self.append_output(m))

        def json_sink(rec: dict):
            if self._json_output.get():
                try:
                    pretty = json.dumps(rec, indent=2)
                    self.after(0, lambda p=pretty: self.append_output(p))
                except Exception:
                    self.after(0, lambda: self.append_output(str(rec)))
            else:
                # compact one-liner for bet events
                if rec.get('event') == 'bet':
                    betno = rec.get('bets_done')
                    win = rec.get('result', {}).get('win')
                    profit = rec.get('result', {}).get('profit')
                    bal = rec.get('balance')
                    self.after(0, lambda: self.append_output(f"bet#{betno} win={'Y' if win else 'N'} profit={profit} bal={bal}"))

        def stop_checker():
            return self._auto_stop_event.is_set()

        def worker():
            try:
                run_auto_bet(api, strategy, params, eng_conf, printer=printer, json_sink=json_sink, stop_checker=stop_checker)
            except SystemExit as se:
                self.after(0, lambda: self.append_output(f"AutoBet Error: exit {getattr(se, 'code', 1)}"))
            except Exception as e:
                self.after(0, lambda: self.append_output(f"AutoBet Error: {e}"))
            finally:
                self.after(0, lambda: self._set_auto_running(False))

        self._set_auto_running(True)
        t = threading.Thread(target=worker, daemon=True)
        self._auto_thread = t
        t.start()

    def on_auto_stop(self) -> None:
        if hasattr(self, '_auto_stop_event') and self._auto_stop_event:
            self._auto_stop_event.set()
        self._set_auto_running(False)



def main() -> None:
    app = DuckDiceGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
