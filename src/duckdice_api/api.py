from __future__ import annotations
"""
DuckDice API client (requests-based)

Provides `DuckDiceConfig` and `DuckDiceAPI` with small, explicit methods used by
CLI and engine packages.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import sys
import requests


@dataclass
class DuckDiceConfig:
    api_key: str
    base_url: str = "https://duckdice.io/api"
    timeout: int = 30


class DuckDiceAPI:
    def __init__(self, config: DuckDiceConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "DuckDiceCLI/1.0.0",
                "Accept": "*/*",
                "Cache-Control": "no-cache",
            }
        )

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[Any, Any]:
        url = f"{self.config.base_url}/{endpoint}"
        params = {"api_key": self.config.api_key}
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=self.config.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params, json=data, timeout=self.config.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}", file=sys.stderr)
            if hasattr(e.response, "text"):
                print(f"Response: {e.response.text}", file=sys.stderr)
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}", file=sys.stderr)
            raise
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}", file=sys.stderr)
            raise

    def play_dice(
        self,
        symbol: str,
        amount: str,
        chance: str,
        is_high: bool,
        faucet: bool = False,
        wagering_bonus_hash: Optional[str] = None,
        tle_hash: Optional[str] = None,
    ) -> Dict[Any, Any]:
        data: Dict[str, Any] = {
            "symbol": symbol,
            "amount": amount,
            "chance": chance,
            "isHigh": is_high,
            "faucet": faucet,
        }
        if wagering_bonus_hash:
            data["userWageringBonusHash"] = wagering_bonus_hash
        if tle_hash:
            data["tleHash"] = tle_hash
        return self._make_request("POST", "dice/play", data)

    def play_range_dice(
        self,
        symbol: str,
        amount: str,
        range_values: List[int],
        is_in: bool,
        faucet: bool = False,
        wagering_bonus_hash: Optional[str] = None,
        tle_hash: Optional[str] = None,
    ) -> Dict[Any, Any]:
        data: Dict[str, Any] = {
            "symbol": symbol,
            "amount": amount,
            "range": range_values,
            "isIn": is_in,
            "faucet": faucet,
        }
        if wagering_bonus_hash:
            data["userWageringBonusHash"] = wagering_bonus_hash
        if tle_hash:
            data["tleHash"] = tle_hash
        return self._make_request("POST", "range-dice/play", data)

    def get_currency_stats(self, symbol: str) -> Dict[Any, Any]:
        return self._make_request("GET", f"bot/stats/{symbol}")

    def get_user_info(self) -> Dict[Any, Any]:
        return self._make_request("GET", "bot/user-info")
