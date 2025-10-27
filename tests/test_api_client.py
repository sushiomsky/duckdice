import json
from types import SimpleNamespace

import duckdice_api as dd


class DummyResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"ok": True}
        self.text = json.dumps(self._payload)

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class DummySession:
    def __init__(self):
        self.headers = {}
        self.last = None

    def get(self, url, params=None, timeout=None):
        self.last = ("GET", url, params, timeout)
        return DummyResponse(200, {"ok": True, "url": url, "params": params})

    def post(self, url, params=None, json=None, timeout=None):
        self.last = ("POST", url, params, timeout, json)
        return DummyResponse(200, {"ok": True, "url": url, "params": params, "json": json})


def test_headers_and_urls(monkeypatch):
    # Patch requests.Session used inside DuckDiceAPI
    import duckdice_api.api as api_mod

    monkeypatch.setattr(api_mod.requests, "Session", DummySession)

    cfg = dd.DuckDiceConfig(api_key="KEY", base_url="https://example/api", timeout=10)
    api = dd.DuckDiceAPI(cfg)

    # GET
    res = api.get_currency_stats("BTC")
    assert res["ok"]
    assert res["url"].startswith("https://example/api/")
    assert res["params"]["api_key"] == "KEY"

    # POST
    res2 = api.play_dice(symbol="BTC", amount="0.1", chance="50", is_high=True)
    assert res2["ok"]
    assert res2["json"]["symbol"] == "BTC"
    assert res2["json"]["isHigh"] is True
