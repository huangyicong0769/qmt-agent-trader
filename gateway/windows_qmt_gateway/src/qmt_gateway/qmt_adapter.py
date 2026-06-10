"""QMT adapter.

The first implementation is a mock-safe adapter. Real xtquant integration should
be added on the Windows machine after `qmt-smoke-test` confirms local support.
"""

from __future__ import annotations

from dataclasses import dataclass

from qmt_gateway.xtquant_loader import can_import_xtquant


@dataclass
class QMTAdapter:
    xtquant_path: object | None = None

    def status(self) -> dict[str, object]:
        return {"xtquant_importable": can_import_xtquant(self.xtquant_path), "connected": False}

    def asset(self) -> dict[str, object]:
        return {"cash": 0.0, "total_asset": 0.0, "mock": True}

    def positions(self) -> dict[str, object]:
        return {"positions": [], "mock": True}

    def orders(self) -> dict[str, object]:
        return {"orders": [], "mock": True}

    def trades(self) -> dict[str, object]:
        return {"trades": [], "mock": True}

    def instruments(self) -> dict[str, object]:
        return {"instruments": [], "mock": True}

    def bars(self, symbols: str, start: str, end: str, freq: str) -> dict[str, object]:
        return {
            "symbols": symbols.split(","),
            "start": start,
            "end": end,
            "freq": freq,
            "bars": [],
            "mock": True,
        }

    def latest(self, symbols: str) -> dict[str, object]:
        return {"symbols": symbols.split(","), "quotes": {}, "mock": True}
