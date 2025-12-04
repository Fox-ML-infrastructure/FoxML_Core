"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brokers.interface import Broker


class PaperBroker(Broker):
    def __init__(
        self, *, slippage_bps: float = 5.0, fee_bps: float = 1.0, log_dir: str = "logs/trades"
    ):
        self._pos: dict[str, float] = {}
        self._cash: float = 100_000.0
        self._slip = float(slippage_bps) * 1e-4
        self._fee = float(fee_bps) * 1e-4
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")

    def submit(self, order: dict[str, Any]) -> dict[str, Any]:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        symbol, side, qty = order["symbol"], order["side"].upper(), float(order["qty"])
        px_theory = float(order.get("px") or 0.0)
        px_fill = px_theory * (1 + self._slip * (1 if side == "BUY" else -1))
        fee = abs(qty * px_fill) * self._fee
        notional = qty * px_fill
        self._pos[symbol] = self._pos.get(symbol, 0.0) + (qty if side == "BUY" else -qty)
        self._cash -= notional + (fee if side == "BUY" else -fee)
        rec = {
            "ts": ts,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "px_theory": px_theory,
            "px_fill": px_fill,
            "slippage_bps": self._slip * 1e4,
            "fee_bps": self._fee * 1e4,
            "notional": abs(notional),
            "run_id": self._run_id,
        }
        self._append(rec)
        return rec

    def cancel(self, order_id: str) -> bool:
        return True

    def positions(self) -> dict[str, float]:
        return dict(self._pos)

    def cash(self) -> float:
        return float(self._cash)

    def now(self):
        return datetime.now(UTC)

    def _append(self, rec: dict[str, Any]) -> None:
        fp = self._log_dir / f"{datetime.now(UTC).date()}.jsonl"
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
