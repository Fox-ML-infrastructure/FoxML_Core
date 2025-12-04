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

from datetime import datetime
from typing import Any, Protocol


class Broker(Protocol):
    """Protocol for live trading broker adapters."""

    def submit_order(
        self, symbol: str, side: str, qty: float, order_type: str = "market"
    ) -> dict[str, Any]:
        """Submit an order to the broker.

        Args:
            symbol: Trading symbol (e.g., "SPY")
            side: "BUY" or "SELL"
            qty: Quantity to trade
            order_type: "market", "limit", etc.

        Returns:
            Dict with order_id, status, timestamp, etc.
        """
        ...

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Dict with status, timestamp, etc.
        """
        ...

    def get_positions(self) -> dict[str, float]:
        """Get current positions by symbol.

        Returns:
            Dict mapping symbol to position size (positive = long, negative = short)
        """
        ...

    def get_cash(self) -> float:
        """Get available cash balance.

        Returns:
            Available cash amount
        """
        ...

    def get_fills(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """Get recent fills.

        Args:
            since: Get fills since this timestamp (UTC)

        Returns:
            List of fill dicts with symbol, side, qty, price, timestamp, etc.
        """
        ...

    def now(self) -> datetime:
        """Get current broker time (UTC).

        Returns:
            Current timestamp from broker
        """
        ...


def get_broker(venue: str = "example") -> Broker:
    """Factory function to get broker instance.

    Args:
        venue: Broker venue ("example", "ibkr", etc.)

    Returns:
        Broker instance implementing the Broker protocol
    """
    if venue == "example":
        from .example_venue import ExampleVenueBroker

        return ExampleVenueBroker()
    elif venue == "ibkr":
        from pathlib import Path

        import yaml

        from .ibkr import IBKRBroker, IBKRConfig

        # Load IBKR config from overlay
        ibkr_cfg = yaml.safe_load(Path("config/brokers/ibkr.yaml").read_text())

        # Create config
        config = IBKRConfig(
            host=ibkr_cfg.get("ibkr", {}).get("host", "127.0.0.1"),
            port=ibkr_cfg.get("ibkr", {}).get("port", 7497),
            client_id=ibkr_cfg.get("ibkr", {}).get("client_id", 123),
            account=ibkr_cfg.get("ibkr", {}).get("account"),
            route=ibkr_cfg.get("ibkr", {}).get("route", "SMART"),
            currency=ibkr_cfg.get("ibkr", {}).get("currency", "USD"),
            allow_fractional=ibkr_cfg.get("ibkr", {}).get("allow_fractional", False),
            px_tick=ibkr_cfg.get("ibkr", {}).get("px_tick", 0.01),
            qty_min=ibkr_cfg.get("ibkr", {}).get("qty_min", 1.0),
        )
        return IBKRBroker(config)
    else:
        raise ValueError(f"Unknown broker venue: {venue}")


def normalize_order(symbol: str, side: str, qty: float, px: float | None = None) -> dict[str, Any]:
    return {
        "symbol": str(symbol),
        "side": str(side).upper(),
        "qty": float(qty),
        "px": (float(px) if px is not None else None),
    }
