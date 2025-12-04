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

"""
Interactive Brokers (IBKR) Broker Integration
Provides connection, order execution, and position management for IBKR TWS/Gateway
"""


import logging
import os
import time
from dataclasses import dataclass

import pandas as pd

try:
    from ib_insync import IB, LimitOrder, MarketOrder, Stock, StopLimitOrder, StopOrder
except ImportError:
    print("Warning: ib_insync not installed. Install with: pip install ib_insync")
    IB = None

logger = logging.getLogger(__name__)


@dataclass
class IBKRConfig:
    """Configuration for IBKR connection."""

    def __init__(self):
        self.paper_trading = os.getenv("IBKR_PAPER_TRADING", "true").lower() == "true"
        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", "7497" if self.paper_trading else "7496"))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))
        self.timeout = int(os.getenv("IBKR_TIMEOUT", "20"))
        self.max_retries = int(os.getenv("IBKR_MAX_RETRIES", "3"))


class IBKRBroker:
    """
    Interactive Brokers broker integration.
    Handles connection, order execution, and position management.
    """

    def __init__(self, config: IBKRConfig = None, auto_connect: bool = True):
        """
        Initialize IBKR broker connection.

        Args:
            config: IBKR configuration
            auto_connect: Whether to connect automatically
        """
        if IB is None:
            raise ImportError("ib_insync is required for IBKR integration")

        self.config = config or IBKRConfig()
        self.ib = IB()
        self.connected = False
        self.positions = {}
        self.orders = {}
        self.account_info = {}

        # Setup error handling
        self.ib.errorEvent += self._handle_error
        self.ib.disconnectedEvent += self._handle_disconnect

        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        """Connect to IBKR TWS or IB Gateway."""
        try:
            logger.info(f"Connecting to IBKR at {self.config.host}:{self.config.port}")

            # Connect with timeout
            self.ib.connect(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
            )

            if self.ib.isConnected():
                self.connected = True
                logger.info(
                    f"Connected to IBKR "
                    f"{'Paper Trading' if self.config.paper_trading else 'Live Trading'}"
                )

                # Request account info
                self._request_account_info()

                return True
            else:
                logger.error("Failed to connect to IBKR")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def _handle_disconnect(self):
        """Handle IBKR disconnection."""
        self.connected = False
        logger.warning("IBKR connection lost, attempting to reconnect...")

        # Attempt to reconnect
        for attempt in range(self.config.max_retries):
            if self.connect():
                logger.info("Successfully reconnected to IBKR")
                return
            time.sleep(2**attempt)  # Exponential backoff

        logger.error("Failed to reconnect to IBKR after multiple attempts")

    def _handle_error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ):
        """Handle IBKR errors."""
        logger.error(f"IBKR Error {errorCode}: {errorString} (reqId: {reqId})")

        # Handle specific error codes
        if errorCode == 1100 or errorCode == 1101:  # Connectivity between IB and TWS lost
            self._handle_disconnect()

    def _request_account_info(self):
        """Request account information."""
        try:
            # Request account summary
            self.ib.reqAccountSummary(1, "All", "NetLiquidation,BuyingPower,TotalCashValue")

            # Wait for account info
            self.ib.sleep(1)

            # Store account info
            for summary in self.ib.accountSummary():
                self.account_info[summary.tag] = summary.value

            logger.info(f"Account info loaded: {self.account_info}")

        except Exception as e:
            logger.error(f"Failed to request account info: {e}")
            # Try alternative method
            try:
                # Get account values directly
                account_values = self.ib.accountValues()
                for value in account_values:
                    self.account_info[value.tag] = value.value
                logger.info(f"Account info loaded via alternative method: {self.account_info}")
            except Exception as e2:
                logger.error(f"Alternative account info method also failed: {e2}")

    def get_account_info(self) -> dict[str, str]:
        """Get current account information."""
        if not self.connected:
            self.connect()

        self._request_account_info()
        return self.account_info.copy()

    def get_positions(self) -> dict[str, dict]:
        """Get current positions."""
        if not self.connected:
            self.connect()

        try:
            positions = {}
            for position in self.ib.positions():
                symbol = position.contract.symbol
                positions[symbol] = {
                    "quantity": position.position,
                    "avg_cost": position.avgCost,
                    "market_value": position.marketValue,
                    "unrealized_pnl": position.unrealizedPNL,
                    "realized_pnl": position.realizedPNL,
                }

            self.positions = positions
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def place_order(
        self,
        symbol: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: str = "DAY",
    ) -> str | None:
        """
        Place an order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares (positive for buy, negative for sell)
            order_type: Order type (MKT, LMT, STP, STP_LMT)
            limit_price: Limit price for LMT orders
            stop_price: Stop price for STP orders
            tif: Time in force (DAY, GTC, IOC, FOK)

        Returns:
            Order ID if successful, None otherwise
        """
        if not self.connected:
            self.connect()

        try:
            # Create contract
            contract = Stock(symbol, "SMART", "USD")

            # Create order
            if order_type == "MKT":
                order = MarketOrder("BUY" if quantity > 0 else "SELL", abs(quantity))
            elif order_type == "LMT":
                if limit_price is None:
                    raise ValueError("Limit price required for LMT orders")
                order = LimitOrder("BUY" if quantity > 0 else "SELL", abs(quantity), limit_price)
            elif order_type == "STP":
                if stop_price is None:
                    raise ValueError("Stop price required for STP orders")
                order = StopOrder("BUY" if quantity > 0 else "SELL", abs(quantity), stop_price)
            elif order_type == "STP_LMT":
                if limit_price is None or stop_price is None:
                    raise ValueError("Both limit and stop prices required for STP_LMT orders")
                order = StopLimitOrder(
                    "BUY" if quantity > 0 else "SELL",
                    abs(quantity),
                    limit_price,
                    stop_price,
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            order.tif = tif

            # Submit order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Wait for order submission

            if trade.orderStatus.status == "Submitted":
                order_id = str(trade.order.orderId)
                self.orders[order_id] = trade
                logger.info(f"Order placed: {symbol} {quantity} shares at {order_type}")
                return order_id
            else:
                logger.error(f"Order failed: {trade.orderStatus.status}")
                return None

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.connected:
            self.connect()

        try:
            if order_id in self.orders:
                trade = self.orders[order_id]
                self.ib.cancelOrder(trade.order)
                self.ib.sleep(1)

                if trade.orderStatus.status == "Cancelled":
                    logger.info(f"Order {order_id} cancelled successfully")
                    return True
                else:
                    logger.error(f"Failed to cancel order {order_id}")
                    return False
            else:
                logger.error(f"Order {order_id} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def get_order_status(self, order_id: str) -> dict | None:
        """Get order status."""
        if order_id in self.orders:
            trade = self.orders[order_id]
            return {
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "last_fill_price": trade.orderStatus.lastFillPrice,
                "why_held": trade.orderStatus.whyHeld,
            }
        return None

    def get_market_data(
        self, symbol: str, duration: str = "1 D", bar_size: str = "1 min"
    ) -> pd.DataFrame | None:
        """
        Get market data for a symbol.

        Args:
            symbol: Stock symbol
            duration: Data duration (e.g., '1 D', '1 W', '1 M')
            bar_size: Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            self.connect()

        try:
            # Create contract
            contract = Stock(symbol, "SMART", "USD")

            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if bars:
                # Convert to DataFrame
                data = []
                for bar in bars:
                    data.append(
                        {
                            "Date": bar.date,
                            "Open": bar.open,
                            "High": bar.high,
                            "Low": bar.low,
                            "Close": bar.close,
                            "Volume": bar.volume,
                        }
                    )

                df = pd.DataFrame(data)
                df.set_index("Date", inplace=True)
                return df
            else:
                logger.warning(f"No market data received for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    def get_real_time_price(self, symbol: str) -> float | None:
        """Get real-time price for a symbol."""
        if not self.connected:
            self.connect()

        try:
            # Create contract
            contract = Stock(symbol, "SMART", "USD")

            # Request real-time data
            self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Wait for data

            # Get last price
            ticker = self.ib.ticker(contract)
            if ticker and ticker.last:
                return ticker.last
            else:
                logger.warning(f"No real-time price available for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Failed to get real-time price for {symbol}: {e}")
            return None

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self.connected and self.ib.isConnected()

    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def test_ibkr_connection():
    """Test IBKR connection."""
    try:
        config = IBKRConfig()
        broker = IBKRBroker(config=config, auto_connect=True)

        if broker.is_connected():
            print(
                f"✅ Connected to IBKR "
                f"{'Paper Trading' if config.paper_trading else 'Live Trading'}"
            )

            # Test account info
            account_info = broker.get_account_info()
            print(f"Account Info: {account_info}")

            # Test positions
            positions = broker.get_positions()
            print(f"Positions: {positions}")

            broker.disconnect()
            return True
        else:
            print("❌ Failed to connect to IBKR")
            return False

    except Exception as e:
        print(f"❌ Error testing IBKR connection: {e}")
        return False


if __name__ == "__main__":
    # Test connection
    test_ibkr_connection()
