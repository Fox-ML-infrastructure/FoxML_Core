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
Risk Guardrails
Risk management and safety checks for the trading system
"""


import logging

import numpy as np


class RiskGuardrails:
    """
    Risk management guardrails for trading operations.
    """

    def __init__(self, config: dict):
        """
        Initialize risk guardrails.

        Args:
            config: Configuration dictionary with risk parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize kill switches
        self.kill_switches = self._initialize_kill_switches()

        # Risk limits
        self.risk_limits = self._initialize_risk_limits()

    def _initialize_kill_switches(self) -> dict:
        """Initialize kill switches for risk management."""
        kill_switches = self.config.get("kill_switches", {})

        # Set defaults if not provided
        defaults = {
            "enabled": True,
            "max_daily_loss_pct": 2.0,
            "max_daily_loss_dollars": 2000,
            "max_drawdown_pct": 10.0,
            "max_position_size_pct": 20.0,
            "max_sector_exposure_pct": 30.0,
        }

        for key, default_value in defaults.items():
            if key not in kill_switches:
                kill_switches[key] = default_value

        return kill_switches

    def _initialize_risk_limits(self) -> dict:
        """Initialize risk limits."""
        risk_params = self.config.get("risk_params", {})

        return {
            "max_position_size": risk_params.get("max_position_size", 0.1),
            "stop_loss_pct": risk_params.get("stop_loss_pct", 0.02),
            "take_profit_pct": risk_params.get("take_profit_pct", 0.04),
            "max_leverage": risk_params.get("max_leverage", 1.0),
            "max_correlation": risk_params.get("max_correlation", 0.7),
        }

    def check_kill_switches(self, daily_returns: list[dict], capital: float) -> bool:
        """
        Check if any kill switches are triggered.

        Args:
            daily_returns: List of daily return records
            capital: Current capital

        Returns:
            True if kill switches are triggered, False otherwise
        """
        if not self.kill_switches.get("enabled", True):
            return False

        # Calculate daily P&L
        if len(daily_returns) > 0:
            daily_pnl = daily_returns[-1].get("pnl", 0)
            daily_pnl_pct = (daily_pnl / capital) * 100

            # Check daily loss limits
            max_daily_loss_pct = self.kill_switches.get("max_daily_loss_pct", 2.0)
            max_daily_loss_dollars = self.kill_switches.get("max_daily_loss_dollars", 2000)

            if daily_pnl_pct < -max_daily_loss_pct:
                self.logger.warning(
                    f"Kill switch triggered: Daily loss {daily_pnl_pct:.2f}% exceeds limit of {max_daily_loss_pct}%"
                )
                return True

            if daily_pnl < -max_daily_loss_dollars:
                self.logger.warning(
                    f"Kill switch triggered: Daily loss ${daily_pnl:.2f} exceeds limit of ${max_daily_loss_dollars}"
                )
                return True

        # Check drawdown
        if len(daily_returns) > 0:
            peak_capital = max([r.get("capital", capital) for r in daily_returns])
            current_capital = capital
            drawdown_pct = ((peak_capital - current_capital) / peak_capital) * 100

            max_drawdown_pct = self.kill_switches.get("max_drawdown_pct", 10.0)
            if drawdown_pct > max_drawdown_pct:
                self.logger.warning(
                    f"Kill switch triggered: Drawdown {drawdown_pct:.2f}% exceeds limit of {max_drawdown_pct}%"
                )
                return True

        return False

    def validate_position_size(
        self, symbol: str, shares: int, price: float, capital: float
    ) -> bool:
        """
        Validate position size against risk limits.

        Args:
            symbol: Trading symbol
            shares: Number of shares
            price: Share price
            capital: Available capital

        Returns:
            True if position size is valid, False otherwise
        """
        position_value = shares * price
        position_pct = (position_value / capital) * 100

        max_position_size_pct = self.kill_switches.get("max_position_size_pct", 20.0)

        if position_pct > max_position_size_pct:
            self.logger.warning(
                f"Position size validation failed: {position_pct:.2f}% exceeds limit of {max_position_size_pct}%"
            )
            return False

        return True

    def validate_trade(self, trade: dict, capital: float, positions: dict) -> bool:
        """
        Validate a trade against risk limits.

        Args:
            trade: Trade dictionary
            capital: Available capital
            positions: Current positions

        Returns:
            True if trade is valid, False otherwise
        """
        symbol = trade["symbol"]
        shares = trade["shares"]
        price = trade["price"]
        action = trade["action"]

        # Check position size
        if not self.validate_position_size(symbol, shares, price, capital):
            return False

        # Check if we have enough capital for buy orders
        if action == "BUY":
            trade_value = shares * price
            if trade_value > capital:
                self.logger.warning(
                    f"Insufficient capital for trade: ${trade_value:.2f} > ${capital:.2f}"
                )
                return False

        # Check if we have enough shares for sell orders
        elif action == "SELL":
            current_position = positions.get(symbol, 0)
            if current_position < shares:
                self.logger.warning(f"Insufficient shares for sell: {shares} > {current_position}")
                return False

        return True

    def calculate_position_limits(self, capital: float, price: float) -> int:
        """
        Calculate maximum position size based on risk limits.

        Args:
            capital: Available capital
            price: Share price

        Returns:
            Maximum number of shares allowed
        """
        max_position_size = self.risk_limits["max_position_size"]
        max_position_value = capital * max_position_size
        max_shares = int(max_position_value / price)

        return max_shares

    def check_portfolio_risk(self, positions: dict, prices: dict, capital: float) -> dict:
        """
        Check portfolio-level risk metrics.

        Args:
            positions: Current positions
            prices: Current prices
            capital: Available capital

        Returns:
            Dictionary of risk metrics
        """
        portfolio_value = capital
        position_values = {}

        for symbol, shares in positions.items():
            if symbol in prices:
                value = shares * prices[symbol]
                position_values[symbol] = value
                portfolio_value += value

        # Calculate concentration metrics
        concentration = {}
        for symbol, value in position_values.items():
            concentration[symbol] = (value / portfolio_value) * 100

        # Find largest position
        max_concentration = max(concentration.values()) if concentration else 0

        # Calculate leverage
        leverage = portfolio_value / capital if capital > 0 else 0

        return {
            "portfolio_value": portfolio_value,
            "leverage": leverage,
            "max_concentration": max_concentration,
            "concentration": concentration,
            "position_count": len(positions),
        }

    def apply_stop_loss(self, positions: dict, prices: dict, stop_loss_pct: float) -> list[dict]:
        """
        Apply stop loss rules to positions.

        Args:
            positions: Current positions
            prices: Current prices
            stop_loss_pct: Stop loss percentage

        Returns:
            List of stop loss orders
        """
        stop_orders = []

        for symbol, shares in positions.items():
            if symbol in prices and shares > 0:
                current_price = prices[symbol]
                # This is a simplified implementation
                # In a real system, you'd track entry prices
                stop_price = current_price * (1 - stop_loss_pct)

                stop_orders.append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "shares": shares,
                        "stop_price": stop_price,
                        "type": "stop_loss",
                    }
                )

        return stop_orders

    def apply_take_profit(
        self, positions: dict, prices: dict, take_profit_pct: float
    ) -> list[dict]:
        """
        Apply take profit rules to positions.

        Args:
            positions: Current positions
            prices: Current prices
            take_profit_pct: Take profit percentage

        Returns:
            List of take profit orders
        """
        take_profit_orders = []

        for symbol, shares in positions.items():
            if symbol in prices and shares > 0:
                current_price = prices[symbol]
                # This is a simplified implementation
                # In a real system, you'd track entry prices
                take_profit_price = current_price * (1 + take_profit_pct)

                take_profit_orders.append(
                    {
                        "symbol": symbol,
                        "action": "SELL",
                        "shares": shares,
                        "take_profit_price": take_profit_price,
                        "type": "take_profit",
                    }
                )

        return take_profit_orders

    def calculate_var(self, returns: list[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: List of returns
            confidence_level: Confidence level for VaR calculation

        Returns:
            VaR value
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_array, var_percentile)

        return var

    def calculate_expected_shortfall(
        self, returns: list[float], confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: List of returns
            confidence_level: Confidence level for calculation

        Returns:
            Expected shortfall value
        """
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        var = self.calculate_var(returns, confidence_level)

        # Calculate expected shortfall
        tail_returns = returns_array[returns_array <= var]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var

        return expected_shortfall

    def get_risk_report(
        self, positions: dict, prices: dict, capital: float, daily_returns: list[dict]
    ) -> dict:
        """
        Generate comprehensive risk report.

        Args:
            positions: Current positions
            prices: Current prices
            capital: Available capital
            daily_returns: Daily return history

        Returns:
            Risk report dictionary
        """
        # Portfolio risk metrics
        portfolio_risk = self.check_portfolio_risk(positions, prices, capital)

        # Calculate returns for VaR
        returns = []
        for i, daily in enumerate(daily_returns):
            if i > 0:
                prev_value = daily_returns[i - 1].get("portfolio_value", capital)
                current_value = daily.get("portfolio_value", capital)
                if prev_value > 0:
                    returns.append((current_value - prev_value) / prev_value)

        # Risk metrics
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        es_95 = self.calculate_expected_shortfall(returns, 0.95)

        # Kill switch status
        kill_switches_triggered = self.check_kill_switches(daily_returns, capital)

        return {
            "portfolio_risk": portfolio_risk,
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall_95": es_95,
            "kill_switches_triggered": kill_switches_triggered,
            "risk_limits": self.risk_limits,
            "kill_switches": self.kill_switches,
        }
