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
Performance and Objective-Driven Risk Management
Handles dynamic position sizing and objective-based risk budgeting.
"""


import logging

import numpy as np
import pandas as pd

from .objectives import build_objective
from .trade_logger import TradeRecord

logger = logging.getLogger(__name__)


def calculate_trade_metrics(
    closed_trades: list[TradeRecord],
) -> dict[str, float | str]:
    """
    Calculate comprehensive trade metrics.

    Args:
        closed_trades: List of closed trade records

    Returns:
        Dictionary with performance metrics
    """
    if not closed_trades:
        return {
            "win_rate": 0.0,
            "profit_factor": "N/A",
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "total_fees": 0.0,
        }

    pnls = [t.realized_pnl for t in closed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_trades = len(closed_trades)
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

    sum_pos = sum(wins)
    sum_neg = abs(sum(losses))

    profit_factor = "N/A" if sum_neg == 0 else sum_pos / sum_neg

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "avg_win": (sum_pos / len(wins)) if wins else 0.0,
        "avg_loss": (-sum(losses) / len(losses)) if losses else 0.0,
        "total_trades": total_trades,
        "total_pnl": sum(pnls),
        "total_fees": sum(t.cum_fees for t in closed_trades),
    }


def calculate_portfolio_metrics(equity_curve: list[dict]) -> dict[str, float]:
    """
    Calculate portfolio-level performance metrics.

    Args:
        equity_curve: List of daily equity values

    Returns:
        Dictionary with portfolio metrics
    """
    if len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    # Calculate daily returns
    returns = []
    for i in range(1, len(equity_curve)):
        prev_equity = equity_curve[i - 1]["equity"]
        curr_equity = equity_curve[i]["equity"]
        if prev_equity > 0:
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)

    if not returns:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    returns_series = pd.Series(returns)

    # Calculate metrics
    total_return = (equity_curve[-1]["equity"] / equity_curve[0]["equity"]) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

    # Calculate max drawdown
    cumulative_returns = (1 + returns_series).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }


class GrowthTargetCalculator:
    """Objective-driven position sizing and risk budgeting manager."""

    def __init__(self, config: dict):
        """
        Initialize objective-driven risk manager.

        Args:
            config: Configuration with growth target parameters
        """
        self.config = config
        self.risk_config = config.get("risk_params", {})
        self.objective = build_objective(config)

        # Objective/position sizing parameters
        self.volatility_adjustment = bool(
            config.get("objective", {}).get("volatility_adjustment", True)
        )
        self.performance_lookback = int(
            config.get("objective", {}).get("performance_lookback_days", 30)
        )

        # Risk parameters
        self.max_position_size = self.risk_config.get("max_position_size", 0.15)
        self.volatility_target = self.risk_config.get("volatility_target", 0.20)
        self.kelly_fraction = self.risk_config.get("kelly_fraction", 0.25)
        self.position_sizing_method = self.risk_config.get(
            "position_sizing_method", "kelly_optimal"
        )

        # Performance tracking
        self.daily_returns = []
        self.performance_history = []

        logger.info("Initialized Objective-driven risk manager")

    def calculate_dynamic_position_size(
        self,
        signal_strength: float,
        current_capital: float,
        symbol_volatility: float = None,
        portfolio_volatility: float = None,
    ) -> float:
        """
        Calculate dynamic position size using objective-derived risk budget and guardrails.

        Args:
            signal_strength: Signal strength (0-1)
            current_capital: Current portfolio capital
            symbol_volatility: Symbol-specific volatility
            portfolio_volatility: Portfolio volatility

        Returns:
            Position size as fraction of capital
        """
        try:
            # Build series of recent returns for objective
            returns_series = (
                pd.Series(self.daily_returns)
                if len(self.daily_returns) > 0
                else pd.Series(dtype=float)
            )
            equity_series = pd.Series(dtype=float)
            risk_metrics = {"portfolio_vol": portfolio_volatility or 0.0}

            # Derive risk budget (0..~1.5) and position multiplier (0.5..1.5)
            risk_budget, pos_mult = self.objective.derive_risk_budget(
                returns_series, equity_series, risk_metrics
            )

            # Estimate edge/variance proxy from signal and volatility
            edge = max(0.0, float(signal_strength) - 0.5)  # centered signal ∈ [0,0.5]
            variance = max(1e-6, (symbol_volatility or portfolio_volatility or 0.02) ** 2)

            # Kelly-style base fraction scaled by cap and risk budget
            kelly_base = edge / variance
            base_position_size = self.kelly_fraction * kelly_base * risk_budget * pos_mult

            # Volatility adjustment relative to target
            if self.volatility_adjustment and portfolio_volatility:
                vol_adjustment = min(1.0, self.volatility_target / max(1e-6, portfolio_volatility))
                base_position_size *= vol_adjustment

            # Clamp to guardrails
            final_position_size = float(np.clip(base_position_size, 0.0, self.max_position_size))

            logger.debug(
                "Objective sizing -> signal=%.3f, risk_budget=%.3f, pos_mult=%.3f, edge=%.4f, var=%.6f, size=%.3f",
                signal_strength,
                risk_budget,
                pos_mult,
                edge,
                variance,
                final_position_size,
            )

            return final_position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _calculate_kelly_position_size(self, signal_strength: float) -> float:
        """
        Calculate Kelly criterion optimal position size.

        Args:
            signal_strength: Signal strength (0-1)

        Returns:
            Kelly optimal position size
        """
        if len(self.daily_returns) < 10:
            return signal_strength * self.max_position_size

        try:
            # Calculate win rate and average win/loss
            returns = pd.Series(self.daily_returns)
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                return signal_strength * self.max_position_size

            win_rate = len(wins) / len(returns)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())

            if avg_loss == 0:
                return signal_strength * self.max_position_size

            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-p
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Apply signal strength and cap at maximum
            kelly_position = max(0.0, kelly_fraction * signal_strength)

            return min(kelly_position, self.max_position_size)

        except Exception as e:
            logger.error(f"Error calculating Kelly position: {e}")
            return signal_strength * self.max_position_size

    def _calculate_performance_adjustment(self) -> float:
        """Deprecated; retained for compatibility, returns neutral 1.0."""
        return 1.0

    def update_performance(self, daily_return: float, portfolio_value: float):
        """
        Update performance tracking.

        Args:
            daily_return: Daily return percentage
            portfolio_value: Current portfolio value
        """
        self.daily_returns.append(daily_return)

        # Keep only recent history
        if len(self.daily_returns) > 252:  # One year
            self.daily_returns = self.daily_returns[-252:]

        # Record performance metrics
        performance_record = {
            "date": pd.Timestamp.now(),
            "daily_return": daily_return,
            "portfolio_value": portfolio_value,
            "cumulative_return": (portfolio_value / self.config.get("initial_capital", 100000)) - 1,
        }

        self.performance_history.append(performance_record)

        logger.debug(
            "Updated performance: return=%.4f, portfolio=%.2f",
            daily_return,
            portfolio_value,
        )

    def get_growth_metrics(self) -> dict:
        """
        Get current growth and performance metrics.

        Returns:
            Dictionary of growth metrics
        """
        if not self.daily_returns:
            return {
                "avg_daily_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "objective_score": 0.0,
                "days_tracked": 0,
            }

        try:
            returns = pd.Series(self.daily_returns)
            avg_return = returns.mean()
            volatility = returns.std()
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
            score = self.objective.score(returns, pd.Series(dtype=float), {})
            return {
                "avg_daily_return": float(avg_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "objective_score": float(score),
                "days_tracked": len(self.daily_returns),
            }

        except Exception as e:
            logger.error(f"Error calculating growth metrics: {e}")
            return {
                "avg_daily_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "objective_score": 0.0,
                "days_tracked": 0,
            }

    def should_adjust_target(self) -> tuple[bool, str]:
        """Deprecated; retained for CLI compatibility."""
        return False, "Objective-based sizing active"
