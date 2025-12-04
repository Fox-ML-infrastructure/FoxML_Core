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
Telemetry Snapshot
System monitoring and state reporting
"""


import json
import logging
from datetime import date as date_class
from datetime import datetime
from pathlib import Path

import pandas as pd


class TelemetrySnapshot:
    """
    System telemetry and monitoring snapshot.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize telemetry snapshot.

        Args:
            output_dir: Output directory for snapshots
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def capture_system_state(
        self,
        capital: float,
        positions: dict,
        trade_history: list[dict],
        daily_returns: list[dict],
        regime_history: list[dict],
        performance_metrics: dict,
        config: dict,
    ) -> dict:
        """
        Capture current system state.

        Args:
            capital: Current capital
            positions: Current positions
            trade_history: Trade history
            daily_returns: Daily returns
            regime_history: Regime history
            performance_metrics: Performance metrics
            config: System configuration

        Returns:
            System state snapshot
        """
        timestamp = datetime.now()

        # Calculate portfolio value
        portfolio_value = capital
        position_values = {}

        for symbol, shares in positions.items():
            # This is simplified - in real system you'd get current prices
            position_values[symbol] = {
                "shares": shares,
                "value": shares * 100,  # Placeholder price
            }
            portfolio_value += shares * 100  # Placeholder calculation

        # Calculate summary metrics
        total_trades = len(trade_history)
        total_volume = sum([t.get("value", 0) for t in trade_history])

        # Calculate daily P&L
        daily_pnl = 0
        if daily_returns:
            daily_pnl = daily_returns[-1].get("pnl", 0)

        # Get current regime
        current_regime = "unknown"
        regime_confidence = 0.0
        if regime_history:
            current_regime = regime_history[-1].get("regime", "unknown")
            regime_confidence = regime_history[-1].get("confidence", 0.0)

        snapshot = {
            "timestamp": timestamp.isoformat(),
            "date": date_class.today().isoformat(),
            "capital": capital,
            "portfolio_value": portfolio_value,
            "total_return_pct": (
                (portfolio_value - config.get("initial_capital", 100000))
                / config.get("initial_capital", 100000)
            )
            * 100,
            "daily_pnl": daily_pnl,
            "positions": position_values,
            "position_count": len(positions),
            "total_trades": total_trades,
            "total_volume": total_volume,
            "current_regime": current_regime,
            "regime_confidence": regime_confidence,
            "performance_metrics": performance_metrics,
            "config_summary": {
                "symbols": config.get("symbols", []),
                "strategies": list(config.get("strategies", {}).keys()),
                "risk_params": config.get("risk_params", {}),
            },
        }

        return snapshot

    def save_snapshot(self, snapshot: dict, filename: str | None = None) -> str:
        """
        Save system snapshot to file.

        Args:
            snapshot: System snapshot
            filename: Optional filename, defaults to timestamp

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_snapshot_{timestamp}.json"

        filepath = Path(self.output_dir) / filename

        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        self.logger.info(f"System snapshot saved to {filepath}")
        return str(filepath)

    def generate_daily_report(
        self,
        capital: float,
        positions: dict,
        trade_history: list[dict],
        daily_returns: list[dict],
        regime_history: list[dict],
        config: dict,
    ) -> str:
        """
        Generate daily trading report.

        Args:
            capital: Current capital
            positions: Current positions
            trade_history: Trade history
            daily_returns: Daily returns
            regime_history: Regime history
            config: System configuration

        Returns:
            Path to generated report
        """
        today = date_class.today()

        # Filter today's data
        today_trades = [t for t in trade_history if t.get("date") == today]
        today_returns = [r for r in daily_returns if r.get("date") == today]
        today_regime = [r for r in regime_history if r.get("date") == today]

        # Calculate today's metrics
        today_volume = sum([t.get("value", 0) for t in today_trades])
        today_pnl = today_returns[-1].get("pnl", 0) if today_returns else 0
        current_regime = today_regime[-1].get("regime", "unknown") if today_regime else "unknown"
        regime_confidence = today_regime[-1].get("confidence", 0.0) if today_regime else 0.0

        # Generate report
        report = {
            "date": today.isoformat(),
            "capital": capital,
            "trades_executed": len(today_trades),
            "volume_traded": today_volume,
            "daily_pnl": today_pnl,
            "current_regime": current_regime,
            "regime_confidence": regime_confidence,
            "positions": dict(positions),
            "trade_details": today_trades,
        }

        # Save report
        filename = f"daily_report_{today.strftime('%Y-%m-%d')}.json"
        filepath = self.save_snapshot(report, filename)

        return filepath

    def generate_performance_report(
        self,
        daily_returns: list[dict],
        trade_history: list[dict],
        regime_history: list[dict],
        config: dict,
    ) -> str:
        """
        Generate comprehensive performance report.

        Args:
            daily_returns: Daily returns
            trade_history: Trade history
            regime_history: Regime history
            config: System configuration

        Returns:
            Path to generated report
        """
        if not daily_returns:
            self.logger.warning("No daily returns data available for performance report")
            return ""

        # Calculate performance metrics
        initial_capital = config.get("initial_capital", 100000)
        final_value = daily_returns[-1]["portfolio_value"]
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        # Calculate daily returns
        daily_returns_list = []
        for i, daily in enumerate(daily_returns):
            if i == 0:
                daily_return_pct = 0
            else:
                prev_value = daily_returns[i - 1]["portfolio_value"]
                daily_return_pct = ((daily["portfolio_value"] - prev_value) / prev_value) * 100
            daily_returns_list.append(daily_return_pct)

        # Calculate statistics
        returns_array = pd.Series(daily_returns_list)
        volatility = returns_array.std() * (252**0.5)  # Annualized
        sharpe_ratio = (returns_array.mean() * 252) / volatility if volatility > 0 else 0

        # Calculate drawdown
        peak = initial_capital
        max_drawdown = 0
        for daily in daily_returns:
            if daily["portfolio_value"] > peak:
                peak = daily["portfolio_value"]
            drawdown = (peak - daily["portfolio_value"]) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate win rate
        winning_days = sum(1 for r in daily_returns_list if r > 0)
        total_days = len(daily_returns_list)
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0

        # Regime distribution
        regime_counts = {}
        for record in regime_history:
            regime = record["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        regime_distribution = {
            regime: (count / len(regime_history) * 100) for regime, count in regime_counts.items()
        }

        # Generate report
        report = {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "volatility_pct": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "win_rate_pct": win_rate,
            "total_trades": len(trade_history),
            "regime_distribution": regime_distribution,
            "daily_returns": daily_returns_list,
            "trade_summary": {
                "total_volume": sum([t.get("value", 0) for t in trade_history]),
                "avg_trade_size": (
                    sum([t.get("value", 0) for t in trade_history]) / len(trade_history)
                    if trade_history
                    else 0
                ),
            },
        }

        # Save report
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.save_snapshot(report, filename)

        return filepath

    def log_system_health(
        self,
        capital: float,
        positions: dict,
        daily_returns: list[dict],
        config: dict,
    ) -> dict:
        """
        Log system health metrics.

        Args:
            capital: Current capital
            positions: Current positions
            daily_returns: Daily returns
            config: System configuration

        Returns:
            Health metrics
        """
        # Calculate health metrics
        initial_capital = config.get("initial_capital", 100000)
        current_return = ((capital - initial_capital) / initial_capital) * 100

        # Check for recent losses
        recent_losses = 0
        if len(daily_returns) >= 5:
            recent_returns = daily_returns[-5:]
            recent_losses = sum(1 for r in recent_returns if r.get("pnl", 0) < 0)

        # Position concentration
        total_position_value = sum(
            [abs(shares * 100) for shares in positions.values()]
        )  # Placeholder price
        concentration = (total_position_value / capital) * 100 if capital > 0 else 0

        health_metrics = {
            "capital_health": "good" if current_return >= 0 else "warning",
            "recent_performance": "good" if recent_losses <= 2 else "warning",
            "position_concentration": "good" if concentration <= 50 else "warning",
            "system_status": "healthy",
            "metrics": {
                "current_return_pct": current_return,
                "recent_losses": recent_losses,
                "position_concentration_pct": concentration,
                "position_count": len(positions),
            },
        }

        # Determine overall health
        if current_return < -5 or recent_losses >= 4 or concentration > 80:
            health_metrics["system_status"] = "critical"
        elif current_return < 0 or recent_losses >= 3 or concentration > 60:
            health_metrics["system_status"] = "warning"

        self.logger.info(f"System health: {health_metrics['system_status']}")

        return health_metrics

    def export_to_csv(self, data: list[dict], filename: str, index: bool = False) -> str:
        """
        Export data to CSV format.

        Args:
            data: Data to export
            filename: Output filename
            index: Whether to include index

        Returns:
            Path to exported file
        """
        if not data:
            self.logger.warning(f"No data to export for {filename}")
            return ""

        df = pd.DataFrame(data)
        filepath = Path(self.output_dir) / filename

        df.to_csv(filepath, index=index)
        self.logger.info(f"Data exported to {filepath}")

        return str(filepath)

    def create_summary_dashboard(
        self,
        capital: float,
        positions: dict,
        daily_returns: list[dict],
        trade_history: list[dict],
        regime_history: list[dict],
        config: dict,
    ) -> str:
        """
        Create summary dashboard data.

        Args:
            capital: Current capital
            positions: Current positions
            daily_returns: Daily returns
            trade_history: Trade history
            regime_history: Regime history
            config: System configuration

        Returns:
            Path to dashboard data
        """
        # Calculate key metrics
        initial_capital = config.get("initial_capital", 100000)
        total_return = ((capital - initial_capital) / initial_capital) * 100

        # Recent performance
        recent_returns = daily_returns[-30:] if len(daily_returns) >= 30 else daily_returns
        recent_pnl = sum([r.get("pnl", 0) for r in recent_returns])

        # Trading activity
        today_trades = len([t for t in trade_history if t.get("date") == date_class.today()])
        week_trades = len(
            [t for t in trade_history if t.get("date") >= date_class.today() - pd.Timedelta(days=7)]
        )

        # Current regime
        current_regime = (
            regime_history[-1].get("regime", "unknown") if regime_history else "unknown"
        )

        dashboard_data = {
            "summary": {
                "capital": capital,
                "total_return_pct": total_return,
                "recent_pnl": recent_pnl,
                "position_count": len(positions),
                "current_regime": current_regime,
            },
            "activity": {
                "today_trades": today_trades,
                "week_trades": week_trades,
                "total_trades": len(trade_history),
            },
            "positions": positions,
            "recent_performance": {
                "daily_returns": [r.get("pnl", 0) for r in recent_returns],
                "dates": [r.get("date", "").isoformat() for r in recent_returns],
            },
        }

        # Save dashboard data
        filename = f"dashboard_data_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.save_snapshot(dashboard_data, filename)

        return filepath
