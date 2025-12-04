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
Core Paper Trading Engine
Handles the main trading logic and state management
"""


import json
from datetime import date as date_class
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from brokers.data_provider import IBKRDataProvider
from brokers.ibkr_broker import IBKRConfig
from core.enhanced_logging import TradingLogger
from core.feature_reweighter import AdaptiveFeatureEngine, FeatureReweighter
from core.notifications import DiscordConfig, DiscordNotifier
from core.performance import GrowthTargetCalculator
from core.regime_detector import RegimeDetector
from core.strategy_selector import StrategySelector
from core.utils import ensure_directories
from strategies.factory import strategy_factory
from strategies.regime_aware_ensemble import (
    RegimeAwareEnsembleParams,
    RegimeAwareEnsembleStrategy,
)


class PaperTradingEngine:
    """
    Core paper trading engine with regime detection and adaptive features.
    """

    def __init__(
        self,
        config_file: str = "config/enhanced_paper_trading_config.json",
        profile_file: str = None,
    ):
        """
        Initialize paper trading engine.

        Args:
            config_file: Configuration file path
            profile_file: Profile configuration file path (optional)
        """
        self.config_file = config_file
        self.profile_file = profile_file

        # Setup enhanced logging
        self.trading_logger = TradingLogger()
        self.logger = self.trading_logger.main_logger

        # Load config after logging is set up
        self.config = self.load_config()

        # Load profile configuration if provided
        if profile_file and Path(profile_file).exists():
            self.load_profile_config(profile_file)

        # Initialize components
        self._initialize_components()

        # Paper trading state
        self.capital = self.config.get("initial_capital", 100000)
        self.positions = {}
        self.trade_history = []
        self.daily_returns = []
        self.regime_history = []
        self.last_prices = {}  # Track current prices for risk management

        # Performance tracking
        self.performance_metrics = {}

        # Log system startup
        self._log_startup()

    def _initialize_components(self):
        """Initialize all system components."""
        # Initialize kill switches
        self.kill_switches = self._initialize_kill_switches()

        # Initialize Discord notifications
        self.discord_notifier = self._setup_discord_notifications()

        # Initialize IBKR data provider
        self.use_ibkr = self.config.get("use_ibkr", False)
        if self.use_ibkr:
            self.ibkr_config = IBKRConfig()
            self.data_provider = IBKRDataProvider(
                config=self.ibkr_config, use_cache=True, fallback_to_yfinance=True
            )
            self.logger.info("Initialized IBKR data provider")
        else:
            self.data_provider = None
            self.logger.info("Using yfinance for data (IBKR disabled)")

        # Initialize other components
        self.regime_detector = RegimeDetector(lookback_period=252)
        self.feature_reweighter = FeatureReweighter(rolling_window=60, reweight_frequency=20)
        self.adaptive_engine = AdaptiveFeatureEngine(reweighter=self.feature_reweighter)

        # Initialize growth target calculator
        self.growth_calculator = GrowthTargetCalculator(self.config)

        # Initialize strategy selector
        self.strategy_selector = StrategySelector(self.config)

        # Initialize strategy
        self.strategy = self._initialize_strategy()

        # Initialize risk management
        self.risk_guardrails = self._initialize_risk_guardrails()

        # Initialize telemetry
        self.telemetry = self._initialize_telemetry()

        self.logger.info("All components initialized successfully")

    def _initialize_kill_switches(self) -> dict[str, Any]:
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

    def _setup_discord_notifications(self) -> DiscordNotifier | None:
        """Setup Discord notifications if configured."""
        notifications_config = self.config.get("notifications", {})

        if notifications_config.get("discord_enabled", False):
            webhook_url = notifications_config.get("webhook_url", "")
            bot_name = notifications_config.get("bot_name", "Trading Bot")

            if webhook_url:
                discord_config = DiscordConfig(webhook_url=webhook_url, bot_name=bot_name)
                return DiscordNotifier(discord_config)
            else:
                self.logger.warning("Discord enabled but no webhook URL provided")

        return None

    def _initialize_risk_guardrails(self):
        """Initialize risk management guardrails."""
        try:
            from core.risk.guardrails import RiskGuardrails

            return RiskGuardrails(self.config)
        except ImportError:
            self.logger.warning("RiskGuardrails not available, using basic risk checks")
            return None

    def _initialize_telemetry(self):
        """Initialize telemetry system."""
        try:
            from core.telemetry.snapshot import TelemetrySnapshot

            return TelemetrySnapshot()
        except ImportError:
            self.logger.warning("TelemetrySnapshot not available")
            return None

    def _log_startup(self):
        """Log system startup information."""
        startup_info = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "initial_capital": self.capital,
            "strategies": [self.strategy.__class__.__name__],
            "symbols": self.config.get("symbols", []),
            "config_file": self.config_file,
        }

        self.logger.info(
            f"🚀 System started: ${self.capital:,} capital, {len(startup_info['strategies'])} strategies active"
        )

        # Log to system telemetry
        if hasattr(self.trading_logger, "log_event"):
            self.trading_logger.log_event("STARTUP", startup_info)

    def load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_file) as f:
                config = json.load(f)

            self.logger.info(f"Loaded config from {self.config_file}")
            return config

        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            raise

    def load_profile_config(self, profile_file: str):
        """Load profile configuration."""
        try:
            with open(profile_file) as f:
                profile_config = json.load(f)

            # Merge profile config with main config
            self.config.update(profile_config)
            self.logger.info(f"Loaded profile config from {profile_file}")

        except FileNotFoundError:
            self.logger.warning(f"Profile file not found: {profile_file}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in profile file: {e}")

    def run_trading_cycle(self, current_date: date_class = None) -> dict[str, Any]:
        """
        Run a complete trading cycle for the current date.

        Args:
            current_date: Date to run trading cycle for (defaults to today)

        Returns:
            Dictionary with cycle results
        """
        if current_date is None:
            current_date = date_class.today()

        self.logger.info(f"🔄 Running trading cycle for {current_date}")

        try:
            # Get market data
            market_data = self._get_market_data(current_date)
            if not market_data:
                self.logger.warning("No market data available")
                return {"status": "no_data"}

            # Detect market regime
            regime_info = self._detect_regime(market_data)

            # Select optimal strategy for current market conditions
            strategy_info = self._select_optimal_strategy(market_data, regime_info)

            # Generate trading signals
            signals = self._generate_signals(market_data, regime_info, strategy_info)

            # Execute trades
            trade_results = self._execute_trades(signals, market_data, current_date)

            # Update portfolio
            self._update_portfolio(market_data, current_date)

            # Record daily return
            self._record_daily_return(current_date)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()

            # Update strategy selector with performance data
            self._update_strategy_performance(strategy_info, performance_metrics)

            # Save results
            self._save_results(current_date, performance_metrics)

            # Send notifications
            self._send_notifications(current_date, performance_metrics)

            return {
                "status": "success",
                "date": current_date,
                "regime": regime_info.get("regime_name", "unknown"),
                "strategy": strategy_info.get("strategy_name", "unknown"),
                "expected_sharpe": strategy_info.get("expected_sharpe", 0.0),
                "trades_executed": len(trade_results),
                "performance_metrics": performance_metrics,
            }

        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            return {"status": "error", "error": str(e)}

    def _get_market_data(self, current_date: date_class) -> dict[str, pd.DataFrame] | None:
        """Get market data for all symbols."""
        symbols = self.config.get("symbols", ["SPY"])
        market_data = {}

        # Get DataSanity wrapper for validation
        from core.data_sanity import get_data_sanity_wrapper

        data_sanity = get_data_sanity_wrapper()

        for symbol in symbols:
            try:
                if self.use_ibkr and self.data_provider:
                    # Use IBKR data provider (already uses DataSanity)
                    symbol_data = self.data_provider.get_historical_data(symbol, "1 M", "1 day")
                else:
                    # Use yfinance as fallback
                    ticker = yf.Ticker(symbol)
                    symbol_data = ticker.history(period="1mo")

                if not symbol_data.empty:
                    # Validate and repair data using DataSanity
                    clean_data = data_sanity.validate_dataframe(symbol_data, symbol)
                    market_data[symbol] = clean_data
                    self.logger.debug(
                        f"Loaded and validated data for {symbol}: {len(clean_data)} bars"
                    )
                else:
                    self.logger.warning(f"No data available for {symbol}")

            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")

        return market_data if market_data else None

    def _detect_regime(self, market_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Detect market regime using regime detector."""
        try:
            # Use the first symbol's data for regime detection
            first_symbol = list(market_data.keys())[0]
            symbol_data = market_data[first_symbol]

            regime_name, confidence, regime_params = self.regime_detector.detect_regime(symbol_data)

            regime_info = {
                "regime_name": regime_name,
                "confidence": confidence,
                "regime_params": regime_params,
                "detection_date": pd.Timestamp.now(),
            }

            # Record regime history
            self.regime_history.append(regime_info)

            self.logger.info(f"Detected regime: {regime_name} (confidence: {confidence:.2f})")
            return regime_info

        except Exception as e:
            self.logger.error(f"Error detecting regime: {e}")
            return {"regime_name": "unknown", "confidence": 0.0, "regime_params": None}

    def _select_optimal_strategy(
        self, market_data: dict[str, pd.DataFrame], regime_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Select optimal strategy for current market conditions."""
        try:
            # Use the first symbol's data for strategy selection
            first_symbol = list(market_data.keys())[0]
            symbol_data = market_data[first_symbol]

            # Select best strategy
            (
                strategy_name,
                strategy_params,
                expected_sharpe,
            ) = self.strategy_selector.select_best_strategy(symbol_data)

            # Check if we need to switch strategies
            current_strategy_name = self.strategy.__class__.__name__.lower().replace("strategy", "")

            if strategy_name != current_strategy_name:
                self.logger.info(
                    f"Switching strategy from {current_strategy_name} to {strategy_name}"
                )
                self.strategy = self._initialize_strategy_by_name(strategy_name, strategy_params)

            strategy_info = {
                "strategy_name": strategy_name,
                "strategy_params": strategy_params,
                "expected_sharpe": expected_sharpe,
                "regime_name": regime_info.get("regime_name", "unknown"),
                "selection_timestamp": pd.Timestamp.now(),
            }

            return strategy_info

        except Exception as e:
            self.logger.error(f"Error selecting optimal strategy: {e}")
            return {
                "strategy_name": "regime_aware_ensemble",
                "strategy_params": {},
                "expected_sharpe": 0.5,
                "regime_name": "unknown",
                "selection_timestamp": pd.Timestamp.now(),
            }

    def _initialize_strategy_by_name(self, strategy_name: str, strategy_params: dict) -> Any:
        """Initialize strategy by name with given parameters."""
        try:
            if strategy_name == "regime_aware_ensemble":
                from strategies.regime_aware_ensemble import (
                    RegimeAwareEnsembleParams,
                    RegimeAwareEnsembleStrategy,
                )

                params = RegimeAwareEnsembleParams(**strategy_params)
                return RegimeAwareEnsembleStrategy(params)
            else:
                return strategy_factory.create_strategy(strategy_name, strategy_params)
        except Exception as e:
            self.logger.error(f"Error initializing strategy {strategy_name}: {e}")
            # Fallback to default regime-aware ensemble
            from strategies.regime_aware_ensemble import (
                RegimeAwareEnsembleParams,
                RegimeAwareEnsembleStrategy,
            )

            params = RegimeAwareEnsembleParams()
            return RegimeAwareEnsembleStrategy(params)

    def _generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
        regime_info: dict[str, Any],
        strategy_info: dict[str, Any],
    ) -> dict[str, float]:
        """Generate trading signals for all symbols using the selected strategy."""
        signals = {}

        try:
            strategy_name = strategy_info.get("strategy_name", "regime_aware_ensemble")
            expected_sharpe = strategy_info.get("expected_sharpe", 0.5)

            self.logger.info(
                f"Generating signals using {strategy_name} (expected Sharpe: {expected_sharpe:.3f})"
            )

            for symbol, symbol_data in market_data.items():
                # Generate signals using the selected strategy
                signal_series = self.strategy.generate_signals(symbol_data)

                # Get the latest signal
                if not signal_series.empty:
                    latest_signal = signal_series.iloc[-1]
                    signals[symbol] = float(latest_signal)
                else:
                    signals[symbol] = 0.0

            self.logger.debug(f"Generated signals: {signals}")
            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {}

    def _execute_trades(
        self,
        signals: dict[str, float],
        market_data: dict[str, pd.DataFrame],
        current_date: date_class,
    ) -> list[dict[str, Any]]:
        """Execute trades based on signals with improved risk management."""
        executed_trades = []

        try:
            # Calculate available capital (excluding current positions)
            available_capital = self.capital
            current_positions_value = 0.0

            for symbol, position in self.positions.items():
                if symbol in market_data and not market_data[symbol].empty:
                    current_price = market_data[symbol]["Close"].iloc[-1]
                    current_positions_value += abs(position) * current_price

            # Reserve some capital for safety
            safety_buffer = 0.1  # 10% safety buffer
            available_capital = available_capital * (1 - safety_buffer)

            # Calculate maximum position size per symbol
            max_position_pct = self.config.get("risk_params", {}).get("max_position_size", 0.1)
            available_capital * max_position_pct

            for symbol, signal in signals.items():
                if abs(signal) < 0.1:  # Skip small signals
                    continue

                if symbol not in market_data:
                    continue

                symbol_data = market_data[symbol]
                if symbol_data.empty:
                    continue

                current_price = symbol_data["Close"].iloc[-1]

                # Calculate signal strength
                signal_strength = min(abs(signal), 1.0)  # Cap at 1.0

                # Calculate dynamic position size using growth target calculator
                position_size_pct = self.growth_calculator.calculate_dynamic_position_size(
                    signal_strength=signal_strength,
                    current_capital=self.capital,
                    symbol_volatility=self._calculate_symbol_volatility(symbol_data),
                    portfolio_volatility=self._calculate_portfolio_volatility(),
                )

                position_value = position_size_pct * available_capital

                # Ensure we don't exceed available capital
                if position_value > available_capital:
                    position_value = available_capital * 0.8  # Use 80% of remaining capital

                shares = int(position_value / current_price)

                if shares == 0:
                    continue

                # Validate position size against risk limits
                if not self._validate_position_size(
                    symbol, shares, current_price, available_capital
                ):
                    self.logger.warning(f"Position size validation failed for {symbol}")
                    continue

                # Determine trade direction
                action = "BUY" if signal > 0 else "SELL"

                # Execute trade
                trade = {
                    "date": current_date,
                    "symbol": symbol,
                    "action": action,
                    "shares": shares,
                    "price": current_price,
                    "value": shares * current_price,
                    "signal": signal,
                    "signal_strength": signal_strength,
                    "position_size_pct": position_size_pct,
                }

                # Update positions
                if action == "BUY":
                    if symbol not in self.positions:
                        self.positions[symbol] = 0
                    self.positions[symbol] += shares
                else:
                    if symbol not in self.positions:
                        self.positions[symbol] = 0
                    self.positions[symbol] -= shares

                # Update available capital
                available_capital -= shares * current_price

                # Record trade
                self.trade_history.append(trade)
                executed_trades.append(trade)

                self.logger.info(
                    f"Executed {action} {shares} shares of {symbol} at ${current_price:.2f} (signal: {signal:.3f}, size: {position_size_pct:.1%})"
                )

        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")

        return executed_trades

    def _validate_position_size(
        self, symbol: str, shares: int, price: float, available_capital: float
    ) -> bool:
        """Validate position size against risk limits."""
        try:
            position_value = shares * price

            # Check against maximum position size
            max_position_pct = self.config.get("risk_params", {}).get("max_position_size", 0.1)
            max_position_value = self.capital * max_position_pct

            if position_value > max_position_value:
                self.logger.warning(
                    f"Position value ${position_value:.2f} exceeds max ${max_position_value:.2f}"
                )
                return False

            # Check against available capital
            if position_value > available_capital:
                self.logger.warning(
                    f"Position value ${position_value:.2f} exceeds available capital ${available_capital:.2f}"
                )
                return False

            # Check against maximum gross exposure
            current_gross_exposure = 0.0
            for sym, pos in self.positions.items():
                if sym in self.last_prices:
                    current_gross_exposure += abs(pos) * self.last_prices[sym]

            max_gross_exposure_pct = self.config.get("risk_params", {}).get(
                "max_gross_exposure_pct", 0.35
            )
            max_gross_exposure = self.capital * max_gross_exposure_pct

            if current_gross_exposure + position_value > max_gross_exposure:
                self.logger.warning(
                    f"Gross exposure would exceed limit: ${current_gross_exposure + position_value:.2f} > ${max_gross_exposure:.2f}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating position size: {e}")
            return False

    def _update_portfolio(self, market_data: dict[str, pd.DataFrame], current_date: date_class):
        """Update portfolio with current market data."""
        try:
            # Update position values and track last prices
            total_value = self.capital

            for symbol, position in self.positions.items():
                if symbol in market_data and not market_data[symbol].empty:
                    current_price = market_data[symbol]["Close"].iloc[-1]
                    self.last_prices[symbol] = current_price  # Track for risk management
                    position_value = position * current_price
                    total_value += position_value

            # Update capital (simplified - in real system would track cash separately)
            self.capital = total_value

        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")

    def _record_daily_return(self, current_date: date_class):
        """Record daily return for performance tracking."""
        try:
            # Calculate daily return
            daily_return_pct = 0.0
            if len(self.daily_returns) > 0:
                prev_value = self.daily_returns[-1]["portfolio_value"]
                daily_return_pct = (
                    (self.capital - prev_value) / prev_value if prev_value > 0 else 0.0
                )

            daily_return = {
                "date": current_date,
                "portfolio_value": self.capital,
                "positions": self.positions.copy(),
                "return_pct": daily_return_pct,
            }

            self.daily_returns.append(daily_return)

            # Update growth target calculator
            self.growth_calculator.update_performance(daily_return_pct, self.capital)

            # Log growth metrics
            growth_metrics = self.growth_calculator.get_growth_metrics()
            self.logger.info(
                "Daily return: %.4f | Avg: %.4f | Objective score: %.4f",
                daily_return_pct,
                growth_metrics.get("avg_daily_return", 0.0),
                growth_metrics.get("objective_score", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Error recording daily return: {e}")

    def _calculate_performance_metrics(self) -> dict[str, float]:
        """Calculate performance metrics."""
        try:
            if len(self.daily_returns) < 2:
                return {
                    "total_return_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown_pct": 0.0,
                }

            # Calculate returns
            returns = []
            for i in range(1, len(self.daily_returns)):
                prev_value = self.daily_returns[i - 1]["portfolio_value"]
                curr_value = self.daily_returns[i]["portfolio_value"]
                daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0.0
                returns.append(daily_return)

            returns_series = pd.Series(returns)

            # Calculate metrics
            total_return_pct = (
                (self.capital / self.config.get("initial_capital", 100000)) - 1
            ) * 100
            sharpe_ratio = (
                returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0.0
            )

            # Calculate max drawdown
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown_pct = abs(drawdown.min()) * 100

            return {
                "total_return_pct": total_return_pct,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "total_trades": len(self.trade_history),
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
            }

    def _save_results(self, current_date: date_class, performance_metrics: dict[str, float]):
        """Save trading results."""
        try:
            # Ensure results directory exists
            results_dir = self.config.get("performance_tracking", {}).get("results_dir", "results")
            ensure_directories(results_dir)

            # Save daily results
            daily_results = {
                "date": current_date.isoformat(),
                "portfolio_value": self.capital,
                "performance_metrics": performance_metrics,
                "positions": self.positions,
                "trades_today": len([t for t in self.trade_history if t["date"] == current_date]),
            }

            results_file = f"{results_dir}/daily_results_{current_date}.json"
            with open(results_file, "w") as f:
                json.dump(daily_results, f, indent=2, default=str)

            self.logger.debug(f"Saved daily results to {results_file}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def _send_notifications(self, current_date: date_class, performance_metrics: dict[str, float]):
        """Send notifications about trading results."""
        try:
            if self.discord_notifier:
                message = f"📊 Trading Update - {current_date}\n"
                message += f"Portfolio Value: ${self.capital:,.2f}\n"
                message += f"Total Return: {performance_metrics.get('total_return_pct', 0):.2f}%\n"
                message += f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}\n"
                message += f"Max Drawdown: {performance_metrics.get('max_drawdown_pct', 0):.2f}%"

                self.discord_notifier.send_message(message)

        except Exception as e:
            self.logger.error(f"Error sending notifications: {e}")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        return {
            "total_return_pct": self.performance_metrics.get("total_return_pct", 0.0),
            "sharpe_ratio": self.performance_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown_pct": self.performance_metrics.get("max_drawdown_pct", 0.0),
            "total_trades": len(self.trade_history),
            "current_capital": self.capital,
            "initial_capital": self.config.get("initial_capital", 100000),
        }

    def get_positions(self) -> dict[str, int]:
        """Get current positions."""
        return self.positions.copy()

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Get trade history."""
        return self.trade_history.copy()

    def get_daily_returns(self) -> list[dict[str, Any]]:
        """Get daily returns."""
        return self.daily_returns.copy()

    def get_regime_history(self) -> list[dict[str, Any]]:
        """Get regime detection history."""
        return self.regime_history.copy()

    def shutdown(self):
        """Shutdown the trading engine."""
        self.logger.info("Shutting down trading engine")

        # Save final results
        final_metrics = self._calculate_performance_metrics()
        self._save_results(date_class.today(), final_metrics)

        # Send final notification
        if self.discord_notifier:
            final_message = (
                f"🔚 Trading Engine Shutdown\nFinal Portfolio Value: ${self.capital:,.2f}"
            )
            self.discord_notifier.send_message(final_message)

        self.logger.info("Trading engine shutdown complete")

    def _initialize_strategy(self):
        """Initialize the trading strategy."""
        strategy_name = self.config.get("strategy", "regime_aware_ensemble")
        strategy_params = self.config.get("strategy_params", {}).get(strategy_name, {})

        if strategy_name == "regime_aware_ensemble":
            params = RegimeAwareEnsembleParams(**strategy_params)
            return RegimeAwareEnsembleStrategy(params)
        else:
            return strategy_factory.create_strategy(strategy_name, strategy_params)

    def _update_strategy_performance(
        self, strategy_info: dict[str, Any], performance_metrics: dict[str, float]
    ):
        """Update the strategy selector with actual performance data."""
        try:
            strategy_name = strategy_info.get("strategy_name", "regime_aware_ensemble")
            regime_name = strategy_info.get("regime_name", "unknown")
            self.strategy_selector.update_performance_data(
                strategy_name, regime_name, performance_metrics
            )
            self.logger.info(
                f"Updated strategy selector with performance data for {strategy_name} in {regime_name} regime"
            )
        except Exception as e:
            self.logger.error(f"Error updating strategy selector performance: {e}")

    def _calculate_symbol_volatility(self, symbol_data: pd.DataFrame) -> float:
        """Calculate symbol-specific volatility."""
        try:
            if len(symbol_data) < 20:
                return 0.0

            returns = symbol_data["Close"].pct_change().dropna()
            return returns.std()
        except Exception as e:
            self.logger.error(f"Error calculating symbol volatility: {e}")
            return 0.0

    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility from recent returns."""
        try:
            if len(self.daily_returns) < 20:
                return 0.0

            # Get recent daily returns
            recent_returns = []
            for i in range(1, len(self.daily_returns)):
                prev_value = self.daily_returns[i - 1]["portfolio_value"]
                curr_value = self.daily_returns[i]["portfolio_value"]
                daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0.0
                recent_returns.append(daily_return)

            return np.std(recent_returns) if recent_returns else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
