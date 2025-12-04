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
Market-Adaptive Strategy Selection
Analyzes market regimes and selects optimal strategies based on performance metrics
"""


import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.learning.selector import BanditSelector, StrategyContext
from core.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Market-adaptive strategy selection based on regime analysis and performance metrics.
    """

    def __init__(self, config: dict):
        """
        Initialize strategy selector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.regime_detector = RegimeDetector(lookback_period=252)
        self.performance_dir = Path("results")
        self.strategies = self._load_available_strategies()
        self.regime_performance = {}
        # ML selector
        self.ml_enabled = bool(self.config.get("ml_selector", {}).get("enabled", False))
        epsilon = float(self.config.get("ml_selector", {}).get("epsilon", 0.1))
        self.bandit = BanditSelector(epsilon=epsilon) if self.ml_enabled else None

        logger.info("Initialized StrategySelector")

    def _load_available_strategies(self) -> dict[str, dict]:
        """Load available strategies and their configurations from config file."""
        from pathlib import Path

        import yaml

        config_path = Path("config/strategies.yaml")
        if not config_path.exists():
            logger.warning(f"Strategy config not found at {config_path}, using defaults")
            return self._get_default_strategies()

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            strategies = config.get("strategies", {})
            if not strategies:
                logger.warning("No strategies found in config, using defaults")
                return self._get_default_strategies()

            logger.info(f"Loaded {len(strategies)} strategies from config")
            return strategies

        except Exception as e:
            logger.error(f"Failed to load strategy config: {e}, using defaults")
            return self._get_default_strategies()

    def _get_default_strategies(self) -> dict[str, dict]:
        """Fallback default strategies if config loading fails."""
        return {
            "regime_aware_ensemble": {
                "name": "Regime-Aware Ensemble",
                "description": "Adaptive ensemble based on market regime detection",
                "default_params": {
                    "combination_method": "rolling_ic",
                    "confidence_threshold": 0.3,
                    "use_regime_switching": True,
                },
            },
            "momentum": {
                "name": "Momentum Strategy",
                "description": "Trend-following momentum strategy",
                "default_params": {
                    "lookback_period": 20,
                    "threshold": 0.02,
                },
            },
            "mean_reversion": {
                "name": "Mean Reversion Strategy",
                "description": "Mean reversion strategy for ranging markets",
                "default_params": {
                    "lookback_period": 20,
                    "std_dev_threshold": 2.0,
                },
            },
            "sma_crossover": {
                "name": "SMA Crossover",
                "description": "Simple moving average crossover strategy",
                "default_params": {
                    "fast_period": 10,
                    "slow_period": 50,
                },
            },
            "ensemble_basic": {
                "name": "Basic Ensemble",
                "description": "Basic ensemble of multiple strategies",
                "default_params": {
                    "combination_method": "equal_weight",
                    "confidence_threshold": 0.3,
                },
            },
        }

    def select_best_strategy(self, market_data: pd.DataFrame) -> tuple[str, dict, float]:
        """
        Select the best strategy for current market conditions.

        Args:
            market_data: Current market data

        Returns:
            Tuple of (strategy_name, strategy_params, expected_sharpe)
        """
        try:
            # Detect current market regime
            regime_name, confidence, regime_params = self.regime_detector.detect_regime(market_data)

            # Get regime-specific performance metrics
            regime_metrics = self._get_regime_performance_metrics(regime_name)

            # Calculate market volatility
            volatility = self._calculate_market_volatility(market_data)

            # Build learning context
            context = StrategyContext(
                regime=regime_name,
                vol_bin=int(np.digitize([volatility], [0.01, 0.02, 0.03, 0.05])[0]),
                trend_strength=float(getattr(regime_params, "trend_strength", 0.0)),
                liquidity=float(market_data.get("Volume", pd.Series([1])).tail(20).mean()),
                spread_bps=10.0,
                time_bucket=int(pd.Timestamp.now().hour),
                corr_cluster=0,
            )

            # Score each strategy based on regime performance and current conditions
            strategy_scores: dict[str, float] = {}
            for strategy_name, _ in self.strategies.items():
                score = self._calculate_strategy_score(
                    strategy_name,
                    regime_name,
                    regime_metrics,
                    volatility,
                    confidence,
                )
                strategy_scores[strategy_name] = score

            # Optionally choose with bandit
            if self.ml_enabled and self.bandit is not None:
                candidates = list(self.strategies.keys())
                strategy_name = self.bandit.recommend(context, candidates)
                expected_sharpe = strategy_scores.get(strategy_name, 0.5)
            else:
                best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
                strategy_name, expected_sharpe = best_strategy

            # Get strategy parameters
            strategy_params = self._get_optimized_params(strategy_name, regime_name, volatility)

            logger.info(
                f"Selected strategy: {strategy_name} (expected Sharpe: {expected_sharpe:.3f}) for regime: {regime_name}"
            )

            return strategy_name, strategy_params, expected_sharpe

        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            # Fallback to regime-aware ensemble
            return (
                "regime_ensemble",
                self.strategies["regime_ensemble"]["default_params"],
                0.5,
            )

    def _get_regime_performance_metrics(self, regime_name: str) -> dict[str, float]:
        """
        Get performance metrics for each strategy in the given regime.

        Args:
            regime_name: Market regime name

        Returns:
            Dictionary of strategy performance metrics
        """
        try:
            # Load performance data from results directory
            performance_file = self.performance_dir / "strategy_performance.json"

            if performance_file.exists():
                with open(performance_file) as f:
                    performance_data = json.load(f)

                regime_data = performance_data.get(regime_name, {})
                return regime_data
            else:
                # Return default metrics if no performance data available
                return self._get_default_regime_metrics(regime_name)

        except Exception as e:
            logger.error(f"Error loading regime performance metrics: {e}")
            return self._get_default_regime_metrics(regime_name)

    def _get_default_regime_metrics(self, regime_name: str) -> dict[str, float]:
        """Get default performance metrics based on regime characteristics."""
        default_metrics = {
            "trend": {
                "regime_ensemble": 0.8,
                "momentum": 0.7,
                "mean_reversion": 0.3,
                "sma": 0.6,
                "ensemble": 0.5,
            },
            "chop": {
                "regime_ensemble": 0.6,
                "momentum": 0.3,
                "mean_reversion": 0.8,
                "sma": 0.4,
                "ensemble": 0.5,
            },
            "volatile": {
                "regime_ensemble": 0.7,
                "momentum": 0.4,
                "mean_reversion": 0.6,
                "sma": 0.3,
                "ensemble": 0.4,
            },
        }

        return default_metrics.get(regime_name, default_metrics["chop"])

    def _calculate_market_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate current market volatility."""
        try:
            if len(market_data) < 20:
                return 0.0

            returns = market_data["Close"].pct_change().dropna()
            return returns.std()

        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            return 0.0

    def _calculate_strategy_score(
        self,
        strategy_name: str,
        regime_name: str,
        regime_metrics: dict[str, float],
        volatility: float,
        confidence: float,
    ) -> float:
        """
        Calculate strategy score based on multiple factors.

        Args:
            strategy_name: Name of the strategy
            regime_name: Current market regime
            regime_metrics: Performance metrics for the regime
            volatility: Current market volatility
            confidence: Regime detection confidence

        Returns:
            Strategy score (expected Sharpe ratio)
        """
        try:
            # Base score from regime performance
            base_score = regime_metrics.get(strategy_name, 0.5)

            # Adjust for volatility
            volatility_adjustment = self._calculate_volatility_adjustment(strategy_name, volatility)

            # Adjust for regime confidence
            confidence_adjustment = 1.0 + (confidence - 0.5) * 0.2  # ±10% adjustment

            # Adjust for strategy-specific factors
            strategy_adjustment = self._calculate_strategy_adjustment(
                strategy_name, regime_name, volatility
            )

            # Calculate final score
            final_score = (
                base_score * volatility_adjustment * confidence_adjustment * strategy_adjustment
            )

            # Ensure reasonable bounds
            final_score = np.clip(final_score, 0.1, 2.0)

            return final_score

        except Exception as e:
            logger.error(f"Error calculating strategy score: {e}")
            return 0.5

    def _calculate_volatility_adjustment(self, strategy_name: str, volatility: float) -> float:
        """Calculate volatility adjustment for strategy."""
        # High volatility strategies perform better in volatile markets
        if strategy_name in ["mean_reversion", "regime_aware_ensemble"]:
            return 1.0 + volatility * 2.0  # Boost in high volatility
        elif strategy_name in ["momentum", "sma_crossover"]:
            return 1.0 - volatility * 1.5  # Reduce in high volatility
        else:
            return 1.0

    def _calculate_strategy_adjustment(
        self, strategy_name: str, regime_name: str, volatility: float
    ) -> float:
        """Calculate strategy-specific adjustments."""
        adjustments = {
            "regime_aware_ensemble": 1.1,  # Slight boost for adaptive strategy
            "momentum": 1.0 if regime_name == "trend" else 0.8,
            "mean_reversion": 1.0 if regime_name == "chop" else 0.7,
            "sma_crossover": 0.9,  # Slight penalty for simplicity
            "ensemble_basic": 0.95,  # Slight penalty for basic ensemble
        }

        return adjustments.get(strategy_name, 1.0)

    def _get_optimized_params(
        self, strategy_name: str, regime_name: str, volatility: float
    ) -> dict:
        """Get optimized parameters for the selected strategy."""
        base_params = self.strategies[strategy_name]["default_params"].copy()

        # Adjust parameters based on regime and volatility
        if strategy_name == "regime_aware_ensemble":
            if regime_name == "trend":
                base_params["trend_following_weight"] = 0.7
                base_params["mean_reversion_weight"] = 0.3
            elif regime_name == "chop":
                base_params["trend_following_weight"] = 0.3
                base_params["mean_reversion_weight"] = 0.7
            else:  # volatile
                base_params["trend_following_weight"] = 0.5
                base_params["mean_reversion_weight"] = 0.5

            # Adjust confidence threshold based on volatility
            if volatility > 0.03:  # High volatility
                base_params["confidence_threshold"] = 0.4
            else:
                base_params["confidence_threshold"] = 0.3

        elif strategy_name == "momentum":
            # Adjust lookback period based on volatility
            if volatility > 0.03:
                base_params["lookback_period"] = 15  # Shorter lookback in high vol
            else:
                base_params["lookback_period"] = 20

        elif strategy_name == "mean_reversion":
            # Adjust std_dev_threshold based on volatility
            if volatility > 0.03:
                base_params["std_dev_threshold"] = 2.5  # Higher threshold in high vol
            else:
                base_params["std_dev_threshold"] = 2.0

        elif strategy_name == "sma_crossover":
            # Adjust periods based on volatility
            if volatility > 0.03:
                base_params["short_period"] = 8
                base_params["long_period"] = 40
            else:
                base_params["short_period"] = 10
                base_params["long_period"] = 50

        return base_params

    def update_performance_data(
        self, strategy_name: str, regime_name: str, performance_metrics: dict
    ):
        """
        Update performance data for strategy selection.

        Args:
            strategy_name: Name of the strategy
            regime_name: Market regime
            performance_metrics: Performance metrics dictionary
        """
        try:
            performance_file = self.performance_dir / "strategy_performance.json"

            # Load existing data
            if performance_file.exists():
                with open(performance_file) as f:
                    performance_data = json.load(f)
            else:
                performance_data = {}

            # Update regime data
            if regime_name not in performance_data:
                performance_data[regime_name] = {}

            # Update strategy performance
            sharpe_ratio = performance_metrics.get("sharpe_ratio", 0.0)
            performance_data[regime_name][strategy_name] = sharpe_ratio

            # Save updated data
            with open(performance_file, "w") as f:
                json.dump(performance_data, f, indent=2)

            logger.info(f"Updated performance data for {strategy_name} in {regime_name} regime")

        except Exception as e:
            logger.error(f"Error updating performance data: {e}")

    def update_online(self, market_data: pd.DataFrame, strategy_name: str, reward: float) -> None:
        """Update ML selector with realized reward if enabled."""
        if not self.ml_enabled or self.bandit is None:
            return
        try:
            regime_name, confidence, regime_params = self.regime_detector.detect_regime(market_data)
            volatility = self._calculate_market_volatility(market_data)
            context = StrategyContext(
                regime=regime_name,
                vol_bin=int(np.digitize([volatility], [0.01, 0.02, 0.03, 0.05])[0]),
                trend_strength=float(getattr(regime_params, "trend_strength", 0.0)),
                liquidity=float(market_data.get("Volume", pd.Series([1])).tail(20).mean()),
                spread_bps=10.0,
                time_bucket=int(pd.Timestamp.now().hour),
                corr_cluster=0,
            )
            self.bandit.update(context, strategy_name, float(reward))
        except Exception as e:
            logger.error(f"Error updating ML selector: {e}")

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get summary of available strategies and their characteristics."""
        summary = {
            "available_strategies": list(self.strategies.keys()),
            "strategy_details": self.strategies,
            "last_update": datetime.now().isoformat(),
        }

        # Add performance data if available
        performance_file = self.performance_dir / "strategy_performance.json"
        if performance_file.exists():
            try:
                with open(performance_file) as f:
                    summary["performance_data"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance data: {e}")

        return summary
