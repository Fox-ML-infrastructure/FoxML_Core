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
Regime-Aware Ensemble Strategy
Blends trend-following and mean-reversion signals with regime-specific weighting
"""


import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.regime_detector import RegimeDetector, RegimeParams
from features.ensemble import SignalCombiner
from strategies.base import BaseStrategy, StrategyParams

logger = logging.getLogger(__name__)


@dataclass
class RegimeAwareEnsembleParams(StrategyParams):
    """Parameters for regime-aware ensemble strategy."""

    # Base parameters
    combination_method: str = "rolling_ic"  # rolling_ic, sharpe, ridge, voting
    confidence_threshold: float = 0.3

    # Regime-specific parameters
    use_regime_switching: bool = True
    regime_lookback: int = 252

    # Feature lookback adjustments
    trend_lookback_multiplier: float = 1.2
    chop_lookback_multiplier: float = 0.8
    volatile_lookback_multiplier: float = 0.6

    # Signal blending parameters
    trend_following_weight: float = 0.6
    mean_reversion_weight: float = 0.4

    # Rolling performance calculation
    rolling_window: int = 60
    min_periods: int = 30


class RegimeAwareEnsembleStrategy(BaseStrategy):
    """
    Regime-aware ensemble strategy that adapts to market conditions.
    """

    def __init__(self, params: RegimeAwareEnsembleParams):
        """
        Initialize regime-aware ensemble strategy.

        Args:
            params: Strategy parameters
        """
        super().__init__(params)
        self.params = params

        # Initialize regime detector
        self.regime_detector = RegimeDetector(lookback_period=params.regime_lookback)

        # Initialize signal combiners for different signal types (will be initialized with data later)
        self.trend_combiner = None
        self.mean_reversion_combiner = None
        self.breakout_combiner = None
        self.combination_method = params.combination_method

        # Performance tracking
        self.rolling_performance = {}
        self.regime_history = []

        logger.info(
            f"Initialized RegimeAwareEnsembleStrategy with {params.combination_method} combination"
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate regime-aware ensemble signals.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Signal series
        """
        if len(data) < self.params.min_periods:
            logger.warning(
                f"Insufficient data for signal generation: {len(data)} < {self.params.min_periods}"
            )
            return pd.Series(0.0, index=data.index)

        # Detect current regime
        regime_name, confidence, regime_params = self.regime_detector.detect_regime(data)
        self.regime_history.append(
            {"date": data.index[-1], "regime": regime_name, "confidence": confidence}
        )

        # Generate regime-adapted features
        features = self._generate_regime_adapted_features(data, regime_params)

        # Generate signals by type
        trend_signals = self._generate_trend_following_signals(data, features)
        mean_reversion_signals = self._generate_mean_reversion_signals(data, features)
        breakout_signals = self._generate_breakout_signals(data, features)

        # Blend signals based on regime
        blended_signals = self._blend_signals_by_regime(
            trend_signals, mean_reversion_signals, breakout_signals, regime_params
        )

        # Apply confidence filter
        filtered_signals = self._apply_confidence_filter(blended_signals, regime_params)

        # Update performance tracking
        self._update_performance_tracking(data, filtered_signals)

        logger.info(
            f"Generated regime-aware signals for {regime_name} regime (confidence: {confidence:.2f})"
        )

        return filtered_signals

    def _generate_regime_adapted_features(
        self, data: pd.DataFrame, regime_params: RegimeParams
    ) -> dict[str, pd.Series]:
        """Generate features adapted to the current regime."""
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        # Adjust lookback periods based on regime
        lookback_adjustment = regime_params.feature_lookback_adjustment

        features = {}

        # Trend-following features
        features.update(self._generate_trend_features(close, lookback_adjustment))

        # Mean reversion features
        features.update(self._generate_mean_reversion_features(close, lookback_adjustment))

        # Breakout features
        features.update(self._generate_breakout_features(high, low, close, lookback_adjustment))

        # Volatility features
        features.update(self._generate_volatility_features(close, volume, lookback_adjustment))

        # Regime-specific features
        features.update(self._generate_regime_specific_features(data, regime_params.regime_name))

        return features

    def _generate_trend_features(
        self, close: pd.Series, lookback_adjustment: float
    ) -> dict[str, pd.Series]:
        """Generate trend-following features."""
        features = {}

        # Adjust periods based on regime
        short_period = int(20 * lookback_adjustment)
        long_period = int(60 * lookback_adjustment)

        # Moving average crossovers
        short_ma = close.rolling(short_period).mean()
        long_ma = close.rolling(long_period).mean()

        features["ma_crossover"] = (short_ma - long_ma) / long_ma
        features["ma_slope"] = short_ma.pct_change(5)

        # Momentum indicators
        features["momentum_5"] = close.pct_change(5)
        features["momentum_10"] = close.pct_change(10)
        features["momentum_20"] = close.pct_change(20)

        # Trend strength
        features["trend_strength"] = self._calculate_trend_strength(
            close, int(20 * lookback_adjustment)
        )

        return features

    def _generate_mean_reversion_features(
        self, close: pd.Series, lookback_adjustment: float
    ) -> dict[str, pd.Series]:
        """Generate mean reversion features."""
        features = {}

        # Adjust periods based on regime
        period = int(20 * lookback_adjustment)

        # Distance from moving average
        ma = close.rolling(period).mean()
        features["ma_distance"] = (close - ma) / ma

        # RSI
        features["rsi"] = self._calculate_rsi(close, period)

        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(close, period)
        features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

        # Z-score
        features["zscore"] = (close - ma) / close.rolling(period).std()

        return features

    def _generate_breakout_features(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback_adjustment: float,
    ) -> dict[str, pd.Series]:
        """Generate breakout features."""
        features = {}

        # Adjust periods based on regime
        period = int(20 * lookback_adjustment)

        # Donchian channels
        upper_channel = high.rolling(period).max()
        lower_channel = low.rolling(period).min()

        features["donchian_position"] = (close - lower_channel) / (upper_channel - lower_channel)
        features["breakout_strength"] = (close - upper_channel) / close

        # Support/resistance levels
        features["resistance_break"] = (close - upper_channel) / close
        features["support_break"] = (close - lower_channel) / close

        return features

    def _generate_volatility_features(
        self, close: pd.Series, volume: pd.Series, lookback_adjustment: float
    ) -> dict[str, pd.Series]:
        """Generate volatility features."""
        features = {}

        # Adjust periods based on regime
        period = int(20 * lookback_adjustment)

        # Volatility
        returns = close.pct_change()
        features["volatility"] = returns.rolling(period).std()

        # Volume
        features["volume_ratio"] = volume / volume.rolling(period).mean()

        # ATR
        features["atr"] = self._calculate_atr(close, period)

        return features

    def _generate_regime_specific_features(
        self, data: pd.DataFrame, regime_name: str
    ) -> dict[str, pd.Series]:
        """Generate regime-specific features."""
        close = data["Close"]
        features = {}

        if regime_name == "trend":
            features["trend_consistency"] = self._calculate_trend_consistency(close)
            features["trend_persistence"] = self._calculate_trend_persistence(close)

        elif regime_name == "chop":
            features["oscillation_frequency"] = self._calculate_oscillation_frequency(close)
            features["mean_reversion_strength"] = self._calculate_mean_reversion_strength(close)

        elif regime_name == "volatile":
            features["volatility_regime"] = self._calculate_volatility_regime(close)
            features["volatility_persistence"] = self._calculate_volatility_persistence(close)

        return features

    def _generate_trend_following_signals(
        self, data: pd.DataFrame, features: dict[str, pd.Series]
    ) -> pd.Series:
        """Generate trend-following signals."""
        # Initialize combiner if needed
        if self.trend_combiner is None:
            self.trend_combiner = SignalCombiner(data["Close"])

        trend_features = {
            "ma_crossover": features.get("ma_crossover", pd.Series(0.0, index=data.index)),
            "momentum_5": features.get("momentum_5", pd.Series(0.0, index=data.index)),
            "momentum_10": features.get("momentum_10", pd.Series(0.0, index=data.index)),
            "trend_strength": features.get("trend_strength", pd.Series(0.0, index=data.index)),
            "trend_consistency": features.get(
                "trend_consistency", pd.Series(0.0, index=data.index)
            ),
        }

        # Add features to combiner
        for name, feature in trend_features.items():
            self.trend_combiner.add_feature(name, feature)

        # Compute weights and combine signals
        self.trend_combiner.compute_weights(method=self.combination_method)
        trend_signals, _ = self.trend_combiner.combine()

        return trend_signals

    def _generate_mean_reversion_signals(
        self, data: pd.DataFrame, features: dict[str, pd.Series]
    ) -> pd.Series:
        """Generate mean reversion signals."""
        # Initialize combiner if needed
        if self.mean_reversion_combiner is None:
            self.mean_reversion_combiner = SignalCombiner(data["Close"])

        mean_reversion_features = {
            "ma_distance": features.get("ma_distance", pd.Series(0.0, index=data.index)),
            "rsi": features.get("rsi", pd.Series(50.0, index=data.index)),
            "bb_position": features.get("bb_position", pd.Series(0.5, index=data.index)),
            "zscore": features.get("zscore", pd.Series(0.0, index=data.index)),
            "mean_reversion_strength": features.get(
                "mean_reversion_strength", pd.Series(0.0, index=data.index)
            ),
        }

        # Add features to combiner
        for name, feature in mean_reversion_features.items():
            self.mean_reversion_combiner.add_feature(name, feature)

        # Compute weights and combine signals
        self.mean_reversion_combiner.compute_weights(method=self.combination_method)
        mean_reversion_signals, _ = self.mean_reversion_combiner.combine()

        return mean_reversion_signals

    def _generate_breakout_signals(
        self, data: pd.DataFrame, features: dict[str, pd.Series]
    ) -> pd.Series:
        """Generate breakout signals."""
        # Initialize combiner if needed
        if self.breakout_combiner is None:
            self.breakout_combiner = SignalCombiner(data["Close"])

        breakout_features = {
            "donchian_position": features.get(
                "donchian_position", pd.Series(0.5, index=data.index)
            ),
            "breakout_strength": features.get(
                "breakout_strength", pd.Series(0.0, index=data.index)
            ),
            "resistance_break": features.get("resistance_break", pd.Series(0.0, index=data.index)),
            "support_break": features.get("support_break", pd.Series(0.0, index=data.index)),
        }

        # Add features to combiner
        for name, feature in breakout_features.items():
            self.breakout_combiner.add_feature(name, feature)

        # Compute weights and combine signals
        self.breakout_combiner.compute_weights(method=self.combination_method)
        breakout_signals, _ = self.breakout_combiner.combine()

        return breakout_signals

    def _blend_signals_by_regime(
        self,
        trend_signals: pd.Series,
        mean_reversion_signals: pd.Series,
        breakout_signals: pd.Series,
        regime_params: RegimeParams,
    ) -> pd.Series:
        """Blend signals based on regime-specific weights."""
        weights = regime_params.ensemble_weights

        # Get weights for each signal type
        trend_weight = weights.get("momentum", 0.3)
        mean_reversion_weight = weights.get("mean_reversion", 0.3)
        breakout_weight = weights.get("breakout", 0.2)
        ensemble_weight = weights.get("ensemble_basic", 0.2)

        # Normalize weights
        total_weight = trend_weight + mean_reversion_weight + breakout_weight + ensemble_weight
        if total_weight > 0:
            trend_weight /= total_weight
            mean_reversion_weight /= total_weight
            breakout_weight /= total_weight
            ensemble_weight /= total_weight

        # Blend signals
        blended_signals = (
            trend_weight * trend_signals
            + mean_reversion_weight * mean_reversion_signals
            + breakout_weight * breakout_signals
            + ensemble_weight * (trend_signals + mean_reversion_signals) / 2
        )

        return blended_signals

    def _apply_confidence_filter(
        self, signals: pd.Series, regime_params: RegimeParams
    ) -> pd.Series:
        """Apply confidence filter based on regime."""
        confidence_threshold = regime_params.confidence_threshold

        # Calculate signal confidence (absolute value)
        signal_confidence = signals.abs()

        # Filter signals below confidence threshold
        filtered_signals = signals.copy()
        filtered_signals[signal_confidence < confidence_threshold] = 0.0

        return filtered_signals

    def _update_performance_tracking(self, data: pd.DataFrame, signals: pd.Series):
        """Update rolling performance tracking."""
        if len(data) < self.params.rolling_window:
            return

        # Calculate returns
        returns = data["Close"].pct_change().shift(-1)  # Next period returns

        # Calculate signal performance
        signal_returns = signals * returns

        # Update rolling performance
        rolling_perf = signal_returns.rolling(self.params.rolling_window).mean()

        self.rolling_performance = {
            "rolling_mean": rolling_perf.iloc[-1] if not rolling_perf.empty else 0.0,
            "rolling_std": signal_returns.rolling(self.params.rolling_window).std().iloc[-1]
            if not signal_returns.empty
            else 0.0,
            "rolling_sharpe": rolling_perf.iloc[-1]
            / (signal_returns.rolling(self.params.rolling_window).std().iloc[-1] + 1e-6)
            if not signal_returns.empty
            else 0.0,
        }

    # Helper methods for technical indicators
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(
        self, close: pd.Series, period: int = 20
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        return upper, lower

    def _calculate_atr(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = close  # Simplified - would need high/low data
        low = close
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    def _calculate_trend_strength(self, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope."""

        def slope(x):
            if len(x) < 2:
                return 0.0
            return np.polyfit(range(len(x)), x, 1)[0]

        return close.rolling(period).apply(slope)

    def _calculate_trend_consistency(self, close: pd.Series) -> pd.Series:
        """Calculate trend consistency."""
        returns = close.pct_change()
        return returns.rolling(20).apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5)

    def _calculate_trend_persistence(self, close: pd.Series) -> pd.Series:
        """Calculate trend persistence."""
        # How long price stays above/below moving average
        ma = close.rolling(20).mean()
        above_ma = (close > ma).rolling(10).sum() / 10
        return above_ma

    def _calculate_oscillation_frequency(self, close: pd.Series) -> pd.Series:
        """Calculate oscillation frequency."""
        # Count of crosses above/below moving average
        ma = close.rolling(20).mean()
        crosses = ((close > ma) != (close.shift(1) > ma)).rolling(10).sum()
        return crosses

    def _calculate_mean_reversion_strength(self, close: pd.Series) -> pd.Series:
        """Calculate mean reversion strength."""
        ma = close.rolling(20).mean()
        return (close - ma) / ma

    def _calculate_volatility_regime(self, close: pd.Series) -> pd.Series:
        """Calculate volatility regime."""
        returns = close.pct_change()
        current_vol = returns.rolling(20).std()
        historical_vol = returns.rolling(252).std()
        return current_vol / historical_vol

    def _calculate_volatility_persistence(self, close: pd.Series) -> pd.Series:
        """Calculate volatility persistence."""
        returns = close.pct_change()
        vol = returns.rolling(20).std()
        vol_threshold = vol.rolling(252).quantile(0.8)
        persistence = (vol > vol_threshold).rolling(10).sum() / 10
        return persistence

    def get_regime_info(self) -> dict:
        """Get current regime information."""
        if not self.regime_history:
            return {"regime": "unknown", "confidence": 0.0}

        latest = self.regime_history[-1]
        return {
            "regime": latest["regime"],
            "confidence": latest["confidence"],
            "performance": self.rolling_performance,
        }

    def get_default_params(self) -> RegimeAwareEnsembleParams:
        """Get default parameters for the strategy."""
        return RegimeAwareEnsembleParams()

    def get_param_ranges(self) -> dict[str, tuple]:
        """Get parameter ranges for optimization."""
        return {
            "confidence_threshold": (0.1, 0.8),
            "trend_following_weight": (0.3, 0.8),
            "mean_reversion_weight": (0.2, 0.7),
            "rolling_window": (30, 120),
            "regime_lookback": (60, 252),
        }
