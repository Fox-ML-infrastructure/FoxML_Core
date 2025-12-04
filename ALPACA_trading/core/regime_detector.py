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
Regime Detection System
Identifies market regimes (trend vs chop) and provides regime-specific parameters
"""


import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeParams:
    """Parameters specific to each market regime."""

    regime_name: str
    confidence_threshold: float
    position_sizing_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    feature_lookback_adjustment: float
    ensemble_weights: dict[str, float]
    trend_strength: float = 0.0


class RegimeDetector:
    """
    Detects market regimes using multiple indicators and provides regime-specific parameters.
    """

    def __init__(self, lookback_period: int = 252):
        """
        Initialize regime detector.

        Args:
            lookback_period: Period for regime detection calculations
        """
        self.lookback_period = lookback_period
        self.regime_params = self._initialize_regime_params()
        logger.info(f"Initialized RegimeDetector with {lookback_period} day lookback")

    def _initialize_regime_params(self) -> dict[str, RegimeParams]:
        """Initialize regime-specific parameters."""
        return {
            "trend": RegimeParams(
                regime_name="trend",
                confidence_threshold=0.3,  # Lower threshold for trend regime
                position_sizing_multiplier=1.5,  # Larger positions in trends
                stop_loss_multiplier=1.2,  # Wider stops in trends
                take_profit_multiplier=1.5,  # Higher targets in trends
                feature_lookback_adjustment=1.2,  # Longer lookbacks for trends
                ensemble_weights={
                    "momentum": 0.4,
                    "breakout": 0.3,
                    "mean_reversion": 0.1,
                    "ensemble_basic": 0.2,
                },
            ),
            "chop": RegimeParams(
                regime_name="chop",
                confidence_threshold=0.5,  # Higher threshold for chop regime
                position_sizing_multiplier=0.7,  # Smaller positions in chop
                stop_loss_multiplier=0.8,  # Tighter stops in chop
                take_profit_multiplier=0.8,  # Lower targets in chop
                feature_lookback_adjustment=0.8,  # Shorter lookbacks for chop
                ensemble_weights={
                    "momentum": 0.1,
                    "breakout": 0.1,
                    "mean_reversion": 0.6,
                    "ensemble_basic": 0.2,
                },
            ),
            "volatile": RegimeParams(
                regime_name="volatile",
                confidence_threshold=0.6,  # Highest threshold for volatile regime
                position_sizing_multiplier=0.5,  # Smallest positions in volatile
                stop_loss_multiplier=0.6,  # Tightest stops in volatile
                take_profit_multiplier=1.0,  # Standard targets in volatile
                feature_lookback_adjustment=0.6,  # Shortest lookbacks for volatile
                ensemble_weights={
                    "momentum": 0.2,
                    "breakout": 0.2,
                    "mean_reversion": 0.4,
                    "ensemble_basic": 0.2,
                },
            ),
        }

    def detect_regime(self, data: pd.DataFrame) -> tuple[str, float, RegimeParams]:
        """
        Detect market regime using multiple indicators.

        Args:
            data: Price data with OHLCV columns

        Returns:
            Tuple of (regime_name, confidence, regime_params)
        """
        if len(data) < self.lookback_period:
            # More informative logging with data length, but rate limited
            available_days = len(data)
            required_days = self.lookback_period

            # Use a simple rate limiting approach - only log every 50th occurrence
            if not hasattr(self, "_insufficient_data_count"):
                self._insufficient_data_count = 0

            self._insufficient_data_count += 1

            if self._insufficient_data_count <= 5:  # Only log first 5 occurrences
                logger.warning(
                    f"Insufficient data for regime detection: {available_days} < {required_days} "
                    f"(need {required_days - available_days} more days)"
                )
            elif self._insufficient_data_count == 6:
                logger.warning("... (suppressing further regime detection warnings)")

            # Return default regime with low confidence
            return "chop", 0.3, self.regime_params["chop"]

        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(data)

        # Determine regime based on indicators
        regime_name, confidence = self._classify_regime(indicators)

        # Get regime-specific parameters
        regime_params = self.regime_params[regime_name]

        # Calculate trend strength for the current data
        close = data["Close"]
        trend_strength = self._calculate_trend_strength(close).iloc[-1] if len(close) > 0 else 0.0

        # Create updated regime params with trend strength
        updated_regime_params = RegimeParams(
            regime_name=regime_params.regime_name,
            confidence_threshold=regime_params.confidence_threshold,
            position_sizing_multiplier=regime_params.position_sizing_multiplier,
            stop_loss_multiplier=regime_params.stop_loss_multiplier,
            take_profit_multiplier=regime_params.take_profit_multiplier,
            feature_lookback_adjustment=regime_params.feature_lookback_adjustment,
            ensemble_weights=regime_params.ensemble_weights,
            trend_strength=trend_strength,
        )

        return regime_name, confidence, updated_regime_params

    def _calculate_trend_strength(self, close: pd.Series) -> pd.Series:
        """Calculate trend strength indicator."""
        # Linear regression slope over different periods
        periods = [10, 20, 50]
        trend_strength = pd.Series(0.0, index=close.index)

        for period in periods:
            if len(close) >= period:
                slope = close.rolling(period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                trend_strength += slope

        return trend_strength / len(periods)

    def _calculate_regime_indicators(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate regime detection indicators."""
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        # 1. ADX (Average Directional Index) - Trend strength
        adx = self._calculate_adx(high, low, close, period=14)
        current_adx = adx.iloc[-1] if not adx.empty else 25

        # 2. Volatility ratio (current vs historical)
        returns = close.pct_change()
        current_vol = returns.rolling(20).std().iloc[-1]
        historical_vol = returns.rolling(252).std().iloc[-1]
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0

        # 3. Price momentum (trend vs mean reversion)
        short_ma = close.rolling(20).mean()
        momentum = (close.iloc[-1] - short_ma.iloc[-1]) / short_ma.iloc[-1]

        # 4. Range expansion/contraction
        atr = self._calculate_atr(high, low, close, period=14)
        current_atr = atr.iloc[-1] if not atr.empty else 0
        historical_atr = atr.rolling(252).mean().iloc[-1] if not atr.empty else 0
        range_ratio = current_atr / historical_atr if historical_atr > 0 else 1.0

        # 5. Volume trend
        volume_ma = volume.rolling(20).mean()
        volume_trend = (volume.iloc[-1] - volume_ma.iloc[-1]) / volume_ma.iloc[-1]

        # 6. Price efficiency (how much price moves in one direction)
        price_efficiency = self._calculate_price_efficiency(close, period=20)

        return {
            "adx": current_adx,
            "vol_ratio": vol_ratio,
            "momentum": momentum,
            "range_ratio": range_ratio,
            "volume_trend": volume_trend,
            "price_efficiency": price_efficiency,
        }

    def _classify_regime(self, indicators: dict[str, float]) -> tuple[str, float]:
        """Classify regime based on indicators."""
        adx = indicators["adx"]
        vol_ratio = indicators["vol_ratio"]
        momentum = indicators["momentum"]
        range_ratio = indicators["range_ratio"]
        volume_trend = indicators["volume_trend"]
        price_efficiency = indicators["price_efficiency"]

        # Calculate regime scores
        trend_score = 0
        chop_score = 0
        volatile_score = 0

        # ADX contribution (higher = more trend)
        if adx > 25:
            trend_score += 0.3
        elif adx < 20:
            chop_score += 0.3
        else:
            chop_score += 0.2

        # Volatility ratio contribution
        if vol_ratio > 1.5:
            volatile_score += 0.3
        elif vol_ratio < 0.8:
            chop_score += 0.2
        else:
            trend_score += 0.1

        # Momentum contribution
        if abs(momentum) > 0.05:
            trend_score += 0.2
        else:
            chop_score += 0.2

        # Range ratio contribution
        if range_ratio > 1.3:
            volatile_score += 0.2
        elif range_ratio < 0.7:
            chop_score += 0.2
        else:
            trend_score += 0.1

        # Volume trend contribution
        if volume_trend > 0.2:
            trend_score += 0.1
        elif volume_trend < -0.2:
            volatile_score += 0.1

        # Price efficiency contribution
        if price_efficiency > 0.6:
            trend_score += 0.2
        elif price_efficiency < 0.3:
            chop_score += 0.2
        else:
            volatile_score += 0.1

        # Normalize scores
        total_score = trend_score + chop_score + volatile_score
        if total_score > 0:
            trend_score /= total_score
            chop_score /= total_score
            volatile_score /= total_score

        # Determine regime
        scores = {"trend": trend_score, "chop": chop_score, "volatile": volatile_score}

        regime = max(scores, key=scores.get)
        confidence = scores[regime]

        return regime, confidence

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Average Directional Index."""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low

        dm_plus = dm_plus.where(dm_plus > dm_minus, 0)
        dm_plus = dm_plus.where(dm_plus > 0, 0)

        dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
        dm_minus = dm_minus.where(dm_minus > 0, 0)

        # Smoothed values
        tr_smooth = tr.rolling(period).mean()
        dm_plus_smooth = dm_plus.rolling(period).mean()
        dm_minus_smooth = dm_minus.rolling(period).mean()

        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth

        # Directional Index
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)

        # ADX
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return atr

    def _calculate_price_efficiency(self, close: pd.Series, period: int = 20) -> float:
        """Calculate price efficiency (how much price moves in one direction)."""
        if len(close) < period:
            return 0.5

        # Calculate net movement vs total movement
        net_movement = abs(close.iloc[-1] - close.iloc[-period])
        total_movement = close.diff().abs().iloc[-period:].sum()

        if total_movement > 0:
            efficiency = net_movement / total_movement
            return min(efficiency, 1.0)
        else:
            return 0.5

    def get_regime_adapted_features(
        self,
        data: pd.DataFrame,
        base_features: dict[str, pd.Series],
        regime_params: RegimeParams,
    ) -> dict[str, pd.Series]:
        """
        Adapt features based on regime parameters.

        Args:
            data: Price data
            base_features: Base feature set
            regime_params: Regime-specific parameters

        Returns:
            Regime-adapted features
        """
        adapted_features = {}

        for feature_name, feature_series in base_features.items():
            # Adjust lookback periods based on regime
            if "lookback" in feature_name.lower() or any(
                period in feature_name for period in ["5", "10", "20", "50", "100", "200"]
            ):
                # Extract period from feature name and adjust
                adjusted_feature = self._adjust_feature_lookback(
                    data, feature_series, regime_params.feature_lookback_adjustment
                )
                adapted_features[feature_name] = adjusted_feature
            else:
                adapted_features[feature_name] = feature_series

        return adapted_features

    def _adjust_feature_lookback(
        self, data: pd.DataFrame, feature: pd.Series, adjustment_factor: float
    ) -> pd.Series:
        """Adjust feature lookback period based on regime."""
        # For now, return the original feature
        # In a more sophisticated implementation, we would recalculate features with different periods
        return feature

    def get_regime_ensemble_weights(self, regime_params: RegimeParams) -> dict[str, float]:
        """Get ensemble weights for the current regime."""
        return regime_params.ensemble_weights.copy()


class RegimeAwareFeatureEngine:
    """
    Feature engine that adapts features based on market regime.
    """

    def __init__(self, regime_detector: RegimeDetector):
        """
        Initialize regime-aware feature engine.

        Args:
            regime_detector: Regime detection system
        """
        self.regime_detector = regime_detector
        logger.info("Initialized RegimeAwareFeatureEngine")

    def generate_regime_adapted_features(
        self, data: pd.DataFrame
    ) -> tuple[dict[str, pd.Series], RegimeParams]:
        """
        Generate features adapted to the current market regime.

        Args:
            data: Price data

        Returns:
            Tuple of (adapted_features, regime_params)
        """
        # Detect regime
        regime_name, confidence, regime_params = self.regime_detector.detect_regime(data)

        # Generate base features
        base_features = self._generate_base_features(data)

        # Adapt features to regime
        adapted_features = self.regime_detector.get_regime_adapted_features(
            data, base_features, regime_params
        )

        # Add regime-specific features
        regime_features = self._generate_regime_specific_features(data, regime_name)
        adapted_features.update(regime_features)

        logger.info(
            f"Generated {len(adapted_features)} regime-adapted features for {regime_name} regime"
        )

        return adapted_features, regime_params

    def _generate_base_features(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Generate base feature set."""
        # This would integrate with your existing feature engine
        # For now, return empty dict - will be implemented in integration
        return {}

    def _generate_regime_specific_features(
        self, data: pd.DataFrame, regime_name: str
    ) -> dict[str, pd.Series]:
        """Generate regime-specific features."""
        close = data["Close"]
        features = {}

        if regime_name == "trend":
            # Trend-specific features
            features["trend_strength"] = self._calculate_trend_strength(close)
            features["trend_consistency"] = self._calculate_trend_consistency(close)

        elif regime_name == "chop":
            # Chop-specific features
            features["chop_oscillation"] = self._calculate_chop_oscillation(close)
            features["mean_reversion_strength"] = self._calculate_mean_reversion_strength(close)

        elif regime_name == "volatile":
            # Volatile-specific features
            features["volatility_regime"] = self._calculate_volatility_regime(close)
            features["volatility_persistence"] = self._calculate_volatility_persistence(close)

        return features

    def _calculate_trend_strength(self, close: pd.Series) -> pd.Series:
        """Calculate trend strength indicator."""
        # Linear regression slope over different periods
        periods = [10, 20, 50]
        trend_strength = pd.Series(0.0, index=close.index)

        for period in periods:
            if len(close) >= period:
                slope = close.rolling(period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
                trend_strength += slope

        return trend_strength / len(periods)

    def _calculate_trend_consistency(self, close: pd.Series) -> pd.Series:
        """Calculate trend consistency indicator."""
        # Percentage of days moving in the same direction
        returns = close.pct_change()
        consistency = returns.rolling(20).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        return consistency

    def _calculate_chop_oscillation(self, close: pd.Series) -> pd.Series:
        """Calculate chop oscillation indicator."""
        # Oscillation around moving average
        ma = close.rolling(20).mean()
        oscillation = (close - ma) / ma
        return oscillation.rolling(10).std()

    def _calculate_mean_reversion_strength(self, close: pd.Series) -> pd.Series:
        """Calculate mean reversion strength indicator."""
        # Distance from moving average
        ma = close.rolling(20).mean()
        reversion_strength = (close - ma) / ma
        return reversion_strength

    def _calculate_volatility_regime(self, close: pd.Series) -> pd.Series:
        """Calculate volatility regime indicator."""
        # Rolling volatility vs historical volatility
        returns = close.pct_change()
        current_vol = returns.rolling(20).std()
        historical_vol = returns.rolling(252).std()
        vol_regime = current_vol / historical_vol
        return vol_regime

    def _calculate_volatility_persistence(self, close: pd.Series) -> pd.Series:
        """Calculate volatility persistence indicator."""
        # How long volatility stays elevated
        returns = close.pct_change()
        vol = returns.rolling(20).std()
        vol_threshold = vol.rolling(252).quantile(0.8)
        persistence = (vol > vol_threshold).rolling(10).sum() / 10
        return persistence
