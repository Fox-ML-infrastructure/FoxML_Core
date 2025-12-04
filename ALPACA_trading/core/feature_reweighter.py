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
Feature Re-weighting System
Adjusts feature importance based on rolling IC and Sharpe ratios by regime
"""


import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeaturePerformance:
    """Performance metrics for a feature."""

    feature_name: str
    regime: str
    rolling_ic: float
    rolling_sharpe: float
    rolling_returns: float
    weight: float
    last_updated: pd.Timestamp


class FeatureReweighter:
    """
    Re-weights features based on rolling performance metrics by regime.
    """

    def __init__(
        self,
        rolling_window: int = 60,
        min_periods: int = 30,
        reweight_frequency: int = 20,
        decay_factor: float = 0.95,
    ):
        """
        Initialize feature re-weighter.

        Args:
            rolling_window: Window for rolling calculations
            min_periods: Minimum periods required for calculations
            reweight_frequency: How often to re-weight features
            decay_factor: Decay factor for old performance
        """
        self.rolling_window = rolling_window
        self.min_periods = min_periods
        self.reweight_frequency = reweight_frequency
        self.decay_factor = decay_factor

        # Performance tracking
        self.feature_performance = {}  # {feature_name: FeaturePerformance}
        self.regime_performance = {}  # {regime: {feature: performance}}
        self.last_reweight = None
        self.update_count = 0

        logger.info(f"Initialized FeatureReweighter with {rolling_window} day window")

    def update_feature_performance(
        self,
        features: dict[str, pd.Series],
        returns: pd.Series,
        regime: str,
        date: pd.Timestamp,
    ):
        """
        Update feature performance metrics.

        Args:
            features: Feature dictionary
            returns: Target returns
            regime: Current market regime
            date: Current date
        """
        if len(returns) < self.min_periods:
            return

        # Calculate rolling performance for each feature
        for feature_name, feature_series in features.items():
            if feature_name not in self.feature_performance:
                self.feature_performance[feature_name] = FeaturePerformance(
                    feature_name=feature_name,
                    regime=regime,
                    rolling_ic=0.0,
                    rolling_sharpe=0.0,
                    rolling_returns=0.0,
                    weight=1.0,
                    last_updated=date,
                )

            # Calculate rolling IC (Information Coefficient)
            rolling_ic = self._calculate_rolling_ic(feature_series, returns)

            # Calculate rolling Sharpe ratio
            rolling_sharpe = self._calculate_rolling_sharpe(feature_series, returns)

            # Calculate rolling returns
            rolling_returns = self._calculate_rolling_returns(feature_series, returns)

            # Update performance with decay
            perf = self.feature_performance[feature_name]
            if perf.last_updated < date:
                # Apply decay to old performance
                perf.rolling_ic = perf.rolling_ic * self.decay_factor + rolling_ic * (
                    1 - self.decay_factor
                )
                perf.rolling_sharpe = perf.rolling_sharpe * self.decay_factor + rolling_sharpe * (
                    1 - self.decay_factor
                )
                perf.rolling_returns = (
                    perf.rolling_returns * self.decay_factor
                    + rolling_returns * (1 - self.decay_factor)
                )
            else:
                # Direct update for same date
                perf.rolling_ic = rolling_ic
                perf.rolling_sharpe = rolling_sharpe
                perf.rolling_returns = rolling_returns

            perf.regime = regime
            perf.last_updated = date

        # Update regime performance tracking
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {}

        for feature_name, perf in self.feature_performance.items():
            if perf.regime == regime:
                self.regime_performance[regime][feature_name] = {
                    "ic": perf.rolling_ic,
                    "sharpe": perf.rolling_sharpe,
                    "returns": perf.rolling_returns,
                }

        self.update_count += 1

        # Re-weight features periodically
        if self.update_count % self.reweight_frequency == 0:
            self._reweight_features_by_regime(regime)

    def _calculate_rolling_ic(self, feature: pd.Series, returns: pd.Series) -> float:
        """Calculate rolling Information Coefficient."""
        if len(feature) < self.rolling_window:
            return 0.0

        # Align feature and returns
        aligned_data = pd.concat([feature, returns], axis=1).dropna()
        if len(aligned_data) < self.rolling_window:
            return 0.0

        feature_col = aligned_data.iloc[:, 0]
        returns_col = aligned_data.iloc[:, 1]

        # Calculate rolling correlation
        rolling_corr = feature_col.rolling(self.rolling_window).corr(returns_col)

        return rolling_corr.iloc[-1] if not rolling_corr.empty else 0.0

    def _calculate_rolling_sharpe(self, feature: pd.Series, returns: pd.Series) -> float:
        """Calculate rolling Sharpe ratio for feature."""
        if len(feature) < self.rolling_window:
            return 0.0

        # Align feature and returns
        aligned_data = pd.concat([feature, returns], axis=1).dropna()
        if len(aligned_data) < self.rolling_window:
            return 0.0

        feature_col = aligned_data.iloc[:, 0]
        returns_col = aligned_data.iloc[:, 1]

        # Calculate feature-weighted returns
        feature_returns = feature_col * returns_col

        # Calculate rolling Sharpe
        rolling_mean = feature_returns.rolling(self.rolling_window).mean()
        rolling_std = feature_returns.rolling(self.rolling_window).std()

        if rolling_std.iloc[-1] > 0:
            sharpe = rolling_mean.iloc[-1] / rolling_std.iloc[-1]
            return sharpe
        else:
            return 0.0

    def _calculate_rolling_returns(self, feature: pd.Series, returns: pd.Series) -> float:
        """Calculate rolling returns for feature."""
        if len(feature) < self.rolling_window:
            return 0.0

        # Align feature and returns
        aligned_data = pd.concat([feature, returns], axis=1).dropna()
        if len(aligned_data) < self.rolling_window:
            return 0.0

        feature_col = aligned_data.iloc[:, 0]
        returns_col = aligned_data.iloc[:, 1]

        # Calculate feature-weighted returns
        feature_returns = feature_col * returns_col

        # Calculate rolling mean returns
        rolling_returns = feature_returns.rolling(self.rolling_window).mean()

        return rolling_returns.iloc[-1] if not rolling_returns.empty else 0.0

    def _reweight_features_by_regime(self, regime: str):
        """Re-weight features based on performance within the regime."""
        if regime not in self.regime_performance:
            return

        regime_perf = self.regime_performance[regime]
        if not regime_perf:
            return

        # Calculate composite performance score
        feature_scores = {}
        for feature_name, perf in regime_perf.items():
            # Normalize metrics
            ic_score = np.tanh(perf["ic"] * 10)  # Scale IC to [-1, 1]
            sharpe_score = np.tanh(perf["sharpe"] * 2)  # Scale Sharpe to [-1, 1]
            returns_score = np.tanh(perf["returns"] * 100)  # Scale returns to [-1, 1]

            # Composite score (weighted average)
            composite_score = 0.4 * ic_score + 0.4 * sharpe_score + 0.2 * returns_score

            feature_scores[feature_name] = composite_score

        # Convert scores to weights using softmax
        scores_array = np.array(list(feature_scores.values()))
        weights = self._softmax(scores_array)

        # Update feature weights
        for i, feature_name in enumerate(feature_scores.keys()):
            if feature_name in self.feature_performance:
                self.feature_performance[feature_name].weight = weights[i]

        logger.info(f"Re-weighted {len(feature_scores)} features for {regime} regime")

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function to convert scores to probabilities."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

    def get_feature_weights(self, regime: str) -> dict[str, float]:
        """Get feature weights for the specified regime."""
        weights = {}

        for feature_name, perf in self.feature_performance.items():
            if perf.regime == regime:
                weights[feature_name] = perf.weight

        return weights

    def get_regime_performance_summary(self, regime: str) -> dict:
        """Get performance summary for a regime."""
        if regime not in self.regime_performance:
            return {}

        regime_perf = self.regime_performance[regime]

        # Calculate summary statistics
        ic_values = [perf["ic"] for perf in regime_perf.values()]
        sharpe_values = [perf["sharpe"] for perf in regime_perf.values()]
        returns_values = [perf["returns"] for perf in regime_perf.values()]

        summary = {
            "num_features": len(regime_perf),
            "avg_ic": np.mean(ic_values) if ic_values else 0.0,
            "avg_sharpe": np.mean(sharpe_values) if sharpe_values else 0.0,
            "avg_returns": np.mean(returns_values) if returns_values else 0.0,
            "top_features": self._get_top_features(regime, 5),
        }

        return summary

    def _get_top_features(self, regime: str, n: int = 5) -> list[dict]:
        """Get top performing features for a regime."""
        if regime not in self.regime_performance:
            return []

        regime_perf = self.regime_performance[regime]

        # Calculate composite scores
        feature_scores = []
        for feature_name, perf in regime_perf.items():
            ic_score = np.tanh(perf["ic"] * 10)
            sharpe_score = np.tanh(perf["sharpe"] * 2)
            returns_score = np.tanh(perf["returns"] * 100)

            composite_score = 0.4 * ic_score + 0.4 * sharpe_score + 0.2 * returns_score

            feature_scores.append(
                {
                    "feature": feature_name,
                    "score": composite_score,
                    "ic": perf["ic"],
                    "sharpe": perf["sharpe"],
                    "returns": perf["returns"],
                }
            )

        # Sort by composite score and return top n
        feature_scores.sort(key=lambda x: x["score"], reverse=True)
        return feature_scores[:n]


class AdaptiveFeatureEngine:
    """
    Feature engine that adapts feature importance based on performance.
    """

    def __init__(self, reweighter: FeatureReweighter):
        """
        Initialize adaptive feature engine.

        Args:
            reweighter: Feature re-weighting system
        """
        self.reweighter = reweighter
        self.scaler = StandardScaler()
        self.ridge_model = Ridge(alpha=1.0)

        logger.info("Initialized AdaptiveFeatureEngine")

    def generate_adaptive_features(
        self, data: pd.DataFrame, regime: str, base_features: dict[str, pd.Series]
    ) -> dict[str, pd.Series]:
        """
        Generate features with adaptive weighting.

        Args:
            data: Price data
            regime: Current market regime
            base_features: Base feature set

        Returns:
            Adaptively weighted features
        """
        # Get feature weights for the regime
        feature_weights = self.reweighter.get_feature_weights(regime)

        # Apply weights to features
        weighted_features = {}
        for feature_name, feature_series in base_features.items():
            weight = feature_weights.get(feature_name, 1.0)
            weighted_features[feature_name] = feature_series * weight

        # Add regime-specific adaptive features
        adaptive_features = self._generate_adaptive_features(data, regime, base_features)
        weighted_features.update(adaptive_features)

        return weighted_features

    def _generate_adaptive_features(
        self, data: pd.DataFrame, regime: str, base_features: dict[str, pd.Series]
    ) -> dict[str, pd.Series]:
        """Generate regime-specific adaptive features."""
        close = data["Close"]
        features = {}

        # Get regime performance summary
        regime_summary = self.reweighter.get_regime_performance_summary(regime)

        if regime == "trend":
            # Trend-adaptive features
            features["trend_adaptive_momentum"] = self._calculate_adaptive_momentum(
                close, regime_summary
            )
            features["trend_adaptive_strength"] = self._calculate_adaptive_trend_strength(
                close, regime_summary
            )

        elif regime == "chop":
            # Chop-adaptive features
            features["chop_adaptive_oscillation"] = self._calculate_adaptive_oscillation(
                close, regime_summary
            )
            features["chop_adaptive_reversion"] = self._calculate_adaptive_reversion(
                close, regime_summary
            )

        elif regime == "volatile":
            # Volatile-adaptive features
            features["volatile_adaptive_vol"] = self._calculate_adaptive_volatility(
                close, regime_summary
            )
            features["volatile_adaptive_risk"] = self._calculate_adaptive_risk(
                close, regime_summary
            )

        return features

    def _calculate_adaptive_momentum(self, close: pd.Series, regime_summary: dict) -> pd.Series:
        """Calculate adaptive momentum based on regime performance."""
        # Adjust momentum calculation based on regime performance
        avg_ic = regime_summary.get("avg_ic", 0.0)
        momentum_period = int(20 * (1 + avg_ic))  # Longer period if IC is positive

        momentum = close.pct_change(momentum_period)

        # Scale by regime performance
        avg_sharpe = regime_summary.get("avg_sharpe", 0.0)
        scaling_factor = 1 + np.tanh(avg_sharpe)

        return momentum * scaling_factor

    def _calculate_adaptive_trend_strength(
        self, close: pd.Series, regime_summary: dict
    ) -> pd.Series:
        """Calculate adaptive trend strength."""
        # Use regime performance to adjust trend strength calculation
        avg_returns = regime_summary.get("avg_returns", 0.0)

        # Adjust lookback period based on performance
        lookback = int(20 * (1 + avg_returns))

        # Calculate trend strength
        ma = close.rolling(lookback).mean()
        trend_strength = (close - ma) / ma

        return trend_strength

    def _calculate_adaptive_oscillation(self, close: pd.Series, regime_summary: dict) -> pd.Series:
        """Calculate adaptive oscillation for chop regime."""
        avg_ic = regime_summary.get("avg_ic", 0.0)

        # Adjust oscillation calculation based on IC
        period = int(10 * (1 - abs(avg_ic)))  # Shorter period if IC is low

        ma = close.rolling(period).mean()
        oscillation = (close - ma) / ma

        return oscillation.rolling(period).std()

    def _calculate_adaptive_reversion(self, close: pd.Series, regime_summary: dict) -> pd.Series:
        """Calculate adaptive mean reversion."""
        avg_sharpe = regime_summary.get("avg_sharpe", 0.0)

        # Adjust reversion strength based on Sharpe
        period = int(20 * (1 + avg_sharpe))

        ma = close.rolling(period).mean()
        reversion = (close - ma) / ma

        return reversion

    def _calculate_adaptive_volatility(self, close: pd.Series, regime_summary: dict) -> pd.Series:
        """Calculate adaptive volatility."""
        avg_returns = regime_summary.get("avg_returns", 0.0)

        # Adjust volatility calculation based on returns
        period = int(15 * (1 + abs(avg_returns)))

        returns = close.pct_change()
        volatility = returns.rolling(period).std()

        return volatility

    def _calculate_adaptive_risk(self, close: pd.Series, regime_summary: dict) -> pd.Series:
        """Calculate adaptive risk measure."""
        avg_sharpe = regime_summary.get("avg_sharpe", 0.0)

        # Adjust risk calculation based on Sharpe
        period = int(10 * (1 + avg_sharpe))

        returns = close.pct_change()
        risk = returns.rolling(period).apply(lambda x: np.percentile(x, 5))  # 5th percentile

        return risk

    def update_performance(
        self,
        features: dict[str, pd.Series],
        returns: pd.Series,
        regime: str,
        date: pd.Timestamp,
    ):
        """Update feature performance metrics."""
        self.reweighter.update_feature_performance(features, returns, regime, date)
