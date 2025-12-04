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
Regime Detection Features

Detects market regimes (trending, choppy, volatile) and creates regime-conditional features.
These features often provide multiplicative alpha as predictive patterns differ by regime.
"""


import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_trend_strength(close: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Compute trend strength as correlation between price and time.
    
    Returns:
        Series with values in [-1, 1]:
        +1 = strong uptrend
        -1 = strong downtrend
        0 = no trend
    """
    def rolling_corr_with_time(x):
        if len(x) < 2:
            return 0.0
        return np.corrcoef(x, np.arange(len(x)))[0, 1]
    
    return close.rolling(lookback, min_periods=2).apply(rolling_corr_with_time)


def compute_volatility_regime(
    close: pd.Series,
    short_window: int = 20,
    long_window: int = 200
) -> pd.Series:
    """
    Compute volatility regime (normalized volatility).
    
    Returns:
        Series with volatility z-scores:
        > 1.5 = high volatility regime
        < -1.0 = low volatility regime
    """
    returns = close.pct_change()
    
    # Short-term volatility
    short_vol = returns.rolling(short_window).std()
    
    # Long-term mean and std
    long_mean = short_vol.rolling(long_window).mean()
    long_std = short_vol.rolling(long_window).std()
    
    # Z-score
    vol_zscore = (short_vol - long_mean) / (long_std + 1e-8)
    
    return vol_zscore


def detect_regime(
    df: pd.DataFrame,
    trend_lookback: int = 50,
    vol_short: int = 20,
    vol_long: int = 200,
    trend_threshold: float = 0.7,
    vol_threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect market regime and add regime columns.
    
    Regimes:
    0 = Choppy (default) - no clear trend, normal vol
    1 = Trending - strong correlation between price and time
    2 = High Volatility - elevated volatility regardless of trend
    
    Args:
        df: DataFrame with 'close' column
        trend_lookback: Window for trend calculation
        vol_short: Short window for volatility
        vol_long: Long window for volatility normalization
        trend_threshold: Abs correlation above this = trending
        vol_threshold: Vol z-score above this = high vol
    
    Returns:
        DataFrame with added regime columns
    """
    df = df.copy()
    
    # Compute regime indicators
    trend_strength = compute_trend_strength(df['close'], trend_lookback)
    vol_zscore = compute_volatility_regime(df['close'], vol_short, vol_long)
    
    # Classify regime (priority: vol > trend > chop)
    regime = pd.Series(0, index=df.index)  # Default: choppy
    
    # Trending regime (strong positive or negative correlation)
    regime[trend_strength.abs() > trend_threshold] = 1
    
    # High volatility regime overrides trending
    regime[vol_zscore > vol_threshold] = 2
    
    # Add regime columns
    df['regime'] = regime
    df['regime_trend'] = (regime == 1).astype(int)
    df['regime_chop'] = (regime == 0).astype(int)
    df['regime_vol'] = (regime == 2).astype(int)
    
    # Add raw indicators for transparency
    df['trend_strength'] = trend_strength
    df['vol_zscore'] = vol_zscore
    
    return df


def add_regime_conditional_features(
    df: pd.DataFrame,
    base_features: list = None
) -> pd.DataFrame:
    """
    Create regime-conditional versions of existing features.
    
    The idea: RSI means different things in trending vs choppy markets.
    By multiplying features by regime indicators, we create regime-specific features.
    
    Args:
        df: DataFrame with regime columns already added
        base_features: List of feature names to make regime-conditional
                      If None, uses common technical indicators
    
    Returns:
        DataFrame with added regime-conditional features
    """
    df = df.copy()
    
    # Ensure regime columns exist
    if 'regime_trend' not in df.columns:
        raise ValueError("Run detect_regime() first to add regime columns")
    
    # Default base features if not specified
    if base_features is None:
        base_features = []
        
        # Returns at different horizons
        for minutes in [5, 10, 15, 20, 30]:
            col = f'ret_{minutes}m'
            if col in df.columns:
                base_features.append(col)
        
        # Common technical indicators
        for indicator in ['rsi_14', 'macd', 'bb_width', 'atr_14']:
            if indicator in df.columns:
                base_features.append(indicator)
    
    # Create regime-conditional features
    for feature in base_features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found, skipping")
            continue
        
        # Feature * regime indicator (element-wise)
        df[f'{feature}_in_trend'] = df[feature] * df['regime_trend']
        df[f'{feature}_in_chop'] = df[feature] * df['regime_chop']
        df[f'{feature}_in_vol'] = df[feature] * df['regime_vol']
    
    return df


def add_regime_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that capture regime transitions (often alpha-rich moments).
    
    Returns:
        DataFrame with transition features
    """
    df = df.copy()
    
    if 'regime' not in df.columns:
        raise ValueError("Run detect_regime() first")
    
    # Regime change indicator
    df['regime_changed'] = (df['regime'] != df['regime'].shift(1)).astype(int)
    
    # Specific transitions
    df['entered_trend'] = ((df['regime'] == 1) & (df['regime'].shift(1) != 1)).astype(int)
    df['exited_trend'] = ((df['regime'] != 1) & (df['regime'].shift(1) == 1)).astype(int)
    df['entered_vol'] = ((df['regime'] == 2) & (df['regime'].shift(1) != 2)).astype(int)
    
    # Time in current regime
    regime_duration = pd.Series(0, index=df.index)
    current_regime = None
    duration = 0
    
    for i, regime in enumerate(df['regime']):
        if regime != current_regime:
            current_regime = regime
            duration = 1
        else:
            duration += 1
        regime_duration.iloc[i] = duration
    
    df['regime_duration'] = regime_duration
    
    return df


def add_all_regime_features(
    df: pd.DataFrame,
    trend_lookback: int = 50,
    base_features: list = None,
    include_transitions: bool = True
) -> pd.DataFrame:
    """
    One-stop function to add all regime-related features.
    
    Args:
        df: DataFrame with at least 'close' column
        trend_lookback: Window for trend detection
        base_features: Features to make regime-conditional
        include_transitions: Whether to add transition features
    
    Returns:
        DataFrame with all regime features added
    """
    logger.info("Adding regime features...")
    
    # 1. Detect regimes
    df = detect_regime(df, trend_lookback=trend_lookback)
    
    # Count regimes
    regime_counts = df['regime'].value_counts().sort_index()
    logger.info(f"Regime distribution:")
    logger.info(f"  Choppy (0): {regime_counts.get(0, 0)} bars ({regime_counts.get(0, 0)/len(df)*100:.1f}%)")
    logger.info(f"  Trending (1): {regime_counts.get(1, 0)} bars ({regime_counts.get(1, 0)/len(df)*100:.1f}%)")
    logger.info(f"  High Vol (2): {regime_counts.get(2, 0)} bars ({regime_counts.get(2, 0)/len(df)*100:.1f}%)")
    
    # 2. Add regime-conditional features
    df = add_regime_conditional_features(df, base_features)
    
    # 3. Add transition features
    if include_transitions:
        df = add_regime_transition_features(df)
    
    n_new_features = len([c for c in df.columns if 'regime' in c.lower()])
    logger.info(f"✅ Added {n_new_features} regime features")
    
    return df


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=10000, freq='5T')
    
    # Simulate price with different regimes
    price = 100.0
    prices = []
    for i in range(len(dates)):
        if i < 3000:  # Trending up
            price += np.random.normal(0.05, 0.5)
        elif i < 6000:  # Choppy
            price += np.random.normal(0, 0.8)
        else:  # High volatility
            price += np.random.normal(0, 2.0)
        prices.append(price)
    
    df = pd.DataFrame({'close': prices}, index=dates)
    
    # Add regime features
    df = add_all_regime_features(df)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Price and regime
    axes[0].plot(df.index, df['close'], alpha=0.7)
    axes[0].fill_between(df.index, df['close'].min(), df['close'].max(), 
                         where=df['regime']==1, alpha=0.2, color='green', label='Trending')
    axes[0].fill_between(df.index, df['close'].min(), df['close'].max(),
                         where=df['regime']==2, alpha=0.2, color='red', label='High Vol')
    axes[0].set_title('Price and Detected Regimes')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Regime indicators
    axes[1].plot(df.index, df['trend_strength'], label='Trend Strength', alpha=0.7)
    axes[1].plot(df.index, df['vol_zscore'], label='Vol Z-score', alpha=0.7)
    axes[1].axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Trend Threshold')
    axes[1].axhline(-0.7, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Vol Threshold')
    axes[1].set_title('Regime Indicators')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regime_detection_example.png', dpi=150)
    print("✅ Saved regime_detection_example.png")
    
    # Show feature correlations
    print("\n📊 Sample regime-conditional features:")
    regime_cols = [c for c in df.columns if 'regime' in c.lower()][:10]
    print(df[regime_cols].head(20))

