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
Barrier/First-Passage Target Generation

Implements robust will_peak/will_valley labels using:
1. Barrier (first-touch) labels
2. Swing-extreme (ZigZag) labels  
3. Max/Min future excursion (MFE/MDD) thresholds

Designed for limit/stop placement, meta-labeling, and order selection.
"""


import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def compute_barrier_targets(
    prices: pd.Series, 
    horizon_minutes: int = 15,
    barrier_size: float = 0.5,  # k * sigma
    vol_window: int = 20,
    min_touch_prob: float = 0.15
) -> pd.DataFrame:
    """
    Compute barrier (first-touch) labels for will_peak/will_valley prediction.
    
    Args:
        prices: Price series (mid or close)
        horizon_minutes: Lookahead horizon in minutes
        barrier_size: Barrier size as multiple of volatility (k * sigma)
        vol_window: Window for volatility estimation
        min_touch_prob: Minimum probability of touch for positive class
        
    Returns:
        DataFrame with columns:
        - y_first_touch: {-1, 0, +1} (down-first, none, up-first)
        - y_will_peak: {0, 1} (will hit up barrier)
        - y_will_valley: {0, 1} (will hit down barrier)
        - p_up: Probability of hitting up barrier first
        - p_down: Probability of hitting down barrier first
        - barrier_up: Up barrier level
        - barrier_down: Down barrier level
    """
    
    # Compute rolling volatility
    returns = prices.pct_change().dropna()
    vol = returns.rolling(window=vol_window, min_periods=5).std()
    
    # Barrier levels
    barrier_up = prices * (1 + barrier_size * vol)
    barrier_down = prices * (1 - barrier_size * vol)
    
    results = []
    
    for i in range(len(prices)):
        if i + horizon_minutes >= len(prices):
            break
            
        current_price = prices.iloc[i]
        current_vol = vol.iloc[i]
        
        if pd.isna(current_vol) or current_vol == 0:
            continue
            
        # Future path
        future_prices = prices.iloc[i+1:i+horizon_minutes+1]
        
        if len(future_prices) < horizon_minutes:
            continue
            
        up_barrier = current_price * (1 + barrier_size * current_vol)
        down_barrier = current_price * (1 - barrier_size * current_vol)
        
        # Check first touch
        up_touch = (future_prices >= up_barrier).any()
        down_touch = (future_prices <= down_barrier).any()
        
        # First touch logic
        if up_touch and down_touch:
            # Both barriers hit - check which first
            up_idx = (future_prices >= up_barrier).idxmax()
            down_idx = (future_prices <= down_barrier).idxmax()
            first_touch = 1 if up_idx < down_idx else -1
        elif up_touch:
            first_touch = 1
        elif down_touch:
            first_touch = -1
        else:
            first_touch = 0
            
        # Binary labels
        will_peak = 1 if up_touch else 0
        will_valley = 1 if down_touch else 0
        
        # Simple probability estimates (can be enhanced with more sophisticated models)
        p_up = 0.5 if up_touch else 0.0
        p_down = 0.5 if down_touch else 0.0
        
        results.append({
            'y_first_touch': first_touch,
            'y_will_peak': will_peak,
            'y_will_valley': will_valley,
            'p_up': p_up,
            'p_down': p_down,
            'barrier_up': up_barrier,
            'barrier_down': down_barrier,
            'vol_at_t': current_vol
        })
    
    return pd.DataFrame(results, index=prices.index[:len(results)])

def compute_zigzag_targets(
    prices: pd.Series,
    horizon_minutes: int = 15,
    reversal_pct: float = 0.1,  # Minimum reversal percentage
    min_swing_bars: int = 3
) -> pd.DataFrame:
    """
    Compute ZigZag-based swing targets for will_peak/will_valley prediction.
    
    Args:
        prices: Price series
        horizon_minutes: Lookahead horizon
        reversal_pct: Minimum reversal percentage for swing
        min_swing_bars: Minimum bars for a swing
        
    Returns:
        DataFrame with swing-based targets
    """
    
    # Simple ZigZag implementation
    zigzag = compute_zigzag(prices, reversal_pct, min_swing_bars)
    
    results = []
    
    for i in range(len(prices)):
        if i + horizon_minutes >= len(prices):
            break
            
        # Check if there will be a swing high/low in the future
        future_prices = prices.iloc[i+1:i+horizon_minutes+1]
        
        if len(future_prices) < min_swing_bars:
            continue
            
        # Check for swing high (peak)
        will_swing_high = 0
        for j in range(min_swing_bars, len(future_prices) - min_swing_bars):
            if (future_prices.iloc[j] > future_prices.iloc[j-min_swing_bars:j]).all() and \
               (future_prices.iloc[j] > future_prices.iloc[j+1:j+min_swing_bars+1]).all():
                will_swing_high = 1
                break
                
        # Check for swing low (valley)
        will_swing_low = 0
        for j in range(min_swing_bars, len(future_prices) - min_swing_bars):
            if (future_prices.iloc[j] < future_prices.iloc[j-min_swing_bars:j]).all() and \
               (future_prices.iloc[j] < future_prices.iloc[j+1:j+min_swing_bars+1]).all():
                will_swing_low = 1
                break
                
        results.append({
            'y_will_swing_high': will_swing_high,
            'y_will_swing_low': will_swing_low
        })
    
    return pd.DataFrame(results, index=prices.index[:len(results)])

def compute_zigzag(prices: pd.Series, reversal_pct: float, min_bars: int) -> pd.Series:
    """Compute ZigZag indicator."""
    zigzag = pd.Series(index=prices.index, dtype=float)
    zigzag.iloc[0] = prices.iloc[0]
    
    last_zigzag = prices.iloc[0]
    last_zigzag_idx = 0
    direction = 1  # 1 for up, -1 for down
    
    for i in range(1, len(prices)):
        price = prices.iloc[i]
        
        if direction == 1:  # Looking for peak
            if price > last_zigzag * (1 + reversal_pct):
                # New peak
                zigzag.iloc[last_zigzag_idx:i] = last_zigzag
                last_zigzag = price
                last_zigzag_idx = i
            elif price < last_zigzag * (1 - reversal_pct):
                # Reversal to down
                direction = -1
                last_zigzag = price
                last_zigzag_idx = i
        else:  # Looking for valley
            if price < last_zigzag * (1 - reversal_pct):
                # New valley
                zigzag.iloc[last_zigzag_idx:i] = last_zigzag
                last_zigzag = price
                last_zigzag_idx = i
            elif price > last_zigzag * (1 + reversal_pct):
                # Reversal to up
                direction = 1
                last_zigzag = price
                last_zigzag_idx = i
    
    # Fill remaining
    zigzag.iloc[last_zigzag_idx:] = last_zigzag
    
    return zigzag

def compute_mfe_mdd_targets(
    prices: pd.Series,
    horizon_minutes: int = 15,
    threshold_up: float = 0.002,  # 20 bps
    threshold_down: float = -0.002
) -> pd.DataFrame:
    """
    Compute Max Future Excursion (MFE) and Max Drawdown (MDD) targets.
    
    Args:
        prices: Price series
        horizon_minutes: Lookahead horizon
        threshold_up: Threshold for will_peak
        threshold_down: Threshold for will_valley
        
    Returns:
        DataFrame with MFE/MDD targets
    """
    
    results = []
    
    for i in range(len(prices)):
        if i + horizon_minutes >= len(prices):
            break
            
        current_price = prices.iloc[i]
        future_prices = prices.iloc[i+1:i+horizon_minutes+1]
        
        if len(future_prices) < horizon_minutes:
            continue
            
        # Compute max/min returns
        future_returns = (future_prices / current_price) - 1
        max_return = future_returns.max()
        min_return = future_returns.min()
        
        # Binary labels
        will_peak = 1 if max_return >= threshold_up else 0
        will_valley = 1 if min_return <= threshold_down else 0
        
        results.append({
            'y_will_peak_mfe': will_peak,
            'y_will_valley_mdd': will_valley,
            'max_return': max_return,
            'min_return': min_return,
            'mfe': max_return,
            'mdd': min_return
        })
    
    return pd.DataFrame(results, index=prices.index[:len(results)])

def add_barrier_targets_to_dataframe(
    df: pd.DataFrame,
    price_col: str = 'close',
    horizons: list = [5, 10, 15, 30, 60],
    barrier_sizes: list = [0.3, 0.5, 0.8],
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Add barrier targets to existing DataFrame.
    
    Args:
        df: DataFrame with price data
        price_col: Name of price column
        horizons: List of horizons in minutes
        barrier_sizes: List of barrier sizes (k * sigma)
        vol_window: Window for volatility estimation
        
    Returns:
        DataFrame with added target columns
    """
    
    result_df = df.copy()
    prices = df[price_col]
    
    for horizon in horizons:
        for barrier_size in barrier_sizes:
            logger.info(f"Computing barrier targets for horizon={horizon}m, barrier_size={barrier_size}")
            
            # Compute barrier targets
            barrier_targets = compute_barrier_targets(
                prices, 
                horizon_minutes=horizon,
                barrier_size=barrier_size,
                vol_window=vol_window
            )
            
            if len(barrier_targets) == 0:
                continue
                
    # Add columns with horizon and barrier size suffix
    suffix = f"_{horizon}m_{barrier_size:.1f}"
    
    # Rename columns with suffix
    renamed_targets = barrier_targets.copy()
    renamed_targets.columns = [f"{col}{suffix}" for col in barrier_targets.columns]
    
    # Concatenate all at once to avoid fragmentation
    result_df = pd.concat([result_df, renamed_targets], axis=1)
    
    return result_df

def add_zigzag_targets_to_dataframe(
    df: pd.DataFrame,
    price_col: str = 'close',
    horizons: list = [5, 10, 15, 30, 60],
    reversal_pcts: list = [0.05, 0.1, 0.2]
) -> pd.DataFrame:
    """Add ZigZag targets to DataFrame."""
    
    result_df = df.copy()
    prices = df[price_col]
    
    for horizon in horizons:
        for reversal_pct in reversal_pcts:
            logger.info(f"Computing ZigZag targets for horizon={horizon}m, reversal_pct={reversal_pct}")
            
            zigzag_targets = compute_zigzag_targets(
                prices,
                horizon_minutes=horizon,
                reversal_pct=reversal_pct
            )
            
            if len(zigzag_targets) == 0:
                continue
                
            suffix = f"_{horizon}m_{reversal_pct:.2f}"
            
            # Rename columns with suffix
            renamed_targets = zigzag_targets.copy()
            renamed_targets.columns = [f"{col}{suffix}" for col in zigzag_targets.columns]
            
            # Concatenate all at once to avoid fragmentation
            result_df = pd.concat([result_df, renamed_targets], axis=1)
    
    return result_df

def add_mfe_mdd_targets_to_dataframe(
    df: pd.DataFrame,
    price_col: str = 'close',
    horizons: list = [5, 10, 15, 30, 60],
    thresholds: list = [0.001, 0.002, 0.005]
) -> pd.DataFrame:
    """Add MFE/MDD targets to DataFrame."""
    
    result_df = df.copy()
    prices = df[price_col]
    
    for horizon in horizons:
        for threshold in thresholds:
            logger.info(f"Computing MFE/MDD targets for horizon={horizon}m, threshold={threshold}")
            
            mfe_mdd_targets = compute_mfe_mdd_targets(
                prices,
                horizon_minutes=horizon,
                threshold_up=threshold,
                threshold_down=-threshold
            )
            
            if len(mfe_mdd_targets) == 0:
                continue
                
            suffix = f"_{horizon}m_{threshold:.3f}"
            
            # Rename columns with suffix
            renamed_targets = mfe_mdd_targets.copy()
            renamed_targets.columns = [f"{col}{suffix}" for col in mfe_mdd_targets.columns]
            
            # Concatenate all at once to avoid fragmentation
            result_df = pd.concat([result_df, renamed_targets], axis=1)
    
    return result_df


# ==============================================================================
# ENHANCED TARGET FAMILIES (High-Signal Additions)
# ==============================================================================

def compute_time_to_hit(
    prices: pd.Series,
    horizon_minutes: int = 15,
    barrier_size: float = 0.5,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Compute time-to-hit (TTH) targets for survival/policy learning.
    
    Returns:
        - tth: signed bars to first touch (+ve = up first, -ve = down first, NaN = censored)
        - tth_abs: absolute bars to first touch (unsigned)
        - hit_direction: {-1, 0, +1} which barrier hit first
    """
    returns = prices.pct_change().dropna()
    vol = returns.rolling(window=vol_window, min_periods=5).std()
    
    results = []
    
    for i in range(len(prices)):
        if i + horizon_minutes >= len(prices):
            break
            
        current_price = prices.iloc[i]
        current_vol = vol.iloc[i]
        
        if pd.isna(current_vol) or current_vol == 0:
            continue
            
        up_barrier = current_price * (1 + barrier_size * current_vol)
        down_barrier = current_price * (1 - barrier_size * current_vol)
        
        # Future path
        future_prices = prices.iloc[i+1:i+horizon_minutes+1]
        
        if len(future_prices) < horizon_minutes:
            continue
            
        # Find first touch
        up_hits = future_prices >= up_barrier
        down_hits = future_prices <= down_barrier
        
        tth = np.nan
        tth_abs = np.nan
        hit_direction = 0
        
        if up_hits.any() or down_hits.any():
            up_idx = up_hits.idxmax() if up_hits.any() else len(future_prices)
            down_idx = down_hits.idxmax() if down_hits.any() else len(future_prices)
            
            up_bars = future_prices.index.get_loc(up_idx) if up_hits.any() else horizon_minutes
            down_bars = future_prices.index.get_loc(down_idx) if down_hits.any() else horizon_minutes
            
            if up_bars < down_bars:
                tth = up_bars + 1
                tth_abs = up_bars + 1
                hit_direction = 1
            elif down_bars < up_bars:
                tth = -(down_bars + 1)
                tth_abs = down_bars + 1
                hit_direction = -1
        
        results.append({
            'tth': tth,
            'tth_abs': tth_abs,
            'hit_direction': hit_direction
        })
    
    return pd.DataFrame(results, index=prices.index[:len(results)])


def compute_ordinal_magnitude(
    prices: pd.Series,
    horizon_minutes: int = 15,
    vol_window: int = 20,
    cuts: tuple = (-2, -1, -0.5, 0.5, 1, 2)
) -> pd.DataFrame:
    """
    Compute vol-scaled ordinal magnitude buckets.
    
    Returns ordinal bins: {-3, -2, -1, 0, +1, +2, +3} based on σ-scaled forward returns.
    """
    returns = prices.pct_change().dropna()
    vol = returns.rolling(window=vol_window, min_periods=5).std()
    
    results = []
    
    for i in range(len(prices)):
        if i + horizon_minutes >= len(prices):
            break
            
        current_price = prices.iloc[i]
        future_price = prices.iloc[i + horizon_minutes]
        current_vol = vol.iloc[i]
        
        if pd.isna(current_vol) or current_vol == 0 or pd.isna(current_price) or pd.isna(future_price):
            continue
            
        # Forward return
        fwd_ret = (future_price - current_price) / current_price
        
        # σ-scaled
        z_score = fwd_ret / max(current_vol, 1e-8)
        
        # Ordinal binning
        if z_score <= cuts[0]:
            ordinal = -3
        elif z_score <= cuts[1]:
            ordinal = -2
        elif z_score <= cuts[2]:
            ordinal = -1
        elif z_score <= cuts[3]:
            ordinal = 0
        elif z_score <= cuts[4]:
            ordinal = 1
        elif z_score <= cuts[5]:
            ordinal = 2
        else:
            ordinal = 3
            
        results.append({
            'ret_ord': ordinal,
            'ret_zscore': z_score
        })
    
    return pd.DataFrame(results, index=prices.index[:len(results)])


def compute_path_quality(
    prices: pd.Series,
    horizon_minutes: int = 15
) -> pd.DataFrame:
    """
    Compute path-aware quality metrics.
    
    Returns:
        - mfe_share: MFE / (MFE + |MDD|) in [0, 1]
        - time_in_profit: fraction of bars with positive return
        - flipcount: number of sign changes
    """
    results = []
    
    for i in range(len(prices)):
        if i + horizon_minutes >= len(prices):
            break
            
        current_price = prices.iloc[i]
        future_prices = prices.iloc[i+1:i+horizon_minutes+1]
        
        if len(future_prices) == 0:
            continue
            
        # Path returns
        path_returns = (future_prices - current_price) / current_price
        
        # MFE/MDD
        mfe = path_returns.max() if len(path_returns) > 0 else 0
        mdd = path_returns.min() if len(path_returns) > 0 else 0
        
        # MFE share (quality metric)
        if mfe > 0 or mdd < 0:
            mfe_share = mfe / (mfe + abs(mdd)) if (mfe + abs(mdd)) > 0 else 0.5
        else:
            mfe_share = 0.5
            
        # Time in profit
        time_in_profit = (path_returns > 0).sum() / len(path_returns) if len(path_returns) > 0 else 0
        
        # Flip count (sign changes)
        signs = np.sign(path_returns.values)
        flipcount = np.sum(np.diff(signs) != 0)
        
        results.append({
            'mfe_share': mfe_share,
            'time_in_profit': time_in_profit,
            'flipcount': flipcount
        })
    
    return pd.DataFrame(results, index=prices.index[:len(results)])


def compute_asymmetric_barriers(
    prices: pd.Series,
    horizon_minutes: int = 15,
    tp_mult: float = 1.0,  # Take-profit multiplier
    sl_mult: float = 0.5,  # Stop-loss multiplier
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Compute asymmetric triple-barrier targets (e.g., 2:1 profit:stop).
    
    Returns:
        - hit_asym: {+1 (tp), -1 (sl), 0 (none)}
        - tth_asym: signed time to hit asymmetric barriers
    """
    returns = prices.pct_change().dropna()
    vol = returns.rolling(window=vol_window, min_periods=5).std()
    
    results = []
    
    for i in range(len(prices)):
        if i + horizon_minutes >= len(prices):
            break
            
        current_price = prices.iloc[i]
        current_vol = vol.iloc[i]
        
        if pd.isna(current_vol) or current_vol == 0:
            continue
            
        # Asymmetric barriers
        tp_barrier = current_price * (1 + tp_mult * current_vol)
        sl_barrier = current_price * (1 - sl_mult * current_vol)
        
        future_prices = prices.iloc[i+1:i+horizon_minutes+1]
        
        if len(future_prices) < horizon_minutes:
            continue
            
        # Find first touch
        tp_hits = future_prices >= tp_barrier
        sl_hits = future_prices <= sl_barrier
        
        hit_asym = 0
        tth_asym = np.nan
        
        if tp_hits.any() or sl_hits.any():
            tp_idx = tp_hits.idxmax() if tp_hits.any() else len(future_prices)
            sl_idx = sl_hits.idxmax() if sl_hits.any() else len(future_prices)
            
            tp_bars = future_prices.index.get_loc(tp_idx) if tp_hits.any() else horizon_minutes
            sl_bars = future_prices.index.get_loc(sl_idx) if sl_hits.any() else horizon_minutes
            
            if tp_bars < sl_bars:
                hit_asym = 1
                tth_asym = tp_bars + 1
            elif sl_bars < tp_bars:
                hit_asym = -1
                tth_asym = -(sl_bars + 1)
        
        results.append({
            'hit_asym': hit_asym,
            'tth_asym': tth_asym
        })
    
    return pd.DataFrame(results, index=prices.index[:len(results)])


# ==============================================================================
# ADD ENHANCED TARGETS TO DATAFRAME
# ==============================================================================

def add_enhanced_targets_to_dataframe(
    df: pd.DataFrame,
    price_col: str = 'close',
    horizons: list = [5, 10, 15, 30, 60],
    barrier_sizes: list = [0.3, 0.5, 0.8],
    tp_sl_ratios: list = [(1.0, 0.5), (1.5, 0.75), (2.0, 1.0)]
) -> pd.DataFrame:
    """
    Add enhanced target families to DataFrame:
    - Time-to-hit (TTH)
    - Ordinal magnitude buckets
    - Path quality metrics
    - Asymmetric barriers
    """
    result_df = df.copy()
    prices = df[price_col]
    
    logger.info("Adding enhanced targets: TTH, ordinal, path quality, asymmetric barriers")
    
    for horizon in horizons:
        # Time-to-hit for each barrier size
        for barrier_size in barrier_sizes:
            logger.info(f"Computing TTH for horizon={horizon}m, barrier={barrier_size}")
            tth_targets = compute_time_to_hit(
                prices,
                horizon_minutes=horizon,
                barrier_size=barrier_size
            )
            
            if len(tth_targets) > 0:
                suffix = f"_{horizon}m_{barrier_size:.1f}"
                renamed = tth_targets.copy()
                renamed.columns = [f"{col}{suffix}" for col in tth_targets.columns]
                result_df = pd.concat([result_df, renamed], axis=1)
        
        # Ordinal magnitude (once per horizon)
        logger.info(f"Computing ordinal magnitude for horizon={horizon}m")
        ordinal_targets = compute_ordinal_magnitude(
            prices,
            horizon_minutes=horizon
        )
        
        if len(ordinal_targets) > 0:
            suffix = f"_{horizon}m"
            renamed = ordinal_targets.copy()
            renamed.columns = [f"{col}{suffix}" for col in ordinal_targets.columns]
            result_df = pd.concat([result_df, renamed], axis=1)
        
        # Path quality (once per horizon)
        logger.info(f"Computing path quality for horizon={horizon}m")
        path_targets = compute_path_quality(
            prices,
            horizon_minutes=horizon
        )
        
        if len(path_targets) > 0:
            suffix = f"_{horizon}m"
            renamed = path_targets.copy()
            renamed.columns = [f"{col}{suffix}" for col in path_targets.columns]
            result_df = pd.concat([result_df, renamed], axis=1)
        
        # Asymmetric barriers
        for tp_mult, sl_mult in tp_sl_ratios:
            logger.info(f"Computing asymmetric barriers for horizon={horizon}m, tp={tp_mult}, sl={sl_mult}")
            asym_targets = compute_asymmetric_barriers(
                prices,
                horizon_minutes=horizon,
                tp_mult=tp_mult,
                sl_mult=sl_mult
            )
            
            if len(asym_targets) > 0:
                suffix = f"_{horizon}m_{tp_mult:.1f}_{sl_mult:.1f}"
                renamed = asym_targets.copy()
                renamed.columns = [f"{col}{suffix}" for col in asym_targets.columns]
                result_df = pd.concat([result_df, renamed], axis=1)
    
    return result_df
