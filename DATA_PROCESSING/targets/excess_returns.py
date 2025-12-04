#!/usr/bin/env python3

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
Target Construction Module

Implements excess-return labels (classification) and regression targets:
- Rolling beta to market
- Future excess returns over horizon H
- Neutral band epsilon computation (train-only)
- 3-class labels via neutral band (SELL=-1, HOLD=0, BUY=+1)
"""


import numpy as np
import pandas as pd
from typing import Tuple, Union


def rolling_beta(asset_ret: pd.Series, mkt_ret: pd.Series, win: int = 60) -> pd.Series:
    """
    Calculate rolling beta of asset returns to market returns
    
    Args:
        asset_ret: Asset returns
        mkt_ret: Market returns
        win: Rolling window size
    
    Returns:
        Rolling beta series
    """
    cov = asset_ret.rolling(win).cov(mkt_ret)
    var = mkt_ret.rolling(win).var()
    beta = cov / var
    return beta


def future_excess_return(asset_ret: pd.Series, mkt_ret: pd.Series, H: int) -> pd.Series:
    """
    Calculate future excess returns over horizon H days
    
    Args:
        asset_ret: Asset returns
        mkt_ret: Market returns  
        H: Horizon in days
    
    Returns:
        Future excess return series
    """
    beta = rolling_beta(asset_ret, mkt_ret, win=60).bfill()
    resid = asset_ret - beta * mkt_ret
    fut = resid.rolling(H).sum().shift(-H)  # H-day future excess return
    return fut


def compute_epsilon_train_only(fut_excess: pd.Series, train_idx: pd.Index, q: float = 0.5) -> float:
    """
    Compute epsilon (neutral band) from training data only
    
    Args:
        fut_excess: Future excess returns
        train_idx: Training data indices
        q: Quantile for epsilon (0.5 = median)
    
    Returns:
        Epsilon value for neutral band
    
    Raises:
        AssertionError: If epsilon is invalid
    """
    x = fut_excess.loc[train_idx].dropna().abs()
    if len(x) == 0:
        raise AssertionError("No valid excess returns in training data")
    
    eps = float(np.quantile(x, q))
    eps = max(eps, 1e-3)  # floor to prevent eps<=0 fallbacks (increased from 1e-4)
    
    if eps <= 0 or not np.isfinite(eps):
        raise AssertionError(f"epsilon invalid: {eps}; check residual inputs")
    
    return eps


def label_excess_band(fut_excess: pd.Series, eps: float) -> pd.Series:
    """
    Create 3-class labels from excess returns using neutral band
    
    Args:
        fut_excess: Future excess returns
        eps: Epsilon (neutral band threshold)
    
    Returns:
        Labels: -1 (SELL), 0 (HOLD), +1 (BUY)
    """
    y = pd.Series(0, index=fut_excess.index, dtype=int)
    y[fut_excess > eps] = 1   # BUY
    y[fut_excess < -eps] = -1  # SELL
    # HOLD (0) is default for |excess| <= eps
    return y


def create_targets(asset_ret: pd.Series, mkt_ret: pd.Series, H: int, 
                  train_idx: pd.Index, eps_quantile: float = 0.5) -> Tuple[pd.Series, pd.Series, float]:
    """
    Create both classification and regression targets
    
    Args:
        asset_ret: Asset returns
        mkt_ret: Market returns
        H: Horizon in days
        train_idx: Training data indices
        eps_quantile: Quantile for epsilon calculation
    
    Returns:
        Tuple of (classification_labels, regression_targets, epsilon)
    """
    # Calculate future excess returns
    fut_excess = future_excess_return(asset_ret, mkt_ret, H)
    
    # Compute epsilon from training data only
    eps = compute_epsilon_train_only(fut_excess, train_idx, eps_quantile)
    
    # Create classification labels
    labels = label_excess_band(fut_excess, eps)
    
    return labels, fut_excess, eps


def validate_targets(labels: pd.Series, targets: pd.Series, eps: float) -> None:
    """
    Validate target construction
    
    Args:
        labels: Classification labels
        targets: Regression targets
        eps: Epsilon value
    
    Raises:
        AssertionError: If targets are invalid
    """
    # Check label distribution
    label_counts = labels.value_counts()
    if len(label_counts) < 2:
        raise AssertionError(f"Labels too imbalanced: {label_counts}")
    
    # Check epsilon is reasonable
    if eps <= 0 or eps > 0.1:  # 10% max epsilon
        raise AssertionError(f"Epsilon out of range: {eps}")
    
    # Check targets are finite
    if not np.all(np.isfinite(targets.dropna())):
        raise AssertionError("Non-finite targets detected")
    
    # Check alignment
    if not labels.index.equals(targets.index):
        raise AssertionError("Label and target indices don't match")


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    spy = yf.download("SPY", start="2023-01-01", end="2024-01-01", progress=False)
    qqq = yf.download("QQQ", start="2023-01-01", end="2024-01-01", progress=False)
    
    # Calculate returns
    spy_ret = spy['Close'].pct_change().dropna()
    qqq_ret = qqq['Close'].pct_change().dropna()
    
    # Align data
    common_idx = spy_ret.index.intersection(qqq_ret.index)
    spy_ret = spy_ret.loc[common_idx]
    qqq_ret = qqq_ret.loc[common_idx]
    
    # Split train/test
    split_idx = int(len(common_idx) * 0.7)
    train_idx = common_idx[:split_idx]
    
    # Create targets
    labels, targets, eps = create_targets(qqq_ret, spy_ret, H=5, train_idx=train_idx)
    
    print(f"Epsilon: {eps:.4f}")
    print(f"Label distribution: {labels.value_counts().to_dict()}")
    print(f"Target stats: mean={targets.mean():.4f}, std={targets.std():.4f}")
    
    # Validate
    validate_targets(labels, targets, eps)
    print("✅ Target validation passed!")
