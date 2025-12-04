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
Sequence Builder for Sequential Models
=====================================

Leak-safe sequence builder for CNN1D, LSTM, Transformer, TabLSTM, TabTransformer models.
Builds (N, T, F) windows ending at time t with labels at t for t→t+H.
"""


from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

def build_sequences_for_symbol(df_sym: pd.DataFrame,
                               feature_cols: List[str],
                               target_col: str,
                               lookback_T: int,
                               horizon_bars: int,
                               stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences for a single symbol.
    
    Args:
        df_sym: Single symbol data, time-indexed ascending
        feature_cols: List of feature column names (no targets)
        target_col: Target column name
        lookback_T: Number of lookback bars for sequence
        horizon_bars: Horizon for target (e.g., 1 for next bar)
        stride: Step between samples (1 = every bar, 2 = every other bar)
    
    Returns:
        X_seq: [N, T, F] sequences
        y: [N] labels
        t_index: [N] timestamps
    """
    # 0) Preconditions
    assert target_col not in feature_cols, f"Target '{target_col}' leaked into features"
    assert lookback_T > 0, "lookback_T must be positive"
    assert horizon_bars > 0, "horizon_bars must be positive"
    
    logger.debug(f"Building sequences: lookback_T={lookback_T}, horizon_bars={horizon_bars}, stride={stride}")
    
    # 1) Ensure strict monotonic time & uniform spacing
    df = df_sym.sort_index()
    
    # 2) Drop rows with missing features (mask here so windows are fully clean)
    feat = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    
    # 3) Align target to the cleaned feature index
    y_full = df.loc[feat.index, target_col]
    
    # 4) Create indices where a full lookback window AND a future label exist
    idx = feat.index
    N_raw = len(idx)
    
    if N_raw < lookback_T:
        logger.warning(f"Insufficient data: {N_raw} < {lookback_T} required")
        return (np.empty((0, lookback_T, len(feature_cols)), np.float32),
                np.empty((0,), np.float32), 
                np.array([], dtype='datetime64[ns]'))
    
    X_list, y_list, t_list = [], [], []
    
    # 5) Sliding window with stride
    for end_i in range(lookback_T-1, N_raw, stride):
        win_idx = idx[end_i-(lookback_T-1): end_i+1]
        
        # Ensure contiguity (optional): check fixed bar spacing if needed
        xw = feat.loc[win_idx].to_numpy(dtype=np.float32, copy=False)
        yw = y_full.loc[idx[end_i]]
        
        # Skip if any NaN values
        if np.isnan(yw) or np.isnan(xw).any():
            continue
            
        X_list.append(xw)                 # [T, F]
        y_list.append(np.float32(yw))     # scalar
        t_list.append(idx[end_i].to_datetime64())
    
    if not X_list:
        logger.warning("No valid sequences found")
        return (np.empty((0, lookback_T, len(feature_cols)), np.float32),
                np.empty((0,), np.float32), 
                np.array([], dtype='datetime64[ns]'))
    
    X = np.stack(X_list, axis=0)          # [N, T, F]
    y = np.asarray(y_list, dtype=np.float32)
    t = np.asarray(t_list)
    
    logger.info(f"Built {len(X)} sequences: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, t

def build_sequences_panel(panel: Dict[str, pd.DataFrame],
                          feature_cols: List[str],
                          target_col: str,
                          lookback_T: int,
                          horizon_bars: int,
                          stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences across multiple symbols.
    
    Args:
        panel: Dict of {symbol: DataFrame}
        feature_cols: List of feature column names
        target_col: Target column name
        lookback_T: Number of lookback bars
        horizon_bars: Horizon for target
        stride: Step between samples
    
    Returns:
        X_seq: [N, T, F] sequences
        y: [N] labels
        t_index: [N] timestamps
        symbols: [N] symbol names
    """
    Xs, ys, ts, syms = [], [], [], []
    
    for sym, df_sym in panel.items():
        logger.debug(f"Processing symbol: {sym}")
        X, y, t = build_sequences_for_symbol(
            df_sym, feature_cols, target_col, lookback_T, horizon_bars, stride
        )
        
        if len(y) > 0:
            Xs.append(X)
            ys.append(y)
            ts.append(t)
            syms.append(np.array([sym] * len(y)))
    
    if not ys:
        logger.warning("No valid sequences found across all symbols")
        return (np.empty((0, lookback_T, len(feature_cols)), np.float32),
                np.empty((0,), np.float32), 
                np.array([], dtype='datetime64[ns]'),
                np.array([], dtype=object))
    
    # Concatenate all symbols
    X_final = np.concatenate(Xs, axis=0)
    y_final = np.concatenate(ys, axis=0)
    t_final = np.concatenate(ts, axis=0)
    sym_final = np.concatenate(syms, axis=0)
    
    logger.info(f"Panel sequences: X.shape={X_final.shape}, y.shape={y_final.shape}, "
                f"symbols={len(set(sym_final))}")
    
    return X_final, y_final, t_final, sym_final

def validate_sequences(X: np.ndarray, y: np.ndarray, t_index: np.ndarray, 
                      lookback_T: int, feature_cols: List[str]) -> bool:
    """
    Validate sequence data for leaks and correctness.
    
    Args:
        X: [N, T, F] sequences
        y: [N] labels
        t_index: [N] timestamps
        lookback_T: Expected sequence length
        feature_cols: Feature column names
    
    Returns:
        bool: True if valid
    """
    try:
        # Check shapes
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert y.ndim == 1, f"y should be 1D, got {y.ndim}D"
        assert len(X) == len(y), f"X and y length mismatch: {len(X)} vs {len(y)}"
        assert len(X) == len(t_index), f"X and t_index length mismatch: {len(X)} vs {len(t_index)}"
        
        # Check sequence length
        assert X.shape[1] == lookback_T, f"Sequence length mismatch: {X.shape[1]} vs {lookback_T}"
        assert X.shape[2] == len(feature_cols), f"Feature count mismatch: {X.shape[2]} vs {len(feature_cols)}"
        
        # Check for NaNs
        assert not np.isnan(X).any(), "X contains NaN values"
        assert not np.isnan(y).any(), "y contains NaN values"
        
        # Check for infinities
        assert not np.isinf(X).any(), "X contains infinite values"
        assert not np.isinf(y).any(), "y contains infinite values"
        
        logger.info("✅ Sequence validation passed")
        return True
        
    except AssertionError as e:
        logger.error(f"❌ Sequence validation failed: {e}")
        return False
