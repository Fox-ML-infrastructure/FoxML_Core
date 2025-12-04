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

"""Data processing utility functions extracted from original 5K line file."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def strip_targets(cols, all_targets=None):
    """

    Remove ALL target-like columns from feature list.
    
    Args:
        cols: List of column names
        all_targets: Set of all discovered target columns (if None, uses heuristics)
    
    Returns:
        List of feature columns only (no targets, no symbol/timestamp)
    """
    if all_targets is None:
        # Fallback to heuristics if all_targets not provided
        EXCLUDE_PREFIXES = ("fwd_ret_", "will_peak", "will_valley", "mdd_", "mfe_", "y_will_")
        return [c for c in cols if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in ("symbol", "timestamp")]
    else:
        # Use explicit target list for precise filtering
        return [c for c in cols if c not in all_targets and c not in ("symbol", "timestamp")]



def collapse_identical_duplicate_columns(df):
    """
    Collapse identical duplicate columns, raise error if conflicting.
    
    Args:
        df: DataFrame with potentially duplicate columns
        
    Returns:
        DataFrame with unique columns, duplicates removed
    """
    if len(df.columns) == len(set(df.columns)):
        return df  # No duplicates
    
    # Group columns by name
    from collections import defaultdict
    col_groups = defaultdict(list)
    for i, col in enumerate(df.columns):
        col_groups[col].append(i)
    
    # Check for conflicts and remove exact duplicates
    new_cols = []
    for col_name, indices in col_groups.items():
        if len(indices) == 1:
            new_cols.append(col_name)
        else:
            # Multiple columns with same name - check if identical
            first_idx = indices[0]
            is_identical = all(df.iloc[:, idx].equals(df.iloc[:, first_idx]) for idx in indices[1:])
            
            if is_identical:
                # Keep one copy
                new_cols.append(col_name)
            else:
                # Conflicting data - this is the root cause of our crashes
                logger.error(f"❌ CONFLICTING DUPLICATE COLUMNS: '{col_name}' has {len(indices)} copies with different data")
                logger.error(f"   This will cause tolist() crashes when df['{col_name}'] returns 2-D DataFrame")
                raise ValueError(f"Conflicting duplicate columns found: '{col_name}' has {len(indices)} copies with different data")
    
    return df[new_cols]



def prepare_sequence_cs(mtf_data, feature_cols, target_cols, lookback=64, min_cs=10,
                        val_start_ts=None, train_ratio=0.8):
    """
    Build temporal sequences for neural network training.
    Creates (batch, lookback, features) sequences with causal structure.
    """
    # 1) detect time col
    time_col = resolve_time_col(next(iter(mtf_data.values())))
    SYM = SYMBOL_COL

    # 2) concat (keep only time, sym, feats, targets)
    dfs = []
    for s, df in mtf_data.items():
        have = [c for c in feature_cols if c in df.columns]
        t_have = [t for t in target_cols if t in df.columns]
        if not t_have: 
            continue
        d = df[[time_col, *have, *t_have]].copy()
        d[SYM] = s
        d = d.sort_values(time_col, kind="mergesort").dropna(subset=t_have)
        dfs.append(d)
    if not dfs:
        raise ValueError("no data for requested targets")

    big = pd.concat(dfs, ignore_index=True)

    # 3) keep timestamps with enough cross-section (so CS eval still makes sense)
    counts = big.groupby(time_col)[SYM].nunique()
    keep_ts = counts[counts >= min_cs].index
    big = big[big[time_col].isin(keep_ts)].copy()

    # 4) build rolling sequences per-symbol (causal: last step is "t", predicts targets at "t")
    Xseq, ymat, ts_last = [], [], []
    for s, g in big.groupby(SYM, sort=False):
        g = g.sort_values(time_col, kind="mergesort")
        F = g[feature_cols].to_numpy(dtype=np.float32)
        T = g[target_cols].to_numpy(dtype=np.float32)
        ts = g[time_col].to_numpy()
        if len(g) < lookback: 
            continue
        for i in range(lookback, len(g)):  # window [i-lookback, i)
            Xseq.append(F[i-lookback:i])
            ymat.append(T[i])              # predict forward returns anchored at time ts[i]
            ts_last.append(ts[i])
    Xseq = np.stack(Xseq)           # (N, L, F)
    ymat = np.stack(ymat)           # (N, n_tasks)
    ts_last = np.asarray(ts_last)   # (N,)

    # 5) time-aware split (no leakage)
    if val_start_ts is not None:
        try:
            cut = np.datetime64(val_start_ts)
        except Exception:
            cut = val_start_ts  # numeric timestamps still ok
        tr_idx = ts_last < cut
        va_idx = ~tr_idx
        train_ts = set(np.unique(ts_last[tr_idx]))
        val_ts   = set(np.unique(ts_last[va_idx]))
    else:
        tr_idx, va_idx, train_ts, val_ts = create_time_aware_split(ts_last, train_ratio=train_ratio)

    pack = {
        "X_tr": Xseq[tr_idx], "X_va": Xseq[va_idx],
        "y_tr": ymat[tr_idx], "y_va": ymat[va_idx],
        "ts_tr": ts_last[tr_idx], "ts_va": ts_last[va_idx],
        "feature_cols": list(feature_cols),
        "task_names": list(target_cols),
        "lookback": lookback, "time_col": time_col,
        "train_ts": train_ts, "val_ts": val_ts
    }
    return pack



