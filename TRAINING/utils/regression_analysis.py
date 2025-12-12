"""
Cohort-Based Regression Analysis

Builds regression datasets from index.parquet, fits recency-weighted models,
and produces per-cohort trend + predictive signals.

Key features:
- Segments cohorts by identity breaks (data_fingerprint, config_hash, etc.)
- Creates lag/diff features for time-series regression
- Applies exponential decay weighting (recency weighting)
- Walk-forward validation within segments (no leakage)
"""

from __future__ import annotations

import math
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, log_loss
import logging

logger = logging.getLogger(__name__)

# Identity columns that trigger segment breaks
IDENTITY_COLS = ["data_fingerprint", "featureset_fingerprint", "config_hash", "git_commit"]


def prepare_segments(
    df: pd.DataFrame,
    time_col: str = "run_started_at",
    identity_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add segment_id and run_seq_in_cohort to dataframe.
    
    Segments are created when any identity field changes within a cohort.
    This prevents regressing across regime breaks (data/config changes).
    
    Args:
        df: DataFrame with cohort_id, time_col, and identity columns
        time_col: Column name for timestamps
        identity_cols: List of columns that trigger segment breaks (default: IDENTITY_COLS)
    
    Returns:
        DataFrame with added segment_id and run_seq_in_cohort columns
    """
    if identity_cols is None:
        identity_cols = IDENTITY_COLS
    
    out = df.copy()
    
    if time_col not in out.columns:
        raise ValueError(f"Missing required time column: {time_col}")
    
    # Parse timestamps
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out.sort_values(["cohort_id", time_col])
    
    def _segmentize(g: pd.DataFrame) -> pd.DataFrame:
        """Create segments within a cohort based on identity changes."""
        g = g.copy()
        changed = np.zeros(len(g), dtype=bool)
        
        # Check each identity column for changes
        for col in identity_cols:
            if col in g.columns:
                # Mark rows where value changes from previous row
                changed |= g[col].ne(g[col].shift(1)).fillna(False).to_numpy()
        
        # First row is always a new segment
        changed[0] = True
        
        # Create segment_id (increments on each change)
        g["segment_id"] = np.cumsum(changed) - 1
        
        # Create run sequence within cohort
        g["run_seq_in_cohort"] = np.arange(len(g), dtype=int)
        
        return g
    
    # Apply segmentization per cohort
    out = out.groupby("cohort_id", group_keys=False).apply(_segmentize)
    
    return out


def exp_decay_weights(
    timestamps: pd.Series,
    half_life_days: float = 14.0
) -> np.ndarray:
    """
    Compute exponential decay weights for recency weighting.
    
    More recent runs get higher weights. Uses exponential decay with configurable half-life.
    
    Args:
        timestamps: Series of timestamps
        half_life_days: Half-life in days (default: 14 days)
    
    Returns:
        Array of weights (sums to ~1, normalized)
    """
    t = pd.to_datetime(timestamps, utc=True, errors="coerce")
    t_max = t.max()
    age_days = (t_max - t).dt.total_seconds() / (3600 * 24)
    lam = math.log(2.0) / half_life_days
    w = np.exp(-lam * age_days.to_numpy())
    
    # Normalize to sum to 1 (for stability)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    
    return w


def make_regression_frame(
    df: pd.DataFrame,
    target: str = "cs_auc",
    feature_cols: Optional[List[str]] = None,
    time_col: str = "run_started_at",
    min_runs_per_segment: int = 8
) -> pd.DataFrame:
    """
    Build regression dataset with lag and diff features.
    
    Creates features that predict current target from previous step signals.
    Each row predicts y_t from X_{t-1} (lag features) and ΔX_t (diff features).
    
    Args:
        df: DataFrame with segments and features
        target: Target column name (Y variable)
        feature_cols: List of feature columns (X variables)
        time_col: Timestamp column for sorting
        min_runs_per_segment: Minimum runs required per segment (default: 8)
    
    Returns:
        DataFrame with lag/diff features and aligned target
    """
    if feature_cols is None:
        feature_cols = [
            "n_features_selected",
            "n_features_post_prune",
            "pos_rate",
            "purge_minutes_used",
            "embargo_minutes_used",
            "horizon_minutes",
            "jaccard_topK",
            "rank_corr_spearman",
            "importance_concentration",
            "runtime_sec",
            "peak_ram_mb",
            "cv_folds_executed",
        ]
    
    # Keep only relevant columns
    keep = ["cohort_id", "segment_id", time_col, target] + [
        c for c in feature_cols if c in df.columns
    ]
    d = df[keep].copy()
    d = d.sort_values(["cohort_id", "segment_id", time_col])
    
    frames = []
    
    for (cohort_id, segment_id), g in d.groupby(["cohort_id", "segment_id"], sort=False):
        g = g.copy().reset_index(drop=True)
        
        # Skip segments with too few runs
        if len(g) < min_runs_per_segment:
            logger.debug(f"Skipping segment {cohort_id}/{segment_id}: only {len(g)} runs (min: {min_runs_per_segment})")
            continue
        
        # Handle missing values: impute with median within segment
        for c in feature_cols:
            if c in g.columns:
                median_val = g[c].median()
                if pd.notna(median_val):
                    g[c] = g[c].fillna(median_val)
        
        # Create lag features (previous timestep)
        for c in feature_cols:
            if c in g.columns:
                g[f"{c}_lag1"] = g[c].shift(1)
        
        # Create diff features (change from previous timestep)
        for c in feature_cols:
            if c in g.columns:
                g[f"d_{c}"] = g[c] - g[c].shift(1)
        
        # Predict current target from previous step signals
        X_cols = [c for c in g.columns if c.endswith("_lag1") or c.startswith("d_")]
        
        # Drop rows with missing target or features
        g = g.dropna(subset=[target] + X_cols)
        
        if len(g) >= min_runs_per_segment:
            # Store X_cols for downstream use
            g["X_cols"] = [X_cols] * len(g)
            frames.append(g)
    
    if not frames:
        logger.warning("No segments with sufficient data for regression")
        return pd.DataFrame()
    
    result = pd.concat(frames, ignore_index=True)
    return result


def fit_weighted_ridge(
    frame: pd.DataFrame,
    target: str = "cs_auc",
    time_col: str = "run_started_at",
    half_life_days: float = 14.0,
    alpha: float = 1.0
) -> Tuple[Ridge, List[str]]:
    """
    Fit weighted Ridge regression with recency weighting.
    
    Args:
        frame: Regression frame from make_regression_frame
        target: Target column name
        time_col: Timestamp column for weighting
        half_life_days: Half-life for exponential decay
        alpha: Ridge regularization strength
    
    Returns:
        (fitted_model, feature_names)
    """
    if len(frame) == 0:
        raise ValueError("Empty frame: cannot fit model")
    
    X_cols = frame["X_cols"].iloc[0]
    X = frame[X_cols].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    w = exp_decay_weights(frame[time_col], half_life_days=half_life_days)
    
    model = Ridge(alpha=alpha)
    model.fit(X, y, sample_weight=w)
    
    return model, X_cols


def walk_forward_ridge(
    frame: pd.DataFrame,
    target: str = "cs_auc",
    time_col: str = "run_started_at",
    half_life_days: float = 14.0,
    alpha: float = 1.0,
    min_train: int = 6
) -> Dict[str, float]:
    """
    Walk-forward validation for Ridge regression.
    
    Trains on runs [0..k], predicts k+1, rolls forward.
    No leakage: only uses past data to predict future.
    
    Args:
        frame: Regression frame (must be sorted by time)
        target: Target column name
        time_col: Timestamp column
        half_life_days: Half-life for weighting
        alpha: Ridge regularization
        min_train: Minimum training samples before predicting
    
    Returns:
        Dict with MAE, R², and n_test
    """
    if len(frame) == 0:
        return {"mae": np.nan, "r2": np.nan, "n_test": 0}
    
    X_cols = frame["X_cols"].iloc[0]
    frame = frame.sort_values(time_col).reset_index(drop=True)
    
    preds, ys = [], []
    
    for k in range(min_train, len(frame)):
        train = frame.iloc[:k]
        test = frame.iloc[k:k+1]
        
        if len(test) == 0:
            continue
        
        Xtr = train[X_cols].to_numpy(dtype=float)
        ytr = train[target].to_numpy(dtype=float)
        wtr = exp_decay_weights(train[time_col], half_life_days=half_life_days)
        
        model = Ridge(alpha=alpha)
        model.fit(Xtr, ytr, sample_weight=wtr)
        
        yhat = model.predict(test[X_cols].to_numpy(dtype=float))[0]
        preds.append(yhat)
        ys.append(test[target].iloc[0])
    
    if len(preds) < 2:
        return {"mae": np.nan, "r2": np.nan, "n_test": len(preds)}
    
    return {
        "mae": float(mean_absolute_error(ys, preds)),
        "r2": float(r2_score(ys, preds)),
        "n_test": int(len(preds)),
    }


def analyze_cohort_trends(
    index_file: Path,
    target: str = "cs_auc",
    half_life_days: float = 14.0,
    min_runs_per_segment: int = 8
) -> pd.DataFrame:
    """
    Analyze trends across all cohorts in index.parquet.
    
    Args:
        index_file: Path to index.parquet
        target: Target metric to analyze
        half_life_days: Half-life for recency weighting
        min_runs_per_segment: Minimum runs per segment
    
    Returns:
        DataFrame with per-cohort/segment analysis results
    """
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        return pd.DataFrame()
    
    df = pd.read_parquet(index_file)
    
    # Prepare segments
    df = prepare_segments(df)
    
    # Build regression frame
    reg_frame = make_regression_frame(
        df, target=target, min_runs_per_segment=min_runs_per_segment
    )
    
    if len(reg_frame) == 0:
        logger.warning("No regression data available")
        return pd.DataFrame()
    
    # Analyze per segment
    results = []
    
    for (cohort_id, segment_id), g in reg_frame.groupby(["cohort_id", "segment_id"], sort=False):
        if len(g) < min_runs_per_segment:
            continue
        
        try:
            # Fit model
            model, X_cols = fit_weighted_ridge(
                g, target=target, half_life_days=half_life_days
            )
            
            # Walk-forward validation
            wf_results = walk_forward_ridge(
                g, target=target, half_life_days=half_life_days
            )
            
            # Predict next run (if we have recent data)
            if len(g) > 0:
                last_row = g.iloc[-1]
                X_next = last_row[X_cols].to_numpy(dtype=float).reshape(1, -1)
                next_pred = model.predict(X_next)[0]
            else:
                next_pred = np.nan
            
            results.append({
                "cohort_id": cohort_id,
                "segment_id": segment_id,
                "n_runs": len(g),
                "target": target,
                "wf_mae": wf_results["mae"],
                "wf_r2": wf_results["r2"],
                "wf_n_test": wf_results["n_test"],
                "next_pred": next_pred,
                "last_actual": g[target].iloc[-1] if len(g) > 0 else np.nan,
                "trend": "improving" if next_pred > g[target].iloc[-2] else "declining" if len(g) >= 2 else "unknown"
            })
        except Exception as e:
            logger.warning(f"Failed to analyze {cohort_id}/{segment_id}: {e}")
            continue
    
    return pd.DataFrame(results)
