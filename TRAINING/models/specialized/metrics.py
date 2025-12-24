# MIT License - see LICENSE file

"""Specialized model classes extracted from original 5K line file."""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


"""Metrics functions for specialized models."""

def cs_metrics_by_time(y_true: np.ndarray, y_pred: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
    """Calculate cross-sectional metrics per timestamp (true CS evaluation)."""
    try:
        from scipy.stats import spearmanr, pearsonr
        scipy_available = True
    except Exception:
        scipy_available = False
        
    ts = np.asarray(ts)
    ic_list, ric_list = [], []
    grp_sizes, grp_hits = [], []
    
    total_timestamps = len(np.unique(ts))
    skipped_timestamps = 0
    
    # Single pass through unique timestamps
    for t in np.unique(ts):
        m = (ts == t)
        if m.sum() <= 2:
            skipped_timestamps += 1
            continue
        y_t, pred_t = y_true[m], y_pred[m]
        
        # skip degenerate groups
        if np.std(y_t) < 1e-12 or np.std(pred_t) < 1e-12:
            skipped_timestamps += 1
            continue
        
        # Compute correlations
        if scipy_available:
            ic = pearsonr(y_t, pred_t)[0]
            ric = spearmanr(y_t, pred_t)[0]
        else:
            # Simple numpy fallback
            def _corr(a, b):
                if a.size < 2: return np.nan
                return float(np.corrcoef(a, b)[0,1])
            ic = _corr(y_t, pred_t)
            # Rank-IC fallback
            ric = _corr(y_t.argsort().argsort(), pred_t.argsort().argsort())
        
        if not np.isnan(ic): ic_list.append(ic)
        if not np.isnan(ric): ric_list.append(ric)
        
        # Hit rate per timestamp: majority vote on direction
        hit_rate_t = float(np.mean(np.sign(y_t) == np.sign(pred_t)))
        grp_sizes.append(m.sum())
        grp_hits.append(hit_rate_t)
    
    # Weight hit rate by group size
    hit_rate = float(np.average(grp_hits, weights=grp_sizes)) if grp_sizes else 0.0
    
    # Log fraction of skipped timestamps
    if total_timestamps > 0:
        skipped_fraction = skipped_timestamps / total_timestamps
        if skipped_fraction > 0.1:  # Log if more than 10% skipped
            logger.warning(f"⚠️  Skipped {skipped_timestamps}/{total_timestamps} timestamps ({skipped_fraction:.1%}) due to degenerate groups")
        else:
            logger.info(f"📊 Skipped {skipped_timestamps}/{total_timestamps} timestamps ({skipped_fraction:.1%}) due to degenerate groups")
    
    # Calculate IC_IR (Information Ratio)
    ic_arr = np.asarray(ic_list)
    ic_ir = float(ic_arr.mean() / (ic_arr.std(ddof=1) + 1e-12)) if ic_list else 0.0
    
    return {
        "mean_IC": float(np.mean(ic_list)) if ic_list else 0.0,
        "mean_RankIC": float(np.mean(ric_list)) if ric_list else 0.0,
        "IC_IR": ic_ir,
        "n_times": int(len(ic_list)),
        "hit_rate": hit_rate,
        "skipped_timestamps": skipped_timestamps,
        "total_timestamps": total_timestamps
    }

