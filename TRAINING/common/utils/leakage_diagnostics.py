# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Leakage Diagnostic Tests

Fast diagnostic checks to distinguish real signal vs CV/splitting/label construction bugs.
These are the tests that "end arguments quickly" per user requirements.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


def placebo_label_test(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_splitter,
    task_type: str = 'classification',
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Placebo label test: Shuffle y after dataset build, rerun exact same CV/eval.
    
    If AUC stays high, evaluation pipeline is leaking (split bug, fold contamination, etc.).
    Expected: ~0.50 AUC for shuffled labels.
    
    Returns:
        Dict with 'auc', 'passed' (True if auc ~0.50), 'diagnosis'
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Shuffle y (preserve distribution, break any relationship with X)
    rng = np.random.RandomState(random_seed)
    y_shuffled = rng.permutation(y)
    
    # Run CV with shuffled labels
    scores = []
    for train_idx, val_idx in cv_splitter.split(X, y_shuffled):
        # Use a simple model for speed
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=random_seed, max_iter=100, solver='liblinear')
        model.fit(X[train_idx], y_shuffled[train_idx])
        
        if task_type == 'classification':
            y_pred_proba = model.predict_proba(X[val_idx])[:, 1]
            if len(np.unique(y_shuffled[val_idx])) == 2:
                score = roc_auc_score(y_shuffled[val_idx], y_pred_proba)
            else:
                score = accuracy_score(y_shuffled[val_idx], model.predict(X[val_idx]))
        else:
            from sklearn.metrics import r2_score
            y_pred = model.predict(X[val_idx])
            score = r2_score(y_shuffled[val_idx], y_pred)
        
        scores.append(score)
    
    auc = np.mean(scores)
    
    # Pass if score is close to random (~0.50 for classification, ~0.0 for regression)
    if task_type == 'classification':
        passed = 0.45 <= auc <= 0.55
        diagnosis = "PASS" if passed else f"FAIL: Shuffled labels gave AUC={auc:.3f} (expected ~0.50). Pipeline is leaking."
    else:
        passed = -0.1 <= auc <= 0.1
        diagnosis = "PASS" if passed else f"FAIL: Shuffled labels gave R²={auc:.3f} (expected ~0.0). Pipeline is leaking."
    
    return {
        'score': auc,
        'passed': passed,
        'diagnosis': diagnosis,
        'scores': scores
    }


def future_direction_sanity_test(
    y: np.ndarray,
    prices: pd.Series,
    target_column: str,
    time_vals: Optional[np.ndarray] = None,
    sample_size: int = 20
) -> Dict[str, Any]:
    """
    Future-direction sanity test: Spot-check random rows.
    
    For random timestamps t, recompute label directly from raw prices using intended definition
    and assert it matches y[t]. Catches "will peak" accidentally implemented as "did peak".
    
    Returns:
        Dict with 'matches', 'mismatches', 'passed', 'diagnosis'
    """
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(y), size=min(sample_size, len(y)), replace=False)
    
    matches = 0
    mismatches = []
    
    # Extract horizon from target column name
    # Format: y_will_peak_60m_0.8 -> horizon=60
    import re
    horizon_match = re.search(r'(\d+)m', target_column)
    horizon_minutes = int(horizon_match.group(1)) if horizon_match else 60
    
    for idx in sample_indices:
        if time_vals is not None and idx < len(time_vals):
            t = time_vals[idx]
        else:
            t = idx
        
        # Get current price and future window
        if isinstance(prices, pd.Series):
            if idx < len(prices):
                current_price = prices.iloc[idx]
                # Future prices: from idx+1 to idx+horizon_minutes+1 (exclusive of current bar)
                if idx + horizon_minutes + 1 <= len(prices):
                    future_prices = prices.iloc[idx+1:idx+horizon_minutes+1]
                    # Simple check: does future max exceed current by threshold?
                    # This is a simplified version - actual barrier logic is more complex
                    max_future = future_prices.max()
                    threshold = 0.8  # Extract from target name if possible
                    expected_label = 1 if (max_future / current_price - 1) >= threshold else 0
                    
                    actual_label = int(y[idx]) if not np.isnan(y[idx]) else None
                    
                    if actual_label is not None and expected_label == actual_label:
                        matches += 1
                    else:
                        mismatches.append({
                            'idx': idx,
                            'expected': expected_label,
                            'actual': actual_label,
                            'current_price': current_price,
                            'max_future': max_future
                        })
    
    passed = len(mismatches) == 0
    diagnosis = f"PASS: {matches}/{len(sample_indices)} labels match expected definition" if passed else \
                f"FAIL: {len(mismatches)}/{len(sample_indices)} labels don't match expected definition. Check label construction."
    
    return {
        'matches': matches,
        'mismatches': mismatches,
        'total_checked': len(sample_indices),
        'passed': passed,
        'diagnosis': diagnosis
    }


def event_offset_histogram(
    y: np.ndarray,
    prices: pd.Series,
    target_column: str,
    horizon_minutes: int
) -> Dict[str, Any]:
    """
    Event offset histogram: For each positive label, compute earliest barrier-hit index.
    
    If big chunk hits at offset 0 (same bar) or 1 (next bar using info already in features),
    label is basically giving the answer away.
    
    Returns:
        Dict with 'histogram', 'offset_0_count', 'offset_1_count', 'passed', 'diagnosis'
    """
    positive_indices = np.where(y == 1)[0]
    
    offsets = []
    offset_0_count = 0
    offset_1_count = 0
    
    for idx in positive_indices:
        if idx + horizon_minutes + 1 <= len(prices):
            current_price = prices.iloc[idx]
            future_prices = prices.iloc[idx+1:idx+horizon_minutes+1]
            
            # Find first bar where barrier is hit
            # Simplified: barrier = current_price * 1.08 (for 0.8 threshold)
            barrier = current_price * 1.08
            hit_indices = np.where(future_prices.values >= barrier)[0]
            
            if len(hit_indices) > 0:
                offset = hit_indices[0]  # First hit
                offsets.append(offset)
                if offset == 0:
                    offset_0_count += 1
                elif offset == 1:
                    offset_1_count += 1
    
    # Create histogram
    if offsets:
        hist, bins = np.histogram(offsets, bins=min(20, max(offsets) + 1), range=(0, max(offsets) + 1))
    else:
        hist, bins = np.array([]), np.array([])
    
    # Pass if <10% hit at offset 0 or 1
    total_positive = len(positive_indices)
    offset_01_ratio = (offset_0_count + offset_1_count) / total_positive if total_positive > 0 else 0.0
    
    passed = offset_01_ratio < 0.10
    diagnosis = f"PASS: {offset_01_ratio:.1%} hit at offset 0/1 (acceptable)" if passed else \
                f"FAIL: {offset_01_ratio:.1%} hit at offset 0/1 (structural leakage - label includes current/next bar)"
    
    return {
        'histogram': (hist, bins),
        'offsets': offsets,
        'offset_0_count': offset_0_count,
        'offset_1_count': offset_1_count,
        'offset_01_ratio': offset_01_ratio,
        'passed': passed,
        'diagnosis': diagnosis
    }


def univariate_feature_auc_scan(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_splitter,
    task_type: str = 'classification',
    threshold: float = 0.90
) -> Dict[str, Any]:
    """
    Univariate feature AUC scan: Compute ROC-AUC for each feature alone.
    
    If any single feature is 0.90+, you found the leak/proxy immediately.
    
    Returns:
        Dict with 'suspicious_features', 'max_auc', 'passed', 'diagnosis'
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    
    suspicious_features = []
    feature_aucs = {}
    
    for feat_idx, feat_name in enumerate(feature_names):
        X_single = X[:, feat_idx:feat_idx+1]  # Single feature
        
        # Run CV
        scores = []
        for train_idx, val_idx in cv_splitter.split(X_single, y):
            try:
                model = LogisticRegression(random_state=42, max_iter=100, solver='liblinear')
                model.fit(X_single[train_idx], y[train_idx])
                
                if task_type == 'classification':
                    y_pred_proba = model.predict_proba(X_single[val_idx])[:, 1]
                    if len(np.unique(y[val_idx])) == 2:
                        score = roc_auc_score(y[val_idx], y_pred_proba)
                    else:
                        continue  # Skip if not binary
                else:
                    from sklearn.metrics import r2_score
                    y_pred = model.predict(X_single[val_idx])
                    score = r2_score(y[val_idx], y_pred)
                
                scores.append(score)
            except Exception as e:
                logger.debug(f"Failed to evaluate {feat_name}: {e}")
                continue
        
        if scores:
            mean_auc = np.mean(scores)
            feature_aucs[feat_name] = mean_auc
            
            if mean_auc >= threshold:
                suspicious_features.append((feat_name, mean_auc))
    
    # Sort by AUC
    suspicious_features.sort(key=lambda x: x[1], reverse=True)
    
    passed = len(suspicious_features) == 0
    diagnosis = "PASS: No single feature exceeds threshold" if passed else \
                f"FAIL: {len(suspicious_features)} feature(s) exceed AUC={threshold}: {[f[0] for f in suspicious_features[:5]]}"
    
    return {
        'suspicious_features': suspicious_features,
        'max_auc': max(feature_aucs.values()) if feature_aucs else 0.0,
        'feature_aucs': feature_aucs,
        'passed': passed,
        'diagnosis': diagnosis
    }


def raw_ohlcv_only_test(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_splitter,
    task_type: str = 'classification',
    raw_features: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    One-switch ablation: "raw OHLCV only" test.
    
    Run same target ranking with only truly time-safe raw features.
    If AUC collapses → leak is in derived feature generation.
    If AUC stays huge → label construction or split/eval bug.
    
    Returns:
        Dict with 'raw_auc', 'full_auc', 'collapsed', 'passed', 'diagnosis'
    """
    # Identify raw OHLCV features
    if raw_features is None:
        raw_features = [f for f in feature_names if any(prefix in f.lower() for prefix in ['open', 'high', 'low', 'close', 'volume', 'ohlc'])]
    
    if not raw_features:
        return {
            'raw_auc': None,
            'full_auc': None,
            'collapsed': None,
            'passed': False,
            'diagnosis': "SKIP: No raw OHLCV features found"
        }
    
    # Get indices of raw features
    raw_indices = [i for i, f in enumerate(feature_names) if f in raw_features]
    X_raw = X[:, raw_indices]
    
    # Run CV with raw features only
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score, r2_score
    
    scores = []
    for train_idx, val_idx in cv_splitter.split(X_raw, y):
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X_raw[train_idx], y[train_idx])
            y_pred_proba = model.predict_proba(X_raw[val_idx])[:, 1]
            if len(np.unique(y[val_idx])) == 2:
                score = roc_auc_score(y[val_idx], y_pred_proba)
            else:
                continue
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            model.fit(X_raw[train_idx], y[train_idx])
            y_pred = model.predict(X_raw[val_idx])
            score = r2_score(y[val_idx], y_pred)
        
        scores.append(score)
    
    raw_auc = np.mean(scores) if scores else None
    
    # Compare to full feature set (would need to be passed in or computed separately)
    # For now, just return raw results
    collapsed = raw_auc < 0.60 if raw_auc is not None else None
    
    passed = collapsed is True if collapsed is not None else None
    diagnosis = f"Raw OHLCV AUC: {raw_auc:.3f}. {'Collapsed (leak likely in derived features)' if collapsed else 'Still high (leak likely in label/split)'}" if raw_auc is not None else "Could not compute"
    
    return {
        'raw_auc': raw_auc,
        'full_auc': None,  # Would need to be computed separately
        'collapsed': collapsed,
        'passed': passed,
        'diagnosis': diagnosis
    }


def time_shift_label_test(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_splitter,
    task_type: str = 'classification',
    shift_bars: int = 288,  # Default: 1 day for 5m bars
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Time-shift label test: Shift labels forward/back by a day.
    
    If AUC stays high after shifting → leak (timing/alignment issue).
    Expected: AUC should drop significantly when labels are shifted.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: Feature names
        cv_splitter: CV splitter
        task_type: 'classification' or 'regression'
        shift_bars: Number of bars to shift (default: 288 = 1 day for 5m bars)
        random_seed: Random seed
    
    Returns:
        Dict with 'auc_original', 'auc_shifted', 'passed' (True if auc drops), 'diagnosis'
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error
    from sklearn.linear_model import LogisticRegression, Ridge
    
    # Run CV with original labels
    scores_original = []
    scores_shifted = []
    
    for train_idx, val_idx in cv_splitter.split(X, y):
        # Original labels
        if task_type == 'classification':
            model = LogisticRegression(random_state=random_seed, max_iter=100, solver='liblinear')
            model.fit(X[train_idx], y[train_idx])
            y_pred_proba_orig = model.predict_proba(X[val_idx])[:, 1]
            if len(np.unique(y[val_idx])) == 2:
                auc_orig = roc_auc_score(y[val_idx], y_pred_proba_orig)
                scores_original.append(auc_orig)
        else:
            model = Ridge()
            model.fit(X[train_idx], y[train_idx])
            y_pred_orig = model.predict(X[val_idx])
            mse_orig = mean_squared_error(y[val_idx], y_pred_orig)
            scores_original.append(-mse_orig)  # Negative MSE for consistency
        
        # Shifted labels (shift forward by shift_bars)
        y_shifted = np.roll(y, shift_bars)
        
        if task_type == 'classification':
            model_shift = LogisticRegression(random_state=random_seed, max_iter=100, solver='liblinear')
            model_shift.fit(X[train_idx], y_shifted[train_idx])
            y_pred_proba_shift = model_shift.predict_proba(X[val_idx])[:, 1]
            if len(np.unique(y_shifted[val_idx])) == 2:
                auc_shift = roc_auc_score(y_shifted[val_idx], y_pred_proba_shift)
                scores_shifted.append(auc_shift)
        else:
            model_shift = Ridge()
            model_shift.fit(X[train_idx], y_shifted[train_idx])
            y_pred_shift = model_shift.predict(X[val_idx])
            mse_shift = mean_squared_error(y_shifted[val_idx], y_pred_shift)
            scores_shifted.append(-mse_shift)
    
    if not scores_original or not scores_shifted:
        return {
            'auc_original': None,
            'auc_shifted': None,
            'passed': False,
            'diagnosis': 'INSUFFICIENT_DATA: Could not compute scores'
        }
    
    mean_orig = np.mean(scores_original)
    mean_shift = np.mean(scores_shifted)
    
    # Pass if shifted AUC drops significantly (by at least 0.10 for classification, or MSE increases for regression)
    if task_type == 'classification':
        passed = mean_shift < (mean_orig - 0.10)
        diagnosis = (
            f"PASS: Shifted AUC ({mean_shift:.3f}) dropped significantly from original ({mean_orig:.3f})"
            if passed else
            f"FAIL: Shifted AUC ({mean_shift:.3f}) stayed high (original: {mean_orig:.3f}) - TIMING LEAKAGE SUSPECTED"
        )
    else:
        passed = mean_shift < mean_orig  # Negative MSE, so lower is worse
        diagnosis = (
            f"PASS: Shifted MSE increased (worse performance) from original"
            if passed else
            f"FAIL: Shifted performance stayed similar - TIMING LEAKAGE SUSPECTED"
        )
    
    return {
        'auc_original': mean_orig,
        'auc_shifted': mean_shift,
        'passed': passed,
        'diagnosis': diagnosis
    }


def run_all_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_splitter,
    prices: Optional[pd.Series] = None,
    target_column: str = "unknown",
    time_vals: Optional[np.ndarray] = None,
    task_type: str = 'classification',
    horizon_minutes: int = 60
) -> Dict[str, Any]:
    """
    Run all diagnostic tests and return consolidated results.
    
    Returns:
        Dict with results from all tests + overall 'passed' flag
    """
    results = {}
    
    logger.info("=" * 60)
    logger.info("LEAKAGE DIAGNOSTICS")
    logger.info("=" * 60)
    
    # Test 1: Placebo label test
    logger.info("Test 1: Placebo label test (shuffled labels)...")
    results['placebo'] = placebo_label_test(X, y, feature_names, cv_splitter, task_type)
    logger.info(f"  {results['placebo']['diagnosis']}")
    
    # Test 2: Future direction sanity test
    if prices is not None:
        logger.info("Test 2: Future direction sanity test...")
        results['future_direction'] = future_direction_sanity_test(y, prices, target_column, time_vals)
        logger.info(f"  {results['future_direction']['diagnosis']}")
    
    # Test 3: Event offset histogram
    if prices is not None:
        logger.info("Test 3: Event offset histogram...")
        results['event_offset'] = event_offset_histogram(y, prices, target_column, horizon_minutes)
        logger.info(f"  {results['event_offset']['diagnosis']}")
    
    # Test 4: Univariate feature AUC scan
    logger.info("Test 4: Univariate feature AUC scan...")
    results['univariate'] = univariate_feature_auc_scan(X, y, feature_names, cv_splitter, task_type)
    logger.info(f"  {results['univariate']['diagnosis']}")
    if results['univariate']['suspicious_features']:
        for feat_name, auc in results['univariate']['suspicious_features'][:5]:
            logger.warning(f"    🚨 {feat_name}: AUC={auc:.3f}")
    
    # Test 5: Raw OHLCV only test
    logger.info("Test 5: Raw OHLCV only test...")
    results['raw_ohlcv'] = raw_ohlcv_only_test(X, y, feature_names, cv_splitter, task_type)
    logger.info(f"  {results['raw_ohlcv']['diagnosis']}")
    
    # Test 6: Time-shift label test (sanity kill test)
    logger.info("Test 6: Time-shift label test...")
    results['time_shift'] = time_shift_label_test(X, y, feature_names, cv_splitter, task_type)
    logger.info(f"  {results['time_shift']['diagnosis']}")
    
    logger.info("=" * 60)
    
    # Overall pass/fail
    all_passed = all(
        r.get('passed', False) 
        for r in results.values() 
        if r.get('passed') is not None
    )
    
    results['overall_passed'] = all_passed
    results['diagnosis'] = "ALL TESTS PASSED" if all_passed else "ONE OR MORE TESTS FAILED - INVESTIGATE"
    
    return results
