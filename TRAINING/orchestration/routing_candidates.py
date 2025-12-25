# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Routing Candidates Aggregation

Aggregates metrics from feature selection, stability tracking, and leakage detection
into a unified routing_candidates snapshot for the training router.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


def load_feature_selection_metadata(
    feature_selections_dir: Path,
    target_column: str
) -> Optional[Dict[str, Any]]:
    """
    Load feature selection metadata for a target.
    
    Args:
        feature_selections_dir: Base directory containing feature selection outputs
        target_column: Target column name
    
    Returns:
        Metadata dict or None if not found
    """
    target_dir = feature_selections_dir / target_column
    # Updated structure: metadata/ subdirectory
    metadata_file = target_dir / "metadata" / "multi_model_metadata.json"
    # Fallback to old location for backward compatibility
    if not metadata_file.exists():
        metadata_file = target_dir / "multi_model_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load FS metadata for {target_column}: {e}")
        return None


def load_cross_sectional_stability(
    feature_selections_dir: Path,
    target_column: str
) -> Optional[Dict[str, Any]]:
    """
    Load cross-sectional stability metrics for a target.
    
    Args:
        feature_selections_dir: Base directory containing feature selection outputs
        target_column: Target column name
    
    Returns:
        Stability dict or None if not found
    """
    target_dir = feature_selections_dir / target_column
    # Updated structure: metadata/ subdirectory
    stability_file = target_dir / "metadata" / "cross_sectional_stability_metadata.json"
    # Fallback to old location for backward compatibility
    if not stability_file.exists():
        stability_file = target_dir / "cross_sectional_stability_metadata.json"
    
    if not stability_file.exists():
        return None
    
    try:
        with open(stability_file, 'r') as f:
            data = json.load(f)
            return data.get('stability', {})
    except Exception as e:
        logger.debug(f"Failed to load CS stability for {target_column}: {e}")
        return None


def load_per_symbol_metadata(
    feature_selections_dir: Path,
    target_column: str,
    symbol: str
) -> Optional[Dict[str, Any]]:
    """
    Load per-symbol model metadata.
    
    Args:
        feature_selections_dir: Base directory containing feature selection outputs
        target_column: Target column name
        symbol: Symbol name
    
    Returns:
        Dict mapping model_family -> metadata, or None if not found
    """
    target_dir = feature_selections_dir / target_column
    metadata_file = target_dir / "model_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)
        
        # Filter for this symbol
        symbol_metadata = {}
        key_prefix = f"{symbol}:{target_column}:"
        for key, value in all_metadata.items():
            if key.startswith(key_prefix):
                model_family = key.split(':')[-1]
                symbol_metadata[model_family] = value
        
        return symbol_metadata if symbol_metadata else None
    except Exception as e:
        logger.debug(f"Failed to load symbol metadata for {symbol}:{target_column}: {e}")
        return None


def aggregate_cross_sectional_metrics(
    feature_selections_dir: Path,
    target_column: str,
    symbols: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Aggregate cross-sectional metrics for a target.
    
    Args:
        feature_selections_dir: Base directory containing feature selection outputs
        target_column: Target column name
        symbols: List of symbols processed
    
    Returns:
        Dict with CS metrics or None if not available
    """
    fs_metadata = load_feature_selection_metadata(feature_selections_dir, target_column)
    cs_stability = load_cross_sectional_stability(feature_selections_dir, target_column)
    
    if not fs_metadata:
        return None
    
    # Extract CS importance score from summary if available
    target_dir = feature_selections_dir / target_column
    summary_file = target_dir / "feature_importance_multi_model.csv"
    
    cs_score = None
    cs_sample_size = None
    
    if summary_file.exists():
        try:
            summary_df = pd.read_csv(summary_file)
            if 'cs_importance_score' in summary_df.columns:
                # Use mean CS importance as proxy for CS model performance
                cs_score = float(summary_df['cs_importance_score'].mean())
        except Exception as e:
            logger.debug(f"Failed to read CS score from summary: {e}")
    
    # Get sample size from metadata
    if fs_metadata:
        cs_sample_size = fs_metadata.get('n_symbols_processed', len(symbols)) * fs_metadata.get('sampling', {}).get('max_samples_per_symbol', 50000)
    
    # Determine stability status
    stability = "UNKNOWN"
    if cs_stability:
        stability = cs_stability.get('status', 'UNKNOWN')
    
    # Leakage status (would need to be extracted from leakage detection results)
    # For now, assume SAFE if no explicit blocking
    leakage_status = "SAFE"
    
    return {
        "target": target_column,
        "symbol": None,  # CS row has no symbol
        "mode": "CROSS_SECTIONAL",
        "score": cs_score,
        "score_ci_low": None,  # Would need CV/bootstrap results
        "score_ci_high": None,
        "stability": stability,
        "sample_size": cs_sample_size,
        "leakage_status": leakage_status,
        "feature_set_id": None,  # Would hash feature set
        "failed_model_families": [],
        "n_symbols": len(symbols)
    }


def aggregate_symbol_metrics(
    feature_selections_dir: Path,
    target_column: str,
    symbol: str
) -> Optional[Dict[str, Any]]:
    """
    Aggregate per-symbol metrics for a (target, symbol) pair.
    
    Args:
        feature_selections_dir: Base directory containing feature selection outputs
        target_column: Target column name
        symbol: Symbol name
    
    Returns:
        Dict with symbol metrics or None if not available
    """
    symbol_metadata = load_per_symbol_metadata(feature_selections_dir, target_column, symbol)
    
    if not symbol_metadata:
        return None
    
    # Aggregate scores across model families
    scores = []
    reproducibilities = []
    failed_families = []
    
    for family, metadata in symbol_metadata.items():
        score = metadata.get('score')
        if score is not None:
            scores.append(float(score))
        else:
            failed_families.append(family)
        
        repro = metadata.get('reproducibility', {})
        if repro:
            reproducibilities.append(repro)
    
    if not scores:
        return None
    
    # Compute aggregate score (mean)
    local_score = float(np.mean(scores))
    
    # Determine stability from reproducibility
    stability = "UNKNOWN"
    if reproducibilities:
        statuses = [r.get('status', 'unknown') for r in reproducibilities]
        if all(s == 'stable' for s in statuses):
            stability = "STABLE"
        elif any(s == 'unstable' for s in statuses):
            stability = "DRIFTING"
        elif any(s == 'diverged' for s in statuses):
            stability = "DIVERGED"
    
    # Sample size (would need to extract from data loading logs or metadata)
    # For now, use default or extract from config
    local_sample_size = None
    
    # Leakage status
    leakage_status = "SAFE"  # Would need to check leakage detection results
    
    return {
        "target": target_column,
        "symbol": symbol,
        "mode": "SYMBOL",
        "score": local_score,
        "score_ci_low": None,
        "score_ci_high": None,
        "stability": stability,
        "sample_size": local_sample_size,
        "leakage_status": leakage_status,
        "feature_set_id": None,
        "failed_model_families": failed_families,
        "n_model_families": len(symbol_metadata),
        "n_successful_families": len(scores)
    }


def build_routing_candidates(
    feature_selections_dir: Path,
    targets: List[str],
    symbols: List[str],
    output_dir: Optional[Path] = None,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None
) -> pd.DataFrame:
    """
    Build routing candidates snapshot from feature selection outputs.
    
    Args:
        feature_selections_dir: Directory containing feature selection outputs
        targets: List of target columns processed
        symbols: List of symbols processed
        output_dir: Optional output directory for routing candidates
        git_commit: Optional git commit hash
        config_hash: Optional config hash
    
    Returns:
        DataFrame with routing candidates (CS rows + per-symbol rows)
    """
    rows = []
    
    for target in targets:
        # Cross-sectional row
        cs_metrics = aggregate_cross_sectional_metrics(feature_selections_dir, target, symbols)
        if cs_metrics:
            cs_metrics['timestamp'] = datetime.utcnow().isoformat()
            cs_metrics['git_commit'] = git_commit
            cs_metrics['config_hash'] = config_hash
            rows.append(cs_metrics)
        
        # Per-symbol rows
        for symbol in symbols:
            symbol_metrics = aggregate_symbol_metrics(feature_selections_dir, target, symbol)
            if symbol_metrics:
                symbol_metrics['timestamp'] = datetime.utcnow().isoformat()
                symbol_metrics['git_commit'] = git_commit
                symbol_metrics['config_hash'] = config_hash
                rows.append(symbol_metrics)
    
    if not rows:
        logger.warning("No routing candidates found")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Save to disk if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        parquet_path = output_dir / "routing_candidates.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"✅ Saved routing candidates: {len(df)} rows → {parquet_path}")
        
        # Save as JSON for humans
        json_path = output_dir / "routing_candidates.json"
        df_dict = df.to_dict(orient='records')
        with open(json_path, 'w') as f:
            json.dump(df_dict, f, indent=2, default=str)
        logger.info(f"✅ Saved routing candidates JSON: {json_path}")
    
    return df











