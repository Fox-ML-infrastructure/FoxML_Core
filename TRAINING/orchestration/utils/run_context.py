"""
Run Context for Audit-Grade Reproducibility

Centralized context object that holds all information needed for reproducibility tracking.
Eliminates manual parameter passing and ensures nothing is forgotten.

Usage:
    from TRAINING.orchestration.utils.run_context import RunContext
    
    ctx = RunContext(
        X=X,
        y=y,
        feature_names=feature_names,
        symbols=symbols,
        time_vals=time_vals,
        target_column=target_column,
        target_config=target_config,
        cv_splitter=cv_splitter,
        horizon_minutes=60,
        feature_lookback_max_minutes=1440
    )
    
    tracker.log_run(ctx, metrics)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """
    Complete context for a single run, containing all data and configuration needed
    for audit-grade reproducibility tracking.
    
    All fields are optional at construction, but required fields will be validated
    when reproducibility_mode == COHORT_AWARE.
    """
    
    # Core data
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None
    y: Optional[Union[np.ndarray, pd.Series]] = None
    feature_names: Optional[List[str]] = None
    symbols: Optional[Union[List[str], np.ndarray, pd.Series]] = None
    time_vals: Optional[Union[np.ndarray, pd.Series, List]] = None
    
    # Target specification
    target_column: Optional[str] = None
    target_name: Optional[str] = None
    target_config: Optional[Dict[str, Any]] = None
    
    # Cross-sectional config
    min_cs: Optional[int] = None
    max_cs_samples: Optional[int] = None
    leakage_filter_version: Optional[str] = None
    universe_id: Optional[str] = None
    
    # CV configuration
    cv_splitter: Optional[Any] = None  # PurgedTimeSeriesSplit or similar
    cv_method: str = "purged_kfold"
    cv_folds: Optional[int] = None
    horizon_minutes: Optional[float] = None
    purge_minutes: Optional[float] = None
    embargo_minutes: Optional[float] = None
    fold_timestamps: Optional[List[Dict[str, Any]]] = None
    
    # Feature configuration
    feature_lookback_max_minutes: Optional[float] = None
    feature_registry_path: Optional[Path] = None
    
    # Additional metadata
    mtf_data: Optional[Dict[str, pd.DataFrame]] = None
    data_interval_minutes: Optional[float] = None
    seed: Optional[int] = None
    output_dir: Optional[Path] = None
    
    # Stage and routing
    stage: str = "target_ranking"
    route_type: Optional[str] = None
    symbol: Optional[str] = None
    view: Optional[str] = None  # "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "LOSO" for target ranking
    model_family: Optional[str] = None
    
    # Mode resolution (SST - Single Source of Truth)
    requested_mode: Optional[str] = None  # Mode requested by caller/config
    resolved_mode: Optional[str] = None  # Mode actually used (after auto-flip logic) - IMMUTABLE after first write
    mode_reason: Optional[str] = None  # Reason for resolution (e.g., "n_symbols=5 (small panel)")
    
    # Data scope (what's loaded right now - can vary per-symbol, non-immutable)
    data_scope: Optional[str] = None  # "PANEL" (multiple symbols) or "SINGLE_SYMBOL" (one symbol)
    
    # Auto-derived fields (computed on demand)
    _data_fingerprint: Optional[str] = field(default=None, init=False, repr=False)
    _feature_registry_hash: Optional[str] = field(default=None, init=False, repr=False)
    _label_definition_hash: Optional[str] = field(default=None, init=False, repr=False)
    
    def derive_purge_embargo(self, buffer_bars: int = 1) -> Tuple[float, float]:
        """
        Automatically derive purge and embargo from horizon.
        
        Uses centralized derivation function for consistency.
        
        Rule: purge_minutes = embargo_minutes = horizon_minutes + buffer
        (Feature lookback is NOT included - it's historical data that's safe to use)
        
        Args:
            buffer_bars: Additional safety buffer in bars (default: 1)
        
        Returns:
            (purge_minutes, embargo_minutes)
        """
        if self.horizon_minutes is None:
            raise ValueError("horizon_minutes is required for automatic derivation")
        
        # Use centralized derivation function to ensure consistency
        from TRAINING.ranking.utils.resolved_config import derive_purge_embargo
        
        if self.data_interval_minutes is None:
            # Default to 5 minutes if not specified
            data_interval_minutes = 5.0
        else:
            data_interval_minutes = self.data_interval_minutes
        
        return derive_purge_embargo(
            horizon_minutes=self.horizon_minutes,
            interval_minutes=data_interval_minutes,
            feature_lookback_max_minutes=self.feature_lookback_max_minutes,
            purge_buffer_bars=buffer_bars,
            default_purge_minutes=None  # Loads from safety_config.yaml (SST)
        )
    
    def get_required_fields(self, reproducibility_mode: str = "COHORT_AWARE") -> List[str]:
        """
        Get list of required fields for the given reproducibility mode.
        
        Args:
            reproducibility_mode: "COHORT_AWARE" or "LEGACY"
        
        Returns:
            List of required field names
        """
        if reproducibility_mode != "COHORT_AWARE":
            return []  # No requirements for legacy mode
        
        required = [
            "X", "y", "feature_names", "symbols", "time_vals",
            "target_column", "horizon_minutes"
        ]
        
        # CV fields are required if CV is being used
        if self.cv_splitter is not None or self.cv_folds is not None:
            required.extend(["cv_folds", "purge_minutes"])
        
        return required
    
    def validate_required_fields(self, reproducibility_mode: str = "COHORT_AWARE") -> List[str]:
        """
        Validate that all required fields are present.
        
        Args:
            reproducibility_mode: "COHORT_AWARE" or "LEGACY"
        
        Returns:
            List of missing required field names (empty if all present)
        """
        required = self.get_required_fields(reproducibility_mode)
        missing = []
        
        for field_name in required:
            value = getattr(self, field_name, None)
            if value is None:
                missing.append(field_name)
        
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "target_column": self.target_column,
            "target_name": self.target_name,
            "target_config": self.target_config,
            "min_cs": self.min_cs,
            "max_cs_samples": self.max_cs_samples,
            "leakage_filter_version": self.leakage_filter_version,
            "universe_id": self.universe_id,
            "cv_method": self.cv_method,
            "cv_folds": self.cv_folds,
            "horizon_minutes": self.horizon_minutes,
            "purge_minutes": self.purge_minutes,
            "embargo_minutes": self.embargo_minutes,
            "feature_lookback_max_minutes": self.feature_lookback_max_minutes,
            "data_interval_minutes": self.data_interval_minutes,
            "seed": self.seed,
            "stage": self.stage,
            "route_type": self.route_type,
            "symbol": self.symbol,
            "model_family": self.model_family,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "mode_reason": self.mode_reason,
            "data_scope": self.data_scope
        }


def save_run_context(
    output_dir: Path,
    requested_mode: Optional[str] = None,
    resolved_mode: Optional[str] = None,
    mode_reason: Optional[str] = None,
    n_symbols: Optional[int] = None,
    data_scope: Optional[str] = None,  # NEW: Data scope (PANEL or SINGLE_SYMBOL) - non-immutable
    **additional_data
) -> Path:
    """
    Save resolved mode to globals/run_context.json (SST - Single Source of Truth).
    
    resolved_mode is immutable after first write to prevent mode drift during a run.
    data_scope can change per-symbol (non-immutable).
    All downstream components should load resolved_mode from this file.
    
    Args:
        output_dir: Run output directory (e.g., RESULTS/runs/.../intelligent_output_...)
        requested_mode: Mode requested by caller/config
        resolved_mode: Mode actually used (after auto-flip logic) - IMMUTABLE after first write
        mode_reason: Reason for resolution
        n_symbols: Number of symbols (for context)
        data_scope: Data scope (PANEL or SINGLE_SYMBOL) - can change per-symbol, non-immutable
        **additional_data: Additional metadata to store
    
    Returns:
        Path to run_context.json file
    
    Raises:
        ValueError: If trying to overwrite existing resolved_mode with different value
    """
    run_context_path = output_dir / "globals" / "run_context.json"
    run_context_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing context if it exists
    existing_context = {}
    if run_context_path.exists():
        try:
            with open(run_context_path, 'r') as f:
                existing_context = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing run_context.json: {e}, creating new one")
    
    # If resolved_mode already exists and is different, raise error (immutable)
    if existing_context.get("resolved_mode") is not None:
        if resolved_mode is not None and existing_context["resolved_mode"] != resolved_mode:
            raise ValueError(
                f"Mode contract violation: Cannot change resolved_mode from '{existing_context['resolved_mode']}' "
                f"to '{resolved_mode}'. Resolved mode is immutable after first write. "
                f"Existing reason: {existing_context.get('mode_reason', 'N/A')}"
            )
        # If same or None, keep existing
        if resolved_mode is None:
            resolved_mode = existing_context.get("resolved_mode")
    
    # Build context dict
    # resolved_mode is immutable (only set once), data_scope can be updated
    context = {
        "requested_mode": requested_mode or existing_context.get("requested_mode"),
        "resolved_mode": resolved_mode or existing_context.get("resolved_mode"),
        "mode_reason": mode_reason or existing_context.get("mode_reason"),
        "n_symbols": n_symbols or existing_context.get("n_symbols"),
        "data_scope": data_scope or existing_context.get("data_scope"),  # Can be updated (non-immutable)
        "resolved_at": existing_context.get("resolved_at") or datetime.utcnow().isoformat() + "Z",
        **additional_data
    }
    
    # Write to file
    with open(run_context_path, 'w') as f:
        json.dump(context, f, indent=2)
    
    logger.info(f"✅ Saved run context (SST): resolved_mode={context['resolved_mode']}, requested_mode={context['requested_mode']}")
    
    return run_context_path


def load_run_context(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load resolved mode from globals/run_context.json (SST).
    
    Args:
        output_dir: Run output directory
    
    Returns:
        Run context dict with resolved_mode, requested_mode, etc., or None if not found
    """
    run_context_path = output_dir / "globals" / "run_context.json"
    
    if not run_context_path.exists():
        return None
    
    try:
        with open(run_context_path, 'r') as f:
            context = json.load(f)
        return context
    except Exception as e:
        logger.warning(f"Could not load run_context.json: {e}")
        return None


def get_resolved_mode(output_dir: Path) -> Optional[str]:
    """
    Get resolved_mode from run context (convenience function).
    
    Args:
        output_dir: Run output directory
    
    Returns:
        Resolved mode string or None if not found
    """
    context = load_run_context(output_dir)
    if context:
        return context.get("resolved_mode")
    return None


def validate_mode_contract(
    resolved_mode: str,
    requested_mode: Optional[str],
    mode_policy: str
) -> bool:
    """
    Validate that resolved_mode matches contract based on mode_policy.
    
    Args:
        resolved_mode: Mode actually used (after auto-flip logic)
        requested_mode: Mode requested by caller/config
        mode_policy: "force" or "auto"
    
    Returns:
        True if contract is satisfied
    
    Raises:
        ValueError: If mode_policy=force and resolved_mode != requested_mode
    """
    if mode_policy == "force":
        if requested_mode is None:
            raise ValueError(
                f"Mode contract violation: mode_policy=force requires requested_mode to be set, "
                f"but got None. Resolved mode: {resolved_mode}"
            )
        if resolved_mode != requested_mode:
            raise ValueError(
                f"Mode contract violation: mode_policy=force requires resolved_mode={requested_mode}, "
                f"but got resolved_mode={resolved_mode}. This indicates the resolver incorrectly "
                f"flipped the mode when it should have been forced."
            )
    # For "auto" policy, any resolved_mode is valid (resolver can flip)
    return True
