"""
Run Context for Audit-Grade Reproducibility

Centralized context object that holds all information needed for reproducibility tracking.
Eliminates manual parameter passing and ensures nothing is forgotten.

Usage:
    from TRAINING.utils.run_context import RunContext
    
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
    model_family: Optional[str] = None
    
    # Auto-derived fields (computed on demand)
    _data_fingerprint: Optional[str] = field(default=None, init=False, repr=False)
    _feature_registry_hash: Optional[str] = field(default=None, init=False, repr=False)
    _label_definition_hash: Optional[str] = field(default=None, init=False, repr=False)
    
    def derive_purge_embargo(self, buffer_bars: int = 1) -> Tuple[float, float]:
        """
        Automatically derive purge and embargo from horizon and feature lookback.
        
        Rule: purge_minutes = embargo_minutes = max(horizon_minutes, feature_lookback_max_minutes) + buffer
        
        Args:
            buffer_bars: Additional safety buffer in bars (default: 1)
        
        Returns:
            (purge_minutes, embargo_minutes)
        """
        if self.horizon_minutes is None:
            raise ValueError("horizon_minutes is required for automatic derivation")
        
        if self.data_interval_minutes is None:
            # Default to 5 minutes if not specified
            data_interval_minutes = 5.0
        else:
            data_interval_minutes = self.data_interval_minutes
        
        buffer_minutes = buffer_bars * data_interval_minutes
        
        # Base purge/embargo = horizon
        base_minutes = self.horizon_minutes
        
        # Add feature lookback if specified
        if self.feature_lookback_max_minutes is not None:
            base_minutes = max(base_minutes, self.feature_lookback_max_minutes)
        
        # Add buffer
        purge_embargo_minutes = base_minutes + buffer_minutes
        
        return purge_embargo_minutes, purge_embargo_minutes
    
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
            "model_family": self.model_family
        }
