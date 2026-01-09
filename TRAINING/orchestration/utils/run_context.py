# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


def compute_universe_signature(symbols: List[str]) -> str:
    """
    Compute a stable hash of the symbol universe.
    
    Properties:
    - Order-independent (sorted)
    - Duplicate-invariant (set)
    - Uses blake2s (fast, no security debates)
    
    Args:
        symbols: List of symbol strings
    
    Returns:
        12-character hex string identifying the universe
    """
    sorted_symbols = sorted(set(symbols))
    return hashlib.blake2s(",".join(sorted_symbols).encode(), digest_size=6).hexdigest()


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
    target: Optional[str] = None
    target_config: Optional[Dict[str, Any]] = None
    
    # Cross-sectional config
    min_cs: Optional[int] = None
    max_cs_samples: Optional[int] = None
    leakage_filter_version: Optional[str] = None
    universe_sig: Optional[str] = None  # SST: Universe signature
    
    # CV configuration
    cv_splitter: Optional[Any] = None  # PurgedTimeSeriesSplit or similar
    cv_method: str = "purged_kfold"
    folds: Optional[int] = None
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
    symbol: Optional[str] = None
    model_family: Optional[str] = None
    
    # View resolution (SST - Single Source of Truth)
    # "view" is the canonical field for modeling granularity
    view: Optional[str] = None  # SST: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    requested_view: Optional[str] = None  # View requested by caller/config
    view_reason: Optional[str] = None  # Reason for resolution (e.g., "n_symbols=5 (small panel)")
    
    # Data scope (what's loaded right now - can vary per-symbol, non-immutable)
    data_scope: Optional[str] = None  # "PANEL" (multiple symbols) or "SINGLE_SYMBOL" (one symbol)
    
    # Auto-derived fields (computed on demand)
    _data_fingerprint: Optional[str] = field(default=None, init=False, repr=False)
    _feature_registry_hash: Optional[str] = field(default=None, init=False, repr=False)
    _label_definition_hash: Optional[str] = field(default=None, init=False, repr=False)
    
    # Run Identity (SST for reproducibility)
    # Partial: computed early (before feature selection)
    # Final: computed after features are locked
    run_identity_partial: Optional[Any] = field(default=None, repr=False)  # RunIdentity (is_final=False)
    run_identity: Optional[Any] = field(default=None, repr=False)  # RunIdentity (is_final=True)
    
    def __post_init__(self):
        """
        Validate and normalize SST fields.
        
        Invariants enforced:
        - view must be None or one of: CROSS_SECTIONAL, SYMBOL_SPECIFIC
        - universe_sig should be set when view is resolved
        """
        # Validate view values if set
        valid_views = {"CROSS_SECTIONAL", "SYMBOL_SPECIFIC"}
        if self.view is not None and self.view not in valid_views:
            raise ValueError(
                f"Invalid view '{self.view}'. Must be one of: {valid_views}"
            )
        if self.requested_view is not None and self.requested_view not in valid_views:
            raise ValueError(
                f"Invalid requested_view '{self.requested_view}'. Must be one of: {valid_views}"
            )
    
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
        if self.cv_splitter is not None or self.folds is not None:
            required.extend(["folds", "purge_minutes"])
        
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
    
    def compute_partial_identity(
        self,
        dataset_signature: str,
        split_signature: str,
        target_signature: str,
        hparams_signature: str,
        routing_signature: str,
        routing_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Compute and store partial RunIdentity (before features are locked).
        
        Call this early in the pipeline before feature selection.
        Call finalize_identity() after features are locked.
        
        Args:
            dataset_signature: 64-char SHA256 from cohort metadata
            split_signature: 64-char SHA256 from CV/split config
            target_signature: 64-char SHA256 from target config
            hparams_signature: 64-char SHA256 from model hyperparameters
            routing_signature: 64-char SHA256 from contracted routing payload
            routing_payload: Optional contracted routing dict for debugging
        """
        from TRAINING.common.utils.fingerprinting import RunIdentity
        
        self.run_identity_partial = RunIdentity(
            schema_version=1,
            dataset_signature=dataset_signature,
            split_signature=split_signature,
            target_signature=target_signature,
            feature_signature=None,  # Not yet known
            hparams_signature=hparams_signature,
            routing_signature=routing_signature,
            routing_payload=routing_payload,
            train_seed=self.seed,
            is_final=False,
        )
        logger.debug(f"Computed partial RunIdentity: {self.run_identity_partial.debug_key}")
    
    def finalize_identity(self, feature_signature: str) -> None:
        """
        Finalize RunIdentity after features are locked.
        
        MUST be called after feature selection/pruning is complete.
        Creates a new finalized identity with computed keys.
        
        Args:
            feature_signature: 64-char SHA256 from resolved feature specs
            
        Raises:
            ValueError: If partial identity not computed yet
        """
        if self.run_identity_partial is None:
            raise ValueError(
                "Cannot finalize identity: partial identity not computed. "
                "Call compute_partial_identity() first."
            )
        
        self.run_identity = self.run_identity_partial.finalize(feature_signature)
        logger.debug(f"Finalized RunIdentity: {self.run_identity.debug_key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "target_column": self.target_column,
            "target": self.target,
            "target_config": self.target_config,
            "min_cs": self.min_cs,
            "max_cs_samples": self.max_cs_samples,
            "leakage_filter_version": self.leakage_filter_version,
            "universe_sig": self.universe_sig,  # SST canonical
            "cv_method": self.cv_method,
            "folds": self.folds,
            "horizon_minutes": self.horizon_minutes,
            "purge_minutes": self.purge_minutes,
            "embargo_minutes": self.embargo_minutes,
            "feature_lookback_max_minutes": self.feature_lookback_max_minutes,
            "data_interval_minutes": self.data_interval_minutes,
            "seed": self.seed,
            "stage": self.stage,
            "view": self.view,
            "symbol": self.symbol,
            "model_family": self.model_family,
            # Canonical fields (SST)
            "view": self.view,
            "requested_view": self.requested_view,
            "view_reason": self.view_reason,
            # Deprecated aliases (for backward compat)
            "requested_view": self.requested_view,
            "view": self.view,
            "view_reason": self.view_reason,
            "data_scope": self.data_scope
        }


def _validate_view(view_str: Optional[str]) -> Optional[str]:
    """
    Validate and normalize view string using View enum.
    
    Args:
        view_str: View string to validate
    
    Returns:
        Normalized view string (CROSS_SECTIONAL or SYMBOL_SPECIFIC) or None
    
    Raises:
        ValueError: If view_str is not a valid view value
    """
    if view_str is None:
        return None
    
    try:
        from TRAINING.orchestration.utils.scope_resolution import View
        return View.from_string(view_str).value
    except ValueError:
        logger.warning(f"Invalid view value '{view_str}', expected CROSS_SECTIONAL or SYMBOL_SPECIFIC")
        raise


def save_run_context(
    output_dir: Path,
    view: Optional[str] = None,
    requested_view: Optional[str] = None,
    view_reason: Optional[str] = None,
    n_symbols: Optional[int] = None,
    data_scope: Optional[str] = None,
    universe_signature: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    **additional_data
) -> Path:
    """
    Save resolved view to globals/run_context.json (SST - Single Source of Truth).
    
    view is immutable PER UNIVERSE (keyed by universe_signature).
    Different universes can have different views.
    data_scope can change per-symbol (non-immutable).
    
    Args:
        output_dir: Run output directory (e.g., RESULTS/runs/.../intelligent_output_...)
        requested_view: DEPRECATED - use requested_view
        view: DEPRECATED - use view
        view_reason: DEPRECATED - use view_reason
        n_symbols: Number of symbols (for context)
        data_scope: Data scope (PANEL or SINGLE_SYMBOL) - can change per-symbol, non-immutable
        universe_signature: Hash of symbol universe (from compute_universe_signature)
        symbols: List of symbols (for logging/debugging)
        view: SST - View actually used (CROSS_SECTIONAL or SYMBOL_SPECIFIC) - IMMUTABLE per universe
        requested_view: SST - View requested by caller/config
        view_reason: SST - Reason for resolution (stored as original_reason for new universes)
        **additional_data: Additional metadata to store
    
    Returns:
        Path to run_context.json file
    
    Raises:
        ValueError: If trying to overwrite existing view for same universe
    """
    # SST: Validate view if provided
    if view is not None:
        view = _validate_view(view)
    
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
    
    # Get existing views dict (keyed by universe signature)
    views = existing_context.get("views", {})
    
    # If universe_signature provided, store per-universe (new behavior)
    if universe_signature and view:
        if universe_signature in views:
            existing_entry = views[universe_signature]
            if existing_entry["view"] != view:
                raise ValueError(
                    f"Mode contract violation for universe {universe_signature}: "
                    f"Cannot change from '{existing_entry['view']}' to '{view}'. "
                    f"Original reason: {existing_entry.get('original_reason', 'N/A')}"
                )
            # Same mode, keep existing entry (don't update timestamp)
        else:
            # New universe - store the entry with original_reason (not nested)
            symbols_sample = symbols[:3] if symbols and len(symbols) > 3 else (symbols or [])
            views[universe_signature] = {
                "view": view,
                "original_reason": view_reason,  # Store the ORIGINAL reason, not nested
                "n_symbols": n_symbols,
                "symbols_sample": symbols_sample,
                "resolved_at": datetime.utcnow().isoformat() + "Z"
            }
            logger.info(f"🔑 New universe {universe_signature}: view={view}, n_symbols={n_symbols}")
    
    # Legacy: check global view if no universe_signature provided
    elif view and existing_context.get("view") is not None:
        if existing_context["view"] != view:
            raise ValueError(
                f"Mode contract violation: Cannot change view from '{existing_context['view']}' "
                f"to '{view}'. Resolved mode is immutable after first write. "
                f"Existing reason: {existing_context.get('view_reason', 'N/A')}"
            )
    
    # Build context dict
    context = {
        "requested_view": requested_view or existing_context.get("requested_view"),
        "views": views,  # Dict keyed by universe signature
        "current_universe": universe_signature or existing_context.get("current_universe"),
        "data_scope": data_scope or existing_context.get("data_scope"),
        # Keep legacy fields for backward compat
        "view": view or existing_context.get("view"),
        "view_reason": view_reason or existing_context.get("view_reason"),
        "n_symbols": n_symbols or existing_context.get("n_symbols"),
        "resolved_at": existing_context.get("resolved_at") or datetime.utcnow().isoformat() + "Z",
        **additional_data
    }
    
    # Write to file
    with open(run_context_path, 'w') as f:
        json.dump(context, f, indent=2)
    
    logger.debug(f"✅ Saved run context: universe={universe_signature}, view={view}")
    
    return run_context_path


def load_run_context(output_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load resolved mode from globals/run_context.json (SST).
    
    Args:
        output_dir: Run output directory
    
    Returns:
        Run context dict with view, requested_view, etc., or None if not found
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


def get_view(output_dir: Path) -> Optional[str]:
    """
    Get view from run context (convenience function).
    
    Args:
        output_dir: Run output directory
    
    Returns:
        Resolved mode string or None if not found
    """
    context = load_run_context(output_dir)
    if context:
        return context.get("view")
    return None


def get_view_for_universe(output_dir: Path, universe_signature: str) -> Optional[Dict[str, Any]]:
    """
    Get the resolved mode entry for a specific universe.
    
    Args:
        output_dir: Run output directory
        universe_signature: Hash of symbol universe (from compute_universe_signature)
    
    Returns:
        Dict with view, original_reason, n_symbols, etc., or None if not found
    """
    context = load_run_context(output_dir)
    if context:
        views = context.get("views", {})
        return views.get(universe_signature)
    return None


def validate_view_contract(
    resolved_view: str,
    requested_view: Optional[str],
    view_policy: str
) -> bool:
    """
    Validate that resolved_view matches contract based on view_policy.
    
    Args:
        resolved_view: View actually used (after auto-flip logic)
        requested_view: View requested by caller/config
        view_policy: "force" or "auto"
    
    Returns:
        True if contract is satisfied
    
    Raises:
        ValueError: If view_policy=force and resolved_view != requested_view
    """
    if view_policy == "force":
        if requested_view is None:
            raise ValueError(
                f"View contract violation: view_policy=force requires requested_view to be set, "
                f"but got None. Resolved view: {resolved_view}"
            )
        if resolved_view != requested_view:
            raise ValueError(
                f"View contract violation: view_policy=force requires resolved_view={requested_view}, "
                f"but got resolved_view={resolved_view}. This indicates the resolver incorrectly "
                f"flipped the view when it should have been forced."
            )
    # For "auto" policy, any resolved_view is valid (resolver can flip)
    return True


# DEPRECATED: Alias for backward compatibility
def validate_mode_contract(
    view: str,
    requested_view: Optional[str],
    view_policy: str
) -> bool:
    """
    DEPRECATED: Use validate_view_contract() instead.
    
    Alias kept for backward compatibility during migration.
    """
    return validate_view_contract(view, requested_view, view_policy)


# =============================================================================
# SST Stage Tracking
# =============================================================================

def _validate_stage(stage_str: str) -> str:
    """
    Validate and normalize stage string using Stage enum.
    
    Args:
        stage_str: Stage string to validate
    
    Returns:
        Normalized stage value (uppercase)
    
    Raises:
        ValueError: If stage is not a valid Stage enum value
    """
    from TRAINING.orchestration.utils.scope_resolution import Stage
    return Stage.from_string(stage_str).value


def save_stage_transition(
    output_dir: Path,
    stage: str,
    reason: Optional[str] = None,
) -> Path:
    """
    Record a stage transition to run_context.json (SST).
    
    Stage history is append-only. Current stage is always the most recent.
    This creates a single source of truth for "what stage are we in?"
    
    Args:
        output_dir: Run output directory (e.g., RESULTS/runs/.../intelligent_output_...)
        stage: Stage name (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
        reason: Optional reason for the transition
    
    Returns:
        Path to run_context.json file
    
    Raises:
        ValueError: If stage is not a valid Stage enum value
    """
    # Validate stage via Stage enum
    stage_normalized = _validate_stage(stage)
    
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
    
    # Get existing stage history (or create new)
    stage_history = existing_context.get("stage_history", [])
    
    # Create new stage entry
    stage_entry = {
        "stage": stage_normalized,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    if reason:
        stage_entry["reason"] = reason
    
    # Append to history (append-only, no overwrites)
    stage_history.append(stage_entry)
    
    # Update context
    existing_context["current_stage"] = stage_normalized
    existing_context["stage_history"] = stage_history
    
    # Write to file
    with open(run_context_path, 'w') as f:
        json.dump(existing_context, f, indent=2)
    
    logger.info(f"🔄 Stage transition: {stage_normalized} (reason: {reason or 'N/A'})")
    
    return run_context_path


def get_current_stage(output_dir: Path) -> Optional[str]:
    """
    Get current stage from SST run_context.json.
    
    Args:
        output_dir: Run output directory
    
    Returns:
        Current stage string (e.g., "TARGET_RANKING") or None if not found
    """
    context = load_run_context(output_dir)
    if context:
        return context.get("current_stage")
    return None


def get_stage_history(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Get full stage transition history from SST run_context.json.
    
    Args:
        output_dir: Run output directory
    
    Returns:
        List of stage transition entries, each with:
        - stage: Stage name
        - started_at: ISO timestamp
        - reason: Optional reason for transition
    """
    context = load_run_context(output_dir)
    if context:
        return context.get("stage_history", [])
    return []


def resolve_stage(
    output_dir: Optional[Path] = None,
    scope: Optional[Any] = None,
    explicit_stage: Optional[str] = None,
) -> Optional[str]:
    """
    SST stage resolution with priority chain.
    
    Priority:
    1. explicit_stage parameter (if provided)
    2. scope.stage (if WriteScope provided and has stage attribute)
    3. get_current_stage(output_dir) from run_context.json
    4. None (legacy fallback)
    
    Args:
        output_dir: Run output directory for SST lookup
        scope: Optional WriteScope object with stage attribute
        explicit_stage: Explicit stage override
    
    Returns:
        Resolved stage string or None
    """
    # Priority 1: Explicit parameter
    if explicit_stage is not None:
        return _validate_stage(explicit_stage)
    
    # Priority 2: WriteScope.stage
    if scope is not None and hasattr(scope, 'stage') and scope.stage is not None:
        # Handle both Stage enum and string
        stage_val = scope.stage
        if hasattr(stage_val, 'value'):
            return stage_val.value  # It's an enum
        return _validate_stage(str(stage_val))
    
    # Priority 3: SST run_context.json
    if output_dir is not None:
        sst_stage = get_current_stage(output_dir)
        if sst_stage:
            return sst_stage
    
    # Priority 4: None (legacy fallback)
    return None
