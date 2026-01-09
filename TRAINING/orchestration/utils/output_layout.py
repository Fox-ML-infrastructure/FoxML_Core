# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
OutputLayout: Canonical output path builder with view+universe scoping.

This is the Single Source of Truth (SST) for all artifact paths.
- CROSS_SECTIONAL paths include universe={universe_sig}/ to identify the symbol set
- SYMBOL_SPECIFIC paths use symbol={symbol}/ only (universe= is redundant)

Usage:
    layout = OutputLayout(
        output_root=run_dir,
        target="fwd_ret_5d",
        view=ScopeView.CROSS_SECTIONAL,
        universe_sig="abc123",
        cohort_id="cs_2024Q1_..."
    )
    repro_dir = layout.repro_dir()
    cohort_dir = layout.cohort_dir()
    metrics_dir = layout.metrics_dir()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)

# Import SST accessor functions
from TRAINING.orchestration.utils.reproducibility.utils import (
    extract_universe_sig,
    extract_target,
)

# Import WriteScope for scope-safe path building
try:
    from TRAINING.orchestration.utils.scope_resolution import (
        WriteScope,
        ScopePurpose,
        View as ScopeView,
        Stage as ScopeStage
    )
    _WRITE_SCOPE_AVAILABLE = True
except ImportError:
    _WRITE_SCOPE_AVAILABLE = False
    WriteScope = None
    ScopePurpose = None
    ScopeView = None
    ScopeStage = None


def _normalize_universe_sig(meta: Dict[str, Any]) -> Optional[str]:
    """Normalize universe signature: delegates to SST accessor.
    
    Single source of truth for universe_sig extraction. Checks:
    1. meta["universe_sig"] (canonical)
    2. meta["universe_sig"] (legacy alias)
    3. meta["cs_config"]["universe_sig"] (nested legacy)
    4. meta["cs_config"]["universe_sig"] (nested legacy alias)
    """
    return extract_universe_sig(meta, meta.get("cs_config"))


def _normalize_view(meta: Dict[str, Any]) -> Optional[str]:
    """Normalize view to uppercase canonical form.
    
    Returns canonical view string or None if invalid/missing.
    """
    raw_view = meta.get("view")
    if not raw_view:
        return None
    normalized = raw_view.upper()
    # Only accept canonical values
    if normalized not in {"CROSS_SECTIONAL", "SYMBOL_SPECIFIC"}:
        return None  # Treat as missing (invalid view)
    return normalized


class OutputLayout:
    """Canonical output path builder with view+universe scoping.

    CROSS_SECTIONAL uses universe={sig}/, SYMBOL_SPECIFIC uses symbol={sym}/ only.
    
    Invariants:
    - view must be "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    - SYMBOL_SPECIFIC requires symbol
    - CROSS_SECTIONAL cannot have symbol
    - universe_sig is required for all scopes
    
    Usage (preferred - with WriteScope):
        scope = WriteScope.for_cross_sectional(universe_sig="abc123", stage=Stage.TRAINING)
        layout = OutputLayout(output_root=run_dir, target="fwd_ret_5d", scope=scope)
    
    Usage (deprecated - loose args):
        layout = OutputLayout(
            output_root=run_dir,
            target="fwd_ret_5d",
            view=ScopeView.CROSS_SECTIONAL,
            universe_sig="abc123"
        )
    """
    
    def __init__(
        self,
        output_root: Path,
        target: str,
        # NEW: Accept WriteScope directly (preferred)
        scope: Optional["WriteScope"] = None,
        # DEPRECATED: Loose args (for backward compat)
        view: Optional[str] = None,
        universe_sig: Optional[str] = None,
        symbol: Optional[str] = None,
        cohort_id: Optional[str] = None,
        stage: Optional[str] = None,  # Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    ):
        """Initialize OutputLayout with either WriteScope (preferred) or loose args.
        
        Args:
            output_root: Base output directory for the run
            target: Target name
            scope: WriteScope object (preferred - derives view, universe_sig, symbol, purpose, stage)
            view: View string (deprecated - use scope instead)
            universe_sig: Universe signature (deprecated - use scope instead)
            symbol: Symbol for SYMBOL_SPECIFIC (deprecated - use scope instead)
            cohort_id: Optional cohort ID
            stage: Pipeline stage for path scoping (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
        """
        self.output_root = Path(output_root) if isinstance(output_root, str) else output_root
        self.target = target
        self.cohort_id = cohort_id
        
        if scope is not None:
            # Preferred path: derive everything from scope
            if not _WRITE_SCOPE_AVAILABLE:
                raise ValueError("WriteScope not available but scope was passed")
            self.scope = scope
            self.view = scope.view.value if hasattr(scope.view, 'value') else scope.view
            self.universe_sig = scope.universe_sig
            self.symbol = scope.symbol
            self._purpose = scope.purpose
            # Get stage from scope if available
            self.stage = scope.stage.value if hasattr(scope, 'stage') and scope.stage and hasattr(scope.stage, 'value') else (scope.stage if hasattr(scope, 'stage') else stage)
        else:
            # Deprecated path: loose args
            warnings.warn(
                "OutputLayout with loose args (view, universe_sig, symbol) is deprecated. "
                "Pass scope=WriteScope(...) instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.scope = None
            # SST: Normalize view and stage to strings (handle enum inputs)
            self.view = view.value if isinstance(view, ScopeView) else (view if isinstance(view, str) else str(view))
            self.universe_sig = universe_sig
            self.symbol = symbol
            self._purpose = ScopePurpose.FINAL if _WRITE_SCOPE_AVAILABLE else None
            # SST: Normalize stage to string (handle enum inputs)
            self.stage = stage.value if isinstance(stage, ScopeStage) else (stage if isinstance(stage, str) else str(stage) if stage else None)
        
        # Normalize view to enum for validation (handles both enum and string)
        view_enum = ScopeView.from_string(self.view) if isinstance(self.view, str) else self.view
        if view_enum not in (ScopeView.CROSS_SECTIONAL, ScopeView.SYMBOL_SPECIFIC):
            raise ValueError(f"Invalid view: {self.view}. Must be ScopeView.CROSS_SECTIONAL or ScopeView.SYMBOL_SPECIFIC")
        # Hard invariant: SYMBOL_SPECIFIC requires symbol
        if view_enum == ScopeView.SYMBOL_SPECIFIC and not self.symbol:
            raise ValueError("ScopeView.SYMBOL_SPECIFIC view requires symbol")
        # Hard invariant: CROSS_SECTIONAL cannot have symbol
        if view_enum == ScopeView.CROSS_SECTIONAL and self.symbol:
            raise ValueError("ScopeView.CROSS_SECTIONAL view cannot have symbol")
        
        # Store as string for path construction (convert enum to string)
        view_str = str(view_enum)  # View enum's __str__ returns .value
        
        # Store as string for path construction (from normalized enum)
        self.view = view_str
        # Hard invariant: universe_sig is required
        if not self.universe_sig:
            raise ValueError("universe_sig is required for all scopes")
    
    @property
    def purpose(self) -> Optional["ScopePurpose"]:
        """Get the purpose (FINAL or ROUTING_EVAL) from scope."""
        return self._purpose
    
    @property
    def is_routing_eval(self) -> bool:
        """Check if this layout is for routing evaluation artifacts."""
        if not _WRITE_SCOPE_AVAILABLE or self._purpose is None:
            return False
        return self._purpose is ScopePurpose.ROUTING_EVAL
    
    def scope_key(self) -> str:
        """Get scope key string for consistent partitioning across metrics/trends/models.
        
        Returns: "view={view}/universe={universe_sig}/[symbol={symbol}]"
        """
        parts = [f"view={self.view}", f"universe={self.universe_sig}"]
        # Normalize view to enum for comparison
        view_enum = ScopeView.from_string(self.view) if isinstance(self.view, str) else self.view
        if view_enum == ScopeView.SYMBOL_SPECIFIC and self.symbol:
            parts.append(f"symbol={self.symbol}")
        return "/".join(parts)
    
    def repro_dir(self) -> Path:
        """Get reproducibility directory for target, scoped by stage/view/universe/symbol.
        
        Returns:
            With stage: targets/{target}/reproducibility/stage={stage}/{view}/universe={universe_sig}/
            Without stage (legacy): targets/{target}/reproducibility/{view}/universe={universe_sig}/
            ROUTING_EVAL: routing_evaluation/{view}/... (same pattern, no stage)
        
        Note: SYMBOL_SPECIFIC uses symbol= only (no universe=) to match cohort path pattern
        and avoid redundant nesting. The symbol already uniquely identifies the scope.
        """
        if self.is_routing_eval:
            # Routing evaluation artifacts go to separate root (no stage scoping)
            base = self.output_root / "routing_evaluation" / self.view
        else:
            # Final artifacts go under target with stage scoping
            repro_base = self.output_root / "targets" / self.target / "reproducibility"
            if self.stage:
                # SST: Ensure stage is string for path construction (defensive)
                stage_str = self.stage.value if isinstance(self.stage, ScopeStage) else str(self.stage)
                base = repro_base / f"stage={stage_str}" / self.view
            else:
                base = repro_base / self.view  # Legacy fallback
        
        # For SYMBOL_SPECIFIC: symbol= is sufficient (universe= is redundant)
        # For CROSS_SECTIONAL: universe= identifies the multi-symbol set
        # Normalize view to enum for comparison
        view_enum = ScopeView.from_string(self.view) if isinstance(self.view, str) else self.view
        if view_enum == ScopeView.SYMBOL_SPECIFIC and self.symbol:
            return base / f"symbol={self.symbol}"
        return base / f"universe={self.universe_sig}"
    
    def cohort_dir(self) -> Path:
        """Get cohort directory within reproducibility.
        
        Returns: repro_dir() / cohort={cohort_id}/
        
        Raises:
            ValueError: If cohort_id not set or doesn't match view
        """
        if not self.cohort_id:
            raise ValueError("cohort_id required for cohort_dir()")
        self.validate_cohort_id(self.cohort_id)
        return self.repro_dir() / f"cohort={self.cohort_id}"
    
    def feature_importance_dir(self) -> Path:
        """Get feature importance directory.
        
        Returns: repro_dir() / feature_importances/
        """
        return self.repro_dir() / "feature_importances"
    
    def metrics_dir(self) -> Path:
        """Get metrics directory for target.
        
        Returns: targets/{target}/metrics/{scope_key}/
        """
        return self.output_root / "targets" / self.target / "metrics" / self.scope_key()
    
    def trends_dir(self) -> Path:
        """Get trends directory for target.
        
        Returns: targets/{target}/trends/{scope_key}/
        """
        return self.output_root / "targets" / self.target / "trends" / self.scope_key()
    
    def model_dir(self, family: str) -> Path:
        """Get model directory for target and family.
        
        Args:
            family: Model family name (e.g., "lightgbm")
        
        Returns: targets/{target}/models/{family}/{scope_key}/
        """
        return self.output_root / "targets" / self.target / "models" / family / self.scope_key()
    
    def validate_cohort_id(self, cohort_id: str) -> None:
        """Validate cohort_id prefix matches view.
        
        Uses startswith() for explicit validation.
        
        Args:
            cohort_id: Cohort identifier to validate
        
        Raises:
            ValueError: If cohort_id is empty or prefix doesn't match view
        """
        if not cohort_id:
            raise ValueError("cohort_id cannot be empty")
        
        # Normalize view to enum for comparison
        view_enum = ScopeView.from_string(self.view) if isinstance(self.view, str) else self.view
        # Explicit prefix check (not split-based)
        if view_enum == ScopeView.CROSS_SECTIONAL:
            if not cohort_id.startswith("cs_"):
                raise ValueError(
                    f"Cohort ID scope violation: cohort_id={cohort_id} does not start with 'cs_' "
                    f"for view={self.view}"
                )
        elif view_enum == ScopeView.SYMBOL_SPECIFIC:
            if not cohort_id.startswith("sy_"):
                raise ValueError(
                    f"Cohort ID scope violation: cohort_id={cohort_id} does not start with 'sy_' "
                    f"for view={self.view}"
                )
        else:
            raise ValueError(f"Invalid view: {self.view}")


def validate_cohort_metadata(
    cohort_metadata: Dict[str, Any],
    view: str,
    symbol: Optional[str] = None
) -> None:
    """Validate that cohort metadata has all required fields for OutputLayout.
    
    Required: view, universe_sig, target
    Required if SYMBOL_SPECIFIC: symbol
    NOT required: cohort_id (passed as separate parameter to _save_to_cohort)
    
    Args:
        cohort_metadata: Metadata dict to validate
        view: Expected view value
        symbol: Expected symbol value (for SYMBOL_SPECIFIC)
    
    Raises:
        ValueError: If required fields are missing or mismatched
    """
    missing = []

    # Normalize expected/actual view
    expected_view = (view or "").upper()
    actual_view = _normalize_view(cohort_metadata)  # canonical or None
    raw_view = cohort_metadata.get("view")

    # Required: view (present + valid)
    if not raw_view:
        missing.append("view")
    elif not actual_view:
        missing.append(f"view (invalid: {raw_view})")

    # Required: universe sig
    if not _normalize_universe_sig(cohort_metadata):
        missing.append("universe_sig")

    # Required: target - use SST accessor
    if not extract_target(cohort_metadata):
        missing.append("target")

    # Required if SYMBOL_SPECIFIC
    # Use normalized view (actual if available, else expected) - normalize to enum for comparison
    normalized_view_str = actual_view or expected_view
    normalized_view_enum = ScopeView.from_string(normalized_view_str) if normalized_view_str else None
    if normalized_view_enum == ScopeView.SYMBOL_SPECIFIC:
        meta_symbol = cohort_metadata.get("symbol")
        if not symbol and not meta_symbol:
            missing.append("symbol (required for SYMBOL_SPECIFIC)")

    if missing:
        raise ValueError(
            f"Cohort metadata missing required fields: {missing}. "
            f"Metadata keys: {list(cohort_metadata.keys())}."
        )

    # Correct view match check: compare normalized values
    if actual_view != expected_view:
        raise ValueError(
            f"View mismatch in metadata: metadata has '{raw_view}' (normalized='{actual_view}') "
            f"but expected '{view}' (normalized='{expected_view}')"
        )

    # Symbol mismatch check (only when symbol is provided AND meta has symbol)
    expected_view_enum = ScopeView.from_string(expected_view) if expected_view else None
    if expected_view_enum == ScopeView.SYMBOL_SPECIFIC:
        meta_symbol = cohort_metadata.get("symbol")
        if symbol and meta_symbol and meta_symbol != symbol:
            raise ValueError(
                f"Symbol mismatch in metadata: metadata has '{meta_symbol}' "
                f"but expected '{symbol}'"
            )

    # No longer need to normalize universe key - legacy key fully deprecated

