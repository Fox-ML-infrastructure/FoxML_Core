# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
OutputLayout: Canonical output path builder with view+universe scoping.

This is the Single Source of Truth (SST) for all artifact paths.
All paths include universe={universe_sig}/ to prevent cross-run collisions.

Usage:
    layout = OutputLayout(
        output_root=run_dir,
        target="fwd_ret_5d",
        view="CROSS_SECTIONAL",
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
    """Normalize universe signature: check both top-level and nested cs_config.
    
    Single source of truth for universe_sig extraction. Checks:
    1. meta["universe_sig"] (canonical)
    2. meta["universe_id"] (legacy alias)
    3. meta["cs_config"]["universe_sig"] (nested legacy)
    4. meta["cs_config"]["universe_id"] (nested legacy alias)
    """
    # Top-level canonical
    sig = meta.get("universe_sig") or meta.get("universe_id")
    if sig:
        return sig
    # Nested fallback (legacy callers)
    cs_config = meta.get("cs_config")
    if isinstance(cs_config, dict):
        return cs_config.get("universe_sig") or cs_config.get("universe_id")
    return None


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
    
    All paths include universe={universe_sig}/ to prevent cross-run collisions.
    
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
            view="CROSS_SECTIONAL",
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
    ):
        """Initialize OutputLayout with either WriteScope (preferred) or loose args.
        
        Args:
            output_root: Base output directory for the run
            target: Target name
            scope: WriteScope object (preferred - derives view, universe_sig, symbol, purpose)
            view: View string (deprecated - use scope instead)
            universe_sig: Universe signature (deprecated - use scope instead)
            symbol: Symbol for SYMBOL_SPECIFIC (deprecated - use scope instead)
            cohort_id: Optional cohort ID
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
        else:
            # Deprecated path: loose args
            warnings.warn(
                "OutputLayout with loose args (view, universe_sig, symbol) is deprecated. "
                "Pass scope=WriteScope(...) instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.scope = None
            self.view = view
            self.universe_sig = universe_sig
            self.symbol = symbol
            self._purpose = ScopePurpose.FINAL if _WRITE_SCOPE_AVAILABLE else None
        
        # Hard invariant: view must be valid canonical value
        if self.view not in {"CROSS_SECTIONAL", "SYMBOL_SPECIFIC"}:
            raise ValueError(f"Invalid view: {self.view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
        # Hard invariant: SYMBOL_SPECIFIC requires symbol
        if self.view == "SYMBOL_SPECIFIC" and not self.symbol:
            raise ValueError("SYMBOL_SPECIFIC view requires symbol")
        # Hard invariant: CROSS_SECTIONAL cannot have symbol
        if self.view == "CROSS_SECTIONAL" and self.symbol:
            raise ValueError("CROSS_SECTIONAL view cannot have symbol")
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
        if self.view == "SYMBOL_SPECIFIC" and self.symbol:
            parts.append(f"symbol={self.symbol}")
        return "/".join(parts)
    
    def repro_dir(self) -> Path:
        """Get reproducibility directory for target, scoped by view/universe/symbol.
        
        Returns:
            FINAL: targets/{target}/reproducibility/{view}/universe={universe_sig}/[symbol={symbol}/]
            ROUTING_EVAL: routing_evaluation/{view}/universe={universe_sig}/[symbol={symbol}/]
        """
        if self.is_routing_eval:
            # Routing evaluation artifacts go to separate root
            base = self.output_root / "routing_evaluation" / self.view
        else:
            # Final artifacts go under target
            base = self.output_root / "targets" / self.target / "reproducibility" / self.view
        
        base = base / f"universe={self.universe_sig}"
        if self.view == "SYMBOL_SPECIFIC" and self.symbol:
            return base / f"symbol={self.symbol}"
        return base
    
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
        
        # Explicit prefix check (not split-based)
        if self.view == "CROSS_SECTIONAL":
            if not cohort_id.startswith("cs_"):
                raise ValueError(
                    f"Cohort ID scope violation: cohort_id={cohort_id} does not start with 'cs_' "
                    f"for view={self.view}"
                )
        elif self.view == "SYMBOL_SPECIFIC":
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
    
    Required: view, universe_sig (or universe_id), target (or target_name)
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
        missing.append("universe_sig (or universe_id)")

    # Required: target
    if not (cohort_metadata.get("target") or cohort_metadata.get("target_name")):
        missing.append("target (or target_name)")

    # Required if SYMBOL_SPECIFIC
    # Use normalized view (actual if available, else expected)
    normalized_view = actual_view or expected_view
    if normalized_view == "SYMBOL_SPECIFIC":
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
    if expected_view == "SYMBOL_SPECIFIC":
        meta_symbol = cohort_metadata.get("symbol")
        if symbol and meta_symbol and meta_symbol != symbol:
            raise ValueError(
                f"Symbol mismatch in metadata: metadata has '{meta_symbol}' "
                f"but expected '{symbol}'"
            )

    # Normalize universe key (write canonical)
    if "universe_id" in cohort_metadata and "universe_sig" not in cohort_metadata:
        cohort_metadata["universe_sig"] = cohort_metadata["universe_id"]

