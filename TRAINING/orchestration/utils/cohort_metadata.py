# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cohort Metadata Utilities

Provides validated cohort metadata construction and universe_sig format guards.
Used by feature selection and other stages to ensure scope partitioning consistency.

Key Functions:
- validate_universe_sig(): Guard against passing view name as universe_sig
- build_cohort_metadata(): Build validated metadata dict with required fields
"""

import os
from typing import Optional, Union
import logging

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View

# SST: Import WriteScope for scope-aware metadata construction
try:
    from TRAINING.orchestration.utils.scope_resolution import WriteScope
    _WRITE_SCOPE_AVAILABLE = True
except ImportError:
    _WRITE_SCOPE_AVAILABLE = False
    WriteScope = None

logger = logging.getLogger(__name__)

# Canonical view names - used to detect view-as-universe bugs
CANON_VIEWS = {"CROSS_SECTIONAL", "SYMBOL_SPECIFIC"}


def validate_universe_sig(universe_sig: Optional[str]) -> None:
    """
    Guard against passing view name as universe_sig.
    
    This prevents the regression where `view` is accidentally passed as `universe_sig`,
    which was causing scope partitioning bugs in feature selection.
    
    Rules (format-agnostic to allow future hash algo changes):
    - Reject if None or empty
    - Reject if in CANON_VIEWS (the actual bug we're preventing)
    - Reject if too short (< 8 chars)
    - Reject if contains path-unsafe chars (os.sep, os.altsep, whitespace, newlines)
    
    Args:
        universe_sig: The universe signature to validate
        
    Raises:
        ValueError: If universe_sig is invalid
    """
    if not universe_sig:
        raise ValueError("universe_sig cannot be empty or None")
    
    if universe_sig in CANON_VIEWS:
        raise ValueError(
            f"SCOPE BUG: universe_sig='{universe_sig}' is a view name, not a hash. "
            f"Extract from resolved_data_config['universe_sig'] instead."
        )
    
    if len(universe_sig) < 8:
        raise ValueError(f"universe_sig too short: '{universe_sig}' (min 8 chars)")
    
    # Robust path-unsafe check (includes newlines and carriage returns)
    bad_chars = {os.sep, '\r', '\n', '\t', ' '}
    if os.altsep:
        bad_chars.add(os.altsep)
    
    if any(ch in universe_sig for ch in bad_chars):
        raise ValueError(f"universe_sig contains path-unsafe chars: '{universe_sig}'")


def build_cohort_metadata(
    *,
    target: str,
    # SST: Preferred - accept WriteScope directly
    scope: Optional["WriteScope"] = None,
    # DEPRECATED: Loose args (for backward compat)
    view: Optional[Union[str, View]] = None,
    universe_sig: Optional[str] = None,
    symbol: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build a validated cohort metadata dict.
    
    This ensures all required fields are present and valid before any artifact writes.
    Use this helper at all feature selection (and other stage) write points to prevent
    "half-pass" metadata bugs.
    
    Args:
        target: Target name (e.g., "fwd_ret_5d")
        scope: WriteScope object (preferred, SST-compliant)
        view: View enum or "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" string (deprecated, use scope)
        universe_sig: Universe signature hash (deprecated, use scope)
        symbol: Symbol name (deprecated, use scope)
        extra: Additional metadata fields to include
        
    Returns:
        Dict with validated metadata fields:
        - target: str
        - view: str (uppercase canonical)
        - universe_sig: str (validated)
        - symbol: str (if SYMBOL_SPECIFIC)
        - ...any extra fields
        
    Raises:
        ValueError: If any required field is missing or invalid
    """
    # SST: Extract from WriteScope if provided
    if scope is not None:
        if not _WRITE_SCOPE_AVAILABLE:
            raise ValueError("WriteScope not available but scope was passed")
        view = scope.view
        universe_sig = scope.universe_sig
        symbol = scope.symbol
    elif view is None or universe_sig is None:
        raise ValueError("Either scope or both view and universe_sig must be provided")
    
    # Normalize view to enum
    view_enum = View.from_string(view) if isinstance(view, str) else view
    if view_enum not in (View.CROSS_SECTIONAL, View.SYMBOL_SPECIFIC):
        raise ValueError(f"Invalid view: {view}. Must be View.CROSS_SECTIONAL or View.SYMBOL_SPECIFIC")
    
    # Validate universe_sig (catches view-as-universe bugs)
    validate_universe_sig(universe_sig)
    
    # SYMBOL_SPECIFIC requires symbol
    if view_enum == View.SYMBOL_SPECIFIC and not symbol:
        raise ValueError("symbol required for View.SYMBOL_SPECIFIC view")
    
    # CROSS_SECTIONAL should not have symbol
    if view_enum == View.CROSS_SECTIONAL and symbol:
        logger.warning(
            f"symbol='{symbol}' provided for CROSS_SECTIONAL view, ignoring. "
            f"CROSS_SECTIONAL artifacts should not be symbol-scoped."
        )
        symbol = None
    
    # Build metadata dict - use enum value for JSON serialization
    meta = {
        "target": target,
        "view": view_enum.value,  # Use enum value for JSON serialization
        "universe_sig": universe_sig,  # Write canonical key only
    }
    
    if symbol:
        meta["symbol"] = symbol
    
    if extra:
        meta.update(extra)
    
    return meta

