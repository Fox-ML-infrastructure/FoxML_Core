# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Scope Resolution: Canonical SST-derived write scope resolver.

This module provides the single source of truth for resolving
(view, symbol, universe_sig) tuples from SST resolved_data_config.

All writers (tracker, feature importance, stability snapshots) MUST
use resolve_write_scope() to ensure consistent scoping.
"""

from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def resolve_write_scope(
    resolved_data_config: Optional[Dict[str, Any]],
    caller_view: str,
    caller_symbol: Optional[str],
    strict: bool = False
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Canonical SST-derived write scope. All writers MUST use this.
    
    Rules:
    - CS → SS fallback: allowed (insufficient symbols), auto-derive symbol if unambiguous
    - SS → CS promotion: BUG (min_cs=1 made CS "valid" for single symbol)
    - Strict mode: raise on any scope ambiguity or missing data
    
    Args:
        resolved_data_config: SST config with resolved_mode, universe_sig, and symbols list
        caller_view: The view requested by caller
        caller_symbol: The symbol requested by caller (may be None)
        strict: If True, raise on any scope invariant violation
    
    Returns:
        (view_for_writes, symbol_for_writes, universe_sig_for_writes)
    
    Raises:
        ValueError: In strict mode, if any scope invariant is violated
    
    Examples:
        # Normal CS case
        view, symbol, sig = resolve_write_scope(sst, "CROSS_SECTIONAL", None, strict=True)
        # Returns: ("CROSS_SECTIONAL", None, "abc123...")
        
        # Normal SS case
        view, symbol, sig = resolve_write_scope(sst, "SYMBOL_SPECIFIC", "AAPL", strict=True)
        # Returns: ("SYMBOL_SPECIFIC", "AAPL", "abc123...")
        
        # CS→SS fallback with symbol derivation
        # sst = {"resolved_mode": "SYMBOL_SPECIFIC", "symbols": ["AAPL"], "universe_sig": "..."}
        view, symbol, sig = resolve_write_scope(sst, "CROSS_SECTIONAL", None, strict=True)
        # Returns: ("SYMBOL_SPECIFIC", "AAPL", "abc123...")
    """
    # Strict mode: require resolved_data_config
    if strict and resolved_data_config is None:
        raise ValueError(
            f"SCOPE BUG: resolved_data_config is None in strict mode. "
            f"Cannot resolve write scope without SST. caller_view={caller_view}, caller_symbol={caller_symbol}"
        )
    
    if resolved_data_config:
        sst_view = resolved_data_config.get('resolved_mode')
        universe_sig = resolved_data_config.get('universe_sig')
        sst_symbols: List[str] = resolved_data_config.get('symbols') or []
    else:
        sst_view = None
        universe_sig = None
        sst_symbols = []
    
    # Asymmetric mode resolution
    # SS → CS promotion is a BUG (min_cs=1 made CS "valid" for single symbol)
    # CS → SS fallback is ALLOWED (insufficient symbols to run cross-sectional)
    if caller_view == "SYMBOL_SPECIFIC" and sst_view == "CROSS_SECTIONAL":
        # SS → CS promotion detected - this is a bug
        if strict:
            raise ValueError(
                f"SCOPE BUG: caller_view=SYMBOL_SPECIFIC but SST resolved_mode=CROSS_SECTIONAL. "
                f"This is invalid SS→CS promotion. Check min_cs config or caller logic."
            )
        else:
            logger.warning(
                f"⚠️  SS→CS promotion detected (caller_view=SS, SST=CS). "
                f"Forcing view_for_writes=SYMBOL_SPECIFIC to prevent corruption."
            )
            view = "SYMBOL_SPECIFIC"
    else:
        # Normal case: trust SST if available, else caller
        view = sst_view or caller_view
    
    # Strict mode: don't silently drop symbol in CS (makes caller bugs visible)
    if strict and view == "CROSS_SECTIONAL" and caller_symbol is not None:
        raise ValueError(
            f"SCOPE BUG: caller_symbol={caller_symbol} provided but view=CROSS_SECTIONAL. "
            f"Caller should not pass symbol for CS writes. This hides a routing bug."
        )
    
    # Symbol resolution for SYMBOL_SPECIFIC
    symbol = None
    if view == "SYMBOL_SPECIFIC":
        if caller_symbol:
            symbol = caller_symbol
        elif len(sst_symbols) == 1:
            # CS → SS fallback: derive symbol from unambiguous SST symbols list
            symbol = sst_symbols[0]
            logger.debug(f"Derived symbol={symbol} from SST symbols list (CS→SS fallback)")
        elif strict:
            raise ValueError(
                f"SCOPE BUG: view=SYMBOL_SPECIFIC but caller_symbol is None and "
                f"cannot derive from SST symbols (len={len(sst_symbols)}). "
                f"Caller must provide symbol or SST symbols must have exactly 1 element."
            )
    
    # Strict mode: require universe_sig
    if strict and not universe_sig:
        raise ValueError(
            f"SCOPE BUG: universe_sig missing from resolved_data_config. "
            f"Cannot write artifacts without universe scoping."
        )
    
    return view, symbol, universe_sig


def populate_additional_data(
    additional_data: Dict[str, Any],
    view_for_writes: str,
    symbol_for_writes: Optional[str],
    universe_sig_for_writes: Optional[str]
) -> Dict[str, Any]:
    """
    Populate additional_data dict with scope fields for tracker/writer.
    
    This is a convenience function that applies the scope tuple to
    additional_data in the correct way (symbol key absent for CS, not null).
    
    Args:
        additional_data: The dict to populate (mutated in place)
        view_for_writes: From resolve_write_scope()
        symbol_for_writes: From resolve_write_scope()
        universe_sig_for_writes: From resolve_write_scope()
    
    Returns:
        The mutated additional_data dict (for chaining)
    """
    additional_data['view'] = view_for_writes
    
    if universe_sig_for_writes:
        additional_data['universe_sig'] = universe_sig_for_writes
        # Mirror into cs_config for legacy readers
        if 'cs_config' not in additional_data:
            additional_data['cs_config'] = {}
        additional_data['cs_config']['universe_sig'] = universe_sig_for_writes
    
    # Only add symbol for SS (key must be ABSENT for CS, not null)
    if view_for_writes == "SYMBOL_SPECIFIC" and symbol_for_writes:
        additional_data['symbol'] = symbol_for_writes
    elif 'symbol' in additional_data:
        # Remove stale symbol key if switching to CS
        del additional_data['symbol']
    
    return additional_data



