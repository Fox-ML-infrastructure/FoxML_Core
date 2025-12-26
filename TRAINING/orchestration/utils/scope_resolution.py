# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Scope Resolution: Canonical SST-derived write scope resolver.

This module provides the single source of truth for resolving
(view, symbol, universe_sig) tuples from SST resolved_data_config.

All writers (tracker, feature importance, stability snapshots) MUST
use resolve_write_scope() or WriteScope to ensure consistent scoping.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# WriteScope: First-class scope object for reproducibility writes
# =============================================================================

class ScopePurpose(Enum):
    """Purpose of the write - determines output directory root."""
    FINAL = "FINAL"              # Final artifacts (reproducibility/{view}/)
    ROUTING_EVAL = "ROUTING_EVAL"  # Routing evaluation artifacts (reproducibility/routing_evaluation/{view}/)


@dataclass(frozen=True)
class WriteScope:
    """
    Canonical scope object for reproducibility writes.
    
    This is a first-class object that encapsulates all scope information
    and validates invariants at construction time. Wrong scope combinations
    are impossible to create.
    
    All tracker/writer methods should accept WriteScope instead of loose
    (view, symbol, universe_sig) args to prevent scope contamination bugs.
    
    Attributes:
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        universe_sig: Hash of symbol universe (required, never None)
        symbol: Symbol ticker (None for CS, required for SS)
        purpose: FINAL or ROUTING_EVAL
        stage: Pipeline stage ("TARGET_RANKING", "FEATURE_SELECTION", "TRAINING")
    
    Examples:
        # Create CS scope
        scope = WriteScope.for_cross_sectional(
            universe_sig="abc123def456",
            stage="TARGET_RANKING"
        )
        
        # Create SS scope
        scope = WriteScope.for_symbol_specific(
            universe_sig="abc123def456",
            symbol="AAPL",
            stage="FEATURE_SELECTION"
        )
        
        # Create routing evaluation scope
        scope = WriteScope.for_routing_eval(
            view="SYMBOL_SPECIFIC",
            universe_sig="abc123def456",
            symbol="AAPL",
            stage="TARGET_RANKING"
        )
    """
    view: str  # "CROSS_SECTIONAL" | "SYMBOL_SPECIFIC"
    universe_sig: str  # Never None - required
    symbol: Optional[str]  # None for CS, required for SS
    purpose: ScopePurpose
    stage: str  # "TARGET_RANKING" | "FEATURE_SELECTION" | "TRAINING"
    
    def __post_init__(self):
        """Validate scope invariants at construction."""
        # Invariant 1: universe_sig is required
        if not self.universe_sig:
            raise ValueError(
                f"WriteScope: universe_sig is required but was None/empty. "
                f"view={self.view}, symbol={self.symbol}, stage={self.stage}"
            )
        
        # Invariant 2: CS must have symbol=None
        if self.view == "CROSS_SECTIONAL" and self.symbol is not None:
            raise ValueError(
                f"WriteScope: CS scope must have symbol=None, got symbol={self.symbol}. "
                f"stage={self.stage}, universe_sig={self.universe_sig}"
            )
        
        # Invariant 3: SS must have non-empty symbol
        if self.view == "SYMBOL_SPECIFIC" and not self.symbol:
            raise ValueError(
                f"WriteScope: SS scope requires non-empty symbol, got symbol={self.symbol}. "
                f"stage={self.stage}, universe_sig={self.universe_sig}"
            )
        
        # Invariant 4: view must be valid
        if self.view not in ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC"):
            raise ValueError(
                f"WriteScope: invalid view={self.view}. "
                f"Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'."
            )
        
        # Invariant 5: stage must be valid
        valid_stages = ("TARGET_RANKING", "FEATURE_SELECTION", "TRAINING")
        if self.stage not in valid_stages:
            raise ValueError(
                f"WriteScope: invalid stage={self.stage}. "
                f"Must be one of {valid_stages}."
            )
    
    @property
    def cohort_prefix(self) -> str:
        """Return expected cohort ID prefix for this scope."""
        return "cs_" if self.view == "CROSS_SECTIONAL" else "sy_"
    
    @property
    def is_final(self) -> bool:
        """Return True if this is a final (non-evaluation) scope."""
        return self.purpose == ScopePurpose.FINAL
    
    @property
    def is_routing_eval(self) -> bool:
        """Return True if this is a routing evaluation scope."""
        return self.purpose == ScopePurpose.ROUTING_EVAL
    
    def validate_cohort_id(self, cohort_id: str) -> None:
        """
        Validate that cohort_id prefix matches this scope's view.
        
        Raises:
            ValueError: If cohort_id prefix doesn't match view
        """
        if not cohort_id:
            return
        
        if self.view == "CROSS_SECTIONAL" and cohort_id.startswith("sy_"):
            raise ValueError(
                f"WriteScope: Cannot use sy_ cohort with CROSS_SECTIONAL view. "
                f"cohort_id={cohort_id}, scope={self}"
            )
        if self.view == "SYMBOL_SPECIFIC" and cohort_id.startswith("cs_"):
            raise ValueError(
                f"WriteScope: Cannot use cs_ cohort with SYMBOL_SPECIFIC view. "
                f"cohort_id={cohort_id}, scope={self}"
            )
    
    def to_additional_data(self, additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Populate additional_data dict with scope fields.
        
        Symbol key is ABSENT for CS (not null), present for SS.
        
        Args:
            additional_data: Dict to populate (creates new if None)
        
        Returns:
            The populated dict
        """
        if additional_data is None:
            additional_data = {}
        
        additional_data['view'] = self.view
        additional_data['universe_sig'] = self.universe_sig
        
        # Mirror into cs_config for legacy readers
        if 'cs_config' not in additional_data:
            additional_data['cs_config'] = {}
        additional_data['cs_config']['universe_sig'] = self.universe_sig
        
        # Symbol key: present for SS, ABSENT for CS
        if self.view == "SYMBOL_SPECIFIC":
            additional_data['symbol'] = self.symbol
        elif 'symbol' in additional_data:
            del additional_data['symbol']
        
        return additional_data
    
    @classmethod
    def for_cross_sectional(
        cls,
        universe_sig: str,
        stage: str,
        purpose: ScopePurpose = ScopePurpose.FINAL
    ) -> "WriteScope":
        """Factory for CROSS_SECTIONAL scope."""
        return cls(
            view="CROSS_SECTIONAL",
            universe_sig=universe_sig,
            symbol=None,
            purpose=purpose,
            stage=stage
        )
    
    @classmethod
    def for_symbol_specific(
        cls,
        universe_sig: str,
        symbol: str,
        stage: str,
        purpose: ScopePurpose = ScopePurpose.FINAL
    ) -> "WriteScope":
        """Factory for SYMBOL_SPECIFIC scope."""
        return cls(
            view="SYMBOL_SPECIFIC",
            universe_sig=universe_sig,
            symbol=symbol,
            purpose=purpose,
            stage=stage
        )
    
    @classmethod
    def for_routing_eval(
        cls,
        view: str,
        universe_sig: str,
        stage: str,
        symbol: Optional[str] = None
    ) -> "WriteScope":
        """Factory for routing evaluation scope (artifacts go to routing_evaluation/ dir)."""
        return cls(
            view=view,
            universe_sig=universe_sig,
            symbol=symbol,
            purpose=ScopePurpose.ROUTING_EVAL,
            stage=stage
        )
    
    @classmethod
    def from_resolved_data_config(
        cls,
        resolved_data_config: Dict[str, Any],
        stage: str,
        symbol: Optional[str] = None,
        purpose: ScopePurpose = ScopePurpose.FINAL
    ) -> "WriteScope":
        """
        Create WriteScope from SST resolved_data_config.
        
        Args:
            resolved_data_config: SST config with resolved_mode, universe_sig, symbols
            stage: Pipeline stage
            symbol: Symbol (optional, auto-derived for SS if SST has 1 symbol)
            purpose: FINAL or ROUTING_EVAL
        
        Returns:
            WriteScope instance
        
        Raises:
            ValueError: If required fields missing or invariants violated
        """
        view = resolved_data_config.get('resolved_mode')
        universe_sig = resolved_data_config.get('universe_sig')
        sst_symbols = resolved_data_config.get('symbols') or []
        
        if not view:
            raise ValueError(
                f"WriteScope.from_resolved_data_config: resolved_mode missing from SST. "
                f"keys={list(resolved_data_config.keys())}"
            )
        
        if not universe_sig:
            raise ValueError(
                f"WriteScope.from_resolved_data_config: universe_sig missing from SST. "
                f"keys={list(resolved_data_config.keys())}"
            )
        
        # For SS, derive symbol if not provided and SST has exactly 1 symbol
        if view == "SYMBOL_SPECIFIC" and not symbol:
            if len(sst_symbols) == 1:
                symbol = sst_symbols[0]
                logger.debug(f"WriteScope: auto-derived symbol={symbol} from SST symbols list")
            else:
                raise ValueError(
                    f"WriteScope.from_resolved_data_config: SS view requires symbol but "
                    f"symbol not provided and SST has {len(sst_symbols)} symbols (need exactly 1 for auto-derive)."
                )
        
        return cls(
            view=view,
            universe_sig=universe_sig,
            symbol=symbol,
            purpose=purpose,
            stage=stage
        )


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



