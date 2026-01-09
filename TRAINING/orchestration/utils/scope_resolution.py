# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Scope Resolution: Canonical SST-derived write scope resolver.

This module provides the single source of truth for resolving
(view, symbol, universe_sig) tuples from SST resolved_data_config.

All writers (tracker, feature importance, stability snapshots) MUST
use resolve_write_scope() or WriteScope to ensure consistent scoping.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums: Centralized constants to eliminate string literal drift
# =============================================================================

class View(str, Enum):
    """
    Canonical view types. Eliminates string literal drift.
    
    Use View.CROSS_SECTIONAL instead of "CROSS_SECTIONAL" everywhere.
    """
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    SYMBOL_SPECIFIC = "SYMBOL_SPECIFIC"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, s: str) -> "View":
        """
        Normalize string to View enum.
        
        Handles common aliases like "INDIVIDUAL" and "LOSO" for SYMBOL_SPECIFIC.
        """
        if s is None:
            raise ValueError("View cannot be None")
        normalized = s.upper().replace("-", "_").replace(" ", "_")
        if normalized == "CROSS_SECTIONAL":
            return cls.CROSS_SECTIONAL
        elif normalized in ("SYMBOL_SPECIFIC", "INDIVIDUAL", "LOSO"):
            return cls.SYMBOL_SPECIFIC
        else:
            raise ValueError(f"Unknown view: {s}. Use 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")


class Stage(str, Enum):
    """
    Pipeline stages. Keeps logs and metadata clean.
    
    Use Stage.TARGET_RANKING instead of "TARGET_RANKING" everywhere.
    """
    TARGET_RANKING = "TARGET_RANKING"
    FEATURE_SELECTION = "FEATURE_SELECTION"
    TRAINING = "TRAINING"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, s: str) -> "Stage":
        """
        Normalize string to Stage enum.
        
        Handles common aliases like "MODEL_TRAINING" for TRAINING.
        """
        if s is None:
            raise ValueError("Stage cannot be None")
        normalized = s.upper().replace("MODEL_TRAINING", "TRAINING").replace(" ", "_").replace("-", "_")
        try:
            return cls(normalized)
        except ValueError:
            raise ValueError(f"Unknown stage: {s}. Use TARGET_RANKING, FEATURE_SELECTION, or TRAINING")


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
        view: View enum (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
        universe_sig: Hash of symbol universe (required, never None)
        symbol: Symbol ticker (None for CS, required for SS)
        purpose: FINAL or ROUTING_EVAL
        stage: Stage enum (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    
    Examples:
        # Create CS scope
        scope = WriteScope.for_cross_sectional(
            universe_sig="abc123def456",
            stage=Stage.TARGET_RANKING
        )
        
        # Create SS scope
        scope = WriteScope.for_symbol_specific(
            universe_sig="abc123def456",
            symbol="AAPL",
            stage=Stage.FEATURE_SELECTION
        )
        
        # Create routing evaluation scope
        scope = WriteScope.for_routing_eval(
            view=View.SYMBOL_SPECIFIC,
            universe_sig="abc123def456",
            symbol="AAPL",
            stage=Stage.TARGET_RANKING
        )
    """
    view: View  # Now enum, not str
    universe_sig: str  # Never None - required
    symbol: Optional[str]  # None for CS, required for SS
    purpose: ScopePurpose
    stage: Stage  # Now enum, not str
    
    def __post_init__(self):
        """Validate scope invariants at construction."""
        # Invariant 1: universe_sig is required
        if not self.universe_sig:
            raise ValueError(
                f"WriteScope: universe_sig is required but was None/empty. "
                f"view={self.view}, symbol={self.symbol}, stage={self.stage}"
            )
        
        # Invariant 2: CS must have symbol=None
        if self.view is View.CROSS_SECTIONAL and self.symbol is not None:
            raise ValueError(
                f"WriteScope: CS scope must have symbol=None, got symbol={self.symbol}. "
                f"stage={self.stage}, universe_sig={self.universe_sig}"
            )
        
        # Invariant 3: SS must have non-empty symbol
        if self.view is View.SYMBOL_SPECIFIC and not self.symbol:
            raise ValueError(
                f"WriteScope: SS scope requires non-empty symbol, got symbol={self.symbol}. "
                f"stage={self.stage}, universe_sig={self.universe_sig}"
            )
        
        # Invariant 4: view must be View enum (not raw string)
        if not isinstance(self.view, View):
            raise ValueError(
                f"WriteScope: view must be View enum, got {type(self.view).__name__}={self.view}. "
                f"Use View.CROSS_SECTIONAL or View.SYMBOL_SPECIFIC."
            )
        
        # Invariant 5: stage must be Stage enum (not raw string)
        if not isinstance(self.stage, Stage):
            raise ValueError(
                f"WriteScope: stage must be Stage enum, got {type(self.stage).__name__}={self.stage}. "
                f"Use Stage.TARGET_RANKING, Stage.FEATURE_SELECTION, or Stage.TRAINING."
            )
    
    @property
    def cohort_prefix(self) -> str:
        """Return expected cohort ID prefix for this scope."""
        return "cs_" if self.view is View.CROSS_SECTIONAL else "sy_"
    
    @property
    def is_final(self) -> bool:
        """Return True if this is a final (non-evaluation) scope."""
        return self.purpose is ScopePurpose.FINAL
    
    @property
    def is_routing_eval(self) -> bool:
        """Return True if this is a routing evaluation scope."""
        return self.purpose is ScopePurpose.ROUTING_EVAL
    
    def validate_cohort_id(self, cohort_id: str) -> None:
        """
        Validate that cohort_id prefix matches this scope's view.
        
        Raises:
            ValueError: If cohort_id prefix doesn't match view
        """
        if not cohort_id:
            return
        
        if self.view is View.CROSS_SECTIONAL and cohort_id.startswith("sy_"):
            raise ValueError(
                f"WriteScope: Cannot use sy_ cohort with CROSS_SECTIONAL view. "
                f"cohort_id={cohort_id}, scope={self}"
            )
        if self.view is View.SYMBOL_SPECIFIC and cohort_id.startswith("cs_"):
            raise ValueError(
                f"WriteScope: Cannot use cs_ cohort with SYMBOL_SPECIFIC view. "
                f"cohort_id={cohort_id}, scope={self}"
            )
    
    def to_additional_data(self, additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Populate additional_data dict with scope fields.
        
        Symbol key is ABSENT for CS (not null), present for SS.
        Also includes purpose for path/metadata consistency checks.
        
        Args:
            additional_data: Dict to populate (creates new if None)
        
        Returns:
            The populated dict
        """
        if additional_data is None:
            additional_data = {}
        
        additional_data['view'] = self.view.value  # Store as string for JSON
        additional_data['universe_sig'] = self.universe_sig
        additional_data['purpose'] = self.purpose.value  # Store purpose for consistency checks
        additional_data['stage'] = self.stage.value  # Store stage for traceability
        
        # Mirror into cs_config for legacy readers
        if 'cs_config' not in additional_data:
            additional_data['cs_config'] = {}
        additional_data['cs_config']['universe_sig'] = self.universe_sig
        
        # Symbol key: present for SS, ABSENT for CS
        if self.view is View.SYMBOL_SPECIFIC:
            additional_data['symbol'] = self.symbol
        elif 'symbol' in additional_data:
            del additional_data['symbol']
        
        return additional_data
    
    @classmethod
    def for_cross_sectional(
        cls,
        universe_sig: str,
        stage: Union[str, Stage],
        purpose: ScopePurpose = ScopePurpose.FINAL
    ) -> "WriteScope":
        """Factory for CROSS_SECTIONAL scope. Accepts string or Stage enum for stage."""
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        return cls(
            view=View.CROSS_SECTIONAL,
            universe_sig=universe_sig,
            symbol=None,
            purpose=purpose,
            stage=stage_enum
        )
    
    @classmethod
    def for_symbol_specific(
        cls,
        universe_sig: str,
        symbol: str,
        stage: Union[str, Stage],
        purpose: ScopePurpose = ScopePurpose.FINAL
    ) -> "WriteScope":
        """Factory for SYMBOL_SPECIFIC scope. Accepts string or Stage enum for stage."""
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        return cls(
            view=View.SYMBOL_SPECIFIC,
            universe_sig=universe_sig,
            symbol=symbol,
            purpose=purpose,
            stage=stage_enum
        )
    
    @classmethod
    def for_routing_eval(
        cls,
        view: Union[str, View],
        universe_sig: str,
        stage: Union[str, Stage],
        symbol: Optional[str] = None
    ) -> "WriteScope":
        """
        Factory for routing evaluation scope (artifacts go to routing_evaluation/ dir).
        
        Accepts string or enum for view and stage.
        """
        view_enum = View.from_string(view) if isinstance(view, str) else view
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        return cls(
            view=view_enum,
            universe_sig=universe_sig,
            symbol=symbol,
            purpose=ScopePurpose.ROUTING_EVAL,
            stage=stage_enum
        )
    
    @classmethod
    def from_resolved_data_config(
        cls,
        resolved_data_config: Dict[str, Any],
        stage: Union[str, Stage],
        symbol: Optional[str] = None,
        purpose: ScopePurpose = ScopePurpose.FINAL
    ) -> "WriteScope":
        """
        Create WriteScope from SST resolved_data_config.
        
        CRITICAL: universe_sig MUST come from resolved_data_config, never computed locally.
        This ensures single source of truth across CS and SS writes.
        
        Args:
            resolved_data_config: SST config with view, universe_sig, symbols
            stage: Pipeline stage (string or Stage enum)
            symbol: Symbol (optional, auto-derived for SS if SST has 1 symbol)
            purpose: FINAL or ROUTING_EVAL
        
        Returns:
            WriteScope instance
        
        Raises:
            ValueError: If required fields missing or invariants violated
        """
        view_str = resolved_data_config.get('view')
        universe_sig = resolved_data_config.get('universe_sig')
        sst_symbols = resolved_data_config.get('symbols') or []
        
        if not view_str:
            raise ValueError(
                f"WriteScope.from_resolved_data_config: view missing from SST. "
                f"keys={list(resolved_data_config.keys())}"
            )
        
        if not universe_sig:
            raise ValueError(
                f"WriteScope.from_resolved_data_config: universe_sig missing from SST. "
                f"keys={list(resolved_data_config.keys())}"
            )
        
        # Normalize to enums
        view_enum = View.from_string(view_str)
        stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
        
        # For SS, derive symbol if not provided and SST has exactly 1 symbol
        if view_enum is View.SYMBOL_SPECIFIC and not symbol:
            if len(sst_symbols) == 1:
                symbol = sst_symbols[0]
                logger.debug(f"WriteScope: auto-derived symbol={symbol} from SST symbols list")
            else:
                raise ValueError(
                    f"WriteScope.from_resolved_data_config: SS view requires symbol but "
                    f"symbol not provided and SST has {len(sst_symbols)} symbols (need exactly 1 for auto-derive)."
                )
        
        return cls(
            view=view_enum,
            universe_sig=universe_sig,
            symbol=symbol,
            purpose=purpose,
            stage=stage_enum
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
        resolved_data_config: SST config with view, universe_sig, and symbols list
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
        # sst = {"view": "SYMBOL_SPECIFIC", "symbols": ["AAPL"], "universe_sig": "..."}
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
        sst_view = resolved_data_config.get('view')
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
                f"SCOPE BUG: caller_view=SYMBOL_SPECIFIC but SST view=CROSS_SECTIONAL. "
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
    view_for_writes: Union[str, View],
    symbol_for_writes: Optional[str],
    universe_sig_for_writes: Optional[str]
) -> Dict[str, Any]:
    """
    Populate additional_data dict with scope fields for tracker/writer.
    
    DEPRECATED: Prefer using WriteScope.to_additional_data() instead.
    
    This is a convenience function that applies the scope tuple to
    additional_data in the correct way (symbol key absent for CS, not null).
    
    Args:
        additional_data: The dict to populate (mutated in place)
        view_for_writes: From resolve_write_scope() (string or View enum)
        symbol_for_writes: From resolve_write_scope()
        universe_sig_for_writes: From resolve_write_scope()
    
    Returns:
        The mutated additional_data dict (for chaining)
    """
    # Normalize view to string for storage
    view_str = view_for_writes.value if isinstance(view_for_writes, View) else view_for_writes
    additional_data['view'] = view_str
    
    if universe_sig_for_writes:
        additional_data['universe_sig'] = universe_sig_for_writes
        # Mirror into cs_config for legacy readers
        if 'cs_config' not in additional_data:
            additional_data['cs_config'] = {}
        additional_data['cs_config']['universe_sig'] = universe_sig_for_writes
    
    # Check if view is SS (handle both string and enum)
    is_ss = (view_for_writes is View.SYMBOL_SPECIFIC if isinstance(view_for_writes, View) 
             else view_for_writes == "SYMBOL_SPECIFIC")
    
    # Only add symbol for SS (key must be ABSENT for CS, not null)
    if is_ss and symbol_for_writes:
        additional_data['symbol'] = symbol_for_writes
    elif 'symbol' in additional_data:
        # Remove stale symbol key if switching to CS
        del additional_data['symbol']
    
    return additional_data



