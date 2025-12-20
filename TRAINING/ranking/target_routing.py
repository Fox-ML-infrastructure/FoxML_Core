"""
Target Routing Logic

Determines which view (CROSS_SECTIONAL, SYMBOL_SPECIFIC, BOTH, or BLOCKED) each target
should use based on dual-view evaluation results.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def _compute_target_routing_decisions(
    results_cs: List[Any],  # List of TargetPredictabilityScore (cross-sectional)
    results_sym: Dict[str, Dict[str, Any]],  # {target_name: {symbol: TargetPredictabilityScore}}
    results_loso: Dict[str, Dict[str, Any]],  # {target_name: {symbol: TargetPredictabilityScore}} (optional)
    symbol_skip_reasons: Dict[str, Dict[str, Dict[str, Any]]] = None  # {target_name: {symbol: {reason, status, ...}}}
) -> Dict[str, Dict[str, Any]]:
    """
    Compute routing decisions for each target based on dual-view scores.
    
    Routing rules:
    - CROSS_SECTIONAL only: cs_auc >= T_cs AND frac_symbols_good >= T_frac
    - SYMBOL_SPECIFIC only: cs_auc < T_cs AND exists symbol with auc >= T_sym
    - BOTH: cs_auc >= T_cs BUT performance is concentrated (high IQR / low frac_symbols_good)
    - BLOCKED: cs_auc >= 0.90 OR any symbol auc >= 0.95 → require label/split sanity tests
    
    Args:
        results_cs: Cross-sectional results
        results_sym: Symbol-specific results by target
        results_loso: LOSO results by target (optional)
        symbol_skip_reasons: Skip reasons by target and symbol (optional)
    
    Returns:
        Dict mapping target_name -> routing decision dict
    """
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        routing_cfg = get_cfg("target_ranking.routing", default={}, config_name="target_ranking_config")
        T_cs = float(routing_cfg.get('cs_auc_threshold', 0.65))
        T_frac = float(routing_cfg.get('frac_symbols_good_threshold', 0.5))
        T_sym = float(routing_cfg.get('symbol_auc_threshold', 0.60))
        T_suspicious_cs = float(routing_cfg.get('suspicious_cs_auc', 0.90))
        T_suspicious_sym = float(routing_cfg.get('suspicious_symbol_auc', 0.95))
    except Exception:
        # Fallback defaults
        T_cs = 0.65
        T_frac = 0.5
        T_sym = 0.60
        T_suspicious_cs = 0.90
        T_suspicious_sym = 0.95
    
    routing_decisions = {}
    
    # Collect all target names (from both CS results and symbol-specific results)
    # This ensures we process targets even if CS failed but symbol-specific succeeded
    all_target_names = set()
    for result_cs in results_cs:
        all_target_names.add(result_cs.target_name)
    for target_name in results_sym.keys():
        all_target_names.add(target_name)
    
    # Process each target (whether it has CS results or not)
    for target_name in all_target_names:
        # Find cross-sectional result if it exists
        result_cs = None
        cs_auc = -999.0  # Default to failed score
        for r in results_cs:
            if r.target_name == target_name:
                result_cs = r
                cs_auc = r.mean_score
                break
        
        # Get symbol-specific results for this target
        sym_results = results_sym.get(target_name, {})
        
        # Compute symbol distribution stats
        symbol_aucs = []
        if sym_results:
            for symbol, result_sym in sym_results.items():
                if result_sym.mean_score != -999.0:  # Valid result
                    symbol_aucs.append(result_sym.mean_score)
        
        if symbol_aucs:
            symbol_auc_mean = np.mean(symbol_aucs)
            symbol_auc_median = np.median(symbol_aucs)
            symbol_auc_min = np.min(symbol_aucs)
            symbol_auc_max = np.max(symbol_aucs)
            symbol_auc_iqr = np.percentile(symbol_aucs, 75) - np.percentile(symbol_aucs, 25)
            frac_symbols_good = sum(1 for auc in symbol_aucs if auc >= T_sym) / len(symbol_aucs)
            winner_symbols = [sym for sym, result in sym_results.items() 
                            if result.mean_score >= T_sym and result.mean_score != -999.0]
        else:
            symbol_auc_mean = None
            symbol_auc_median = None
            symbol_auc_min = None
            symbol_auc_max = None
            symbol_auc_iqr = None
            frac_symbols_good = 0.0
            winner_symbols = []
        
        # Handle case where CS failed (cs_auc = -999.0 or result_cs is None)
        if result_cs is None or cs_auc == -999.0:
            # CS failed - check if symbol-specific works
            if symbol_aucs and max(symbol_aucs) >= T_sym:
                route = "SYMBOL_SPECIFIC"
                reason = f"cs_failed (mean_score=-999.0) BUT exists symbol with auc >= {T_sym}"
                winner_symbols_str = ', '.join(winner_symbols[:5])
                if len(winner_symbols) > 5:
                    winner_symbols_str += f", ... ({len(winner_symbols)} total)"
                reason += f" (winners: {winner_symbols_str})"
            else:
                # CS failed and no good symbol-specific results
                route = "BLOCKED"
                if symbol_aucs:
                    reason = f"cs_failed AND max_symbol_auc={max(symbol_aucs):.3f} < {T_sym} (no viable route)"
                else:
                    reason = f"cs_failed AND no symbol-specific results (no viable route)"
        
        # Check for suspicious scores (BLOCKED)
        elif cs_auc >= T_suspicious_cs or (symbol_aucs and max(symbol_aucs) >= T_suspicious_sym):
            route = "BLOCKED"
            reason = f"cs_auc={cs_auc:.3f} >= {T_suspicious_cs}" if cs_auc >= T_suspicious_cs else \
                    f"max_symbol_auc={max(symbol_aucs):.3f} >= {T_suspicious_sym}"
        
        # CROSS_SECTIONAL only: strong CS performance + good symbol coverage
        elif cs_auc >= T_cs and frac_symbols_good >= T_frac:
            route = "CROSS_SECTIONAL"
            reason = f"cs_auc={cs_auc:.3f} >= {T_cs} AND frac_symbols_good={frac_symbols_good:.2f} >= {T_frac}"
        
        # SYMBOL_SPECIFIC only: weak CS but some symbols work
        elif cs_auc < T_cs and symbol_aucs and max(symbol_aucs) >= T_sym:
            route = "SYMBOL_SPECIFIC"
            reason = f"cs_auc={cs_auc:.3f} < {T_cs} BUT exists symbol with auc >= {T_sym}"
            winner_symbols_str = ', '.join(winner_symbols[:5])
            if len(winner_symbols) > 5:
                winner_symbols_str += f", ... ({len(winner_symbols)} total)"
            reason += f" (winners: {winner_symbols_str})"
        
        # BOTH: strong CS but concentrated performance
        elif cs_auc >= T_cs and symbol_aucs and (symbol_auc_iqr > 0.15 or frac_symbols_good < T_frac):
            route = "BOTH"
            reason = f"cs_auc={cs_auc:.3f} >= {T_cs} BUT concentrated (IQR={symbol_auc_iqr:.3f}, frac_good={frac_symbols_good:.2f})"
        
        # Default: CROSS_SECTIONAL (fallback)
        else:
            route = "CROSS_SECTIONAL"
            if len(symbol_aucs) == 0:
                reason = f"default (cs_auc={cs_auc:.3f}, symbol_eval=0 symbols evaluable)"
            else:
                reason = f"default (cs_auc={cs_auc:.3f}, no strong symbol-specific signal)"
        
        # Get skip reasons for this target
        target_skip_reasons = {}
        if symbol_skip_reasons and target_name in symbol_skip_reasons:
            target_skip_reasons = symbol_skip_reasons[target_name]
        
        routing_decisions[target_name] = {
            'route': route,
            'reason': reason,
            'cs_auc': cs_auc,
            'symbol_auc_mean': symbol_auc_mean,
            'symbol_auc_median': symbol_auc_median,
            'symbol_auc_min': symbol_auc_min,
            'symbol_auc_max': symbol_auc_max,
            'symbol_auc_iqr': symbol_auc_iqr,
            'frac_symbols_good': frac_symbols_good,
            'winner_symbols': winner_symbols,
            'n_symbols_evaluated': len(symbol_aucs) if symbol_aucs else 0,
            'symbol_skip_reasons': target_skip_reasons if target_skip_reasons else None
        }
    
    return routing_decisions


def _save_dual_view_rankings(
    results_cs: List[Any],
    results_sym: Dict[str, Dict[str, Any]],
    results_loso: Dict[str, Dict[str, Any]],
    routing_decisions: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """
    Save dual-view ranking results and routing decisions.
    
    Target-first structure:
    - Global routing decisions → globals/routing_decisions.json (global summary)
    - Per-target routing decision → targets/<target>/decision/routing_decision.json (optional, for fast local inspection)
    
    Also maintains backward compatibility:
    - DECISION/TARGET_RANKING/routing_decisions.json (legacy location)
    - REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json (convenience copy)
    
    Args:
        output_dir: Base output directory (RESULTS/{run}/), not target_rankings subdirectory
    """
    import json
    from pathlib import Path
    from TRAINING.orchestration.utils.target_first_paths import (
        get_globals_dir, get_target_decision_dir, ensure_target_structure
    )
    
    # Determine base output directory (handle both old and new call patterns)
    if output_dir.name == "target_rankings":
        base_output_dir = output_dir.parent
    else:
        base_output_dir = output_dir
    
    # Ensure we have the actual run directory (where "targets" or "RESULTS" would be)
    # Walk up to find it if needed
    for _ in range(10):  # Limit depth
        if (base_output_dir / "targets").exists() or base_output_dir.name == "RESULTS":
            break
        if not base_output_dir.parent.exists() or base_output_dir.parent == base_output_dir:
            break
        base_output_dir = base_output_dir.parent
    
    # Prepare routing data
    routing_data = {
        'routing_decisions': routing_decisions,
        'summary': {
            'total_targets': len(routing_decisions),
            'cross_sectional_only': sum(1 for r in routing_decisions.values() if r.get('route') == 'CROSS_SECTIONAL'),
            'symbol_specific_only': sum(1 for r in routing_decisions.values() if r.get('route') == 'SYMBOL_SPECIFIC'),
            'both': sum(1 for r in routing_decisions.values() if r.get('route') == 'BOTH'),
            'blocked': sum(1 for r in routing_decisions.values() if r.get('route') == 'BLOCKED')
        }
    }
    
    # Save to globals/ (target-first primary location)
    globals_dir = get_globals_dir(base_output_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    globals_file = globals_dir / "routing_decisions.json"
    with open(globals_file, 'w') as f:
        json.dump(routing_data, f, indent=2, default=str)
    logger.info(f"Saved routing decisions to {globals_file}")
    
    # Save per-target slices for fast local inspection
    for target, decision in routing_decisions.items():
        try:
            ensure_target_structure(base_output_dir, target)
            target_decision_dir = get_target_decision_dir(base_output_dir, target)
            target_decision_file = target_decision_dir / "routing_decision.json"
            with open(target_decision_file, 'w') as f:
                json.dump({target: decision}, f, indent=2, default=str)
            logger.debug(f"Saved per-target routing decision to {target_decision_file}")
        except Exception as e:
            logger.debug(f"Failed to save per-target routing decision for {target}: {e}")
    
    # Target-first structure only - no legacy writes
    
    # Note: Individual view results are already saved by evaluate_target_predictability
    # via reproducibility tracker (with view/symbol metadata in RunContext)


def load_routing_decisions(routing_file: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load routing decisions from file.
    
    Tries multiple locations in order:
    1. Target-first structure: globals/routing_decisions.json
    2. Legacy structure: DECISION/TARGET_RANKING/routing_decisions.json
    3. Legacy structure: REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json
    4. Explicit routing_file path if provided
    
    Args:
        routing_file: Optional explicit path to routing_decisions.json
        output_dir: Optional base output directory (will search for routing decisions)
    
    Returns:
        Dict mapping target_name -> routing decision dict
    """
    import json
    
    # If explicit file provided, use it
    if routing_file and routing_file.exists():
        try:
            with open(routing_file, 'r') as f:
                data = json.load(f)
            return data.get('routing_decisions', {})
        except Exception as e:
            logger.error(f"Failed to load routing decisions from {routing_file}: {e}")
            return {}
    
    # If output_dir provided, search for routing decisions in new and legacy locations
    if output_dir:
        output_dir = Path(output_dir)
        
        # Try target-first structure first (globals/routing_decisions.json)
        globals_file = output_dir / "globals" / "routing_decisions.json"
        if globals_file.exists():
            try:
                with open(globals_file, 'r') as f:
                    data = json.load(f)
                logger.debug(f"Loaded routing decisions from target-first structure: {globals_file}")
                return data.get('routing_decisions', {})
            except Exception as e:
                logger.debug(f"Failed to load from globals: {e}")
        
        # Try legacy locations
        legacy_locations = [
            output_dir / "DECISION" / "TARGET_RANKING" / "routing_decisions.json",
            output_dir / "REPRODUCIBILITY" / "TARGET_RANKING" / "routing_decisions.json",
            output_dir / "target_rankings" / "REPRODUCIBILITY" / "TARGET_RANKING" / "routing_decisions.json"
        ]
        
        for legacy_file in legacy_locations:
            if legacy_file.exists():
                try:
                    with open(legacy_file, 'r') as f:
                        data = json.load(f)
                    logger.debug(f"Loaded routing decisions from legacy location: {legacy_file}")
                    return data.get('routing_decisions', {})
                except Exception as e:
                    logger.debug(f"Failed to load from {legacy_file}: {e}")
                    continue
    
    # If routing_file was provided but doesn't exist, warn
    if routing_file:
        logger.warning(f"Routing decisions file not found: {routing_file}")
    
    return {}
