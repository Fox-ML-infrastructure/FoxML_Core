# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
    results_sym: Dict[str, Dict[str, Any]],  # {target: {symbol: TargetPredictabilityScore}}
    results_loso: Dict[str, Dict[str, Any]],  # {target: {symbol: TargetPredictabilityScore}} (optional)
    symbol_skip_reasons: Dict[str, Dict[str, Dict[str, Any]]] = None  # {target: {symbol: {reason, status, ...}}}
) -> Dict[str, Dict[str, Any]]:
    """
    Compute routing decisions for each target based on dual-view scores.
    
    Uses skill01 (normalized [0,1] score) for unified routing across task types:
    - Regression: skill01 = 0.5 * (IC + 1.0) where IC ∈ [-1, 1]
    - Classification: skill01 = 0.5 * (AUC-excess + 1.0) where AUC-excess ∈ [-0.5, 0.5]
    
    Routing rules (using skill01 thresholds):
    - CROSS_SECTIONAL only: skill01 >= T_cs AND frac_symbols_good >= T_frac
    - SYMBOL_SPECIFIC only: skill01 < T_cs AND exists symbol with skill01 >= T_sym
    - BOTH: skill01 >= T_cs BUT performance is concentrated (high IQR / low frac_symbols_good)
    - BLOCKED: skill01 >= 0.90 (suspicious) UNLESS tstat > 3.0 (stable signal)
    
    Args:
        results_cs: Cross-sectional results (TargetPredictabilityScore objects)
        results_sym: Symbol-specific results by target
        results_loso: LOSO results by target (optional)
        symbol_skip_reasons: Skip reasons by target and symbol (optional)
    
    Returns:
        Dict mapping target -> routing decision dict (includes skill01_cs, skill01_sym_mean)
    """
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        routing_cfg = get_cfg("target_ranking.routing", default={}, config_name="target_ranking_config")
        T_cs = float(routing_cfg.get('auc_threshold', 0.65))
        T_frac = float(routing_cfg.get('frac_symbols_good_threshold', 0.5))
        T_sym = float(routing_cfg.get('symbol_auc_threshold', 0.60))
        T_suspicious_cs = float(routing_cfg.get('suspicious_auc', 0.90))
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
    all_targets = set()
    for result_cs in results_cs:
        all_targets.add(result_cs.target)
    for target in results_sym.keys():
        all_targets.add(target)
    
    # Process each target (whether it has CS results or not)
    for target in all_targets:
        # Find cross-sectional result if it exists
        result_cs = None
        skill01_cs = 0.0  # Default to failed score (skill01 is [0,1], so 0.0 = failed)
        auc = -999.0  # Keep for backward compatibility (deprecated for routing)
        for r in results_cs:
            if r.target == target:
                result_cs = r
                # Use skill01 for routing (normalized [0,1] score works for both regression and classification)
                skill01_cs = r.skill01 if hasattr(r, 'skill01') and r.skill01 is not None else 0.0
                auc = r.auc  # Keep for backward compatibility
                break
        
        # Get symbol-specific results for this target
        sym_results = results_sym.get(target, {})
        
        # Compute symbol distribution stats using skill01
        symbol_skill01s = []
        if sym_results:
            for symbol, result_sym in sym_results.items():
                skill01_val = result_sym.skill01 if hasattr(result_sym, 'skill01') and result_sym.skill01 is not None else None
                if skill01_val is not None and skill01_val > 0.0:  # Valid result (skill01 > 0)
                    symbol_skill01s.append(skill01_val)
        
        if symbol_skill01s:
            symbol_skill01_mean = np.mean(symbol_skill01s)
            symbol_skill01_median = np.median(symbol_skill01s)
            symbol_skill01_min = np.min(symbol_skill01s)
            symbol_skill01_max = np.max(symbol_skill01s)
            symbol_skill01_iqr = np.percentile(symbol_skill01s, 75) - np.percentile(symbol_skill01s, 25)
            frac_symbols_good = sum(1 for s01 in symbol_skill01s if s01 >= T_sym) / len(symbol_skill01s)
            winner_symbols = [sym for sym, result in sym_results.items() 
                            if hasattr(result, 'skill01') and result.skill01 is not None and result.skill01 >= T_sym]
        else:
            symbol_skill01_mean = None
            symbol_skill01_median = None
            symbol_skill01_min = None
            symbol_skill01_max = None
            symbol_skill01_iqr = None
            frac_symbols_good = 0.0
            winner_symbols = []
        
        # Initialize route to None (will be set by conditions below)
        route = None
        reason = None
        
        # Initialize route to None (will be set by conditions below)
        route = None
        reason = None
        
        # Handle case where CS failed (skill01 = 0.0 or result_cs is None)
        if result_cs is None or skill01_cs <= 0.0:
            # CS failed - check if symbol-specific works
            if symbol_skill01s and max(symbol_skill01s) >= T_sym:
                route = "SYMBOL_SPECIFIC"
                reason = f"cs_failed (skill01={skill01_cs:.3f}) BUT exists symbol with skill01 >= {T_sym}"
                winner_symbols_str = ', '.join(winner_symbols[:5])
                if len(winner_symbols) > 5:
                    winner_symbols_str += f", ... ({len(winner_symbols)} total)"
                reason += f" (winners: {winner_symbols_str})"
            else:
                # CS failed and no good symbol-specific results
                route = "BLOCKED"
                if symbol_skill01s:
                    reason = f"cs_failed AND max_symbol_skill01={max(symbol_skill01s):.3f} < {T_sym} (no viable route)"
                else:
                    reason = f"cs_failed AND no symbol-specific results (no viable route)"
        
        # Check for suspicious scores (BLOCKED) - task-aware: high skill01 + low tstat = suspicious
        if route is None and (skill01_cs >= T_suspicious_cs or (symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym)):
            # Additional check: if tstat available, verify signal stability
            is_suspicious = True
            if result_cs and hasattr(result_cs, 'primary_metric_tstat'):
                tstat = result_cs.primary_metric_tstat
                if tstat is not None and tstat > 3.0:  # Strong, stable signal
                    # High skill01 + high tstat = legitimate strong signal
                    is_suspicious = False
                    logger.debug(f"High skill01 ({skill01_cs:.3f}) but stable (tstat={tstat:.2f}), not blocking")
            
            if is_suspicious:
                route = "BLOCKED"
                if skill01_cs >= T_suspicious_cs:
                    reason = f"skill01={skill01_cs:.3f} >= {T_suspicious_cs} (suspicious high score)"
                elif symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym:
                    reason = f"max_symbol_skill01={max(symbol_skill01s):.3f} >= {T_suspicious_sym} (suspicious high score)"
                else:
                    reason = f"skill01={skill01_cs:.3f} or symbol_skill01 >= suspicious threshold"
        
        # CROSS_SECTIONAL only: strong CS performance + good symbol coverage
        if route is None and skill01_cs >= T_cs and frac_symbols_good >= T_frac:
            route = "CROSS_SECTIONAL"
            reason = f"skill01={skill01_cs:.3f} >= {T_cs} AND frac_symbols_good={frac_symbols_good:.2f} >= {T_frac}"
        
        # SYMBOL_SPECIFIC only: weak CS but some symbols work
        if route is None and skill01_cs < T_cs and symbol_skill01s and max(symbol_skill01s) >= T_sym:
            route = "SYMBOL_SPECIFIC"
            reason = f"skill01={skill01_cs:.3f} < {T_cs} BUT exists symbol with skill01 >= {T_sym}"
            winner_symbols_str = ', '.join(winner_symbols[:5])
            if len(winner_symbols) > 5:
                winner_symbols_str += f", ... ({len(winner_symbols)} total)"
            reason += f" (winners: {winner_symbols_str})"
        
        # BOTH: strong CS but concentrated performance
        if route is None and skill01_cs >= T_cs and symbol_skill01s and symbol_skill01_iqr is not None and (symbol_skill01_iqr > 0.15 or frac_symbols_good < T_frac):
            route = "BOTH"
            reason = f"skill01={skill01_cs:.3f} >= {T_cs} BUT concentrated (IQR={symbol_skill01_iqr:.3f}, frac_good={frac_symbols_good:.2f})"
        
        # Default: CROSS_SECTIONAL (fallback)
        if route is None:
            route = "CROSS_SECTIONAL"
            if len(symbol_skill01s) == 0:
                reason = f"default (skill01={skill01_cs:.3f}, symbol_eval=0 symbols evaluable)"
            else:
                reason = f"default (skill01={skill01_cs:.3f}, no strong symbol-specific signal)"
        
        # Get skip reasons for this target
        target_skip_reasons = {}
        if symbol_skip_reasons and target in symbol_skip_reasons:
            target_skip_reasons = symbol_skip_reasons[target]
        
        routing_decisions[target] = {
            'route': route,
            'reason': reason,
            'skill01_cs': skill01_cs,  # New: normalized skill score for routing
            'skill01_sym_mean': symbol_skill01_mean,  # New: mean symbol skill01
            'auc': auc,  # Deprecated: kept for backward compatibility (R² for regression, AUC for classification)
            'symbol_auc_mean': symbol_skill01_mean,  # Deprecated: now contains skill01_mean, kept for backward compat
            'symbol_auc_median': symbol_skill01_median,  # Deprecated: now contains skill01_median
            'symbol_auc_min': symbol_skill01_min,  # Deprecated: now contains skill01_min
            'symbol_auc_max': symbol_skill01_max,  # Deprecated: now contains skill01_max
            'symbol_auc_iqr': symbol_skill01_iqr,  # Deprecated: now contains skill01_iqr
            'frac_symbols_good': frac_symbols_good,
            'winner_symbols': winner_symbols,
            'n_symbols_evaluated': len(symbol_skill01s) if symbol_skill01s else 0,
            'symbol_skip_reasons': target_skip_reasons if target_skip_reasons else None
        }
    
    return routing_decisions


def _compute_single_target_routing_decision(
    target: str,
    result_cs: Optional[Any],  # TargetPredictabilityScore or None
    sym_results: Dict[str, Any],  # {symbol: TargetPredictabilityScore}
    symbol_skip_reasons: Optional[Dict[str, Dict[str, Any]]] = None  # {symbol: {reason, status, ...}}
) -> Dict[str, Any]:
    """
    Compute routing decision for a single target.
    
    This is a single-target version of _compute_target_routing_decisions for incremental saving.
    Uses skill01 (normalized [0,1] score) for unified routing across task types.
    
    Args:
        target: Target name
        result_cs: Cross-sectional result (TargetPredictabilityScore or None if failed)
        sym_results: Symbol-specific results for this target
        symbol_skip_reasons: Skip reasons for symbols (optional)
    
    Returns:
        Routing decision dict for this target (includes skill01_cs, skill01_sym_mean)
    """
    # Load thresholds from config (support both new skill01_threshold and legacy auc_threshold)
    try:
        from CONFIG.config_loader import get_cfg
        routing_cfg = get_cfg("target_ranking.routing", default={}, config_name="target_ranking_config")
        # New unified skill01 thresholds (works for both regression IC and classification AUC)
        T_cs = float(routing_cfg.get('skill01_threshold', routing_cfg.get('auc_threshold', 0.65)))
        T_frac = float(routing_cfg.get('frac_symbols_good_threshold', 0.5))
        T_sym = float(routing_cfg.get('symbol_skill01_threshold', routing_cfg.get('symbol_auc_threshold', 0.60)))
        T_suspicious_cs = float(routing_cfg.get('suspicious_skill01', routing_cfg.get('suspicious_auc', 0.90)))
        T_suspicious_sym = float(routing_cfg.get('suspicious_symbol_skill01', routing_cfg.get('suspicious_symbol_auc', 0.95)))
    except Exception:
        # Fallback defaults
        T_cs = 0.65
        T_frac = 0.5
        T_sym = 0.60
        T_suspicious_cs = 0.90
        T_suspicious_sym = 0.95
    
    # Get CS skill01 (normalized [0,1] score for unified routing)
    skill01_cs = 0.0  # Default to failed score
    auc = -999.0  # Keep for backward compatibility (deprecated for routing)
    if result_cs:
        skill01_cs = result_cs.skill01 if hasattr(result_cs, 'skill01') and result_cs.skill01 is not None else 0.0
        auc = result_cs.auc  # Keep for backward compatibility
    
    # Compute symbol distribution stats using skill01
    symbol_skill01s = []
    if sym_results:
        for symbol, result_sym in sym_results.items():
            skill01_val = result_sym.skill01 if hasattr(result_sym, 'skill01') and result_sym.skill01 is not None else None
            if skill01_val is not None and skill01_val > 0.0:  # Valid result (skill01 > 0)
                symbol_skill01s.append(skill01_val)
    
    if symbol_skill01s:
        symbol_skill01_mean = np.mean(symbol_skill01s)
        symbol_skill01_median = np.median(symbol_skill01s)
        symbol_skill01_min = np.min(symbol_skill01s)
        symbol_skill01_max = np.max(symbol_skill01s)
        symbol_skill01_iqr = np.percentile(symbol_skill01s, 75) - np.percentile(symbol_skill01s, 25)
        frac_symbols_good = sum(1 for s01 in symbol_skill01s if s01 >= T_sym) / len(symbol_skill01s)
        winner_symbols = [sym for sym, result in sym_results.items() 
                        if hasattr(result, 'skill01') and result.skill01 is not None and result.skill01 >= T_sym]
    else:
        symbol_skill01_mean = None
        symbol_skill01_median = None
        symbol_skill01_min = None
        symbol_skill01_max = None
        symbol_skill01_iqr = None
        frac_symbols_good = 0.0
        winner_symbols = []
    
    # Initialize route to None (will be set by conditions below)
    route = None
    reason = None
    
    # Handle case where CS failed (skill01 = 0.0 or result_cs is None)
    if result_cs is None or skill01_cs <= 0.0:
        # CS failed - check if symbol-specific works
        if symbol_skill01s and max(symbol_skill01s) >= T_sym:
            route = "SYMBOL_SPECIFIC"
            reason = f"cs_failed (skill01={skill01_cs:.3f}) BUT exists symbol with skill01 >= {T_sym}"
            winner_symbols_str = ', '.join(winner_symbols[:5])
            if len(winner_symbols) > 5:
                winner_symbols_str += f", ... ({len(winner_symbols)} total)"
            reason += f" (winners: {winner_symbols_str})"
        else:
            # CS failed and no good symbol-specific results
            route = "BLOCKED"
            if symbol_skill01s:
                reason = f"cs_failed AND max_symbol_skill01={max(symbol_skill01s):.3f} < {T_sym} (no viable route)"
            else:
                reason = f"cs_failed AND no symbol-specific results (no viable route)"
    
    # Check for suspicious scores (BLOCKED) - task-aware: high skill01 + low tstat = suspicious
    if route is None and (skill01_cs >= T_suspicious_cs or (symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym)):
        # Additional check: if tstat available, verify signal stability
        is_suspicious = True
        if result_cs and hasattr(result_cs, 'primary_metric_tstat'):
            tstat = result_cs.primary_metric_tstat
            if tstat is not None and tstat > 3.0:  # Strong, stable signal
                # High skill01 + high tstat = legitimate strong signal
                is_suspicious = False
                logger.debug(f"High skill01 ({skill01_cs:.3f}) but stable (tstat={tstat:.2f}), not blocking")
        
        if is_suspicious:
            route = "BLOCKED"
            if skill01_cs >= T_suspicious_cs:
                reason = f"skill01={skill01_cs:.3f} >= {T_suspicious_cs} (suspicious high score)"
            elif symbol_skill01s and max(symbol_skill01s) >= T_suspicious_sym:
                reason = f"max_symbol_skill01={max(symbol_skill01s):.3f} >= {T_suspicious_sym} (suspicious high score)"
            else:
                reason = f"skill01={skill01_cs:.3f} or symbol_skill01 >= suspicious threshold"
    
    # CROSS_SECTIONAL only: strong CS performance + good symbol coverage
    if route is None and skill01_cs >= T_cs and frac_symbols_good >= T_frac:
        route = "CROSS_SECTIONAL"
        reason = f"skill01={skill01_cs:.3f} >= {T_cs} AND frac_symbols_good={frac_symbols_good:.2f} >= {T_frac}"
    
    # SYMBOL_SPECIFIC only: weak CS but some symbols work
    if route is None and skill01_cs < T_cs and symbol_skill01s and max(symbol_skill01s) >= T_sym:
        route = "SYMBOL_SPECIFIC"
        reason = f"skill01={skill01_cs:.3f} < {T_cs} BUT exists symbol with skill01 >= {T_sym}"
        winner_symbols_str = ', '.join(winner_symbols[:5])
        if len(winner_symbols) > 5:
            winner_symbols_str += f", ... ({len(winner_symbols)} total)"
        reason += f" (winners: {winner_symbols_str})"
    
    # BOTH: strong CS but concentrated performance
    if route is None and skill01_cs >= T_cs and symbol_skill01s and symbol_skill01_iqr is not None and (symbol_skill01_iqr > 0.15 or frac_symbols_good < T_frac):
        route = "BOTH"
        reason = f"skill01={skill01_cs:.3f} >= {T_cs} BUT concentrated (IQR={symbol_skill01_iqr:.3f}, frac_good={frac_symbols_good:.2f})"
    
    # Default: CROSS_SECTIONAL (fallback)
    if route is None:
        route = "CROSS_SECTIONAL"
        if len(symbol_skill01s) == 0:
            reason = f"default (skill01={skill01_cs:.3f}, symbol_eval=0 symbols evaluable)"
        else:
            reason = f"default (skill01={skill01_cs:.3f}, no strong symbol-specific signal)"
    
    # Get skip reasons for this target
    target_skip_reasons = symbol_skip_reasons if symbol_skip_reasons else {}
    
    return {
        'route': route,
        'reason': reason,
        'skill01_cs': skill01_cs,  # New: normalized skill score for routing
        'skill01_sym_mean': symbol_skill01_mean,  # New: mean symbol skill01
        'auc': auc,  # Deprecated: kept for backward compatibility (R² for regression, AUC for classification)
        'symbol_auc_mean': symbol_skill01_mean,  # Deprecated: now contains skill01_mean, kept for backward compat
        'symbol_auc_median': symbol_skill01_median,  # Deprecated: now contains skill01_median
        'symbol_auc_min': symbol_skill01_min,  # Deprecated: now contains skill01_min
        'symbol_auc_max': symbol_skill01_max,  # Deprecated: now contains skill01_max
        'symbol_auc_iqr': symbol_skill01_iqr,  # Deprecated: now contains skill01_iqr
        'frac_symbols_good': frac_symbols_good,
        'winner_symbols': winner_symbols,
        'n_symbols_evaluated': len(symbol_skill01s) if symbol_skill01s else 0,
        'symbol_skip_reasons': target_skip_reasons if target_skip_reasons else None
    }


def _save_single_target_decision(
    target: str,
    decision: Dict[str, Any],
    output_dir: Optional[Path]
) -> None:
    """
    Save routing decision for a single target immediately after evaluation.
    
    This allows incremental saving so decisions are available as soon as each target completes.
    
    Args:
        target: Target name
        decision: Routing decision dict for this target
        output_dir: Base output directory (RESULTS/{run}/) - can be None, will try to infer
    """
    import json
    from TRAINING.orchestration.utils.target_first_paths import (
        get_target_decision_dir, ensure_target_structure
    )
    
    if output_dir is None:
        logger.warning(f"⚠️  Cannot save routing decision for {target}: output_dir is None")
        return
    
    # Determine base output directory
    if output_dir.name == "target_rankings":
        base_output_dir = output_dir.parent
    else:
        base_output_dir = output_dir
    
    # Walk up to find run directory using SST helper
    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
    base_output_dir = get_run_root(base_output_dir)
    
    try:
        ensure_target_structure(base_output_dir, target)
        target_decision_dir = get_target_decision_dir(base_output_dir, target)
        target_decision_file = target_decision_dir / "routing_decision.json"
        with open(target_decision_file, 'w') as f:
            json.dump({target: decision}, f, indent=2, default=str)
        logger.debug(f"✅ Saved routing decision for {target} to {target_decision_file}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to save routing decision for {target}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")


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
    
    Reading logic maintains backward compatibility (reads from legacy locations if needed):
    - DECISION/TARGET_RANKING/routing_decisions.json (legacy location)
    - REPRODUCIBILITY/TARGET_RANKING/routing_decisions.json (legacy location)
    
    Args:
        output_dir: Base output directory (RESULTS/{run}/), not target_rankings subdirectory
    """
    import json
    # Path is already imported globally at line 13
    from TRAINING.orchestration.utils.target_first_paths import (
        get_globals_dir, get_target_decision_dir, ensure_target_structure
    )
    
    # Determine base output directory (handle both old and new call patterns)
    if output_dir.name == "target_rankings":
        base_output_dir = output_dir.parent
    else:
        base_output_dir = output_dir
    
    # Ensure we have the actual run directory using SST helper
    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
    base_output_dir = get_run_root(base_output_dir)
    
    # Compute fingerprint for routing decisions to prevent stale data reuse
    # Fingerprint includes: target set, symbol set, and config hash
    import hashlib
    target_set = sorted(set(routing_decisions.keys()))
    # Extract symbols from routing decisions (if available)
    symbols_set = set()
    for decision in routing_decisions.values():
        if 'symbols' in decision and isinstance(decision['symbols'], dict):
            symbols_set.update(decision['symbols'].keys())
    symbols_list = sorted(symbols_set) if symbols_set else []
    
    # Load view from run context (SST) for fingerprint
    run_view = None
    try:
        from TRAINING.orchestration.utils.run_context import load_run_context
        context = load_run_context(output_dir)
        if context:
            run_view = context.get("view")
    except Exception:
        pass
    
    # Create fingerprint from targets, symbols, and view
    fingerprint_data = {
        'targets': target_set,
        'symbols': symbols_list,
        'target_count': len(target_set),
        'symbol_count': len(symbols_list),
        'view': run_view  # Include view in fingerprint (SST)
    }
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    # Prepare routing data
    routing_data = {
        'routing_decisions': routing_decisions,
        'fingerprint': fingerprint_hash,
        'fingerprint_data': fingerprint_data,  # Store for debugging
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
    # SST: Sanitize routing data to normalize enums to strings before JSON serialization
    from enum import Enum
    import pandas as pd
    def _sanitize_for_json(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_sanitize_for_json(v) for v in obj]
        else:
            return obj
    sanitized_routing_data = _sanitize_for_json(routing_data)
    
    globals_file = globals_dir / "routing_decisions.json"
    with open(globals_file, 'w') as f:
        json.dump(sanitized_routing_data, f, indent=2, default=str)
    logger.info(f"Saved routing decisions to {globals_file}")
    
    # Save per-target slices for fast local inspection
    # CRITICAL: This ensures ALL targets in routing_decisions get decision files,
    # even if incremental save failed earlier. This is a safety net.
    for target, decision in routing_decisions.items():
        try:
            ensure_target_structure(base_output_dir, target)
            target_decision_dir = get_target_decision_dir(base_output_dir, target)
            target_decision_file = target_decision_dir / "routing_decision.json"
            with open(target_decision_file, 'w') as f:
                json.dump({target: decision}, f, indent=2, default=str)
            logger.debug(f"Saved per-target routing decision to {target_decision_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save per-target routing decision for {target}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Target-first structure only - no legacy writes
    
    # Note: Individual view results are already saved by evaluate_target_predictability
    # via reproducibility tracker (with view/symbol metadata in RunContext)


def load_routing_decisions(
    routing_file: Optional[Path] = None, 
    output_dir: Optional[Path] = None,
    expected_targets: Optional[List[str]] = None,
    validate_fingerprint: bool = True
) -> Dict[str, Dict[str, Any]]:
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
        expected_targets: Optional list of expected targets (for fingerprint validation)
        validate_fingerprint: If True, validate fingerprint matches expected targets (default: True)
    
    Returns:
        Dict mapping target -> routing decision dict
    """
    import json
    import hashlib
    
    # If explicit file provided, use it
    if routing_file and routing_file.exists():
        try:
            with open(routing_file, 'r') as f:
                data = json.load(f)
            routing_decisions = data.get('routing_decisions', {})
            
            # Validate fingerprint if expected_targets provided
            if validate_fingerprint and expected_targets:
                stored_fingerprint = data.get('fingerprint')
                if stored_fingerprint:
                    # Load view from run context (SST) for fingerprint validation
                    run_view = None
                    try:
                        from TRAINING.orchestration.utils.run_context import load_run_context
                        context = load_run_context(output_dir)
                        if context:
                            run_view = context.get("view")
                    except Exception:
                        pass
                    
                    # Compute expected fingerprint (include view)
                    target_set = sorted(set(expected_targets))
                    fingerprint_data = {
                        'targets': target_set,
                        'target_count': len(target_set),
                        'view': run_view  # Include view in fingerprint (SST)
                    }
                    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
                    expected_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
                    
                    if stored_fingerprint != expected_fingerprint:
                        # Check dev_mode to decide whether to raise or return empty
                        dev_mode = False
                        try:
                            from CONFIG.config_loader import get_cfg
                            routing_config = get_cfg("training_config.routing", default={}, config_name="training_config")
                            dev_mode = routing_config.get("dev_mode", False)
                        except Exception:
                            pass
                        
                        error_msg = (
                            f"🚨 Routing decisions fingerprint mismatch: stored={stored_fingerprint[:8]}... "
                            f"expected={expected_fingerprint[:8]}... "
                            f"This indicates stale routing decisions. "
                            f"Re-run feature selection to generate fresh routing decisions."
                        )
                        
                        if dev_mode:
                            logger.warning(f"{error_msg} Dev mode: Ignoring stale decisions, attempting regeneration...")
                            # Try to regenerate from current candidates
                            try:
                                # Check if fresh candidates exist
                                from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                                globals_dir = get_globals_dir(output_dir) if output_dir else None
                                if globals_dir:
                                    candidates_file = globals_dir / "routing" / "routing_candidates.parquet"
                                    if candidates_file.exists():
                                        logger.info("Fresh routing candidates found, but regeneration not yet implemented. Returning empty decisions.")
                                        # TODO: Implement regeneration from candidates
                                        return {"_STALE_DECISIONS_IGNORED": True, "_regenerate": True}
                            except Exception as e:
                                logger.warning(f"Failed to check for fresh candidates: {e}")
                            # Return marker indicating stale decisions were ignored
                            return {"_STALE_DECISIONS_IGNORED": True}
                        else:
                            logger.error(error_msg)
                            raise ValueError(
                                "Stale routing decisions detected. Re-run feature selection to generate fresh decisions. "
                                f"Fingerprint mismatch: stored={stored_fingerprint[:8]}... expected={expected_fingerprint[:8]}..."
                            )
                else:
                    logger.debug("Routing decisions file has no fingerprint - skipping validation")
            
            return routing_decisions
        except Exception as e:
            logger.error(f"Failed to load routing decisions from {routing_file}: {e}")
            return {}
    
    # If output_dir provided, enforce single known path (globals/routing_decisions.json)
    if output_dir:
        output_dir = Path(output_dir)
        
        # PRIMARY: globals/routing_decisions.json (current run only)
        globals_file = output_dir / "globals" / "routing_decisions.json"
        if globals_file.exists():
            try:
                with open(globals_file, 'r') as f:
                    data = json.load(f)
                routing_decisions = data.get('routing_decisions', {})
                
                # Validate fingerprint if expected_targets provided
                if validate_fingerprint and expected_targets:
                    stored_fingerprint = data.get('fingerprint')
                    if stored_fingerprint:
                        # Compute expected fingerprint
                        # Load view from run context (SST) for fingerprint validation
                        run_view = None
                        try:
                            from TRAINING.orchestration.utils.run_context import load_run_context
                            context = load_run_context(output_dir)
                            if context:
                                run_view = context.get("view")
                        except Exception:
                            pass
                        
                        target_set = sorted(set(expected_targets))
                        fingerprint_data = {
                            'targets': target_set,
                            'target_count': len(target_set),
                            'view': run_view  # Include view in fingerprint (SST)
                        }
                        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
                        expected_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
                        
                        if stored_fingerprint != expected_fingerprint:
                            # Check dev_mode to decide whether to raise or return empty
                            dev_mode = False
                            try:
                                from CONFIG.config_loader import get_cfg
                                routing_config = get_cfg("training_config.routing", default={}, config_name="training_config")
                                dev_mode = routing_config.get("dev_mode", False)
                            except Exception:
                                pass
                            
                            error_msg = (
                                f"🚨 Routing decisions fingerprint mismatch: stored={stored_fingerprint[:8]}... "
                                f"expected={expected_fingerprint[:8]}... "
                                f"Loaded {len(routing_decisions)} decisions but fingerprint doesn't match expected targets. "
                                f"This indicates stale routing decisions. "
                                f"Re-run feature selection to generate fresh routing decisions."
                            )
                            
                            if dev_mode:
                                logger.warning(f"{error_msg} Dev mode: Ignoring stale decisions, attempting regeneration...")
                                # Try to regenerate from current candidates
                                try:
                                    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                                    globals_dir = get_globals_dir(output_dir) if output_dir else None
                                    if globals_dir:
                                        candidates_file = globals_dir / "routing" / "routing_candidates.parquet"
                                        if candidates_file.exists():
                                            logger.info("Fresh routing candidates found, but regeneration not yet implemented. Returning empty decisions.")
                                            # TODO: Implement regeneration from candidates
                                            return {"_STALE_DECISIONS_IGNORED": True, "_regenerate": True}
                                except Exception as e:
                                    logger.warning(f"Failed to check for fresh candidates: {e}")
                                # Return marker indicating stale decisions were ignored
                                return {"_STALE_DECISIONS_IGNORED": True}
                            else:
                                logger.error(error_msg)
                                raise ValueError(
                                    "Stale routing decisions detected. Re-run feature selection to generate fresh decisions. "
                                    f"Fingerprint mismatch: stored={stored_fingerprint[:8]}... expected={expected_fingerprint[:8]}..."
                                )
                    else:
                        logger.debug("Routing decisions file has no fingerprint - skipping validation")
                
                logger.debug(f"Loaded routing decisions from target-first structure: {globals_file}")
                return routing_decisions
            except Exception as e:
                logger.debug(f"Failed to load from globals: {e}")
        
        # If not found, fail loudly (no legacy fallback to prevent stale decisions)
        if validate_fingerprint and expected_targets:
            raise FileNotFoundError(
                f"Routing decisions not found for current run at {globals_file}. "
                f"Expected targets: {expected_targets}. "
                f"Re-run feature selection to generate fresh decisions."
            )
        else:
            logger.warning(f"Routing decisions not found at {globals_file}. Returning empty dict.")
            return {}
    
    # If routing_file was provided but doesn't exist, warn
    if routing_file:
        logger.warning(f"Routing decisions file not found: {routing_file}")
    
    return {}
