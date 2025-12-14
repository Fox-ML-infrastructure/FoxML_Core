"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Shared Lookback Cap Enforcement

Single function for applying lookback cap enforcement that can be used by both
ranking and feature selection. Ensures consistent behavior and prevents split-brain.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LookbackCapResult:
    """Result of lookback cap enforcement."""
    safe_features: List[str]  # Features that passed the cap
    quarantined_features: List[str]  # Features that exceeded cap (quarantined)
    budget: Any  # LeakageBudget object
    canonical_map: Dict[str, float]  # Canonical lookback map (single source of truth)
    fingerprint: str  # Feature set fingerprint
    actual_max_lookback: float  # Actual max lookback from safe features
    quarantine_count: int  # Number of features quarantined


def apply_lookback_cap(
    features: List[str],
    interval_minutes: float,
    cap_minutes: Optional[float],
    policy: str = "strict",
    stage: str = "unknown",
    canonical_map: Optional[Dict[str, float]] = None,
    registry: Optional[Any] = None,
    feature_time_meta_map: Optional[Dict[str, Any]] = None,
    base_interval_minutes: Optional[float] = None,
    log_mode: str = "summary"  # "summary" or "debug"
) -> LookbackCapResult:
    """
    Apply lookback cap enforcement to a feature set.
    
    This is the single source of truth for lookback cap enforcement, used by both
    ranking and feature selection to ensure consistent behavior.
    
    Pipeline:
    1. Build canonical lookback map (or use provided)
    2. Quarantine features exceeding cap
    3. Compute budget from safe features
    4. Validate invariants (hard-fail in strict mode)
    5. Return safe features + metadata
    
    Args:
        features: List of feature names to enforce cap on
        interval_minutes: Data interval in minutes
        cap_minutes: Lookback cap in minutes (None = no cap)
        policy: Enforcement policy ("strict", "drop", "warn")
        stage: Stage name for logging (e.g., "FS_PRE", "FS_POST", "GATEKEEPER")
        canonical_map: Optional pre-computed canonical map (if None, will compute)
        registry: Optional feature registry
        feature_time_meta_map: Optional map of feature_name -> FeatureTimeMeta
        base_interval_minutes: Optional base training grid interval
        log_mode: Logging mode ("summary" for one-liners, "debug" for per-feature traces)
    
    Returns:
        LookbackCapResult with safe_features, quarantined_features, budget, canonical_map, etc.
    
    Raises:
        RuntimeError: In strict mode if cap violation detected or invariants fail
    """
    from TRAINING.utils.leakage_budget import compute_feature_lookback_max, compute_budget, _feat_key
    from TRAINING.utils.cross_sectional_data import _compute_feature_fingerprint
    
    if not features:
        # Empty feature set - return empty result
        set_fp, _ = _compute_feature_fingerprint([], set_invariant=True)
        from TRAINING.utils.leakage_budget import LeakageBudget
        empty_budget = LeakageBudget(
            interval_minutes=interval_minutes,
            horizon_minutes=60.0,  # Default
            max_feature_lookback_minutes=0.0,
            cap_max_lookback_minutes=cap_minutes,
            allowed_max_lookback_minutes=None
        )
        return LookbackCapResult(
            safe_features=[],
            quarantined_features=[],
            budget=empty_budget,
            canonical_map={},
            fingerprint=set_fp,
            actual_max_lookback=0.0,
            quarantine_count=0
        )
    
    # Step 1: Build canonical lookback map (single source of truth)
    # If provided, use it; otherwise compute it
    # CRITICAL: Don't pass cap_minutes to compute_feature_lookback_max - we enforce the cap ourselves
    # This allows us to quarantine features without raising (policy-dependent)
    if canonical_map is None:
        lookback_result = compute_feature_lookback_max(
            features,
            interval_minutes=interval_minutes,
            max_lookback_cap_minutes=None,  # Don't enforce cap here - we do it below
            registry=registry,
            stage=stage,
            feature_time_meta_map=feature_time_meta_map,
            base_interval_minutes=base_interval_minutes
        )
        canonical_map = lookback_result.canonical_lookback_map if hasattr(lookback_result, 'canonical_lookback_map') else {}
        initial_max = lookback_result.max_minutes if hasattr(lookback_result, 'max_minutes') else None
    else:
        # Use provided canonical map (already computed)
        initial_max = None
    
    # Step 2: Quarantine features exceeding cap
    safe_features = []
    quarantined_features = []
    
    if cap_minutes is not None:
        for feat_name in features:
            feat_key = _feat_key(feat_name)
            lookback = canonical_map.get(feat_key)
            
            if lookback is None:
                # Missing from canonical map - treat as unsafe (inf)
                if policy == "strict":
                    logger.error(
                        f"🚨 {stage}: Feature '{feat_name}' missing from canonical map. "
                        f"This indicates a bug in lookback computation."
                    )
                    if policy == "strict":
                        raise RuntimeError(
                            f"Feature '{feat_name}' missing from canonical map in {stage}. "
                            f"This indicates a bug in lookback computation."
                        )
                quarantined_features.append(feat_name)
            elif lookback == float("inf"):
                # Unknown lookback - treat as unsafe
                if log_mode == "debug":
                    logger.debug(f"   {stage}: {feat_name} → unknown lookback (quarantined)")
                quarantined_features.append(feat_name)
            elif lookback > cap_minutes:
                # Exceeds cap - quarantine
                if log_mode == "debug":
                    logger.debug(f"   {stage}: {feat_name} → {lookback:.1f}m > cap={cap_minutes:.1f}m (quarantined)")
                quarantined_features.append(feat_name)
            else:
                # Safe - keep
                safe_features.append(feat_name)
    else:
        # No cap - all features are safe (but still check for unknown)
        for feat_name in features:
            feat_key = _feat_key(feat_name)
            lookback = canonical_map.get(feat_key)
            
            if lookback is None or lookback == float("inf"):
                # Unknown lookback - still quarantine in strict mode
                if policy == "strict":
                    quarantined_features.append(feat_name)
                else:
                    safe_features.append(feat_name)
            else:
                safe_features.append(feat_name)
    
    # Step 3: Compute budget from safe features
    if safe_features:
        budget, budget_fp, _ = compute_budget(
            safe_features,
            interval_minutes,
            60.0,  # Default horizon
            registry=registry,
            max_lookback_cap_minutes=cap_minutes,
            stage=f"{stage}_budget",
            canonical_lookback_map=canonical_map,  # Use same canonical map
            feature_time_meta_map=feature_time_meta_map,
            base_interval_minutes=base_interval_minutes
        )
        actual_max_lookback = budget.max_feature_lookback_minutes if budget.max_feature_lookback_minutes is not None else 0.0
    else:
        # No safe features - create empty budget
        from TRAINING.utils.leakage_budget import LeakageBudget
        budget = LeakageBudget(
            interval_minutes=interval_minutes,
            horizon_minutes=60.0,
            max_feature_lookback_minutes=0.0,
            cap_max_lookback_minutes=cap_minutes,
            allowed_max_lookback_minutes=None
        )
        budget_fp, _ = _compute_feature_fingerprint([], set_invariant=True)
        actual_max_lookback = 0.0
    
    # Step 4: Validate invariants (hard-fail in strict mode)
    # Invariant 1: actual_max <= cap (if cap is set)
    if cap_minutes is not None and actual_max_lookback > cap_minutes:
        error_msg = (
            f"🚨 CAP VIOLATION ({stage}): actual_max={actual_max_lookback:.1f}m > cap={cap_minutes:.1f}m. "
            f"{len(quarantined_features)} features quarantined, but {len(safe_features)} safe features still exceed cap. "
            f"This indicates a bug in quarantine logic."
        )
        logger.error(error_msg)
        
        if policy == "strict":
            raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
    
    # Invariant 2: Oracle consistency (budget.max == actual_max from canonical map)
    if safe_features:
        # Compute max from canonical map directly
        max_from_map = 0.0
        for feat_name in safe_features:
            feat_key = _feat_key(feat_name)
            lookback = canonical_map.get(feat_key)
            if lookback is not None and lookback != float("inf"):
                max_from_map = max(max_from_map, lookback)
        
        if abs(max_from_map - actual_max_lookback) > 1.0:
            error_msg = (
                f"🚨 INVARIANT VIOLATION ({stage}): "
                f"max(canonical_map[safe_features])={max_from_map:.1f}m != "
                f"budget.max={actual_max_lookback:.1f}m. "
                f"This indicates canonical map inconsistency."
            )
            logger.error(error_msg)
            
            if policy == "strict":
                raise RuntimeError(error_msg)
    
    # Step 5: Log summary (one-liner per stage)
    if log_mode == "summary":
        # One-line summary
        cap_str = f"cap={cap_minutes:.1f}m" if cap_minutes is not None else "cap=None"
        logger.info(
            f"📊 {stage}: n_features={len(features)} → safe={len(safe_features)} "
            f"quarantined={len(quarantined_features)} {cap_str} actual_max={actual_max_lookback:.1f}m"
        )
        
        # Log top offenders if any (only if quarantined)
        if quarantined_features:
            # Get top 5 offenders with lookback values
            offenders_with_lookback = []
            for feat_name in quarantined_features[:10]:  # Top 10
                feat_key = _feat_key(feat_name)
                lookback = canonical_map.get(feat_key)
                if lookback is not None and lookback != float("inf"):
                    offenders_with_lookback.append((feat_name, lookback))
            
            if offenders_with_lookback:
                offenders_with_lookback.sort(key=lambda x: x[1], reverse=True)
                top_5 = offenders_with_lookback[:5]
                offenders_str = ', '.join([f'{f}({l:.0f}m)' for f, l in top_5])
                logger.info(f"   Top offenders: {offenders_str}")
    else:
        # Debug mode: detailed per-feature logging (already done above in quarantine loop)
        logger.debug(f"📊 {stage} (DEBUG): n_features={len(features)} → safe={len(safe_features)} quarantined={len(quarantined_features)}")
    
    # Step 6: Return result
    return LookbackCapResult(
        safe_features=safe_features,
        quarantined_features=quarantined_features,
        budget=budget,
        canonical_map=canonical_map,
        fingerprint=budget_fp,
        actual_max_lookback=actual_max_lookback,
        quarantine_count=len(quarantined_features)
    )
