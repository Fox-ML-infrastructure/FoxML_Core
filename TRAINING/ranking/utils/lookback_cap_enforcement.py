# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Shared Lookback Cap Enforcement

Single function for applying lookback cap enforcement that can be used by both
ranking and feature selection. Ensures consistent behavior and prevents split-brain.
"""

import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnforcedFeatureSet:
    """
    Single Source of Truth (SST) contract for enforced feature sets.
    
    After any stage that can change features, the pipeline MUST produce an EnforcedFeatureSet,
    and downstream code must take THAT, not raw feature_names lists.
    
    This prevents split-brain: featureset mis-wire, lookback oracle mismatch, etc.
    """
    features: List[str]  # Safe, ordered feature list (the truth)
    fingerprint_set: str  # Set-invariant fingerprint (hash(sorted(features))) - for cache keys
    fingerprint_ordered: str  # Order-sensitive fingerprint (hash(tuple(features))) - for validation
    cap_minutes: Optional[float]  # Cap that was enforced
    actual_max_minutes: float  # Actual max lookback from safe features
    canonical_map: Dict[str, float]  # Canonical lookback map (single source of truth)
    quarantined: Dict[str, float]  # Feature → lookback for quarantined features
    unknown: List[str]  # Features with unknown lookback (inf) - tracked separately
    stage: str  # Stage name (e.g., "GATEKEEPER", "POST_PRUNE", "FS_PRE")
    budget: Any  # LeakageBudget object (for purge/embargo computation)
    
    def __post_init__(self):
        """Validate invariants."""
        assert len(self.features) > 0 or len(self.quarantined) == 0, \
            "Cannot have quarantined features if no safe features"
        assert self.actual_max_minutes <= (self.cap_minutes or float('inf')), \
            f"actual_max={self.actual_max_minutes:.1f}m > cap={self.cap_minutes}"
    
    @property
    def fingerprint(self) -> str:
        """Backward compatibility: return set fingerprint."""
        return self.fingerprint_set


@dataclass
class LookbackCapResult:
    """Result of lookback cap enforcement (legacy - use EnforcedFeatureSet)."""
    safe_features: List[str]  # Features that passed the cap
    quarantined_features: List[str]  # Features that exceeded cap (quarantined)
    budget: Any  # LeakageBudget object
    canonical_map: Dict[str, float]  # Canonical lookback map (single source of truth)
    fingerprint: str  # Feature set fingerprint
    actual_max_lookback: float  # Actual max lookback from safe features
    quarantine_count: int  # Number of features quarantined
    
    def to_enforced_set(self, stage: str, cap_minutes: Optional[float] = None) -> EnforcedFeatureSet:
        """Convert to EnforcedFeatureSet (SST contract)."""
        from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
        
        # Use stored unknown/quarantined if available (from apply_lookback_cap)
        if hasattr(self, '_unknown_features') and hasattr(self, '_quarantined_dict'):
            unknown = self._unknown_features
            quarantined = self._quarantined_dict
        else:
            # Fallback: separate unknown (inf) from over-cap
            unknown = []
            quarantined = {}
            from TRAINING.ranking.utils.leakage_budget import _feat_key
            for feat_name in self.quarantined_features:
                feat_key = _feat_key(feat_name)
                lookback = self.canonical_map.get(feat_key)
                if lookback == float("inf"):
                    unknown.append(feat_name)
                else:
                    quarantined[feat_name] = lookback if lookback is not None else float("inf")
        
        # Compute both set and ordered fingerprints
        fp_set, fp_ordered = _compute_feature_fingerprint(self.safe_features, set_invariant=True)
        
        return EnforcedFeatureSet(
            features=self.safe_features,
            fingerprint_set=fp_set,
            fingerprint_ordered=fp_ordered,
            cap_minutes=cap_minutes,
            actual_max_minutes=self.actual_max_lookback,
            canonical_map=self.canonical_map,
            quarantined=quarantined,
            unknown=unknown,
            stage=stage,
            budget=self.budget
        )


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
    log_mode: str = "summary",  # "summary" or "debug"
    output_dir: Optional[Path] = None  # Optional: write full drop list to artifact
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
    from TRAINING.ranking.utils.leakage_budget import compute_feature_lookback_max, compute_budget, _feat_key
    from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
    
    if not features:
        # Empty feature set - return empty result
        set_fp, _ = _compute_feature_fingerprint([], set_invariant=True)
        from TRAINING.ranking.utils.leakage_budget import LeakageBudget
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
    # CRITICAL: Treat unknown (inf) as violation EXACTLY like lookback > cap
    # This ensures gatekeeper and POST_PRUNE use the same logic
    safe_features = []
    quarantined_features = []  # All quarantined (over-cap + unknown)
    unknown_features = []  # Track unknown separately for logging
    quarantined_dict = {}  # Feature → lookback for quarantined
    
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
                    raise RuntimeError(
                        f"Feature '{feat_name}' missing from canonical map in {stage}. "
                        f"This indicates a bug in lookback computation."
                    )
                # Treat as unknown
                unknown_features.append(feat_name)
                quarantined_features.append(feat_name)
                quarantined_dict[feat_name] = float("inf")
            elif lookback == float("inf"):
                # Unknown lookback - treat as violation (same as over-cap)
                # PHASE 1: Stage-aware logging (INFO pre-enforcement, WARNING post-enforcement)
                is_pre_enforcement = any(stage.startswith(prefix) for prefix in ["SAFE_CANDIDATES", "FS_PRE", "FS_POST"])
                if is_pre_enforcement:
                    # Pre-enforcement: Expected, log as INFO
                    if log_mode == "debug":
                        logger.debug(f"   {stage}: {feat_name} → unknown lookback (will be quarantined at gatekeeper)")
                else:
                    # Post-enforcement: Actual violation, log as WARNING
                    if log_mode == "debug":
                        logger.warning(f"   {stage}: {feat_name} → unknown lookback (quarantined - violation)")
                unknown_features.append(feat_name)
                quarantined_features.append(feat_name)
                quarantined_dict[feat_name] = float("inf")
            elif lookback > cap_minutes:
                # Exceeds cap - quarantine
                if log_mode == "debug":
                    logger.debug(f"   {stage}: {feat_name} → {lookback:.1f}m > cap={cap_minutes:.1f}m (quarantined)")
                quarantined_features.append(feat_name)
                quarantined_dict[feat_name] = lookback
            else:
                # Safe - keep
                safe_features.append(feat_name)
    else:
        # No cap - all features are safe (but still check for unknown in strict mode)
        for feat_name in features:
            feat_key = _feat_key(feat_name)
            lookback = canonical_map.get(feat_key)
            
            if lookback is None or lookback == float("inf"):
                # Unknown lookback - still quarantine in strict mode
                if policy == "strict":
                    unknown_features.append(feat_name)
                    quarantined_features.append(feat_name)
                    quarantined_dict[feat_name] = float("inf")
                else:
                    safe_features.append(feat_name)
            else:
                safe_features.append(feat_name)
    
    # Step 3: Compute budget from safe features ONLY
    # CRITICAL: If unknowns exist, they must be treated as violation
    # Either hard-stop (strict) or drop them (drop_features), but don't compute budget with them
    if unknown_features and policy == "strict":
        # In strict mode, unknowns are violations - hard-stop
        error_msg = (
            f"🚨 UNKNOWN LOOKBACK VIOLATION ({stage}): {len(unknown_features)} features have unknown lookback (inf). "
            f"In strict mode, unknown lookback is UNSAFE and must be dropped/quarantined. "
            f"Sample: {unknown_features[:10]}"
        )
        logger.error(error_msg)
        raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
    
    # Compute budget from safe features only (unknowns already quarantined)
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
        
        # CRITICAL: If unknowns were quarantined, actual_max should reflect that unknowns exist
        # In strict mode, this is already handled (hard-stop above)
        # In drop mode, we've dropped them, so actual_max is correct (finite only)
        # But log both finite max and unknown count for clarity
        # PHASE 1: Stage-aware logging for unknown features
        if unknown_features:
            is_pre_enforcement = any(stage.startswith(prefix) for prefix in ["SAFE_CANDIDATES", "FS_PRE", "FS_POST"])
            if is_pre_enforcement:
                logger.info(
                    f"   📊 {stage}: actual_max_finite={actual_max_lookback:.1f}m, "
                    f"unknown_count={len(unknown_features)} (will be quarantined at gatekeeper)"
                )
            else:
                logger.warning(
                    f"   📊 {stage}: actual_max_finite={actual_max_lookback:.1f}m, "
                    f"unknown_count={len(unknown_features)} (quarantined - violation)"
                )
    else:
        # No safe features - create empty budget
        from TRAINING.ranking.utils.leakage_budget import LeakageBudget
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
        # One-line summary with unknown count
        cap_str = f"cap={cap_minutes:.1f}m" if cap_minutes is not None else "cap=None"
        unknown_str = f" unknown={len(unknown_features)}" if unknown_features else ""
        logger.info(
            f"📊 {stage}: n_features={len(features)} → safe={len(safe_features)} "
            f"quarantined={len(quarantined_features)}{unknown_str} {cap_str} actual_max={actual_max_lookback:.1f}m"
        )
        
                # Log top offenders if any (only if quarantined)
        if quarantined_features:
            # Get top 10 offenders with lookback values (prioritize over-cap, then unknown)
            offenders_with_lookback = []
            for feat_name in quarantined_features:
                feat_key = _feat_key(feat_name)
                lookback = canonical_map.get(feat_key)
                if lookback is not None:
                    # Handle inf as "unknown"
                    lookback_val = lookback if lookback != float("inf") else None
                    offenders_with_lookback.append((feat_name, lookback_val))
            
            if offenders_with_lookback:
                # Sort by lookback (inf/None last) - show over-cap first
                offenders_with_lookback.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0), reverse=True)
                top_10 = offenders_with_lookback[:10]
                # Format for log (show top 5)
                top_5 = top_10[:5]
                offenders_str = ', '.join([f'{f}({l:.0f}m)' if l is not None else f'{f}(inf)' for f, l in top_5])
                logger.info(f"   Top offenders: {offenders_str}")
                
                # Write full drop list to artifact if output_dir provided
                if output_dir is not None:
                    try:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        artifact_path = output_dir / f"lookback_cap_quarantined_{stage}.json"
                        # Build full list with all quarantined features
                        full_quarantined_list = []
                        for feat_name in quarantined_features:
                            feat_key = _feat_key(feat_name)
                            lookback = canonical_map.get(feat_key)
                            if lookback is None:
                                reason = "missing from canonical map"
                                lookback_val = None
                            elif lookback == float("inf"):
                                reason = "unknown lookback (cannot infer - treated as unsafe)"
                                lookback_val = None
                            else:
                                reason = f"lookback ({lookback:.1f}m) > cap ({cap_minutes:.1f}m)"
                                lookback_val = lookback
                            full_quarantined_list.append({
                                "feature_name": feat_name,
                                "lookback_minutes": lookback_val,
                                "reason": reason
                            })
                        # Sort by lookback (descending, None last)
                        full_quarantined_list.sort(key=lambda x: (x["lookback_minutes"] is None, x["lookback_minutes"] if x["lookback_minutes"] is not None else 0), reverse=True)
                        
                        artifact_data = {
                            "stage": stage,
                            "cap_minutes": cap_minutes,
                            "interval_minutes": interval_minutes,
                            "n_quarantined": len(quarantined_features),
                            "n_safe": len(safe_features),
                            "actual_max_lookback": actual_max_lookback,
                            "quarantined_features": full_quarantined_list
                        }
                        with open(artifact_path, 'w') as f:
                            json.dump(artifact_data, f, indent=2)
                        logger.info(f"   📄 Full drop list written to: {artifact_path}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to write artifact file: {e}")
    else:
        # Debug mode: detailed per-feature logging (already done above in quarantine loop)
        logger.debug(f"📊 {stage} (DEBUG): n_features={len(features)} → safe={len(safe_features)} quarantined={len(quarantined_features)}")
    
    # Step 6: Return result
    result = LookbackCapResult(
        safe_features=safe_features,
        quarantined_features=quarantined_features,
        budget=budget,
        canonical_map=canonical_map,
        fingerprint=budget_fp,
        actual_max_lookback=actual_max_lookback,
        quarantine_count=len(quarantined_features)
    )
    
    # Store unknown/quarantined dict for EnforcedFeatureSet conversion
    result._unknown_features = unknown_features
    result._quarantined_dict = quarantined_dict
    
    return result

