"""
Audit Enforcer for Reproducibility Validation

Enforces audit-grade validation rules to catch data leakage, configuration errors,
and reproducibility violations before they cause silent failures.

Usage:
    from TRAINING.common.utils.audit_enforcer import AuditEnforcer
    
    enforcer = AuditEnforcer(mode="strict")  # or "warn" or "off"
    enforcer.validate(metadata, metrics, previous_metadata=None)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum

from TRAINING.common.utils.duration_parser import parse_duration, Duration, DurationLike

logger = logging.getLogger(__name__)


class AuditMode(str, Enum):
    """Audit enforcement mode."""
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


class AuditEnforcer:
    """
    Enforces audit-grade validation rules for reproducibility tracking.
    
    Hard fails (always enforced in strict mode):
    - purge_minutes < horizon_minutes
    - embargo_minutes < horizon_minutes
    - purge_minutes < feature_lookback_max_minutes
    - cohort_id unchanged but data_fingerprint changed
    - cohort_id unchanged but fold_boundaries_hash changed
    
    Soft fails / warnings (configurable):
    - AUC > threshold (default: 0.90)
    - feature_registry_hash changed within same cohort
    """
    
    def __init__(
        self,
        mode: str = "warn",
        suspicious_auc_threshold: float = 0.90,
        allow_fold_boundary_changes: bool = False
    ):
        """
        Initialize audit enforcer.
        
        Args:
            mode: "off" | "warn" | "strict" (default: "warn")
            suspicious_auc_threshold: AUC threshold for suspicious score warning (default: 0.90)
            allow_fold_boundary_changes: If True, allow fold_boundaries_hash changes within same cohort (default: False)
        """
        try:
            self.mode = AuditMode(mode.lower())
        except ValueError:
            logger.warning(f"Invalid audit mode '{mode}', defaulting to 'warn'")
            self.mode = AuditMode.WARN
        
        self.suspicious_auc_threshold = suspicious_auc_threshold
        self.allow_fold_boundary_changes = allow_fold_boundary_changes
        self.violations: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def validate(
        self,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        previous_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate metadata and metrics against audit rules.
        
        Args:
            metadata: Current run metadata
            metrics: Current run metrics
            previous_metadata: Previous run metadata (for regression detection)
        
        Returns:
            (is_valid, audit_report) where:
            - is_valid: True if validation passed (or mode is "off")
            - audit_report: Dict with violations, warnings, and recommendations
        """
        self.violations = []
        self.warnings = []
        
        if self.mode == AuditMode.OFF:
            return True, {"mode": "off", "violations": [], "warnings": []}
        
        # Hard validation rules
        self._validate_purge_embargo(metadata)
        self._validate_feature_lookback(metadata)
        
        # Regression detection (if previous metadata available)
        if previous_metadata:
            self._validate_cohort_consistency(metadata, previous_metadata)
            self._validate_fold_consistency(metadata, previous_metadata)
        
        # Soft validation rules (warnings)
        self._validate_suspicious_scores(metrics)
        self._validate_feature_registry_changes(metadata, previous_metadata)
        
        # Check for non-auditable status (STICKY MARKER)
        is_non_auditable = metadata.get("is_auditable", True) == False or metadata.get("audit_status") == "non_auditable"
        
        # Build audit report
        audit_report = {
            "mode": self.mode.value,
            "violations": self.violations,
            "warnings": self.warnings,
            "is_valid": len(self.violations) == 0 and not is_non_auditable,
            "has_warnings": len(self.warnings) > 0,
            "is_auditable": not is_non_auditable,  # STICKY MARKER: explicit boolean
            "audit_status": metadata.get("audit_status", "auditable"),
            "audit_failure_reason": metadata.get("audit_failure_reason")
        }
        
        # Determine if validation passed
        is_valid = len(self.violations) == 0 and not is_non_auditable
        
        # STICKY MARKER: Print non-auditable status prominently in header
        if is_non_auditable:
            reason = metadata.get("audit_failure_reason", "Unknown reason")
            logger.error(
                "=" * 80 + "\n"
                "🚨 RUN MARKED AS NON-AUDITABLE 🚨\n"
                f"Reason: {reason}\n"
                "This run cannot be used for production decisions or reproducibility tracking.\n"
                "=" * 80
            )
        
        # In strict mode, violations cause failure
        if self.mode == AuditMode.STRICT and not is_valid:
            violation_summary = "; ".join([v["message"] for v in self.violations])
            if is_non_auditable:
                raise ValueError(f"Audit validation failed (strict mode): Run is NON-AUDITABLE. {violation_summary}")
            else:
                raise ValueError(f"Audit validation failed (strict mode): {violation_summary}")
        
        # In warn mode, log violations but don't fail
        if not is_valid:
            for violation in self.violations:
                logger.error(f"🚨 AUDIT VIOLATION: {violation['message']} (rule: {violation['rule']})")
        
        # Log warnings
        for warning in self.warnings:
            logger.warning(f"⚠️  AUDIT WARNING: {warning['message']} (rule: {warning['rule']})")
        
        # STICKY MARKER: Include in summary line
        if is_non_auditable:
            logger.error("❌ AUDIT STATUS: NON-AUDITABLE - Results cannot be trusted")
        elif is_valid:
            logger.info("✅ AUDIT STATUS: PASSED - Results are auditable")
        else:
            logger.warning("⚠️  AUDIT STATUS: VIOLATIONS DETECTED - Review required")
        
        return is_valid, audit_report
    
    def _validate_purge_embargo(self, metadata: Dict[str, Any]) -> None:
        """Validate purge and embargo are >= horizon."""
        cv_details = metadata.get("cv_details", {})
        horizon = cv_details.get("horizon_minutes") or metadata.get("horizon_minutes")
        purge = cv_details.get("purge_minutes") or metadata.get("purge_minutes")
        
        # Schema v2: Extract scalar from tagged union (backward compatible with v1)
        from TRAINING.orchestration.utils.reproducibility_tracker import extract_embargo_minutes
        embargo = extract_embargo_minutes(metadata, cv_details)
        
        if horizon is None:
            return  # Can't validate without horizon
        
        if purge is not None and purge < horizon:
            self.violations.append({
                "rule": "purge_minutes >= horizon_minutes",
                "message": f"purge_minutes ({purge}) < horizon_minutes ({horizon}) - DATA LEAKAGE RISK",
                "severity": "critical",
                "purge_minutes": purge,
                "horizon_minutes": horizon
            })
        
        if embargo is not None and embargo < horizon:
            self.violations.append({
                "rule": "embargo_minutes >= horizon_minutes",
                "message": f"embargo_minutes ({embargo}) < horizon_minutes ({horizon}) - DATA LEAKAGE RISK",
                "severity": "critical",
                "embargo_minutes": embargo,
                "horizon_minutes": horizon
            })
    
    def _validate_feature_lookback(self, metadata: Dict[str, Any]) -> None:
        """
        Validate purge/embargo cover feature lookback.
        
        Uses duration-aware comparison to handle any time period format
        (strings like "85.0m", "1h30m", or numeric values in minutes).
        
        **Fail-closed policy**: If duration parsing fails, this marks the run as
        non-auditable rather than silently falling back to potentially incorrect comparisons.
        """
        cv_details = metadata.get("cv_details", {})
        purge_raw = cv_details.get("purge_minutes") or metadata.get("purge_minutes")
        
        # Schema v2: Extract scalar from tagged union (backward compatible with v1)
        from TRAINING.orchestration.utils.reproducibility_tracker import extract_embargo_minutes
        embargo_raw = extract_embargo_minutes(metadata, cv_details)
        lookback_raw = cv_details.get("feature_lookback_max_minutes") or metadata.get("feature_lookback_max_minutes")
        
        if lookback_raw is None:
            return  # Can't validate without lookback
        
        # Parse durations (handle both float minutes and duration strings)
        # Current codebase uses float minutes, but we support DurationLike for future extensibility
        try:
            # If numeric, interpret as minutes (convert to seconds for parse_duration)
            # If string, parse directly
            if isinstance(lookback_raw, (int, float)):
                lookback_d = Duration.from_seconds(lookback_raw * 60.0)
            else:
                lookback_d = parse_duration(lookback_raw)
            
            if purge_raw is not None:
                if isinstance(purge_raw, (int, float)):
                    purge_d = Duration.from_seconds(purge_raw * 60.0)
                else:
                    purge_d = parse_duration(purge_raw)
                
                if purge_d < lookback_d:
                    # Format for error message (preserve original format if possible)
                    purge_str = f"{purge_raw}" if isinstance(purge_raw, (int, float)) else str(purge_raw)
                    lookback_str = f"{lookback_raw}" if isinstance(lookback_raw, (int, float)) else str(lookback_raw)
                    
                    self.violations.append({
                        "rule": "purge_minutes >= feature_lookback_max_minutes",
                        "message": f"purge_minutes ({purge_str}) < feature_lookback_max_minutes ({lookback_str}) - ROLLING WINDOW LEAKAGE RISK",
                        "severity": "critical",
                        "purge_minutes": purge_raw,
                        "feature_lookback_max_minutes": lookback_raw
                    })
        except (ValueError, TypeError) as e:
            # FAIL CLOSED: Don't silently fall back to potentially incorrect comparisons
            # This is a configuration error that must be fixed
            error_msg = (
                f"Failed to parse durations for feature lookback validation: {e}. "
                f"This indicates a configuration error. Run marked as NON-AUDITABLE."
            )
            
            if self.mode == AuditMode.STRICT:
                # In strict mode, raise immediately
                raise ValueError(error_msg) from e
            else:
                # In warn mode, log loud warning and mark as violation
                logger.error(f"🚨 {error_msg}")
                self.violations.append({
                    "rule": "duration_parsing_failed",
                    "message": error_msg,
                    "severity": "critical",
                    "purge_minutes": purge_raw,
                    "feature_lookback_max_minutes": lookback_raw,
                    "parsing_error": str(e)
                })
                # Mark metadata as non-auditable with STICKY markers
                metadata["audit_status"] = "non_auditable"
                metadata["audit_failure_reason"] = f"Duration parsing failed: {e}"
                metadata["is_auditable"] = False  # Explicit boolean flag
                metadata["audit_warnings"] = metadata.get("audit_warnings", [])
                metadata["audit_warnings"].append({
                    "type": "non_auditable",
                    "message": error_msg,
                    "timestamp": pd.Timestamp.now().isoformat() if 'pd' in globals() else None
                })
        
        # NOTE: Embargo is NOT required to cover feature lookback
        # Embargo is for label/horizon overlap, not rolling window features
        # Only purge needs to cover feature lookback to prevent rolling window leakage
        # Removed embargo >= feature_lookback check (was incorrect)
    
    def _validate_cohort_consistency(
        self,
        metadata: Dict[str, Any],
        previous_metadata: Dict[str, Any]
    ) -> None:
        """Validate cohort_id consistency with data_fingerprint."""
        current_cohort = metadata.get("cohort_id")
        previous_cohort = previous_metadata.get("cohort_id")
        
        if current_cohort != previous_cohort:
            return  # Different cohorts, no consistency check needed
        
        current_fingerprint = metadata.get("data_fingerprint")
        previous_fingerprint = previous_metadata.get("data_fingerprint")
        
        if current_fingerprint and previous_fingerprint:
            if current_fingerprint != previous_fingerprint:
                self.violations.append({
                    "rule": "cohort_id unchanged => data_fingerprint unchanged",
                    "message": f"cohort_id unchanged ({current_cohort}) but data_fingerprint changed - DATA DRIFT DETECTED",
                    "severity": "critical",
                    "cohort_id": current_cohort,
                    "previous_fingerprint": previous_fingerprint,
                    "current_fingerprint": current_fingerprint
                })
    
    def _validate_fold_consistency(
        self,
        metadata: Dict[str, Any],
        previous_metadata: Dict[str, Any]
    ) -> None:
        """Validate fold_boundaries_hash consistency."""
        if self.allow_fold_boundary_changes:
            return  # Explicitly allowed
        
        current_cohort = metadata.get("cohort_id")
        previous_cohort = previous_metadata.get("cohort_id")
        
        if current_cohort != previous_cohort:
            return  # Different cohorts, no consistency check needed
        
        current_cv = metadata.get("cv_details", {})
        previous_cv = previous_metadata.get("cv_details", {})
        
        current_hash = current_cv.get("fold_boundaries_hash")
        previous_hash = previous_cv.get("fold_boundaries_hash")
        
        if current_hash and previous_hash:
            if current_hash != previous_hash:
                self.violations.append({
                    "rule": "cohort_id unchanged => fold_boundaries_hash unchanged",
                    "message": f"cohort_id unchanged ({current_cohort}) but fold_boundaries_hash changed - SPLIT DRIFT DETECTED",
                    "severity": "critical",
                    "cohort_id": current_cohort,
                    "previous_hash": previous_hash,
                    "current_hash": current_hash
                })
    
    def _validate_suspicious_scores(self, metrics: Dict[str, Any]) -> None:
        """Warn on suspiciously high scores (potential leakage)."""
        metric_name = metrics.get("metric_name", "").upper()
        mean_score = metrics.get("mean_score")
        
        if mean_score is None:
            return
        
        # Check AUC specifically
        if "AUC" in metric_name or "ROC" in metric_name:
            if mean_score >= self.suspicious_auc_threshold:
                self.warnings.append({
                    "rule": "suspicious_score_threshold",
                    "message": f"{metric_name} = {mean_score:.3f} >= {self.suspicious_auc_threshold} - VERIFY FOR LEAKAGE",
                    "severity": "warning",
                    "metric_name": metric_name,
                    "score": mean_score,
                    "threshold": self.suspicious_auc_threshold
                })
    
    def _validate_feature_registry_changes(
        self,
        metadata: Dict[str, Any],
        previous_metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Warn if feature_registry_hash changed within same cohort."""
        if previous_metadata is None:
            return
        
        current_cohort = metadata.get("cohort_id")
        previous_cohort = previous_metadata.get("cohort_id")
        
        if current_cohort != previous_cohort:
            return  # Different cohorts, no warning needed
        
        current_hash = metadata.get("feature_registry_hash")
        previous_hash = previous_metadata.get("feature_registry_hash")
        
        if current_hash and previous_hash:
            if current_hash != previous_hash:
                self.warnings.append({
                    "rule": "feature_registry_hash_changed",
                    "message": f"cohort_id unchanged ({current_cohort}) but feature_registry_hash changed - FEATURE SET CHANGED",
                    "severity": "warning",
                    "cohort_id": current_cohort,
                    "previous_hash": previous_hash,
                    "current_hash": current_hash
                })
