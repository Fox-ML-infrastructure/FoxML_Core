"""
Audit Enforcer for Reproducibility Validation

Enforces audit-grade validation rules to catch data leakage, configuration errors,
and reproducibility violations before they cause silent failures.

Usage:
    from TRAINING.utils.audit_enforcer import AuditEnforcer
    
    enforcer = AuditEnforcer(mode="strict")  # or "warn" or "off"
    enforcer.validate(metadata, metrics, previous_metadata=None)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

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
        
        # Build audit report
        audit_report = {
            "mode": self.mode.value,
            "violations": self.violations,
            "warnings": self.warnings,
            "is_valid": len(self.violations) == 0,
            "has_warnings": len(self.warnings) > 0
        }
        
        # Determine if validation passed
        is_valid = len(self.violations) == 0
        
        # In strict mode, violations cause failure
        if self.mode == AuditMode.STRICT and not is_valid:
            violation_summary = "; ".join([v["message"] for v in self.violations])
            raise ValueError(f"Audit validation failed (strict mode): {violation_summary}")
        
        # In warn mode, log violations but don't fail
        if not is_valid:
            for violation in self.violations:
                logger.error(f"🚨 AUDIT VIOLATION: {violation['message']} (rule: {violation['rule']})")
        
        # Log warnings
        for warning in self.warnings:
            logger.warning(f"⚠️  AUDIT WARNING: {warning['message']} (rule: {warning['rule']})")
        
        return is_valid, audit_report
    
    def _validate_purge_embargo(self, metadata: Dict[str, Any]) -> None:
        """Validate purge and embargo are >= horizon."""
        cv_details = metadata.get("cv_details", {})
        horizon = cv_details.get("horizon_minutes") or metadata.get("horizon_minutes")
        purge = cv_details.get("purge_minutes") or metadata.get("purge_minutes")
        embargo = cv_details.get("embargo_minutes") or metadata.get("embargo_minutes")
        
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
        """Validate purge/embargo cover feature lookback."""
        cv_details = metadata.get("cv_details", {})
        purge = cv_details.get("purge_minutes") or metadata.get("purge_minutes")
        embargo = cv_details.get("embargo_minutes") or metadata.get("embargo_minutes")
        lookback = cv_details.get("feature_lookback_max_minutes") or metadata.get("feature_lookback_max_minutes")
        
        if lookback is None:
            return  # Can't validate without lookback
        
        if purge is not None and purge < lookback:
            self.violations.append({
                "rule": "purge_minutes >= feature_lookback_max_minutes",
                "message": f"purge_minutes ({purge}) < feature_lookback_max_minutes ({lookback}) - ROLLING WINDOW LEAKAGE RISK",
                "severity": "critical",
                "purge_minutes": purge,
                "feature_lookback_max_minutes": lookback
            })
        
        if embargo is not None and embargo < lookback:
            self.violations.append({
                "rule": "embargo_minutes >= feature_lookback_max_minutes",
                "message": f"embargo_minutes ({embargo}) < feature_lookback_max_minutes ({lookback}) - ROLLING WINDOW LEAKAGE RISK",
                "severity": "critical",
                "embargo_minutes": embargo,
                "feature_lookback_max_minutes": lookback
            })
    
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
