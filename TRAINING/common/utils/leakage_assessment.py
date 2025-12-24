# MIT License - see LICENSE file

"""
Leakage Assessment Dataclass

Single source of truth for leakage assessment flags.
Prevents contradictory reason strings like "overfit_likely; cv_not_suspicious".
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class LeakageAssessment:
    """
    Comprehensive leakage assessment with all flags computed once.
    
    This prevents contradictory reason strings by computing all flags
    from a single assessment rather than building strings ad-hoc.
    """
    leak_scan_pass: bool
    cv_suspicious: bool
    overfit_likely: bool
    auc_too_high_models: List[str]
    
    def reason(self) -> str:
        """
        Generate reason string from flags.
        
        Returns:
            Semicolon-separated list of flags, or "none" if all flags are False
        """
        flags = []
        
        if self.cv_suspicious:
            flags.append("cv_suspicious")
        
        if self.overfit_likely:
            flags.append("overfit_likely")
        
        if self.auc_too_high_models:
            flags.append(f"auc>0.90:{','.join(self.auc_too_high_models)}")
        
        return "; ".join(flags) if flags else "none"
    
    def should_auto_fix(self) -> bool:
        """
        Determine if auto-fix should run based on assessment.
        
        Auto-fix should run if:
        - Leak scan failed (leaky features detected)
        - CV is suspicious (suggests real leakage, not just overfitting)
        - Smoke tests failed (if available)
        
        Auto-fix should NOT run if:
        - Only overfit_likely (classic overfitting, not leakage)
        - CV is normal (suggests legitimate signal)
        """
        # Run auto-fix if leak scan failed or CV is suspicious
        return not self.leak_scan_pass or self.cv_suspicious
    
    def auto_fix_reason(self) -> Optional[str]:
        """
        Generate reason for why auto-fix was skipped (if applicable).
        
        Returns:
            Reason string if should_auto_fix() is False, None otherwise
        """
        if self.should_auto_fix():
            return None
        
        # Build reason for skipping
        reasons = []
        
        if self.overfit_likely and not self.cv_suspicious:
            reasons.append("overfit_likely")
        
        if not self.cv_suspicious:
            reasons.append("cv_not_suspicious")
        
        return "; ".join(reasons) if reasons else "none"
