# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Audit Helpers for Downstream Components

Provides utilities for downstream components to check audit status and refuse
non-auditable data.
"""

from typing import Dict, Any, Optional


def is_auditable(metadata: Dict[str, Any]) -> bool:
    """
    Check if run metadata indicates auditable status.
    
    This is the canonical check that downstream components should use
    to refuse non-auditable data.
    
    Args:
        metadata: Run metadata dictionary
    
    Returns:
        True if run is auditable, False if non-auditable
    """
    # Check explicit boolean flag (most reliable)
    if "is_auditable" in metadata:
        return metadata["is_auditable"] is True
    
    # Check audit_status string
    audit_status = metadata.get("audit_status", "auditable")
    if audit_status == "non_auditable":
        return False
    
    # Default to auditable if not explicitly marked
    return True


def require_auditable(metadata: Dict[str, Any], component_name: str = "component") -> None:
    """
    Require that run is auditable, raise if not.
    
    Downstream components should call this to refuse non-auditable data.
    
    Args:
        metadata: Run metadata dictionary
        component_name: Name of component requiring auditable data (for error message)
    
    Raises:
        ValueError: If run is not auditable
    """
    if not is_auditable(metadata):
        reason = metadata.get("audit_failure_reason", "Unknown reason")
        raise ValueError(
            f"{component_name} requires auditable data, but run is marked as NON-AUDITABLE. "
            f"Reason: {reason}. "
            f"This run cannot be used for production decisions or reproducibility tracking."
        )


def get_audit_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get audit status summary for logging/display.
    
    Args:
        metadata: Run metadata dictionary
    
    Returns:
        Dictionary with audit status information
    """
    is_audit = is_auditable(metadata)
    return {
        "is_auditable": is_audit,
        "audit_status": metadata.get("audit_status", "auditable" if is_audit else "non_auditable"),
        "audit_failure_reason": metadata.get("audit_failure_reason"),
        "audit_warnings": metadata.get("audit_warnings", [])
    }
