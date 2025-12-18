"""
Dropped Features Tracker

Centralized tracking of features dropped at various stages for telemetry and reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Any


@dataclass
class DropReason:
    """Structured reason for feature drop."""
    reason_code: str  # LOOKBACK_CAP, ALL_NAN, QUARANTINED_LOOKBACK, LOW_IMPORTANCE, SCHEMA_FILTER, etc.
    stage: str  # gatekeeper, sanitizer, pruning, nan_removal, schema_filter, etc.
    human_reason: str  # Human-readable string for logs
    measured_value: Optional[float] = None  # e.g., lookback_minutes
    threshold_value: Optional[float] = None  # e.g., max_safe_lookback_minutes
    config_provenance: Optional[str] = None  # Which config knob caused it


@dataclass
class StageRecord:
    """Record of feature set transition at a stage."""
    stage_id: str  # Unique stage identifier
    stage_name: str  # Human-readable stage name
    input_fingerprint: Optional[str] = None
    output_fingerprint: Optional[str] = None
    input_count: int = 0
    output_count: int = 0
    dropped_count: int = 0
    dropped_sample: List[str] = field(default_factory=list)  # First N dropped features
    order_changed: bool = False
    config_provenance: Optional[Dict[str, Any]] = None  # Config knobs that affected this stage


@dataclass
class DroppedFeaturesTracker:
    """
    Tracks features dropped at various stages of the pipeline.
    
    Used for telemetry, debugging, and reproducibility tracking.
    """
    # Stage records (ordered by execution)
    stage_records: List[StageRecord] = field(default_factory=list)
    
    # Gatekeeper drops (lookback violations)
    gatekeeper_dropped: List[str] = field(default_factory=list)
    gatekeeper_reasons: Dict[str, DropReason] = field(default_factory=dict)  # feature_name -> structured reason
    
    # Sanitizer quarantines (pre-emptive lookback filtering)
    sanitizer_quarantined: List[str] = field(default_factory=list)
    sanitizer_reasons: Dict[str, DropReason] = field(default_factory=dict)  # feature_name -> structured reason
    
    # Pruning drops (importance-based)
    pruning_dropped: List[str] = field(default_factory=list)
    pruning_stats: Optional[Dict[str, Any]] = None  # Full pruning stats dict
    
    # NaN drops (all-NaN columns)
    nan_dropped: List[str] = field(default_factory=list)
    
    # Early filter drops (schema/pattern/registry) - summary only
    early_filter_summary: Dict[str, Any] = field(default_factory=dict)  # counts, top_samples, rule_hits
    
    def record_stage_transition(
        self,
        stage_id: str,
        stage_name: str,
        input_features: List[str],
        output_features: List[str],
        dropped_features: Optional[List[str]] = None,
        config_provenance: Optional[Dict[str, Any]] = None
    ) -> StageRecord:
        """
        Record a feature set transition at a stage.
        
        Uses set-based comparison to correctly detect drops vs reordering.
        """
        from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
        
        input_set = set(input_features)
        output_set = set(output_features)
        
        # Set-based drop detection (handles reordering correctly)
        if dropped_features is None:
            dropped_set = input_set - output_set
            dropped_features = sorted(list(dropped_set))
        else:
            dropped_set = set(dropped_features)
        
        # Detect order changes
        order_changed = (input_features != output_features) and (input_set == output_set)
        
        # Compute fingerprints
        input_fp, _ = _compute_feature_fingerprint(input_features, set_invariant=True)
        output_fp, _ = _compute_feature_fingerprint(output_features, set_invariant=True)
        
        # Create stage record
        record = StageRecord(
            stage_id=stage_id,
            stage_name=stage_name,
            input_fingerprint=input_fp,
            output_fingerprint=output_fp,
            input_count=len(input_features),
            output_count=len(output_features),
            dropped_count=len(dropped_features),
            dropped_sample=dropped_features[:10],  # First 10 for metadata
            order_changed=order_changed,
            config_provenance=config_provenance
        )
        
        self.stage_records.append(record)
        return record
    
    def add_gatekeeper_drops(
        self,
        features: List[str],
        reasons: Dict[str, DropReason],
        input_features: Optional[List[str]] = None,
        output_features: Optional[List[str]] = None,
        config_provenance: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add features dropped by gatekeeper."""
        self.gatekeeper_dropped.extend(features)
        self.gatekeeper_reasons.update(reasons)
        
        # Record stage transition if input/output provided
        if input_features is not None and output_features is not None:
            self.record_stage_transition(
                stage_id="gatekeeper",
                stage_name="Final Gatekeeper (Lookback Enforcement)",
                input_features=input_features,
                output_features=output_features,
                dropped_features=features,
                config_provenance=config_provenance
            )
    
    def add_sanitizer_quarantines(
        self,
        features: List[str],
        reasons: Dict[str, DropReason],
        input_features: Optional[List[str]] = None,
        output_features: Optional[List[str]] = None,
        config_provenance: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add features quarantined by sanitizer."""
        self.sanitizer_quarantined.extend(features)
        self.sanitizer_reasons.update(reasons)
        
        # Record stage transition if input/output provided
        if input_features is not None and output_features is not None:
            self.record_stage_transition(
                stage_id="sanitizer",
                stage_name="Active Sanitization (Pre-emptive Lookback Filter)",
                input_features=input_features,
                output_features=output_features,
                dropped_features=features,
                config_provenance=config_provenance
            )
    
    def add_pruning_drops(
        self,
        features: List[str],
        stats: Optional[Dict[str, Any]] = None,
        input_features: Optional[List[str]] = None,
        output_features: Optional[List[str]] = None,
        config_provenance: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add features dropped by pruning."""
        self.pruning_dropped.extend(features)
        if stats is not None:
            self.pruning_stats = stats
        
        # Record stage transition if input/output provided
        if input_features is not None and output_features is not None:
            self.record_stage_transition(
                stage_id="pruning",
                stage_name="Importance-Based Pruning",
                input_features=input_features,
                output_features=output_features,
                dropped_features=features,
                config_provenance=config_provenance
            )
    
    def add_nan_drops(
        self,
        features: List[str],
        input_features: Optional[List[str]] = None,
        output_features: Optional[List[str]] = None,
        config_provenance: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add features dropped for all-NaN (set-based comparison)."""
        # Use set-based comparison to avoid false positives from reordering
        if input_features is not None and output_features is not None:
            input_set = set(input_features)
            output_set = set(output_features)
            actual_dropped = sorted(list(input_set - output_set))
            self.nan_dropped.extend(actual_dropped)
            
            # Record stage transition
            self.record_stage_transition(
                stage_id="nan_removal",
                stage_name="All-NaN Column Removal",
                input_features=input_features,
                output_features=output_features,
                dropped_features=actual_dropped,
                config_provenance=config_provenance
            )
        else:
            # Fallback: use provided list (assumes caller did set-based comparison)
            self.nan_dropped.extend(features)
    
    def add_early_filter_summary(
        self,
        filter_name: str,
        dropped_count: int,
        top_samples: List[str],
        rule_hits: Optional[Dict[str, int]] = None
    ) -> None:
        """Add summary of early filter drops (schema/pattern/registry)."""
        if filter_name not in self.early_filter_summary:
            self.early_filter_summary[filter_name] = {
                "dropped_count": 0,
                "top_samples": [],
                "rule_hits": {}
            }
        
        self.early_filter_summary[filter_name]["dropped_count"] += dropped_count
        self.early_filter_summary[filter_name]["top_samples"].extend(top_samples[:10])
        if rule_hits:
            for rule, count in rule_hits.items():
                self.early_filter_summary[filter_name]["rule_hits"][rule] = \
                    self.early_filter_summary[filter_name]["rule_hits"].get(rule, 0) + count
    
    def get_all_dropped(self) -> List[str]:
        """Get all unique dropped features (across all stages)."""
        all_dropped = set()
        all_dropped.update(self.gatekeeper_dropped)
        all_dropped.update(self.sanitizer_quarantined)
        all_dropped.update(self.pruning_dropped)
        all_dropped.update(self.nan_dropped)
        return sorted(list(all_dropped))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary dict for metadata.json with structured reasons and stage records."""
        # Convert DropReason objects to dicts for JSON serialization
        def reason_to_dict(reason: DropReason) -> Dict[str, Any]:
            return {
                "reason_code": reason.reason_code,
                "stage": reason.stage,
                "human_reason": reason.human_reason,
                "measured_value": reason.measured_value,
                "threshold_value": reason.threshold_value,
                "config_provenance": reason.config_provenance
            }
        
        gatekeeper_reasons_dict = {
            name: reason_to_dict(reason) if isinstance(reason, DropReason) else reason
            for name, reason in self.gatekeeper_reasons.items()
        }
        
        sanitizer_reasons_dict = {
            name: reason_to_dict(reason) if isinstance(reason, DropReason) else reason
            for name, reason in self.sanitizer_reasons.items()
        }
        
        # Convert stage records to dicts
        stage_records_dict = []
        for record in self.stage_records:
            stage_records_dict.append({
                "stage_id": record.stage_id,
                "stage_name": record.stage_name,
                "input_fingerprint": record.input_fingerprint,
                "output_fingerprint": record.output_fingerprint,
                "input_count": record.input_count,
                "output_count": record.output_count,
                "dropped_count": record.dropped_count,
                "dropped_sample": record.dropped_sample,
                "order_changed": record.order_changed,
                "config_provenance": record.config_provenance
            })
        
        return {
            "stage_records": stage_records_dict,
            "gatekeeper": {
                "count": len(self.gatekeeper_dropped),
                "features": self.gatekeeper_dropped,
                "reasons": gatekeeper_reasons_dict
            },
            "sanitizer": {
                "count": len(self.sanitizer_quarantined),
                "features": self.sanitizer_quarantined,
                "reasons": sanitizer_reasons_dict
            },
            "pruning": {
                "count": len(self.pruning_dropped),
                "features": self.pruning_dropped,
                "stats": self.pruning_stats
            },
            "nan": {
                "count": len(self.nan_dropped),
                "features": self.nan_dropped
            },
            "early_filters": self.early_filter_summary,
            "total_unique": len(self.get_all_dropped())
        }
    
    def is_empty(self) -> bool:
        """Check if any features were dropped."""
        return (
            len(self.gatekeeper_dropped) == 0 and
            len(self.sanitizer_quarantined) == 0 and
            len(self.pruning_dropped) == 0 and
            len(self.nan_dropped) == 0
        )
