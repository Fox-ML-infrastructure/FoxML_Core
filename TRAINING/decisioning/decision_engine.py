"""
Decision Engine

Evaluates regression/trend signals and produces actionable decisions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from TRAINING.decisioning.policies import DecisionPolicy, evaluate_policies
from TRAINING.decisioning.bayesian_policy import (
    BayesianPatchPolicy, PatchTemplate, compute_reward
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Result of decision evaluation."""
    run_id: str
    cohort_id: str
    segment_id: Optional[int] = None
    
    # Decision levels (0=no action, 1=warning, 2=recommendation, 3=action)
    decision_level: int = 0
    decision_action_mask: List[str] = None  # List of action codes (e.g., ["freeze_features", "tighten_leakage"])
    decision_reason_codes: List[str] = None  # List of reason codes (e.g., ["jaccard_collapse", "route_instability"])
    
    # Predictions and trends
    predicted_cs_auc: Optional[float] = None
    predicted_sym_auc: Optional[float] = None
    trend_direction: Optional[str] = None  # "improving", "declining", "stable"
    
    # Policy evaluations
    policy_results: Dict[str, Any] = None
    
    # Metadata
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert None to null for JSON
        for k, v in result.items():
            if v is None:
                result[k] = None
        return result


class DecisionEngine:
    """
    Decision engine that evaluates regression/trend signals and produces decisions.
    
    Operates in two modes:
    - Assist mode (default): Computes and persists decisions, but doesn't modify pipeline
    - Apply mode: Applies decisions to config (requires explicit opt-in)
    """
    
    def __init__(
        self,
        index_path: Path,
        policies: Optional[List[DecisionPolicy]] = None,
        apply_mode: bool = False,
        use_bayesian: bool = False,
        base_dir: Optional[Path] = None
    ):
        """
        Initialize decision engine.
        
        Args:
            index_path: Path to index.parquet
            policies: List of decision policies (default: use default policies)
            apply_mode: If True, decisions can modify config (default: False = assist mode)
            use_bayesian: If True, enable Bayesian patch policy (default: False)
            base_dir: Base directory for Bayesian state (required if use_bayesian=True)
        """
        self.index_path = Path(index_path)
        self.apply_mode = apply_mode
        self.policies = policies or DecisionPolicy.get_default_policies()
        self.use_bayesian = use_bayesian
        self.base_dir = Path(base_dir) if base_dir else index_path.parent
        
        # Initialize Bayesian policy if enabled
        self.bayesian_policy = None
        self.bayesian_config = None
        if use_bayesian:
            try:
                # Load config for Bayesian policy (SST: Single Source of Truth)
                from CONFIG.config_loader import get_cfg
                self.bayesian_config = {
                    'bayesian': {
                        'min_runs_for_learning': get_cfg('training.decisions.bayesian.min_runs_for_learning', default=5, config_name='training_config'),
                        'p_improve_threshold': get_cfg('training.decisions.bayesian.p_improve_threshold', default=0.8, config_name='training_config'),
                        'min_expected_gain': get_cfg('training.decisions.bayesian.min_expected_gain', default=0.01, config_name='training_config'),
                        'reward_metric': get_cfg('training.decisions.bayesian.reward_metric', default='cs_auc', config_name='training_config'),
                        'recency_decay': get_cfg('training.decisions.bayesian.recency_decay', default=0.95, config_name='training_config'),
                        'level_3_threshold': get_cfg('training.decisions.bayesian.level_3_threshold', default=0.8, config_name='training_config'),
                        'level_3_gain': get_cfg('training.decisions.bayesian.level_3_gain', default=0.01, config_name='training_config'),
                        'level_2_threshold': get_cfg('training.decisions.bayesian.level_2_threshold', default=0.6, config_name='training_config'),
                        'level_2_gain': get_cfg('training.decisions.bayesian.level_2_gain', default=0.005, config_name='training_config'),
                        'level_1_threshold': get_cfg('training.decisions.bayesian.level_1_threshold', default=0.4, config_name='training_config'),
                        'baseline_window': get_cfg('training.decisions.bayesian.baseline_window', default=10, config_name='training_config'),
                        'templates': get_cfg('training.decisions.bayesian.templates', default=None, config_name='training_config')
                    }
                }
                self.bayesian_policy = BayesianPatchPolicy(
                    index_path=index_path,
                    base_dir=self.base_dir,
                    config=self.bayesian_config
                )
                logger.info("✅ Bayesian patch policy enabled (all config-driven)")
            except Exception as e:
                logger.warning(f"Failed to initialize Bayesian policy: {e}. Continuing without it.")
                self.use_bayesian = False
    
    def evaluate(
        self,
        cohort_id: str,
        run_id: str,
        segment_id: Optional[int] = None
    ) -> DecisionResult:
        """
        Evaluate decisions for a run.
        
        Args:
            cohort_id: Cohort identifier
            run_id: Run identifier
            segment_id: Optional segment identifier
        
        Returns:
            DecisionResult
        """
        if not self.index_path.exists():
            logger.warning(f"Index file not found: {self.index_path}, returning no-op decision")
            return DecisionResult(
                run_id=run_id,
                cohort_id=cohort_id,
                segment_id=segment_id,
                decision_level=0,
                decision_action_mask=[],
                decision_reason_codes=[]
            )
        
        try:
            df = pd.read_parquet(self.index_path)
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return DecisionResult(
                run_id=run_id,
                cohort_id=cohort_id,
                segment_id=segment_id,
                decision_level=0,
                decision_action_mask=[],
                decision_reason_codes=[]
            )
        
        # Filter to cohort/segment
        mask = df['cohort_id'] == cohort_id
        if segment_id is not None:
            mask &= df['segment_id'] == segment_id
        
        cohort_data = df[mask].sort_values('run_started_at')
        
        if len(cohort_data) == 0:
            logger.debug(f"No data for cohort {cohort_id}, returning no-op decision")
            return DecisionResult(
                run_id=run_id,
                cohort_id=cohort_id,
                segment_id=segment_id,
                decision_level=0,
                decision_action_mask=[],
                decision_reason_codes=[]
            )
        
        # Get latest run metrics
        latest = cohort_data.iloc[-1]
        
        # Evaluate rule-based policies
        policy_results = evaluate_policies(cohort_data, latest, self.policies)
        
        # Evaluate Bayesian policy if enabled
        bayesian_result = None
        recommended_patch_template = None
        if self.use_bayesian and self.bayesian_policy:
            try:
                bayesian_result = self.bayesian_policy.evaluate(
                    cohort_id=cohort_id,
                    segment_id=segment_id,
                    run_id=run_id,
                    cohort_data=cohort_data
                )
                if bayesian_result.get('triggered', False):
                    recommended_patch_template = bayesian_result.get('recommended_patch')
                    policy_results['bayesian_patch'] = bayesian_result
            except Exception as e:
                logger.warning(f"Bayesian policy evaluation failed: {e}")
                bayesian_result = None
        
        # Determine decision level and actions
        decision_level = 0
        action_mask = []
        reason_codes = []
        
        # Prioritize Bayesian recommendations if they exist and are high-confidence
        if bayesian_result and bayesian_result.get('triggered', False):
            bayesian_level = bayesian_result.get('level', 0)
            if bayesian_level >= 2:  # Only use Bayesian if confidence is high enough
                decision_level = max(decision_level, bayesian_level)
                if bayesian_result.get('action'):
                    action_mask.append(bayesian_result['action'])
                if bayesian_result.get('reason'):
                    reason_codes.append(bayesian_result['reason'])
        
        # Also check rule-based policies (but Bayesian takes precedence if both trigger)
        for policy_name, result in policy_results.items():
            if policy_name == 'bayesian_patch':
                continue  # Already handled
            if result.get('triggered', False):
                level = result.get('level', 0)
                # Only add rule-based actions if Bayesian didn't already recommend
                if not action_mask or decision_level < level:
                    decision_level = max(decision_level, level)
                    if result.get('action') and result['action'] not in action_mask:
                        action_mask.append(result['action'])
                    if result.get('reason') and result['reason'] not in reason_codes:
                        reason_codes.append(result['reason'])
        
        # Get predictions if available (from regression analysis)
        predicted_cs_auc = latest.get('next_pred') if 'next_pred' in latest else None
        predicted_sym_auc = latest.get('next_pred_sym_auc') if 'next_pred_sym_auc' in latest else None
        
        # Determine trend
        if len(cohort_data) >= 2:
            recent_auc = cohort_data['cs_auc'].iloc[-1] if 'cs_auc' in cohort_data.columns else None
            prev_auc = cohort_data['cs_auc'].iloc[-2] if 'cs_auc' in cohort_data.columns else None
            if recent_auc is not None and prev_auc is not None:
                if recent_auc > prev_auc * 1.01:  # 1% improvement
                    trend_direction = "improving"
                elif recent_auc < prev_auc * 0.99:  # 1% decline
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = None
        else:
            trend_direction = None
        
        from datetime import datetime
        
        # Store Bayesian metadata if available
        bayesian_metadata = {}
        if bayesian_result:
            bayesian_metadata = {
                'confidence': bayesian_result.get('confidence', 0.0),
                'expected_gain': bayesian_result.get('expected_gain', 0.0),
                'baseline_reward': bayesian_result.get('baseline_reward', 0.0),
                'bayes_stats': bayesian_result.get('bayes_stats', {})
            }
        
        return DecisionResult(
            run_id=run_id,
            cohort_id=cohort_id,
            segment_id=segment_id,
            decision_level=decision_level,
            decision_action_mask=action_mask,
            decision_reason_codes=reason_codes,
            predicted_cs_auc=float(predicted_cs_auc) if predicted_cs_auc is not None and not np.isnan(predicted_cs_auc) else None,
            predicted_sym_auc=float(predicted_sym_auc) if predicted_sym_auc is not None and not np.isnan(predicted_sym_auc) else None,
            trend_direction=trend_direction,
            policy_results={**policy_results, 'bayesian_metadata': bayesian_metadata} if bayesian_metadata else policy_results,
            created_at=datetime.now().isoformat()
        )
    
    def persist(
        self,
        decision_result: DecisionResult,
        base_dir: Path
    ) -> Path:
        """
        Persist decision result to file.
        
        Args:
            decision_result: Decision result to persist
            base_dir: Base directory for reproducibility artifacts
        
        Returns:
            Path to persisted decision file
        """
        decisions_dir = base_dir / "REPRODUCIBILITY" / "decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        
        decision_file = decisions_dir / f"{decision_result.run_id}.json"
        
        try:
            with open(decision_file, 'w') as f:
                json.dump(decision_result.to_dict(), f, indent=2, default=str)
            logger.debug(f"Persisted decision to {decision_file}")
            return decision_file
        except Exception as e:
            logger.error(f"Failed to persist decision: {e}")
            raise
    
    def load_latest(
        self,
        cohort_id: str,
        base_dir: Optional[Path] = None
    ) -> Optional[DecisionResult]:
        """
        Load latest decision for a cohort.
        
        Args:
            cohort_id: Cohort identifier
            base_dir: Base directory (if None, uses index_path parent)
        
        Returns:
            Latest DecisionResult or None
        """
        if base_dir is None:
            base_dir = self.index_path.parent
        
        decisions_dir = base_dir / "REPRODUCIBILITY" / "decisions"
        
        if not decisions_dir.exists():
            return None
        
        # Find all decision files and load the latest one for this cohort
        decision_files = list(decisions_dir.glob("*.json"))
        if not decision_files:
            return None
        
        # Load and filter by cohort_id
        latest = None
        latest_time = None
        
        for df_path in decision_files:
            try:
                with open(df_path, 'r') as f:
                    data = json.load(f)
                if data.get('cohort_id') == cohort_id:
                    created_at = data.get('created_at')
                    if created_at:
                        from datetime import datetime
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if latest_time is None or dt > latest_time:
                            latest_time = dt
                            latest = DecisionResult(**data)
            except Exception:
                continue
        
        return latest
    
    def apply_patch(
        self,
        resolved_config: Dict[str, Any],
        decision_result: DecisionResult
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Apply decision patch to resolved config.
        
        Only applies if apply_mode=True and decision_level >= 2.
        
        Args:
            resolved_config: Current resolved config
            decision_result: Decision result
        
        Returns:
            (new_config, patch_dict) - new config and patch that was applied
        """
        if not self.apply_mode:
            logger.debug("Apply mode disabled, skipping config patch")
            return resolved_config, {}, []
        
        if decision_result.decision_level < 2:
            logger.debug(f"Decision level {decision_result.decision_level} < 2, skipping config patch")
            return resolved_config, {}, []
        
        from TRAINING.decisioning.policies import apply_decision_patch
        new_config, patch, warnings = apply_decision_patch(resolved_config, decision_result)
        
        # Log warnings if any
        for warning in warnings:
            logger.warning(f"⚠️  Patch clamp warning: {warning}")
        
        return new_config, patch, warnings
    
    def update_bayesian_state(
        self,
        decision_result: DecisionResult,
        current_run_metrics: Dict[str, Any],
        applied_patch_template: Optional[Any] = None  # PatchTemplate
    ):
        """
        Update Bayesian state with observed reward after run completes.
        
        Args:
            decision_result: Decision that was used
            current_run_metrics: Metrics from current run (from index.parquet)
            applied_patch_template: Template that was applied (None if no patch)
        """
        if not self.use_bayesian or not self.bayesian_policy:
            return
        
        if not self.index_path.exists():
            return
        
        try:
            df = pd.read_parquet(self.index_path)
        except Exception:
            return
        
        # Get cohort data to compute baseline
        mask = df['cohort_id'] == decision_result.cohort_id
        if decision_result.segment_id is not None:
            mask &= df['segment_id'] == decision_result.segment_id
        
        cohort_data = df[mask].sort_values('run_started_at')
        if len(cohort_data) < 2:  # Need at least 2 runs to compute reward
            return
        
        # Compute reward (current - baseline median)
        from TRAINING.decisioning.bayesian_policy import compute_reward
        baseline_runs = cohort_data.tail(min(10, len(cohort_data) - 1))  # Exclude current run
        reward = compute_reward(
            current_run=pd.Series(current_run_metrics),
            baseline_runs=baseline_runs,
            metric=self.bayesian_policy.reward_metric
        )
        
        # Update Bayesian state
        self.bayesian_policy.update(
            cohort_id=decision_result.cohort_id,
            segment_id=decision_result.segment_id,
            run_id=decision_result.run_id,
            applied_patch_template=applied_patch_template,
            reward=reward
        )
        
        logger.debug(f"Updated Bayesian state: reward={reward:.4f} for {decision_result.cohort_id}/{decision_result.segment_id}")
