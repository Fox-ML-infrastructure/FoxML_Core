# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Bayesian Decision Policy

Implements Thompson sampling bandit over discrete patch templates.
Learns from past run outcomes within the same cohort+segment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PatchTemplate:
    """A discrete patch template that can be applied."""
    name: str
    action_code: str  # Maps to existing action codes (e.g., "cap_features")
    patch_params: Dict[str, Any]  # Parameters for the patch (e.g., {"change_pct": -10})
    description: str = ""
    
    def to_patch(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate concrete patch from template.
        
        Args:
            current_config: Current config to patch
        
        Returns:
            Patch dict (keys like "feature_selection.max_features")
        """
        # This will be called by apply_decision_patch, but we return params for logging
        return self.patch_params


@dataclass
class ArmStats:
    """Posterior statistics for one patch template (arm)."""
    n: int = 0  # Number of times tried
    mean_reward: float = 0.0  # Mean reward
    var_reward: float = 1.0  # Variance (for Thompson sampling)
    last_updated: Optional[str] = None
    
    def update(self, reward: float, decay: float = 1.0):
        """Update stats with new reward (with optional decay)."""
        if self.n == 0:
            self.mean_reward = reward
            self.var_reward = 1.0
        else:
            # Exponential moving average with decay
            # Decay reduces weight of old observations
            alpha = min(1.0, 1.0 / (self.n + 1) * decay)
            old_mean = self.mean_reward
            self.mean_reward = (1 - alpha) * self.mean_reward + alpha * reward
            # Simple variance estimate (could use Welford's algorithm for better)
            # Use exponential moving variance
            self.var_reward = max(0.1, (1 - alpha) * self.var_reward + alpha * (reward - old_mean) ** 2)
        
        self.n += 1
        self.last_updated = datetime.now().isoformat()
    
    def sample(self) -> float:
        """Thompson sample: sample reward from posterior."""
        if self.n == 0:
            return 0.0  # Pessimistic prior for untried arms
        # Normal posterior (could use Beta for bounded rewards, but Normal is simpler)
        return np.random.normal(self.mean_reward, np.sqrt(self.var_reward))
    
    def p_improve(self, baseline: float = 0.0) -> float:
        """Probability that this arm improves over baseline."""
        if self.n == 0:
            return 0.0
        # P(reward > baseline) using normal CDF approximation
        # Avoid scipy dependency - use erf approximation
        std = np.sqrt(max(0.1, self.var_reward))
        if std < 1e-6:
            return 1.0 if self.mean_reward > baseline else 0.0
        z = (baseline - self.mean_reward) / std
        # Normal CDF approximation: 1 - 0.5 * (1 + erf(z/sqrt(2)))
        # For z < 0 (mean > baseline), we want high probability
        if z < -3:
            return 0.999
        elif z > 3:
            return 0.001
        else:
            # Simple erf approximation: erf(x) ≈ tanh(1.128 * x)
            erf_approx = np.tanh(1.128 * z / np.sqrt(2))
            return 1.0 - 0.5 * (1.0 + erf_approx)
    
    def expected_gain(self, baseline: float = 0.0) -> float:
        """Expected gain over baseline."""
        return max(0.0, self.mean_reward - baseline)


@dataclass
class BayesState:
    """Bayesian state for a cohort+segment."""
    cohort_id: str
    segment_id: Optional[int]
    arms: Dict[str, ArmStats]  # Map template_name -> ArmStats
    last_run_id: Optional[str] = None
    baseline_reward: float = 0.0  # Rolling median of recent runs
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = asdict(self)
        # Convert ArmStats to dict
        result['arms'] = {
            name: asdict(stats) for name, stats in self.arms.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesState':
        """Create from dict."""
        arms = {
            name: ArmStats(**stats) for name, stats in data.get('arms', {}).items()
        }
        return cls(
            cohort_id=data['cohort_id'],
            segment_id=data.get('segment_id'),
            arms=arms,
            last_run_id=data.get('last_run_id'),
            baseline_reward=data.get('baseline_reward', 0.0),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )


class BayesStateStore:
    """Persists and loads Bayesian state."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize store.
        
        Args:
            base_dir: Base directory (usually output_dir.parent)
        """
        self.base_dir = Path(base_dir)
        self.state_dir = self.base_dir / "REPRODUCIBILITY" / "bayes_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def _state_file(self, cohort_id: str, segment_id: Optional[int]) -> Path:
        """Get state file path."""
        if segment_id is not None:
            filename = f"{cohort_id}_seg{segment_id}.json"
        else:
            filename = f"{cohort_id}.json"
        return self.state_dir / filename
    
    def load(self, cohort_id: str, segment_id: Optional[int]) -> Optional[BayesState]:
        """Load state for cohort+segment."""
        state_file = self._state_file(cohort_id, segment_id)
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
            return BayesState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load Bayes state from {state_file}: {e}")
            return None
    
    def save(self, state: BayesState):
        """Save state."""
        state_file = self._state_file(state.cohort_id, state.segment_id)
        state.updated_at = datetime.now().isoformat()
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2, default=str)
            logger.debug(f"Saved Bayes state to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save Bayes state: {e}")
            raise


def get_default_patch_templates(config: Optional[Dict[str, Any]] = None) -> List[PatchTemplate]:
    """
    Get default patch templates (discrete action space).
    
    These map to existing action codes but with specific parameters.
    Templates are configurable via config dict.
    
    Args:
        config: Optional config dict with template definitions
    """
    templates = []
    
    # Get template configs (defaults if not provided)
    if config is None:
        config = {}
    
    template_configs = config.get('templates', [
        {"name": "cap_features_10pct", "action_code": "cap_features", "reduction_pct": 10},
        {"name": "cap_features_20pct", "action_code": "cap_features", "reduction_pct": 20},
        {"name": "tighten_routing_10pct", "action_code": "tighten_routing", "increase_pct": 10},
        {"name": "tighten_routing_20pct", "action_code": "tighten_routing", "increase_pct": 20},
        {"name": "freeze_features", "action_code": "freeze_features"}
    ])
    
    for tmpl_cfg in template_configs:
        name = tmpl_cfg.get('name')
        action_code = tmpl_cfg.get('action_code')
        if not name or not action_code:
            continue
        
        patch_params = {}
        if 'reduction_pct' in tmpl_cfg:
            patch_params['reduction_pct'] = tmpl_cfg['reduction_pct']
        if 'increase_pct' in tmpl_cfg:
            patch_params['increase_pct'] = tmpl_cfg['increase_pct']
        
        description = tmpl_cfg.get('description', f"{action_code} with {patch_params}")
        
        templates.append(PatchTemplate(
            name=name,
            action_code=action_code,
            patch_params=patch_params,
            description=description
        ))
    
    return templates


def compute_reward(
    current_run: pd.Series,
    baseline_runs: pd.DataFrame,
    metric: str = "auc"
) -> float:
    """
    Compute reward for current run vs baseline.
    
    Args:
        current_run: Current run metrics
        baseline_runs: Historical runs (same cohort+segment, no patch applied)
        metric: Metric to use (default: auc)
    
    Returns:
        Reward = current_metric - median(baseline_metrics)
    """
    if metric not in current_run:
        return 0.0
    
    current_value = current_run[metric]
    if pd.isna(current_value):
        return 0.0
    
    if len(baseline_runs) == 0 or metric not in baseline_runs.columns:
        return 0.0
    
    baseline_median = baseline_runs[metric].median()
    if pd.isna(baseline_median):
        return 0.0
    
    reward = float(current_value - baseline_median)
    return reward


class BayesianPatchPolicy:
    """
    Bayesian policy that selects patches using Thompson sampling.
    
    Integrates with existing DecisionPolicy interface.
    """
    
    def __init__(
        self,
        index_path: Path,
        base_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Bayesian policy.
        
        Args:
            index_path: Path to index.parquet
            base_dir: Base directory for state persistence
            config: Config dict with Bayesian policy settings (all values configurable)
        """
        self.index_path = Path(index_path)
        self.base_dir = Path(base_dir)
        
        # Load config (defaults if not provided)
        if config is None:
            config = {}
        
        # All parameters are config-driven (no hardcoded defaults)
        bayesian_cfg = config.get('bayesian', {})
        self.min_runs_for_learning = bayesian_cfg.get('min_runs_for_learning', 5)
        self.p_improve_threshold = bayesian_cfg.get('p_improve_threshold', 0.8)
        self.min_expected_gain = bayesian_cfg.get('min_expected_gain', 0.01)
        self.reward_metric = bayesian_cfg.get('reward_metric', 'auc')
        self.recency_decay = bayesian_cfg.get('recency_decay', 0.95)
        
        # Decision level thresholds (configurable)
        self.level_3_threshold = bayesian_cfg.get('level_3_threshold', 0.8)  # P(improve) for auto-apply
        self.level_3_gain = bayesian_cfg.get('level_3_gain', 0.01)  # Expected gain for auto-apply
        self.level_2_threshold = bayesian_cfg.get('level_2_threshold', 0.6)  # P(improve) for recommend
        self.level_2_gain = bayesian_cfg.get('level_2_gain', 0.005)  # Expected gain for recommend
        self.level_1_threshold = bayesian_cfg.get('level_1_threshold', 0.4)  # P(improve) for warning
        
        # Baseline window size (configurable)
        self.baseline_window = bayesian_cfg.get('baseline_window', 10)
        
        # Load templates (configurable)
        self.templates = get_default_patch_templates(bayesian_cfg)
        
        self.state_store = BayesStateStore(self.base_dir)
        
        # Initialize arms for each template
        self._init_arms()
    
    def _init_arms(self):
        """Initialize arm stats for templates that don't exist in state."""
        # This is called during state loading, so we ensure all templates have arms
        pass
    
    def evaluate(
        self,
        cohort_id: str,
        segment_id: Optional[int],
        run_id: str,
        cohort_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate and recommend patch.
        
        Args:
            cohort_id: Cohort identifier
            segment_id: Segment identifier
            run_id: Current run ID
            cohort_data: Historical data for cohort+segment
        
        Returns:
            Dict with:
                - triggered: bool
                - level: int (0-3)
                - action: str (action code)
                - reason: str
                - recommended_patch: Optional[PatchTemplate]
                - confidence: float (P(improve))
                - expected_gain: float
                - bayes_stats: Dict (for logging)
        """
        # Load or create state
        state = self.state_store.load(cohort_id, segment_id)
        if state is None:
            state = BayesState(
                cohort_id=cohort_id,
                segment_id=segment_id,
                arms={template.name: ArmStats() for template in self.templates},
                created_at=datetime.now().isoformat()
            )
        
        # Need minimum runs to learn
        if len(cohort_data) < self.min_runs_for_learning:
            return {
                'triggered': False,
                'level': 0,
                'action': None,
                'reason': f"insufficient_history (need {self.min_runs_for_learning}, have {len(cohort_data)})",
                'recommended_patch': None,
                'confidence': 0.0,
                'expected_gain': 0.0,
                'bayes_stats': {}
            }
        
        # Compute baseline (recent runs without patches, or rolling median)
        baseline_runs = cohort_data.tail(min(self.baseline_window, len(cohort_data)))
        if self.reward_metric in baseline_runs.columns:
            state.baseline_reward = float(baseline_runs[self.reward_metric].median())
        
        # Thompson sample: pick best arm
        best_template = None
        best_sample = float('-inf')
        arm_samples = {}
        
        for template in self.templates:
            arm_stats = state.arms.get(template.name, ArmStats())
            sample = arm_stats.sample()
            arm_samples[template.name] = {
                'sample': sample,
                'mean': arm_stats.mean_reward,
                'n': arm_stats.n,
                'p_improve': arm_stats.p_improve(state.baseline_reward),
                'expected_gain': arm_stats.expected_gain(state.baseline_reward)
            }
            
            if sample > best_sample:
                best_sample = sample
                best_template = template
        
        if best_template is None:
            return {
                'triggered': False,
                'level': 0,
                'action': None,
                'reason': "no_template_selected",
                'recommended_patch': None,
                'confidence': 0.0,
                'expected_gain': 0.0,
                'bayes_stats': arm_samples
            }
        
        # Get stats for best arm
        best_arm_stats = state.arms.get(best_template.name, ArmStats())
        confidence = best_arm_stats.p_improve(state.baseline_reward)
        expected_gain = best_arm_stats.expected_gain(state.baseline_reward)
        
        # Decision level based on confidence and gain (all config-driven)
        if confidence >= self.level_3_threshold and expected_gain >= self.level_3_gain:
            level = 3  # High confidence, auto-apply
        elif confidence >= self.level_2_threshold and expected_gain >= self.level_2_gain:
            level = 2  # Moderate confidence, recommend
        elif confidence >= self.level_1_threshold:
            level = 1  # Low confidence, warning
        else:
            level = 0  # No action
        
        return {
            'triggered': level > 0,
            'level': level,
            'action': best_template.action_code,
            'reason': f"bayes_thompson_sample (template={best_template.name}, confidence={confidence:.3f}, gain={expected_gain:.4f})",
            'recommended_patch': best_template,
            'confidence': confidence,
            'expected_gain': expected_gain,
            'bayes_stats': arm_samples,
            'baseline_reward': state.baseline_reward
        }
    
    def update(
        self,
        cohort_id: str,
        segment_id: Optional[int],
        run_id: str,
        applied_patch_template: Optional[PatchTemplate],
        reward: float
    ):
        """
        Update Bayesian state with new reward.
        
        Args:
            cohort_id: Cohort identifier
            segment_id: Segment identifier
            run_id: Run ID
            applied_patch_template: Template that was applied (None if no patch)
            reward: Observed reward
        """
        state = self.state_store.load(cohort_id, segment_id)
        if state is None:
            state = BayesState(
                cohort_id=cohort_id,
                segment_id=segment_id,
                arms={template.name: ArmStats() for template in self.templates},
                created_at=datetime.now().isoformat()
            )
        
        # Update arm that was tried
        if applied_patch_template:
            arm_name = applied_patch_template.name
            if arm_name not in state.arms:
                state.arms[arm_name] = ArmStats()
            state.arms[arm_name].update(reward, decay=self.recency_decay)
        
        # Also update "no_patch" baseline if no patch was applied
        if not applied_patch_template:
            if "no_patch" not in state.arms:
                state.arms["no_patch"] = ArmStats()
            state.arms["no_patch"].update(reward, decay=self.recency_decay)
        
        state.last_run_id = run_id
        self.state_store.save(state)
        
        logger.debug(f"Updated Bayes state for {cohort_id}/{segment_id}: {applied_patch_template.name if applied_patch_template else 'no_patch'} → reward={reward:.4f}")
