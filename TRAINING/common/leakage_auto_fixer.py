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
Automated Leakage Detection and Auto-Fix System

Automatically detects leaking features from:
1. Leakage sentinels (shifted target, symbol holdout, randomized time tests)
2. Feature importance analysis (perfect scores, suspicious importance)
3. Importance diff detector (comparing full vs safe feature sets)

Then auto-populates excluded_features.yaml and feature_registry.yaml
and re-runs training until no leakage is detected.
"""

import sys
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json

# Add project root to path
# TRAINING/common/leakage_auto_fixer.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from TRAINING.common.leakage_sentinels import LeakageSentinel, SentinelResult
from TRAINING.common.importance_diff_detector import ImportanceDiffDetector
from TRAINING.utils.leakage_filtering import filter_features_for_target, _load_leakage_config

logger = logging.getLogger(__name__)

# Try to import config loader for path configuration
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LeakageDetection:
    """Result of leakage detection for a single feature."""
    feature_name: str
    confidence: float  # 0.0 to 1.0
    reason: str  # Why it's considered a leak
    source: str  # Which detector found it (sentinels, importance, diff)
    suggested_action: str  # 'exact', 'prefix', 'regex', 'registry_reject'


@dataclass
class AutoFixInfo:
    """Information about what was modified by auto-fixer."""
    modified_configs: bool  # True if any configs were modified
    modified_files: List[str]  # List of config files that were modified
    modified_features: List[str]  # List of feature names that were excluded/rejected
    excluded_features_updates: Dict[str, Any]  # Updates to excluded_features.yaml
    feature_registry_updates: Dict[str, Any]  # Updates to feature_registry.yaml
    backup_files: List[str] = None  # List of backup files created


class LeakageAutoFixer:
    """
    Automatically detects and fixes data leakage by:
    1. Running leakage diagnostics
    2. Identifying leaking features
    3. Auto-updating config files
    4. Re-running until clean
    """
    
    def __init__(
        self,
        excluded_features_path: Optional[Path] = None,
        feature_registry_path: Optional[Path] = None,
        backup_configs: bool = True
    ):
        """
        Initialize auto-fixer.
        
        Args:
            excluded_features_path: Path to excluded_features.yaml (default: CONFIG/excluded_features.yaml)
            feature_registry_path: Path to feature_registry.yaml (default: CONFIG/feature_registry.yaml)
            backup_configs: If True, backup configs before modifying
        """
        # Load paths from config if available, otherwise use defaults
        if excluded_features_path is None:
            if _CONFIG_AVAILABLE:
                try:
                    system_cfg = get_system_config()
                    config_path = system_cfg.get('system', {}).get('paths', {})
                    excluded_path = config_path.get('excluded_features')
                    if excluded_path:
                        excluded_features_path = Path(excluded_path)
                        if not excluded_features_path.is_absolute():
                            excluded_features_path = _REPO_ROOT / excluded_path
                    else:
                        # Use default: CONFIG/excluded_features.yaml
                        config_dir = config_path.get('config_dir', 'CONFIG')
                        excluded_features_path = _REPO_ROOT / config_dir / "excluded_features.yaml"
                except Exception:
                    # Fallback to default
                    excluded_features_path = _REPO_ROOT / "CONFIG" / "excluded_features.yaml"
            else:
                excluded_features_path = _REPO_ROOT / "CONFIG" / "excluded_features.yaml"
        
        if feature_registry_path is None:
            if _CONFIG_AVAILABLE:
                try:
                    system_cfg = get_system_config()
                    config_path = system_cfg.get('system', {}).get('paths', {})
                    registry_path = config_path.get('feature_registry')
                    if registry_path:
                        feature_registry_path = Path(registry_path)
                        if not feature_registry_path.is_absolute():
                            feature_registry_path = _REPO_ROOT / registry_path
                    else:
                        # Use default: CONFIG/feature_registry.yaml
                        config_dir = config_path.get('config_dir', 'CONFIG')
                        feature_registry_path = _REPO_ROOT / config_dir / "feature_registry.yaml"
                except Exception:
                    # Fallback to default
                    feature_registry_path = _REPO_ROOT / "CONFIG" / "feature_registry.yaml"
            else:
                feature_registry_path = _REPO_ROOT / "CONFIG" / "feature_registry.yaml"
        
        self.excluded_features_path = Path(excluded_features_path)
        self.feature_registry_path = Path(feature_registry_path)
        self.backup_configs = backup_configs
        
        # Get backup directory from config
        if _CONFIG_AVAILABLE:
            try:
                system_cfg = get_system_config()
                config_path = system_cfg.get('system', {}).get('paths', {})
                backup_dir = config_path.get('config_backup_dir')
                if backup_dir:
                    self.backup_dir = Path(backup_dir)
                    if not self.backup_dir.is_absolute():
                        self.backup_dir = _REPO_ROOT / backup_dir
                else:
                    # Default: CONFIG/backups/
                    config_dir = config_path.get('config_dir', 'CONFIG')
                    self.backup_dir = _REPO_ROOT / config_dir / "backups"
            except Exception:
                # Fallback to default
                self.backup_dir = self.excluded_features_path.parent / "backups"
        else:
            # Fallback to default
            self.backup_dir = self.excluded_features_path.parent / "backups"
        
        # Track detected leaks across iterations
        self.detected_leaks: Dict[str, LeakageDetection] = {}
        self.iteration_count = 0
        
        # Cache of already-excluded features (loaded on demand)
        self._excluded_features_cache: Optional[Set[str]] = None
        self._excluded_prefixes_cache: Optional[Set[str]] = None
        
        # Load backup settings from config
        self.max_backups_per_target = self._load_backup_config()
    
    def _load_excluded_features(self) -> Tuple[Set[str], Set[str]]:
        """Load already-excluded features from config files."""
        if self._excluded_features_cache is not None:
            return self._excluded_features_cache, self._excluded_prefixes_cache
        
        excluded_exact = set()
        excluded_prefixes = set()
        
        # Load from excluded_features.yaml
        if self.excluded_features_path.exists():
            try:
                with open(self.excluded_features_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    always_exclude = config.get('always_exclude', {})
                    excluded_exact = set(always_exclude.get('exact_patterns', []))
                    excluded_prefixes = set(always_exclude.get('prefix_patterns', []))
            except Exception as e:
                logger.debug(f"Could not load excluded_features.yaml: {e}")
        
        # Load from feature_registry.yaml (rejected features)
        if self.feature_registry_path.exists():
            try:
                with open(self.feature_registry_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    features = config.get('features', {})
                    for feat_name, metadata in features.items():
                        if metadata.get('rejected', False):
                            excluded_exact.add(feat_name)
            except Exception as e:
                logger.debug(f"Could not load feature_registry.yaml: {e}")
        
        self._excluded_features_cache = excluded_exact
        self._excluded_prefixes_cache = excluded_prefixes
        return excluded_exact, excluded_prefixes
    
    def _is_already_excluded(self, feature_name: str) -> bool:
        """Check if a feature is already excluded."""
        excluded_exact, excluded_prefixes = self._load_excluded_features()
        
        # Check exact match
        if feature_name in excluded_exact:
            return True
        
        # Check prefix match
        for prefix in excluded_prefixes:
            if feature_name.startswith(prefix):
                return True
        
        return False
    
    def detect_leaking_features(
        self,
        X: Any,  # Feature matrix (pd.DataFrame or np.ndarray)
        y: Any,  # Target (pd.Series or np.ndarray)
        feature_names: List[str],
        target_column: str,
        symbols: Optional[Any] = None,  # pd.Series with symbol labels
        task_type: str = 'classification',  # 'classification' or 'regression'
        data_interval_minutes: int = 5,
        model_importance: Optional[Dict[str, float]] = None,  # feature -> importance
        train_score: Optional[float] = None,  # Perfect score indicates leakage
        test_score: Optional[float] = None
    ) -> List[LeakageDetection]:
        """
        Detect leaking features using multiple methods.
        
        Filters out features that are already excluded to avoid redundant detections.
        
        Returns:
            List of LeakageDetection objects
        """
        # Filter out already-excluded features from detection
        # (These shouldn't be in feature_names if filtering worked, but check anyway)
        excluded_exact, excluded_prefixes = self._load_excluded_features()
        candidate_features = [
            f for f in feature_names 
            if not self._is_already_excluded(f)
        ]
        
        if len(candidate_features) < len(feature_names):
            logger.debug(
                f"Filtered out {len(feature_names) - len(candidate_features)} "
                f"already-excluded features from detection"
            )
        
        detections = []
        
        # Method 1: Perfect scores indicate leakage
        if train_score is not None and train_score >= 0.99:
            logger.debug(f"Method 1: Perfect score detected ({train_score:.4f} >= 0.99)")
            # High importance features in perfect-score models are suspicious
            if model_importance and len(model_importance) > 0:
                sorted_features = sorted(model_importance.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:10]  # Top 10 most important
                logger.debug(f"Method 1: Found {len(top_features)} top features from model_importance")
                
                for feat_name, importance in top_features:
                    if feat_name in candidate_features:
                        detections.append(LeakageDetection(
                            feature_name=feat_name,
                            confidence=min(0.9, importance),  # Scale importance to confidence
                            reason=f"High importance ({importance:.2%}) in perfect-score model (train_score={train_score:.4f})",
                            source="perfect_score_importance",
                            suggested_action=self._suggest_action(feat_name)
                        ))
                logger.debug(f"Method 1: Created {len(detections)} detections from perfect score")
            else:
                logger.debug(f"Method 1: No model_importance provided or empty (len={len(model_importance) if model_importance else 0})")
                # Even without importance, if we have perfect score, check for known leaky patterns in top features
                # This is a fallback - check all features for known patterns
                logger.debug("Method 1: Falling back to pattern-based detection for all features")
                for feat_name in candidate_features:
                    if self._is_known_leaky_pattern(feat_name):
                        detections.append(LeakageDetection(
                            feature_name=feat_name,
                            confidence=0.9,  # High confidence for known patterns with perfect score
                            reason=f"Known leaky pattern in perfect-score model (train_score={train_score:.4f})",
                            source="perfect_score_pattern",
                            suggested_action=self._suggest_action(feat_name)
                        ))
                logger.debug(f"Method 1: Pattern-based fallback found {len(detections)} detections")
        
        # Method 2: Leakage sentinels
        try:
            sentinel = LeakageSentinel()
            
            # Convert to DataFrame if needed
            import pandas as pd
            import numpy as np
            
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = X
            
            if not isinstance(y, pd.Series):
                y_series = pd.Series(y)
            else:
                y_series = y
            
            # Train a simple model for sentinel tests
            # Use sklearn models for fast training
            try:
                from sklearn.linear_model import LogisticRegression, LinearRegression
                from sklearn.model_selection import train_test_split
                
                if task_type == 'classification':
                    simple_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=100)
                else:
                    simple_model = LinearRegression()
                
                # Train on subset for speed
                if len(X_df) > 10000:
                    X_sample, _, y_sample, _ = train_test_split(X_df, y_series, train_size=10000, random_state=42, stratify=y_series if task_type == 'classification' else None)
                else:
                    X_sample, y_sample = X_df, y_series
                
                simple_model.fit(X_sample, y_sample)
                
                # Shifted target test (requires model)
                shifted_result = sentinel.shifted_target_test(
                    simple_model, X_sample.values, y_sample.values, horizon=1
                )
                if not shifted_result.passed and shifted_result.score > 0.7:
                    # High score on shifted target = features encode future info
                    # Mark top features as suspicious
                    if model_importance:
                        top_suspicious = sorted(model_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        for feat_name, importance in top_suspicious:
                            if feat_name in candidate_features:
                                detections.append(LeakageDetection(
                                    feature_name=feat_name,
                                    confidence=0.8,
                                    reason=f"High importance in shifted-target test failure (score={shifted_result.score:.3f})",
                                    source="shifted_target_test",
                                    suggested_action=self._suggest_action(feat_name)
                                ))
                
                # Symbol holdout test (requires train/test split by symbol)
                if symbols is not None and len(symbols.unique()) >= 2:
                    try:
                        from sklearn.model_selection import train_test_split as sk_train_test_split
                        unique_symbols = symbols.unique()
                        train_syms, test_syms = sk_train_test_split(
                            unique_symbols, test_size=0.2, random_state=42
                        )
                        X_train_sym = X_df[symbols.isin(train_syms)]
                        y_train_sym = y_series[symbols.isin(train_syms)]
                        X_test_sym = X_df[symbols.isin(test_syms)]
                        y_test_sym = y_series[symbols.isin(test_syms)]
                        
                        if len(X_train_sym) > 100 and len(X_test_sym) > 100:
                            holdout_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=100) if task_type == 'classification' else LinearRegression()
                            holdout_model.fit(X_train_sym, y_train_sym)
                            
                            holdout_result = sentinel.symbol_holdout_test(
                                holdout_model, X_train_sym.values, y_train_sym.values,
                                X_test_sym.values, y_test_sym.values,
                                train_symbols=list(train_syms), test_symbols=list(test_syms)
                            )
                            if not holdout_result.passed:
                                # Large train/test gap = symbol-specific leakage
                                if model_importance:
                                    top_suspicious = sorted(model_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                                    for feat_name, importance in top_suspicious:
                                        if feat_name in candidate_features:
                                            detections.append(LeakageDetection(
                                                feature_name=feat_name,
                                                confidence=0.7,
                                                reason=f"High importance in symbol-holdout test failure (diff={holdout_result.details.get('gap', 0):.3f})",
                                                source="symbol_holdout_test",
                                                suggested_action=self._suggest_action(feat_name)
                                            ))
                    except Exception as e:
                        logger.debug(f"Symbol holdout test skipped: {e}")
                
                # Randomized time test (requires model)
                randomized_result = sentinel.randomized_time_test(
                    simple_model, X_sample.values, y_sample.values
                )
                if not randomized_result.passed and randomized_result.score > 0.7:
                    # High score on randomized time = features encode temporal info incorrectly
                    if model_importance:
                        top_suspicious = sorted(model_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        for feat_name, importance in top_suspicious:
                            if feat_name in candidate_features:
                                detections.append(LeakageDetection(
                                    feature_name=feat_name,
                                    confidence=0.75,
                                    reason=f"High importance in randomized-time test failure (score={randomized_result.score:.3f})",
                                    source="randomized_time_test",
                                    suggested_action=self._suggest_action(feat_name)
                                ))
            except Exception as e:
                logger.debug(f"Sentinel tests skipped (need model): {e}")
        except Exception as e:
            logger.warning(f"Leakage sentinels failed: {e}")
        
        # Method 3: Pattern-based detection (known leaky patterns)
        # Only run if we haven't already detected leaks from perfect scores
        # (to avoid duplicate detections, but still check patterns if no importance data)
        if not detections or (train_score is None or train_score < 0.99):
            logger.debug("Method 3: Checking for known leaky patterns")
            pattern_detections = []
            for feat_name in candidate_features:
                if self._is_known_leaky_pattern(feat_name):
                    pattern_detections.append(LeakageDetection(
                        feature_name=feat_name,
                        confidence=0.95,  # High confidence for known patterns
                        reason="Matches known leaky pattern",
                        source="pattern_detection",
                        suggested_action=self._suggest_action(feat_name)
                    ))
            detections.extend(pattern_detections)
            logger.debug(f"Method 3: Found {len(pattern_detections)} known leaky patterns")
        
        # Deduplicate and merge confidence scores
        merged = self._merge_detections(detections)
        logger.debug(f"Total detections after merge: {len(merged)}")
        
        return merged
    
    def _is_known_leaky_pattern(self, feature_name: str) -> bool:
        """Check if feature matches known leaky patterns."""
        leaky_prefixes = ['p_', 'y_', 'fwd_ret_', 'tth_', 'mfe_', 'mdd_', 'barrier_', 'next_', 'future_']
        leaky_exact = ['ts', 'timestamp', 'symbol', 'date', 'time']
        
        if feature_name in leaky_exact:
            return True
        
        for prefix in leaky_prefixes:
            if feature_name.startswith(prefix):
                return True
        
        return False
    
    def _suggest_action(self, feature_name: str) -> str:
        """Suggest the best action for excluding a feature."""
        # Exact match for common metadata
        if feature_name in ['ts', 'timestamp', 'symbol', 'date', 'time']:
            return 'exact'
        
        # Prefix patterns for known leaky families
        if feature_name.startswith('p_'):
            return 'prefix'  # Add 'p_' to prefix_patterns
        if feature_name.startswith('y_'):
            return 'prefix'  # Add 'y_' to prefix_patterns
        if feature_name.startswith('fwd_ret_'):
            return 'prefix'  # Add 'fwd_ret_' to prefix_patterns
        
        # For others, use exact match (safer)
        return 'exact'
    
    def _merge_detections(self, detections: List[LeakageDetection]) -> List[LeakageDetection]:
        """Merge duplicate detections, taking max confidence."""
        merged_dict: Dict[str, LeakageDetection] = {}
        
        for det in detections:
            if det.feature_name not in merged_dict:
                merged_dict[det.feature_name] = det
            else:
                # Merge: take max confidence, combine reasons
                existing = merged_dict[det.feature_name]
                if det.confidence > existing.confidence:
                    merged_dict[det.feature_name] = LeakageDetection(
                        feature_name=det.feature_name,
                        confidence=det.confidence,
                        reason=f"{existing.reason}; {det.reason}",
                        source=f"{existing.source}+{det.source}",
                        suggested_action=det.suggested_action
                    )
        
        return list(merged_dict.values())
    
    def apply_fixes(
        self,
        detections: List[LeakageDetection],
        min_confidence: float = 0.7,
        max_features: Optional[int] = None,
        dry_run: bool = False,
        target_name: Optional[str] = None,
        max_backups_per_target: int = 20
    ) -> Tuple[Dict[str, Any], AutoFixInfo]:
        """
        Apply detected fixes to config files.
        
        Args:
            detections: List of detected leaks
            min_confidence: Minimum confidence to auto-fix (default: 0.7)
            max_features: Maximum number of features to fix per run (default: None = no limit)
            dry_run: If True, don't actually modify files, just return what would be done
        
        Returns:
            Tuple of (updates_dict, AutoFixInfo) where:
            - updates_dict: Dict with 'excluded_features_updates' and 'feature_registry_updates'
            - AutoFixInfo: Information about what was modified
        """
        # Filter by confidence
        high_confidence = [d for d in detections if d.confidence >= min_confidence]
        
        # Backup configs BEFORE checking if there are leaks to fix
        # This ensures we have a backup even when auto-fix mode is enabled but no leaks detected
        backup_files = []
        if self.backup_configs and not dry_run:
            # Use provided max_backups or fall back to instance config
            backup_max = max_backups_per_target if max_backups_per_target is not None else self.max_backups_per_target
            backup_files = self._backup_configs(
                target_name=target_name,
                max_backups_per_target=backup_max
            )
            if not high_confidence:
                logger.info(
                    f"Created backup (no leaks detected with confidence >= {min_confidence}): "
                    f"{len(backup_files)} backup files"
                )
        
        if not high_confidence:
            logger.info(f"No leaks detected with confidence >= {min_confidence}")
            empty_autofix_info = AutoFixInfo(
                modified_configs=False,
                modified_files=[],
                modified_features=[],
                excluded_features_updates={},
                feature_registry_updates={},
                backup_files=backup_files  # Include backup files even when no leaks
            )
            return {'excluded_features_updates': {}, 'feature_registry_updates': {}}, empty_autofix_info
        
        # Sort by confidence (descending) and limit to max_features
        high_confidence.sort(key=lambda x: x.confidence, reverse=True)
        if max_features is not None and len(high_confidence) > max_features:
            logger.info(
                f"Limiting auto-fix to top {max_features} features (by confidence) "
                f"out of {len(high_confidence)} detected leaks"
            )
            high_confidence = high_confidence[:max_features]
        
        logger.info(
            f"Auto-fixing {len(high_confidence)} leaks "
            f"(confidence >= {min_confidence}, max_features={max_features})"
        )
        
        # Group by action type
        exact_matches = []
        prefix_patterns = set()
        
        for det in high_confidence:
            if det.suggested_action == 'exact':
                exact_matches.append(det.feature_name)
            elif det.suggested_action == 'prefix':
                # Extract prefix
                if det.feature_name.startswith('p_'):
                    prefix_patterns.add('p_')
                elif det.feature_name.startswith('y_'):
                    prefix_patterns.add('y_')
                elif det.feature_name.startswith('fwd_ret_'):
                    prefix_patterns.add('fwd_ret_')
                else:
                    # Fallback to exact
                    exact_matches.append(det.feature_name)
        
        updates = {
            'excluded_features_updates': {
                'exact_patterns': exact_matches,
                'prefix_patterns': list(prefix_patterns)
            },
            'feature_registry_updates': {
                'rejected_features': [d.feature_name for d in high_confidence]
            }
        }
        
        modified_files = []
        
        if not dry_run:
            self._apply_excluded_features_updates(updates['excluded_features_updates'])
            self._apply_feature_registry_updates(updates['feature_registry_updates'])
            
            # Invalidate cache so next detection reloads excluded features
            self._excluded_features_cache = None
            self._excluded_prefixes_cache = None
            
            # Track which files were modified
            if updates['excluded_features_updates'].get('exact_patterns') or updates['excluded_features_updates'].get('prefix_patterns'):
                modified_files.append(str(self.excluded_features_path))
            if updates['feature_registry_updates'].get('rejected_features'):
                modified_files.append(str(self.feature_registry_path))
            
            # Backup files are already created by _backup_configs() above
        
        # Create AutoFixInfo
        autofix_info = AutoFixInfo(
            modified_configs=len(modified_files) > 0,
            modified_files=modified_files,
            modified_features=[d.feature_name for d in high_confidence],
            excluded_features_updates=updates['excluded_features_updates'],
            feature_registry_updates=updates['feature_registry_updates'],
            backup_files=backup_files if backup_files else None
        )
        
        return updates, autofix_info
    
    def _load_backup_config(self) -> int:
        """Load backup configuration from system_config.yaml."""
        default_max_backups = 20
        if _CONFIG_AVAILABLE:
            try:
                system_cfg = get_system_config()
                backup_cfg = system_cfg.get('system', {}).get('backup', {})
                max_backups = backup_cfg.get('max_backups_per_target', default_max_backups)
                return int(max_backups) if max_backups is not None else default_max_backups
            except Exception as e:
                logger.debug(f"Could not load backup config: {e}, using default {default_max_backups}")
        return default_max_backups
    
    def _get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=_REPO_ROOT
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _backup_configs(self, target_name: Optional[str] = None, max_backups_per_target: Optional[int] = None):
        """
        Backup config files before modification.
        
        Uses timestamp subdirectory structure:
        - With target: CONFIG/backups/{target}/{timestamp}/files + manifest.json
        - Without target: CONFIG/backups/{timestamp}/files (legacy flat mode)
        
        Args:
            target_name: Optional target name to organize backups per-target.
                        If provided, backups are stored in CONFIG/backups/{target_name}/{timestamp}/
            max_backups_per_target: Maximum number of backups to keep per target 
                                   (None = use config/default, 0 = no limit)
        
        Returns:
            List of backup file paths
        """
        import shutil
        from datetime import datetime
        import json
        
        # Use configured backup directory (set in __init__)
        base_backup_dir = self.backup_dir
        
        # Use config value if not explicitly provided
        if max_backups_per_target is None:
            max_backups_per_target = self.max_backups_per_target
        
        # Generate high-resolution timestamp to avoid collisions
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
        
        # Organize by target if provided
        if target_name:
            # Sanitize target name for filesystem (remove invalid chars)
            safe_target = "".join(c for c in target_name if c.isalnum() or c in ('_', '-', '.'))[:50]
            target_backup_dir = base_backup_dir / safe_target
            snapshot_dir = target_backup_dir / timestamp
        else:
            # Legacy flat mode (warn about this)
            logger.warning(
                "Backup created with no target_name; using legacy flat layout. "
                "Consider passing target_name for better organization."
            )
            snapshot_dir = base_backup_dir / timestamp
        
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        backup_files = []
        
        # Copy config files to snapshot directory
        if self.excluded_features_path.exists():
            backup_path = snapshot_dir / "excluded_features.yaml"
            shutil.copy2(self.excluded_features_path, backup_path)
            backup_files.append(str(backup_path))
            logger.info(f"Backed up excluded_features.yaml to {backup_path}")
        
        if self.feature_registry_path.exists():
            backup_path = snapshot_dir / "feature_registry.yaml"
            shutil.copy2(self.feature_registry_path, backup_path)
            backup_files.append(str(backup_path))
            logger.info(f"Backed up feature_registry.yaml to {backup_path}")
        
        # Log backup creation with full context
        git_commit = self._get_git_commit_hash()
        logger.info(
            f"📦 Backup created: target={target_name or 'N/A'}, "
            f"timestamp={timestamp}, git_commit={git_commit or 'N/A'}, "
            f"source=auto_fix_leakage"
        )
        
        # Create manifest file
        manifest_path = snapshot_dir / "manifest.json"
        try:
            manifest = {
                "backup_version": 1,
                "source": "auto_fix_leakage",
                "target_name": target_name,
                "timestamp": timestamp,
                "backup_files": backup_files,
                "excluded_features_path": str(self.excluded_features_path),
                "feature_registry_path": str(self.feature_registry_path),
                "git_commit": self._get_git_commit_hash()
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            backup_files.append(str(manifest_path))
        except Exception as e:
            logger.debug(f"Could not create manifest file: {e}")
        
        # Apply retention policy (prune old backups for this target)
        if target_name and max_backups_per_target > 0:
            pruned_count = self._prune_old_backups(target_backup_dir, max_backups_per_target)
            if pruned_count > 0:
                logger.info(
                    f"🧹 Pruned {pruned_count} old backup(s) for target={target_name} "
                    f"(kept {max_backups_per_target} most recent)"
                )
        
        return backup_files
    
    def _prune_old_backups(self, target_backup_dir: Path, max_backups: int) -> int:
        """
        Prune old backups for a target, keeping only the most recent N.
        
        Args:
            target_backup_dir: Directory containing backups for a target
            max_backups: Maximum number of backups to keep
        
        Returns:
            Number of backups pruned
        """
        if not target_backup_dir.exists():
            return 0
        
        try:
            # Get all timestamp subdirectories
            backup_dirs = [
                d for d in target_backup_dir.iterdir()
                if d.is_dir() and d.name.replace('_', '').replace('.', '').isdigit()
            ]
            
            if len(backup_dirs) <= max_backups:
                return 0  # No pruning needed
            
            # Sort by timestamp (directory name is timestamp)
            backup_dirs.sort(key=lambda d: d.name, reverse=True)
            
            # Remove oldest backups
            to_remove = backup_dirs[max_backups:]
            pruned_count = 0
            for old_backup in to_remove:
                try:
                    import shutil
                    shutil.rmtree(old_backup)
                    pruned_count += 1
                    logger.debug(f"Pruned old backup: {old_backup.name}")
                except Exception as e:
                    logger.warning(f"Could not prune backup {old_backup}: {e}")
            
            return pruned_count
        except Exception as e:
            logger.debug(f"Could not prune backups: {e}")
            return 0
    
    def _apply_excluded_features_updates(self, updates: Dict[str, Any]):
        """Apply updates to excluded_features.yaml."""
        if not self.excluded_features_path.exists():
            logger.warning(f"excluded_features.yaml not found at {self.excluded_features_path}, creating new file")
            config = {
                'always_exclude': {
                    'regex_patterns': [],
                    'prefix_patterns': [],
                    'keyword_patterns': [],
                    'exact_patterns': []
                }
            }
        else:
            with open(self.excluded_features_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        # Ensure structure exists
        if 'always_exclude' not in config:
            config['always_exclude'] = {}
        
        always_exclude = config['always_exclude']
        
        # Add exact patterns
        existing_exact = set(always_exclude.get('exact_patterns', []))
        new_exact = set(updates.get('exact_patterns', []))
        always_exclude['exact_patterns'] = sorted(list(existing_exact | new_exact))
        
        # Add prefix patterns
        existing_prefix = set(always_exclude.get('prefix_patterns', []))
        new_prefix = set(updates.get('prefix_patterns', []))
        always_exclude['prefix_patterns'] = sorted(list(existing_prefix | new_prefix))
        
        # Write back
        with open(self.excluded_features_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated {self.excluded_features_path}: added {len(new_exact)} exact patterns, {len(new_prefix)} prefix patterns")
    
    def _apply_feature_registry_updates(self, updates: Dict[str, Any]):
        """Apply updates to feature_registry.yaml."""
        if not self.feature_registry_path.exists():
            logger.warning(f"feature_registry.yaml not found at {self.feature_registry_path}, creating new file")
            config = {'features': {}}
        else:
            with open(self.feature_registry_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        if 'features' not in config:
            config['features'] = {}
        
        # Mark features as rejected
        for feat_name in updates.get('rejected_features', []):
            if feat_name not in config['features']:
                config['features'][feat_name] = {
                    'lag_bars': 0,
                    'allowed_horizons': [],
                    'source': 'unknown',
                    'rejected': True,
                    'description': f'AUTO-REJECTED: Detected as leaky feature'
                }
            else:
                # Update existing entry
                config['features'][feat_name]['rejected'] = True
                config['features'][feat_name]['allowed_horizons'] = []
        
        # Write back
        with open(self.feature_registry_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated {self.feature_registry_path}: marked {len(updates.get('rejected_features', []))} features as rejected")
    
    def run_auto_fix_loop(
        self,
        training_function,  # Function that runs training and returns (X, y, feature_names, model_importance, scores)
        max_iterations: int = 5,
        min_confidence: float = 0.7,
        target_column: str = None,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Run training in a loop, detecting and fixing leaks until clean.
        
        Args:
            training_function: Function that runs training and returns results
            max_iterations: Maximum number of fix iterations
            min_confidence: Minimum confidence to auto-fix
            target_column: Target column name
            **training_kwargs: Additional arguments to pass to training function
        
        Returns:
            Dict with final results and fix history
        """
        self.iteration_count = 0
        fix_history = []
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration + 1
            logger.info(f"\n{'='*70}")
            logger.info(f"Auto-Fix Iteration {self.iteration_count}/{max_iterations}")
            logger.info(f"{'='*70}")
            
            # Run training
            logger.info("Running training...")
            training_results = training_function(**training_kwargs)
            
            # Extract results
            X = training_results.get('X')
            y = training_results.get('y')
            feature_names = training_results.get('feature_names', [])
            model_importance = training_results.get('model_importance', {})
            train_score = training_results.get('train_score')
            test_score = training_results.get('test_score')
            symbols = training_results.get('symbols')
            task_type = training_results.get('task_type', 'classification')
            data_interval_minutes = training_results.get('data_interval_minutes', 5)
            
            # Detect leaks
            logger.info("Detecting leaking features...")
            detections = self.detect_leaking_features(
                X=X, y=y, feature_names=feature_names,
                target_column=target_column or training_results.get('target_column', 'unknown'),
                symbols=symbols, task_type=task_type,
                data_interval_minutes=data_interval_minutes,
                model_importance=model_importance,
                train_score=train_score, test_score=test_score
            )
            
            if not detections:
                logger.info("✅ No leaks detected! Training is clean.")
                return {
                    'success': True,
                    'iterations': self.iteration_count,
                    'fix_history': fix_history,
                    'final_results': training_results
                }
            
            # Check if we've seen these leaks before (avoid infinite loop)
            leak_names = {d.feature_name for d in detections}
            if leak_names.issubset(set(self.detected_leaks.keys())):
                logger.warning(f"⚠️  Same leaks detected again - may need manual intervention")
                logger.warning(f"   Detected: {leak_names}")
                break
            
            # Record detected leaks
            for det in detections:
                self.detected_leaks[det.feature_name] = det
            
            # Apply fixes
            logger.info(f"Applying fixes for {len(detections)} leaks...")
            updates = self.apply_fixes(
                detections, 
                min_confidence=min_confidence, 
                dry_run=False,
                target_name=target_column,  # Use target_column from training_kwargs if available
                max_backups_per_target=20
            )
            
            fix_history.append({
                'iteration': self.iteration_count,
                'detections': [d.__dict__ for d in detections],
                'updates': updates
            })
            
            logger.info(f"✅ Applied fixes. Re-running training in next iteration...")
        
        logger.warning(f"⚠️  Reached max iterations ({max_iterations}). Some leaks may remain.")
        return {
            'success': False,
            'iterations': self.iteration_count,
            'fix_history': fix_history,
            'remaining_leaks': list(self.detected_leaks.keys())
        }
    
    @staticmethod
    def list_backups(target_name: Optional[str] = None, backup_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        List available backups for a target (or all targets if None).
        
        Args:
            target_name: Target name to list backups for (None = all targets)
            backup_dir: Backup directory (default: CONFIG/backups)
        
        Returns:
            List of backup info dicts with: target_name, timestamp, manifest_path, snapshot_dir
        """
        import json
        
        if backup_dir is None:
            backup_dir = _REPO_ROOT / "CONFIG" / "backups"
        
        if not backup_dir.exists():
            return []
        
        backups = []
        
        if target_name:
            # List backups for specific target
            safe_target = "".join(c for c in target_name if c.isalnum() or c in ('_', '-', '.'))[:50]
            target_dir = backup_dir / safe_target
            if target_dir.exists():
                for snapshot_dir in target_dir.iterdir():
                    if snapshot_dir.is_dir():
                        manifest_path = snapshot_dir / "manifest.json"
                        if manifest_path.exists():
                            try:
                                with open(manifest_path, 'r') as f:
                                    manifest = json.load(f)
                                backups.append({
                                    'target_name': manifest.get('target_name'),
                                    'timestamp': manifest.get('timestamp'),
                                    'manifest_path': str(manifest_path),
                                    'snapshot_dir': str(snapshot_dir),
                                    'git_commit': manifest.get('git_commit'),
                                    'source': manifest.get('source')
                                })
                            except Exception:
                                pass
        else:
            # List all backups across all targets
            for target_dir in backup_dir.iterdir():
                if target_dir.is_dir():
                    for snapshot_dir in target_dir.iterdir():
                        if snapshot_dir.is_dir():
                            manifest_path = snapshot_dir / "manifest.json"
                            if manifest_path.exists():
                                try:
                                    with open(manifest_path, 'r') as f:
                                        manifest = json.load(f)
                                    backups.append({
                                        'target_name': manifest.get('target_name'),
                                        'timestamp': manifest.get('timestamp'),
                                        'manifest_path': str(manifest_path),
                                        'snapshot_dir': str(snapshot_dir),
                                        'git_commit': manifest.get('git_commit'),
                                        'source': manifest.get('source')
                                    })
                                except Exception:
                                    pass
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda b: b.get('timestamp', ''), reverse=True)
        return backups
    
    @staticmethod
    def restore_backup(
        target_name: str,
        timestamp: Optional[str] = None,
        backup_dir: Optional[Path] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Restore config files from a backup.
        
        Args:
            target_name: Target name
            timestamp: Timestamp of backup to restore (None = most recent)
            backup_dir: Backup directory (default: CONFIG/backups)
            dry_run: If True, only show what would be restored without actually restoring
        
        Returns:
            True if restore succeeded, False otherwise
        """
        import shutil
        import json
        
        if backup_dir is None:
            backup_dir = _REPO_ROOT / "CONFIG" / "backups"
        
        safe_target = "".join(c for c in target_name if c.isalnum() or c in ('_', '-', '.'))[:50]
        target_backup_dir = backup_dir / safe_target
        
        if not target_backup_dir.exists():
            logger.error(f"❌ No backups found for target: {target_name}")
            logger.error(f"   Backup directory does not exist: {target_backup_dir}")
            return False
        
        # Find backup to restore
        if timestamp:
            snapshot_dir = target_backup_dir / timestamp
            if not snapshot_dir.exists():
                # List available timestamps for better error message
                available = LeakageAutoFixer.list_backups(target_name=target_name, backup_dir=backup_dir)
                available_timestamps = [b['timestamp'] for b in available]
                logger.error(f"❌ Backup not found: {target_name}/{timestamp}")
                if available_timestamps:
                    logger.error(f"   Available timestamps for {target_name}:")
                    for ts in available_timestamps[:10]:  # Show first 10
                        logger.error(f"     - {ts}")
                    if len(available_timestamps) > 10:
                        logger.error(f"     ... and {len(available_timestamps) - 10} more")
                else:
                    logger.error(f"   No backups found for target: {target_name}")
                return False
        else:
            # Find most recent backup
            backups = LeakageAutoFixer.list_backups(target_name=target_name, backup_dir=backup_dir)
            if not backups:
                logger.error(f"❌ No backups found for target: {target_name}")
                logger.error(f"   Backup directory exists but contains no valid backups: {target_backup_dir}")
                return False
            snapshot_dir = Path(backups[0]['snapshot_dir'])
            timestamp = backups[0]['timestamp']
            logger.info(f"📦 Using most recent backup: {timestamp} (git: {backups[0].get('git_commit', 'N/A')})")
        
        # Load manifest
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"❌ Manifest not found in backup: {snapshot_dir}")
            logger.error(f"   Expected manifest at: {manifest_path}")
            logger.error(f"   This backup may be corrupted or incomplete")
            return False
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Manifest is malformed (invalid JSON): {manifest_path}")
            logger.error(f"   Error: {e}")
            logger.error(f"   Cannot restore from corrupted backup")
            return False
        except Exception as e:
            logger.error(f"❌ Could not load manifest: {e}")
            logger.error(f"   Manifest path: {manifest_path}")
            return False
        
        # Validate manifest structure
        required_fields = ['excluded_features_path', 'feature_registry_path']
        missing_fields = [f for f in required_fields if f not in manifest]
        if missing_fields:
            logger.error(f"❌ Manifest missing required fields: {missing_fields}")
            logger.error(f"   Manifest version: {manifest.get('backup_version', 'unknown')}")
            logger.error(f"   Cannot restore from incomplete backup")
            return False
        
        # Restore files with atomic writes
        excluded_features_backup = snapshot_dir / "excluded_features.yaml"
        feature_registry_backup = snapshot_dir / "feature_registry.yaml"
        
        excluded_features_path = Path(manifest['excluded_features_path'])
        feature_registry_path = Path(manifest['feature_registry_path'])
        
        restored = []
        import os
        import tempfile
        
        # Atomic restore helper
        def atomic_restore(backup_file: Path, target_path: Path, file_name: str) -> bool:
            """Restore a file atomically (write to temp, then atomic rename)."""
            if not backup_file.exists():
                logger.warning(f"⚠️  Backup file not found: {backup_file}")
                return False
            
            if dry_run:
                logger.info(f"[DRY RUN] Would restore: {backup_file} -> {target_path}")
                return True
            
            try:
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to temporary file first
                temp_suffix = f".tmp-{os.getpid()}-{os.urandom(4).hex()}"
                temp_path = target_path.parent / f"{target_path.name}{temp_suffix}"
                
                # Copy backup to temp file
                shutil.copy2(backup_file, temp_path)
                
                # Atomic rename (POSIX: rename is atomic)
                os.replace(temp_path, target_path)
                
                restored.append(file_name)
                logger.info(f"✅ Restored: {target_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to restore {file_name}: {e}")
                # Clean up temp file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                return False
        
        # Restore both files
        atomic_restore(excluded_features_backup, excluded_features_path, 'excluded_features.yaml')
        atomic_restore(feature_registry_backup, feature_registry_path, 'feature_registry.yaml')
        
        if restored:
            logger.info(
                f"✅ Restored {len(restored)} config file(s) from backup "
                f"(target={target_name}, timestamp={timestamp}, "
                f"git_commit={manifest.get('git_commit', 'N/A')})"
            )
            return True
        else:
            logger.warning("⚠️  No files were restored")
            return False


def auto_fix_leakage(
    training_function,
    target_column: str,
    max_iterations: int = 5,
    min_confidence: float = 0.7,
    **training_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run auto-fix loop.
    
    Example:
        def my_training():
            # ... run training ...
            return {
                'X': X, 'y': y, 'feature_names': feature_names,
                'model_importance': importance_dict,
                'train_score': 0.99, 'test_score': 0.85,
                'symbols': symbols_series, 'task_type': 'classification'
            }
        
        results = auto_fix_leakage(my_training, target_column='y_will_peak_60m_0.8')
    """
    fixer = LeakageAutoFixer()
    return fixer.run_auto_fix_loop(
        training_function=training_function,
        max_iterations=max_iterations,
        min_confidence=min_confidence,
        target_column=target_column,
        **training_kwargs
    )

