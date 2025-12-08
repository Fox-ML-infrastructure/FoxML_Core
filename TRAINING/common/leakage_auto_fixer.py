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
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from TRAINING.common.leakage_sentinels import LeakageSentinel, SentinelResult
from TRAINING.common.importance_diff_detector import ImportanceDiffDetector
from TRAINING.utils.leakage_filtering import filter_features_for_target, _load_leakage_config

logger = logging.getLogger(__name__)


@dataclass
class LeakageDetection:
    """Result of leakage detection for a single feature."""
    feature_name: str
    confidence: float  # 0.0 to 1.0
    reason: str  # Why it's considered a leak
    source: str  # Which detector found it (sentinels, importance, diff)
    suggested_action: str  # 'exact', 'prefix', 'regex', 'registry_reject'


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
        if excluded_features_path is None:
            excluded_features_path = _REPO_ROOT / "CONFIG" / "excluded_features.yaml"
        if feature_registry_path is None:
            feature_registry_path = _REPO_ROOT / "CONFIG" / "feature_registry.yaml"
        
        self.excluded_features_path = Path(excluded_features_path)
        self.feature_registry_path = Path(feature_registry_path)
        self.backup_configs = backup_configs
        
        # Track detected leaks across iterations
        self.detected_leaks: Dict[str, LeakageDetection] = {}
        self.iteration_count = 0
    
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
        
        Returns:
            List of LeakageDetection objects
        """
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
                    if feat_name in feature_names:
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
                for feat_name in feature_names:
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
                            if feat_name in feature_names:
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
                                        if feat_name in feature_names:
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
                            if feat_name in feature_names:
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
            for feat_name in feature_names:
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
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply detected fixes to config files.
        
        Args:
            detections: List of detected leaks
            min_confidence: Minimum confidence to auto-fix (default: 0.7)
            dry_run: If True, don't actually modify files, just return what would be done
        
        Returns:
            Dict with 'excluded_features_updates' and 'feature_registry_updates'
        """
        # Filter by confidence
        high_confidence = [d for d in detections if d.confidence >= min_confidence]
        
        if not high_confidence:
            logger.info(f"No leaks detected with confidence >= {min_confidence}")
            return {'excluded_features_updates': {}, 'feature_registry_updates': {}}
        
        logger.info(f"Auto-fixing {len(high_confidence)} leaks (confidence >= {min_confidence})")
        
        # Backup configs if requested
        if self.backup_configs and not dry_run:
            self._backup_configs()
        
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
        
        if not dry_run:
            self._apply_excluded_features_updates(updates['excluded_features_updates'])
            self._apply_feature_registry_updates(updates['feature_registry_updates'])
        
        return updates
    
    def _backup_configs(self):
        """Backup config files before modification."""
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.excluded_features_path.exists():
            backup_path = self.excluded_features_path.with_suffix(f'.yaml.backup_{timestamp}')
            shutil.copy2(self.excluded_features_path, backup_path)
            logger.info(f"Backed up excluded_features.yaml to {backup_path}")
        
        if self.feature_registry_path.exists():
            backup_path = self.feature_registry_path.with_suffix(f'.yaml.backup_{timestamp}')
            shutil.copy2(self.feature_registry_path, backup_path)
            logger.info(f"Backed up feature_registry.yaml to {backup_path}")
    
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
            updates = self.apply_fixes(detections, min_confidence=min_confidence, dry_run=False)
            
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

