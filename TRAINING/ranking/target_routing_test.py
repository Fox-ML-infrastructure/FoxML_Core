# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Unit tests for task-aware routing logic.

Tests that routing correctly handles:
- Regression targets with negative R² but positive IC
- Classification targets (unchanged behavior)
- Suspicious detection with tstat checks
"""

import unittest
from typing import Dict, Any
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore
from TRAINING.ranking.target_routing import _compute_single_target_routing_decision
from TRAINING.common.utils.task_types import TaskType


class TestTaskAwareRouting(unittest.TestCase):
    """Test task-aware routing with skill01"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.default_config = {
            'skill01_threshold': 0.65,
            'symbol_skill01_threshold': 0.60,
            'frac_symbols_good_threshold': 0.5,
            'suspicious_skill01': 0.90,
            'suspicious_symbol_skill01': 0.95
        }
    
    def create_result(self, target: str, task_type: TaskType, primary_metric_mean: float, 
                     auc: float = None, tstat: float = None) -> TargetPredictabilityScore:
        """Helper to create TargetPredictabilityScore for testing"""
        if auc is None:
            # Default: for regression, auc = R² (can be negative)
            # For classification, auc = ROC-AUC (0-1)
            if task_type == TaskType.REGRESSION:
                auc = -0.2  # Negative R² example
            else:
                auc = 0.70  # Classification AUC
        
        result = TargetPredictabilityScore(
            target=target,
            target_column=f"y_{target}",
            task_type=task_type,
            auc=auc,
            std_score=0.1,
            mean_importance=0.5,
            consistency=0.8,
            n_models=3,
            model_scores={'lightgbm': 0.5, 'xgboost': 0.6, 'random_forest': 0.4},
            primary_metric_mean=primary_metric_mean,
            primary_metric_tstat=tstat
        )
        return result
    
    def test_regression_negative_r2_positive_ic(self):
        """Test regression with negative R² but positive IC routes correctly"""
        # Regression: R² = -0.2 (bad), but IC = 0.04 (positive, skill01 = 0.52)
        # skill01 = 0.5 * (0.04 + 1.0) = 0.52
        result_cs = self.create_result(
            target="test_regression",
            task_type=TaskType.REGRESSION,
            primary_metric_mean=0.04,  # IC = 0.04
            auc=-0.2,  # R² = -0.2 (negative, but not used for routing)
            tstat=2.0
        )
        
        sym_results = {}
        
        decision = _compute_single_target_routing_decision(
            target="test_regression",
            result_cs=result_cs,
            sym_results=sym_results
        )
        
        # Should route to default (skill01=0.52 < 0.65 threshold)
        self.assertIsNotNone(decision['route'])
        self.assertEqual(decision['skill01_cs'], 0.52)  # 0.5 * (0.04 + 1.0)
        # auc should still be -0.2 (R²) for backward compatibility
        self.assertEqual(decision['auc'], -0.2)
    
    def test_regression_high_ic_routes_cross_sectional(self):
        """Test regression with high IC routes to CROSS_SECTIONAL"""
        # Regression: IC = 0.10 (skill01 = 0.55), but skill01 < 0.65, so should default
        # Let's use IC = 0.30 (skill01 = 0.65) to hit threshold
        result_cs = self.create_result(
            target="test_regression_high",
            task_type=TaskType.REGRESSION,
            primary_metric_mean=0.30,  # IC = 0.30
            auc=-0.1,  # R² can be negative, doesn't matter
            tstat=3.5
        )
        
        # Create symbol results with good coverage
        sym_results = {
            'AAPL': self.create_result('AAPL', TaskType.REGRESSION, 0.25, auc=-0.05),
            'MSFT': self.create_result('MSFT', TaskType.REGRESSION, 0.28, auc=-0.08),
            'GOOGL': self.create_result('GOOGL', TaskType.REGRESSION, 0.30, auc=-0.10),
        }
        
        decision = _compute_single_target_routing_decision(
            target="test_regression_high",
            result_cs=result_cs,
            sym_results=sym_results
        )
        
        # skill01 = 0.5 * (0.30 + 1.0) = 0.65 (meets threshold)
        self.assertEqual(decision['skill01_cs'], 0.65)
        # Should route to CROSS_SECTIONAL if frac_symbols_good >= 0.5
        # Symbol skill01s: [0.625, 0.64, 0.65] - all >= 0.60, so frac = 1.0 >= 0.5
        self.assertEqual(decision['route'], 'CROSS_SECTIONAL')
    
    def test_classification_unchanged_behavior(self):
        """Test classification routing behavior unchanged"""
        # Classification: AUC = 0.70, AUC-excess = 0.20, skill01 = 0.60
        result_cs = self.create_result(
            target="test_classification",
            task_type=TaskType.BINARY_CLASSIFICATION,
            primary_metric_mean=0.20,  # AUC-excess = 0.20 (AUC - 0.5)
            auc=0.70,  # ROC-AUC
            tstat=4.0
        )
        
        sym_results = {
            'AAPL': self.create_result('AAPL', TaskType.BINARY_CLASSIFICATION, 0.15, auc=0.65),
            'MSFT': self.create_result('MSFT', TaskType.BINARY_CLASSIFICATION, 0.18, auc=0.68),
        }
        
        decision = _compute_single_target_routing_decision(
            target="test_classification",
            result_cs=result_cs,
            sym_results=sym_results
        )
        
        # skill01 = 0.5 * (0.20 + 1.0) = 0.60
        self.assertEqual(decision['skill01_cs'], 0.60)
        # skill01 < 0.65, but symbol skill01s >= 0.60, so should route to SYMBOL_SPECIFIC
        self.assertEqual(decision['route'], 'SYMBOL_SPECIFIC')
    
    def test_suspicious_high_skill01_with_high_tstat_not_blocked(self):
        """Test that high skill01 with high tstat is not blocked (legitimate signal)"""
        # High skill01 (0.95) but stable (tstat=4.0) should not be blocked
        result_cs = self.create_result(
            target="test_stable_high",
            task_type=TaskType.REGRESSION,
            primary_metric_mean=0.90,  # IC = 0.90 (very high)
            auc=0.85,  # R² = 0.85
            tstat=4.0  # High tstat = stable signal
        )
        
        sym_results = {}
        
        decision = _compute_single_target_routing_decision(
            target="test_stable_high",
            result_cs=result_cs,
            sym_results=sym_results
        )
        
        # skill01 = 0.5 * (0.90 + 1.0) = 0.95 (above suspicious threshold)
        self.assertEqual(decision['skill01_cs'], 0.95)
        # Should NOT be blocked because tstat > 3.0 (stable signal)
        self.assertNotEqual(decision['route'], 'BLOCKED')
    
    def test_suspicious_high_skill01_with_low_tstat_blocked(self):
        """Test that high skill01 with low tstat is blocked (suspicious)"""
        # High skill01 (0.95) but unstable (tstat=1.0) should be blocked
        result_cs = self.create_result(
            target="test_unstable_high",
            task_type=TaskType.BINARY_CLASSIFICATION,
            primary_metric_mean=0.90,  # AUC-excess = 0.90 (AUC = 1.40, impossible but for test)
            auc=0.95,  # ROC-AUC
            tstat=1.0  # Low tstat = unstable signal
        )
        
        sym_results = {}
        
        decision = _compute_single_target_routing_decision(
            target="test_unstable_high",
            result_cs=result_cs,
            sym_results=sym_results
        )
        
        # skill01 = 0.5 * (0.90 + 1.0) = 0.95 (above suspicious threshold)
        self.assertEqual(decision['skill01_cs'], 0.95)
        # Should be blocked because tstat < 3.0 (unstable signal)
        self.assertEqual(decision['route'], 'BLOCKED')
    
    def test_skill01_normalization_regression(self):
        """Test skill01 normalization for regression IC"""
        # IC = -0.5 → skill01 = 0.25
        result = self.create_result('test', TaskType.REGRESSION, -0.5, auc=-1.0)
        self.assertAlmostEqual(result.skill01, 0.25, places=5)
        
        # IC = 0.0 → skill01 = 0.50
        result = self.create_result('test', TaskType.REGRESSION, 0.0, auc=-0.5)
        self.assertAlmostEqual(result.skill01, 0.50, places=5)
        
        # IC = 0.5 → skill01 = 0.75
        result = self.create_result('test', TaskType.REGRESSION, 0.5, auc=0.3)
        self.assertAlmostEqual(result.skill01, 0.75, places=5)
    
    def test_skill01_normalization_classification(self):
        """Test skill01 normalization for classification AUC-excess"""
        # AUC-excess = -0.3 → skill01 = 0.35
        result = self.create_result('test', TaskType.BINARY_CLASSIFICATION, -0.3, auc=0.2)
        self.assertAlmostEqual(result.skill01, 0.35, places=5)
        
        # AUC-excess = 0.0 → skill01 = 0.50
        result = self.create_result('test', TaskType.BINARY_CLASSIFICATION, 0.0, auc=0.5)
        self.assertAlmostEqual(result.skill01, 0.50, places=5)
        
        # AUC-excess = 0.3 → skill01 = 0.65
        result = self.create_result('test', TaskType.BINARY_CLASSIFICATION, 0.3, auc=0.8)
        self.assertAlmostEqual(result.skill01, 0.65, places=5)


if __name__ == '__main__':
    unittest.main()
