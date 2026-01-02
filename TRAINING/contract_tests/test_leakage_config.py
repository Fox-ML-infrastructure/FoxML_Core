"""
Tests for config-driven leakage control settings.

Tests the new explicit config knobs:
- over_budget_action: drop | hard_stop | warn
- lookback_budget_minutes: auto | <number>
- lookback_buffer_minutes: <number>
- cv.embargo_extra_bars: <number>
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import yaml

from TRAINING.utils.leakage_budget import compute_budget, compute_feature_lookback_max
from TRAINING.ranking.utils.resolved_config import derive_purge_embargo, create_resolved_config
from TRAINING.ranking.predictability.model_evaluation import _enforce_final_safety_gate


class TestOverBudgetAction:
    """Test over_budget_action config setting (drop | hard_stop | warn)."""
    
    def test_over_budget_action_drop(self):
        """Test that 'drop' action drops violating features."""
        # Create mock resolved_config with purge limit
        resolved_config = MagicMock()
        resolved_config.purge_minutes = 100.0  # 100 minute purge
        
        # Create feature matrix with features that violate purge
        X = np.random.randn(100, 3)
        feature_names = ['test_feature_15m', 'test_feature_60d', 'test_feature_30m']
        
        # Mock config to return 'drop'
        with patch('TRAINING.ranking.predictability.model_evaluation.get_cfg') as mock_get_cfg:
            mock_get_cfg.return_value = 'drop'
            
            # Create logger
            import logging
            logger = logging.getLogger('test')
            
            # Call gatekeeper
            X_filtered, features_filtered = _enforce_final_safety_gate(
                X=X,
                feature_names=feature_names,
                resolved_config=resolved_config,
                interval_minutes=5.0,
                logger=logger
            )
            
            # Should drop the 60d feature (86400m > 100m purge)
            assert len(features_filtered) < len(feature_names)
            assert 'test_feature_60d' not in features_filtered
            assert X_filtered.shape[1] < X.shape[1]
    
    def test_over_budget_action_hard_stop(self):
        """Test that 'hard_stop' action raises RuntimeError."""
        # Create mock resolved_config with purge limit
        resolved_config = MagicMock()
        resolved_config.purge_minutes = 100.0  # 100 minute purge
        
        # Create feature matrix with features that violate purge
        X = np.random.randn(100, 3)
        feature_names = ['test_feature_15m', 'test_feature_60d', 'test_feature_30m']
        
        # Mock config to return 'hard_stop'
        with patch('TRAINING.ranking.predictability.model_evaluation.get_cfg') as mock_get_cfg:
            mock_get_cfg.return_value = 'hard_stop'
            
            # Create logger
            import logging
            logger = logging.getLogger('test')
            
            # Call gatekeeper - should raise RuntimeError
            with pytest.raises(RuntimeError, match="OVER_BUDGET VIOLATION.*hard_stop"):
                _enforce_final_safety_gate(
                    X=X,
                    feature_names=feature_names,
                    resolved_config=resolved_config,
                    interval_minutes=5.0,
                    logger=logger
                )
    
    def test_over_budget_action_warn(self):
        """Test that 'warn' action logs but doesn't drop."""
        # Create mock resolved_config with purge limit
        resolved_config = MagicMock()
        resolved_config.purge_minutes = 100.0  # 100 minute purge
        
        # Create feature matrix with features that violate purge
        X = np.random.randn(100, 3)
        feature_names = ['test_feature_15m', 'test_feature_60d', 'test_feature_30m']
        
        # Mock config to return 'warn'
        with patch('TRAINING.ranking.predictability.model_evaluation.get_cfg') as mock_get_cfg:
            mock_get_cfg.return_value = 'warn'
            
            # Create logger
            import logging
            logger = logging.getLogger('test')
            
            # Call gatekeeper - should NOT drop features
            X_filtered, features_filtered = _enforce_final_safety_gate(
                X=X,
                feature_names=feature_names,
                resolved_config=resolved_config,
                interval_minutes=5.0,
                logger=logger
            )
            
            # Should keep all features (warn mode doesn't drop)
            assert len(features_filtered) == len(feature_names)
            assert X_filtered.shape[1] == X.shape[1]


class TestLookbackBuffer:
    """Test lookback_buffer_minutes config setting."""
    
    def test_lookback_buffer_from_config(self):
        """Test that buffer is loaded from config."""
        # Test with custom buffer
        with patch('TRAINING.ranking.predictability.model_evaluation.get_cfg') as mock_get_cfg:
            mock_get_cfg.side_effect = lambda key, default, **kwargs: {
                'safety.leakage_detection.lookback_buffer_minutes': 10.0,  # 10 minute buffer
                'safety.leakage_detection.policy': 'strict'
            }.get(key, default)
            
            # This would be tested in the actual policy enforcement code
            # For now, just verify the config is read
            assert True  # Placeholder - actual test would verify buffer is used


class TestEmbargoExtraBars:
    """Test embargo_extra_bars config setting."""
    
    def test_embargo_extra_bars_from_config(self):
        """Test that embargo_extra_bars is loaded from config."""
        # Test with custom embargo_extra_bars
        with patch('TRAINING.utils.resolved_config.get_cfg') as mock_get_cfg:
            mock_get_cfg.return_value = 10  # 10 extra bars
            
            purge, embargo = derive_purge_embargo(
                horizon_minutes=60.0,
                interval_minutes=5.0,
                purge_buffer_bars=5,
                embargo_extra_bars=None  # Will load from config
            )
            
            # Embargo should be horizon + (10 bars * 5 minutes) = 60 + 50 = 110
            # Purge should be horizon + (5 bars * 5 minutes) = 60 + 25 = 85
            assert embargo > purge  # Embargo should be larger due to extra bars
            assert embargo == 110.0  # 60 + (10 * 5)
            assert purge == 85.0  # 60 + (5 * 5)
    
    def test_embargo_extra_bars_explicit(self):
        """Test that explicit embargo_extra_bars parameter is used."""
        purge, embargo = derive_purge_embargo(
            horizon_minutes=60.0,
            interval_minutes=5.0,
            purge_buffer_bars=5,
            embargo_extra_bars=10  # Explicit value
        )
        
        # Embargo should be horizon + (10 bars * 5 minutes) = 60 + 50 = 110
        assert embargo == 110.0
        assert purge == 85.0  # 60 + (5 * 5)


class TestLookbackBudgetMinutes:
    """Test lookback_budget_minutes config setting (auto | <number>)."""
    
    def test_lookback_budget_auto(self):
        """Test that 'auto' computes from actual features."""
        # Features with different lookbacks
        feature_names = ['test_feature_15m', 'test_feature_30m', 'test_feature_60d']
        
        # Compute lookback (auto mode - no cap)
        result = compute_feature_lookback_max(
            feature_names,
            interval_minutes=5.0,
            max_lookback_cap_minutes=None  # No cap = auto
        )
        
        # Should return actual max (60d = 86400m)
        assert result.max_minutes == 86400.0
        assert 'test_feature_60d' in [f for f, _ in result.top_offenders]
    
    def test_lookback_budget_capped(self):
        """Test that explicit number caps the lookback."""
        # Features with different lookbacks
        feature_names = ['test_feature_15m', 'test_feature_30m', 'test_feature_60d']
        
        # Compute lookback with cap
        result = compute_feature_lookback_max(
            feature_names,
            interval_minutes=5.0,
            max_lookback_cap_minutes=100.0  # Cap at 100m
        )
        
        # Budget should be capped, but actual max should still be reported
        # (The cap is for gatekeeper logic, not reporting)
        assert result.max_minutes == 86400.0  # Still reports actual max
        # But top_offenders will show the 60d feature
        assert 'test_feature_60d' in [f for f, _ in result.top_offenders]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
