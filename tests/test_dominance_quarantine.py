"""
Unit tests for dominance quarantine system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, List

from TRAINING.ranking.utils.dominance_quarantine import (
    DominanceConfig,
    Suspect,
    ConfirmResult,
    detect_suspects,
    confirm_quarantine,
    write_suspects_artifact_with_data,
    persist_confirmed_quarantine,
    load_confirmed_quarantine,
    _dominance_metrics
)


class TestDominanceMetrics:
    """Test dominance metrics computation."""
    
    def test_dominance_metrics_normal(self):
        """Test normal dominance metrics computation."""
        imp_pct = {
            "feature_a": 40.0,
            "feature_b": 10.0,
            "feature_c": 5.0,
            "feature_d": 45.0  # Top feature
        }
        result = _dominance_metrics(imp_pct)
        assert result is not None
        feature, share, ratio = result
        assert feature == "feature_d"
        assert share == 0.45
        assert ratio == 45.0 / 10.0  # feature_d / feature_b (second highest)
    
    def test_dominance_metrics_insufficient(self):
        """Test with insufficient features."""
        imp_pct = {"feature_a": 100.0}
        result = _dominance_metrics(imp_pct)
        assert result is None
    
    def test_dominance_metrics_empty(self):
        """Test with empty dict."""
        imp_pct = {}
        result = _dominance_metrics(imp_pct)
        assert result is None


class TestDetectSuspects:
    """Test suspect detection."""
    
    def test_detect_suspects_hard_threshold(self):
        """Test detection with hard threshold (40%+)."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        per_model_importance_pct = {
            "lightgbm": {
                "leaky_feature": 45.0,  # 45% - exceeds hard threshold
                "feature_b": 10.0,
                "feature_c": 5.0
            }
        }
        
        suspects = detect_suspects(per_model_importance_pct, cfg)
        assert len(suspects) == 1
        assert suspects[0].feature == "leaky_feature"
        assert suspects[0].top1_share == 0.45
        assert suspects[0].model_name == "lightgbm"
    
    def test_detect_suspects_soft_threshold(self):
        """Test detection with soft threshold (30%+ and 3× ratio)."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        per_model_importance_pct = {
            "xgboost": {
                "suspicious_feature": 35.0,  # 35% (exceeds 30%) and 35/10 = 3.5× (exceeds 3×)
                "feature_b": 10.0,
                "feature_c": 5.0
            }
        }
        
        suspects = detect_suspects(per_model_importance_pct, cfg)
        assert len(suspects) == 1
        assert suspects[0].feature == "suspicious_feature"
    
    def test_detect_suspects_no_suspects(self):
        """Test when no suspects are detected."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        per_model_importance_pct = {
            "lightgbm": {
                "feature_a": 25.0,  # 25% (below 30%)
                "feature_b": 20.0,  # Ratio 25/20 = 1.25× (below 3×)
                "feature_c": 15.0
            }
        }
        
        suspects = detect_suspects(per_model_importance_pct, cfg)
        assert len(suspects) == 0
    
    def test_detect_suspects_max_features(self):
        """Test max_features limit."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=2,  # Limit to 2
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        per_model_importance_pct = {
            "model1": {"feature_a": 50.0, "feature_b": 10.0},
            "model2": {"feature_c": 45.0, "feature_d": 10.0},
            "model3": {"feature_e": 40.0, "feature_f": 10.0}
        }
        
        suspects = detect_suspects(per_model_importance_pct, cfg)
        assert len(suspects) <= 2
    
    def test_detect_suspects_disabled(self):
        """Test when dominance quarantine is disabled."""
        cfg = DominanceConfig(
            enabled=False,  # Disabled
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        per_model_importance_pct = {
            "lightgbm": {"leaky_feature": 50.0, "feature_b": 10.0}
        }
        
        suspects = detect_suspects(per_model_importance_pct, cfg)
        assert len(suspects) == 0


class TestConfirmQuarantine:
    """Test confirm quarantine logic."""
    
    def test_confirm_quarantine_confirmed_abs_drop(self):
        """Test confirmation with absolute score drop."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,  # Drop of 0.20 exceeds 0.15
            mean_score_drop_rel=0.25
        )
        
        suspects = [Suspect("lightgbm", "leaky_feature", 0.45, 4.5)]
        
        result = confirm_quarantine(
            pre_mean_score=0.80,
            post_mean_score=0.60,  # Drop of 0.20
            suspects=suspects,
            n_samples=1000,
            n_symbols=5,
            cfg=cfg
        )
        
        assert result.confirmed is True
        assert result.reason == "score_collapse"
        assert result.drop_abs == 0.20
        assert abs(result.drop_rel - 0.25) < 0.01  # 0.20 / 0.80 = 0.25
    
    def test_confirm_quarantine_confirmed_rel_drop(self):
        """Test confirmation with relative score drop."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25  # Drop of 30% exceeds 25%
        )
        
        suspects = [Suspect("xgboost", "suspicious_feature", 0.35, 3.5)]
        
        result = confirm_quarantine(
            pre_mean_score=0.50,
            post_mean_score=0.35,  # Drop of 0.15 (30% relative)
            suspects=suspects,
            n_samples=1000,
            n_symbols=5,
            cfg=cfg
        )
        
        assert result.confirmed is True
        assert result.drop_rel == 0.30
    
    def test_confirm_quarantine_not_confirmed(self):
        """Test when confirmation fails (no significant drop)."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        suspects = [Suspect("lightgbm", "feature_a", 0.35, 3.5)]
        
        result = confirm_quarantine(
            pre_mean_score=0.80,
            post_mean_score=0.75,  # Drop of 0.05 (below both thresholds)
            suspects=suspects,
            n_samples=1000,
            n_symbols=5,
            cfg=cfg
        )
        
        assert result.confirmed is False
        assert result.reason == "no_collapse"
    
    def test_confirm_quarantine_insufficient_data(self):
        """Test when data is insufficient for confirmation."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=True,
            rerun_once=True,
            min_samples=500,  # Need 500
            min_symbols=3,  # Need 3
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        suspects = [Suspect("lightgbm", "feature_a", 0.45, 4.5)]
        
        result = confirm_quarantine(
            pre_mean_score=0.80,
            post_mean_score=0.60,
            suspects=suspects,
            n_samples=300,  # Below min_samples
            n_symbols=5,
            cfg=cfg
        )
        
        assert result.confirmed is False
        assert result.reason == "insufficient_data_for_confirm"
    
    def test_confirm_quarantine_disabled(self):
        """Test when confirm is disabled."""
        cfg = DominanceConfig(
            enabled=True,
            top1_share=0.30,
            top1_over_top2=3.0,
            hard_top1_share=0.40,
            max_features=3,
            confirm_enabled=False,  # Disabled
            rerun_once=True,
            min_samples=500,
            min_symbols=3,
            mean_score_drop_abs=0.15,
            mean_score_drop_rel=0.25
        )
        
        suspects = [Suspect("lightgbm", "feature_a", 0.45, 4.5)]
        
        result = confirm_quarantine(
            pre_mean_score=0.80,
            post_mean_score=0.60,
            suspects=suspects,
            n_samples=1000,
            n_symbols=5,
            cfg=cfg
        )
        
        assert result.confirmed is False
        assert result.reason == "confirm_disabled"


class TestArtifactWriting:
    """Test artifact writing and loading."""
    
    def test_write_and_load_suspects(self):
        """Test writing and loading suspects artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            target = "fwd_ret_10m"
            view = "CROSS_SECTIONAL"
            
            suspects = [
                Suspect("lightgbm", "leaky_feature", 0.45, 4.5),
                Suspect("xgboost", "suspicious_feature", 0.35, 3.5)
            ]
            
            # Write suspects
            artifact_path = write_suspects_artifact_with_data(
                out_dir=out_dir,
                target=target,
                view=view,
                suspects=suspects,
                symbol=None
            )
            
            assert artifact_path.exists()
            
            # Verify content
            with open(artifact_path, 'r') as f:
                data = json.load(f)
            
            assert data["target"] == target
            assert data["view"] == view
            assert len(data["suspects"]) == 2
            assert data["suspects"][0]["feature"] == "leaky_feature"
    
    def test_write_and_load_confirmed_quarantine(self):
        """Test writing and loading confirmed quarantine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            target = "fwd_ret_10m"
            view = "CROSS_SECTIONAL"
            
            suspects = [
                Suspect("lightgbm", "leaky_feature", 0.45, 4.5),
                Suspect("xgboost", "leaky_feature", 0.40, 4.0)  # Same feature, different model
            ]
            
            # Write confirmed quarantine
            artifact_path = persist_confirmed_quarantine(
                out_dir=out_dir,
                target=target,
                suspects=suspects,
                view=view,
                symbol=None
            )
            
            assert artifact_path.exists()
            
            # Load confirmed quarantine
            confirmed_features = load_confirmed_quarantine(
                out_dir=out_dir,
                target=target,
                view=view,
                symbol=None
            )
            
            assert len(confirmed_features) == 1  # Only one unique feature
            assert "leaky_feature" in confirmed_features
            
            # Verify content
            with open(artifact_path, 'r') as f:
                data = json.load(f)
            
            assert data["target"] == target
            assert len(data["confirmed_features"]) == 1
            assert "leaky_feature" in data["confirmed_features"]
    
    def test_load_nonexistent_quarantine(self):
        """Test loading non-existent quarantine returns empty set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            
            confirmed_features = load_confirmed_quarantine(
                out_dir=out_dir,
                target="nonexistent_target",
                view="CROSS_SECTIONAL",
                symbol=None
            )
            
            assert len(confirmed_features) == 0

