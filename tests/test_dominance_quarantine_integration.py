"""
Integration test for dominance quarantine full flow: suspect → confirm → quarantine → escalation
"""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

from TRAINING.ranking.utils.dominance_quarantine import (
    DominanceConfig,
    detect_suspects,
    confirm_quarantine,
    write_suspects_artifact_with_data,
    persist_confirmed_quarantine,
    load_confirmed_quarantine
)


class TestDominanceQuarantineIntegration:
    """Integration test for full dominance quarantine workflow."""
    
    def test_full_workflow_suspect_confirm_quarantine(self):
        """Test full workflow: suspect detection → confirm → quarantine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            target = "fwd_ret_10m"
            view = "CROSS_SECTIONAL"
            
            # Step 1: Detect suspects
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
                },
                "xgboost": {
                    "leaky_feature": 40.0,  # Same feature, different model
                    "feature_b": 12.0,
                    "feature_c": 8.0
                }
            }
            
            suspects = detect_suspects(per_model_importance_pct, cfg)
            assert len(suspects) >= 1
            assert any(s.feature == "leaky_feature" for s in suspects)
            
            # Step 2: Write suspects artifact
            artifact_path = write_suspects_artifact_with_data(
                out_dir=out_dir,
                target=target,
                view=view,
                suspects=suspects,
                symbol=None
            )
            assert artifact_path.exists()
            
            # Step 3: Confirm quarantine (simulate rerun with suspects removed)
            confirm_result = confirm_quarantine(
                pre_mean_score=0.80,
                post_mean_score=0.60,  # Significant drop
                suspects=suspects,
                n_samples=1000,
                n_symbols=5,
                cfg=cfg
            )
            
            assert confirm_result.confirmed is True
            
            # Step 4: Persist confirmed quarantine
            if confirm_result.confirmed:
                confirmed_path = persist_confirmed_quarantine(
                    out_dir=out_dir,
                    target=target,
                    suspects=suspects,
                    view=view,
                    symbol=None
                )
                assert confirmed_path.exists()
                
                # Step 5: Load confirmed quarantine (simulate downstream stage)
                confirmed_features = load_confirmed_quarantine(
                    out_dir=out_dir,
                    target=target,
                    view=view,
                    symbol=None
                )
                
                assert len(confirmed_features) > 0
                assert "leaky_feature" in confirmed_features
                
                # Step 6: Verify escalation policy (quarantine exists, so don't block)
                # This would be checked in training_router, but we verify the artifact exists
                assert confirmed_path.exists()
    
    def test_workflow_not_confirmed(self):
        """Test workflow when confirm fails (no significant drop)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            target = "fwd_ret_5d"
            view = "CROSS_SECTIONAL"
            
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
            
            # Detect suspects
            per_model_importance_pct = {
                "lightgbm": {"feature_a": 35.0, "feature_b": 10.0}
            }
            suspects = detect_suspects(per_model_importance_pct, cfg)
            assert len(suspects) == 1
            
            # Write suspects
            write_suspects_artifact_with_data(
                out_dir=out_dir,
                target=target,
                view=view,
                suspects=suspects,
                symbol=None
            )
            
            # Confirm fails (no significant drop)
            confirm_result = confirm_quarantine(
                pre_mean_score=0.80,
                post_mean_score=0.75,  # Small drop
                suspects=suspects,
                n_samples=1000,
                n_symbols=5,
                cfg=cfg
            )
            
            assert confirm_result.confirmed is False
            
            # No confirmed quarantine should be persisted
            confirmed_features = load_confirmed_quarantine(
                out_dir=out_dir,
                target=target,
                view=view,
                symbol=None
            )
            assert len(confirmed_features) == 0

