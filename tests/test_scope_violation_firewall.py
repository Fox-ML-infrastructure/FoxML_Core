"""
Test scope violation firewall (PR1).

This killer integration test verifies:
1. OutputLayout enforces view+universe scoping
2. cohort=sy_* artifacts are NEVER written under CROSS_SECTIONAL/ paths
3. cohort=cs_* artifacts are NEVER written under SYMBOL_SPECIFIC/ paths
4. Mode resolution is universe-scoped (different universes can have different modes)
5. validate_cohort_id() catches scope violations at write time
6. validate_universe_sig() prevents view-as-universe bugs
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any

from TRAINING.orchestration.utils.output_layout import (
    OutputLayout,
    validate_cohort_metadata,
    _normalize_universe_sig,
    _normalize_view
)
from TRAINING.orchestration.utils.cohort_metadata import (
    validate_universe_sig,
    build_cohort_metadata,
    CANON_VIEWS
)


class TestOutputLayoutInvariants:
    """Test OutputLayout dataclass invariants."""
    
    def test_valid_cross_sectional_layout(self):
        """Valid CROSS_SECTIONAL layout should work."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc123",
            cohort_id="cs_2024Q1_abc12345"
        )
        assert layout.view == "CROSS_SECTIONAL"
        assert layout.universe_sig == "abc123"
        assert layout.symbol is None
        
    def test_valid_symbol_specific_layout(self):
        """Valid SYMBOL_SPECIFIC layout should work."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="abc123",
            symbol="AAPL",
            cohort_id="sy_2024Q1_abc12345"
        )
        assert layout.view == "SYMBOL_SPECIFIC"
        assert layout.symbol == "AAPL"
    
    def test_invalid_view_raises(self):
        """Invalid view should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid view"):
            OutputLayout(
                output_root=Path("/tmp/test"),
                target="fwd_ret_5d",
                view="INVALID",
                universe_sig="abc123"
            )
    
    def test_symbol_specific_requires_symbol(self):
        """SYMBOL_SPECIFIC without symbol should raise."""
        with pytest.raises(ValueError, match="requires symbol"):
            OutputLayout(
                output_root=Path("/tmp/test"),
                target="fwd_ret_5d",
                view="SYMBOL_SPECIFIC",
                universe_sig="abc123"
            )
    
    def test_cross_sectional_cannot_have_symbol(self):
        """CROSS_SECTIONAL with symbol should raise."""
        with pytest.raises(ValueError, match="cannot have symbol"):
            OutputLayout(
                output_root=Path("/tmp/test"),
                target="fwd_ret_5d",
                view="CROSS_SECTIONAL",
                universe_sig="abc123",
                symbol="AAPL"
            )
    
    def test_universe_sig_required(self):
        """universe_sig is required."""
        with pytest.raises(ValueError, match="universe_sig is required"):
            OutputLayout(
                output_root=Path("/tmp/test"),
                target="fwd_ret_5d",
                view="CROSS_SECTIONAL",
                universe_sig=""
            )


class TestCohortIdValidation:
    """Test cohort_id prefix validation against view."""
    
    def test_cross_sectional_accepts_cs_prefix(self):
        """CROSS_SECTIONAL should accept cs_ prefix."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc123",
            cohort_id="cs_2024Q1_abc12345"
        )
        # Should not raise
        layout.validate_cohort_id("cs_2024Q1_abc12345")
    
    def test_cross_sectional_rejects_sy_prefix(self):
        """CROSS_SECTIONAL should reject sy_ prefix (SCOPE VIOLATION)."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc123"
        )
        with pytest.raises(ValueError, match="Cohort ID scope violation.*sy_.*CROSS_SECTIONAL"):
            layout.validate_cohort_id("sy_2024Q1_abc12345")
    
    def test_symbol_specific_accepts_sy_prefix(self):
        """SYMBOL_SPECIFIC should accept sy_ prefix."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="abc123",
            symbol="AAPL",
            cohort_id="sy_2024Q1_abc12345"
        )
        # Should not raise
        layout.validate_cohort_id("sy_2024Q1_abc12345")
    
    def test_symbol_specific_rejects_cs_prefix(self):
        """SYMBOL_SPECIFIC should reject cs_ prefix (SCOPE VIOLATION)."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="abc123",
            symbol="AAPL"
        )
        with pytest.raises(ValueError, match="Cohort ID scope violation.*cs_.*SYMBOL_SPECIFIC"):
            layout.validate_cohort_id("cs_2024Q1_abc12345")
    
    def test_empty_cohort_id_raises(self):
        """Empty cohort_id should raise."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc123"
        )
        with pytest.raises(ValueError, match="cannot be empty"):
            layout.validate_cohort_id("")


class TestPathGeneration:
    """Test path generation includes universe_sig."""
    
    def test_repro_dir_includes_universe(self):
        """repro_dir should include universe={universe_sig}."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc123"
        )
        repro_dir = layout.repro_dir()
        assert "universe=abc123" in str(repro_dir)
    
    def test_cohort_dir_includes_universe(self):
        """cohort_dir should include universe={universe_sig}."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc123",
            cohort_id="cs_2024Q1_abc12345"
        )
        cohort_dir = layout.cohort_dir()
        assert "universe=abc123" in str(cohort_dir)
        assert "cohort=cs_2024Q1_abc12345" in str(cohort_dir)
    
    def test_symbol_specific_path_includes_symbol(self):
        """SYMBOL_SPECIFIC path should include symbol={symbol}."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="abc123",
            symbol="AAPL",
            cohort_id="sy_2024Q1_abc12345"
        )
        cohort_dir = layout.cohort_dir()
        assert "symbol=AAPL" in str(cohort_dir)
        assert "universe=abc123" in str(cohort_dir)
    
    def test_scope_key_format(self):
        """scope_key should have consistent format."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc123"
        )
        scope_key = layout.scope_key()
        assert scope_key == "view=CROSS_SECTIONAL/universe=abc123"
        
        layout2 = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="abc123",
            symbol="AAPL"
        )
        scope_key2 = layout2.scope_key()
        assert scope_key2 == "view=SYMBOL_SPECIFIC/universe=abc123/symbol=AAPL"


class TestNormalization:
    """Test view and universe_sig normalization."""
    
    def test_normalize_view_uppercase(self):
        """_normalize_view should uppercase valid views."""
        assert _normalize_view({"view": "cross_sectional"}) == "CROSS_SECTIONAL"
        assert _normalize_view({"view": "CROSS_SECTIONAL"}) == "CROSS_SECTIONAL"
        assert _normalize_view({"view": "symbol_specific"}) == "SYMBOL_SPECIFIC"
    
    def test_normalize_view_invalid_returns_none(self):
        """_normalize_view should return None for invalid views."""
        assert _normalize_view({"view": "invalid"}) is None
        assert _normalize_view({"view": "LOSO"}) is None  # LOSO not canonical
        assert _normalize_view({}) is None
    
    def test_normalize_universe_sig_prefers_universe_sig(self):
        """_normalize_universe_sig should prefer universe_sig over universe_id."""
        assert _normalize_universe_sig({"universe_sig": "abc", "universe_id": "xyz"}) == "abc"
    
    def test_normalize_universe_sig_fallback_to_universe_id(self):
        """_normalize_universe_sig should fall back to universe_id."""
        assert _normalize_universe_sig({"universe_id": "xyz"}) == "xyz"
    
    def test_normalize_universe_sig_missing_returns_none(self):
        """_normalize_universe_sig should return None if both missing."""
        assert _normalize_universe_sig({}) is None


class TestValidateCohortMetadata:
    """Test validate_cohort_metadata function."""
    
    def test_valid_cross_sectional_metadata(self):
        """Valid CROSS_SECTIONAL metadata should pass."""
        metadata = {
            "view": "CROSS_SECTIONAL",
            "universe_sig": "abc123",
            "target": "fwd_ret_5d"
        }
        # Should not raise
        validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
    
    def test_valid_symbol_specific_metadata(self):
        """Valid SYMBOL_SPECIFIC metadata should pass."""
        metadata = {
            "view": "SYMBOL_SPECIFIC",
            "universe_sig": "abc123",
            "target": "fwd_ret_5d",
            "symbol": "AAPL"
        }
        # Should not raise
        validate_cohort_metadata(metadata, view="SYMBOL_SPECIFIC", symbol="AAPL")
    
    def test_missing_view_raises(self):
        """Missing view should raise."""
        metadata = {
            "universe_sig": "abc123",
            "target": "fwd_ret_5d"
        }
        with pytest.raises(ValueError, match="missing required fields.*view"):
            validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
    
    def test_invalid_view_raises(self):
        """Invalid view should raise with details."""
        metadata = {
            "view": "invalid",
            "universe_sig": "abc123",
            "target": "fwd_ret_5d"
        }
        with pytest.raises(ValueError, match=r"view \(invalid: invalid\)"):
            validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
    
    def test_missing_universe_sig_raises(self):
        """Missing universe_sig should raise."""
        metadata = {
            "view": "CROSS_SECTIONAL",
            "target": "fwd_ret_5d"
        }
        with pytest.raises(ValueError, match="missing required fields.*universe_sig"):
            validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
    
    def test_missing_target_raises(self):
        """Missing target should raise."""
        metadata = {
            "view": "CROSS_SECTIONAL",
            "universe_sig": "abc123"
        }
        with pytest.raises(ValueError, match="missing required fields.*target"):
            validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
    
    def test_symbol_specific_missing_symbol_raises(self):
        """SYMBOL_SPECIFIC missing symbol should raise."""
        metadata = {
            "view": "SYMBOL_SPECIFIC",
            "universe_sig": "abc123",
            "target": "fwd_ret_5d"
        }
        with pytest.raises(ValueError, match="missing required fields.*symbol"):
            validate_cohort_metadata(metadata, view="SYMBOL_SPECIFIC")
    
    def test_view_mismatch_raises(self):
        """View mismatch between metadata and expected should raise."""
        metadata = {
            "view": "SYMBOL_SPECIFIC",
            "universe_sig": "abc123",
            "target": "fwd_ret_5d",
            "symbol": "AAPL"
        }
        with pytest.raises(ValueError, match="View mismatch"):
            validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
    
    def test_case_insensitive_view_comparison(self):
        """View comparison should be case-insensitive."""
        metadata = {
            "view": "cross_sectional",  # lowercase
            "universe_sig": "abc123",
            "target": "fwd_ret_5d"
        }
        # Should not raise (normalized comparison)
        validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
    
    def test_normalizes_universe_id_to_universe_sig(self):
        """Should normalize universe_id to universe_sig."""
        metadata = {
            "view": "CROSS_SECTIONAL",
            "universe_id": "abc123",  # legacy key
            "target": "fwd_ret_5d"
        }
        validate_cohort_metadata(metadata, view="CROSS_SECTIONAL")
        # After validation, universe_sig should be set
        assert metadata["universe_sig"] == "abc123"


class TestKillerScenario:
    """
    KILLER TEST: Verifies the exact bug that was happening.
    
    Scenario:
    1. Process universe A with CROSS_SECTIONAL view
    2. Process universe B with SYMBOL_SPECIFIC view
    3. Verify no sy_* cohorts appear under CROSS_SECTIONAL paths
    """
    
    def test_no_sy_cohort_under_cross_sectional(self):
        """sy_* cohort should NEVER be under CROSS_SECTIONAL path."""
        # This should fail at validate_cohort_id level
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="universe_a_hash"
        )
        
        # Attempting to use sy_ cohort with CROSS_SECTIONAL view should fail
        with pytest.raises(ValueError, match="Cohort ID scope violation"):
            layout.validate_cohort_id("sy_2024Q1_abc12345")
    
    def test_no_cs_cohort_under_symbol_specific(self):
        """cs_* cohort should NEVER be under SYMBOL_SPECIFIC path."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="universe_b_hash",
            symbol="AAPL"
        )
        
        # Attempting to use cs_ cohort with SYMBOL_SPECIFIC view should fail
        with pytest.raises(ValueError, match="Cohort ID scope violation"):
            layout.validate_cohort_id("cs_2024Q1_abc12345")
    
    def test_different_universes_can_have_different_modes(self):
        """Different universes can have different views (modes)."""
        # Universe A: CROSS_SECTIONAL with cs_ cohort - OK
        layout_a = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="universe_a_hash",
            cohort_id="cs_2024Q1_abc12345"
        )
        layout_a.validate_cohort_id("cs_2024Q1_abc12345")
        
        # Universe B: SYMBOL_SPECIFIC with sy_ cohort - OK
        layout_b = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="universe_b_hash",
            symbol="AAPL",
            cohort_id="sy_2024Q1_xyz67890"
        )
        layout_b.validate_cohort_id("sy_2024Q1_xyz67890")
        
        # Paths should be in completely different directories
        path_a = str(layout_a.repro_dir())
        path_b = str(layout_b.repro_dir())
        
        assert "CROSS_SECTIONAL" in path_a
        assert "SYMBOL_SPECIFIC" in path_b
        assert "universe=universe_a_hash" in path_a
        assert "universe=universe_b_hash" in path_b
        
        # No overlap in paths
        assert "SYMBOL_SPECIFIC" not in path_a
        assert "CROSS_SECTIONAL" not in path_b


class TestUniverseSigFormatGuard:
    """Test validate_universe_sig() format guard from cohort_metadata.py.
    
    This guard prevents the regression where `view` is accidentally passed as `universe_sig`,
    which was causing scope partitioning bugs in feature selection.
    """
    
    def test_view_as_universe_sig_raises(self):
        """Passing view name as universe_sig should raise SCOPE BUG error."""
        with pytest.raises(ValueError, match="SCOPE BUG.*view name"):
            validate_universe_sig("CROSS_SECTIONAL")
        with pytest.raises(ValueError, match="SCOPE BUG.*view name"):
            validate_universe_sig("SYMBOL_SPECIFIC")
    
    def test_none_universe_sig_raises(self):
        """None universe_sig should raise."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_universe_sig(None)
    
    def test_empty_universe_sig_raises(self):
        """Empty string universe_sig should raise."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_universe_sig("")
    
    def test_too_short_universe_sig_raises(self):
        """Too short universe_sig (< 8 chars) should raise."""
        with pytest.raises(ValueError, match="too short"):
            validate_universe_sig("abc")
        with pytest.raises(ValueError, match="too short"):
            validate_universe_sig("1234567")  # 7 chars
    
    def test_path_unsafe_chars_raise(self):
        """Path-unsafe characters should raise."""
        # os.sep (e.g., '/' on Linux)
        with pytest.raises(ValueError, match="path-unsafe"):
            validate_universe_sig("abc" + os.sep + "def12345")
        # Newline
        with pytest.raises(ValueError, match="path-unsafe"):
            validate_universe_sig("abc\ndef12345")
        # Carriage return
        with pytest.raises(ValueError, match="path-unsafe"):
            validate_universe_sig("abc\rdef12345")
        # Tab
        with pytest.raises(ValueError, match="path-unsafe"):
            validate_universe_sig("abc\tdef12345")
        # Space
        with pytest.raises(ValueError, match="path-unsafe"):
            validate_universe_sig("abc def12345")
    
    def test_valid_universe_sig_passes(self):
        """Valid universe_sig should not raise."""
        # Should not raise
        validate_universe_sig("abc12345")  # 8 chars
        validate_universe_sig("universe_a_hash_long_enough")
        validate_universe_sig("1234567890abcdef")  # hex-like
    
    def test_build_cohort_metadata_validates_universe_sig(self):
        """build_cohort_metadata should validate universe_sig."""
        # Should raise on view as universe_sig
        with pytest.raises(ValueError, match="SCOPE BUG"):
            build_cohort_metadata(
                target="fwd_ret_5d",
                view="CROSS_SECTIONAL",
                universe_sig="CROSS_SECTIONAL"  # Bug!
            )
    
    def test_build_cohort_metadata_valid(self):
        """build_cohort_metadata should return correct dict for valid inputs."""
        meta = build_cohort_metadata(
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc12345678"
        )
        assert meta["target"] == "fwd_ret_5d"
        assert meta["view"] == "CROSS_SECTIONAL"
        assert meta["universe_sig"] == "abc12345678"
        assert "symbol" not in meta
    
    def test_build_cohort_metadata_symbol_specific(self):
        """build_cohort_metadata should require symbol for SYMBOL_SPECIFIC."""
        with pytest.raises(ValueError, match="symbol required"):
            build_cohort_metadata(
                target="fwd_ret_5d",
                view="SYMBOL_SPECIFIC",
                universe_sig="abc12345678"
            )
        
        # With symbol
        meta = build_cohort_metadata(
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig="abc12345678",
            symbol="AAPL"
        )
        assert meta["symbol"] == "AAPL"


class TestFeatureSelectionPathsMatchSST:
    """Test that feature selection paths contain universe_sig matching SST value."""
    
    def test_output_layout_repro_dir_contains_universe_sig(self, tmp_path):
        """Artifacts from OutputLayout.repro_dir() should contain universe={sig}."""
        universe_sig = "sst_computed_hash_12345"
        
        layout = OutputLayout(
            output_root=tmp_path,
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig=universe_sig
        )
        
        repro_dir = layout.repro_dir()
        assert f"universe={universe_sig}" in str(repro_dir)
    
    def test_output_layout_feature_importance_dir_contains_universe_sig(self, tmp_path):
        """Artifacts from OutputLayout.feature_importance_dir() should contain universe={sig}."""
        universe_sig = "sst_computed_hash_12345"
        
        layout = OutputLayout(
            output_root=tmp_path,
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig=universe_sig
        )
        
        fi_dir = layout.feature_importance_dir()
        assert f"universe={universe_sig}" in str(fi_dir)
        assert "feature_importances" in str(fi_dir)
    
    def test_symbol_specific_paths_contain_symbol(self, tmp_path):
        """SYMBOL_SPECIFIC paths should contain symbol= partition."""
        universe_sig = "sst_computed_hash_12345"
        
        layout = OutputLayout(
            output_root=tmp_path,
            target="fwd_ret_5d",
            view="SYMBOL_SPECIFIC",
            universe_sig=universe_sig,
            symbol="AAPL"
        )
        
        repro_dir = layout.repro_dir()
        assert f"universe={universe_sig}" in str(repro_dir)
        assert "symbol=AAPL" in str(repro_dir)
        assert "SYMBOL_SPECIFIC" in str(repro_dir)
    
    def test_cross_sectional_paths_do_not_contain_symbol(self, tmp_path):
        """CROSS_SECTIONAL paths should NOT contain symbol= partition."""
        universe_sig = "sst_computed_hash_12345"
        
        layout = OutputLayout(
            output_root=tmp_path,
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig=universe_sig
        )
        
        repro_dir = layout.repro_dir()
        assert "symbol=" not in str(repro_dir)
        assert "CROSS_SECTIONAL" in str(repro_dir)


class TestPatch0SSTViewOverride:
    """
    PATCH 0 Regression Test: Verifies SST view override works correctly.
    
    Scenario: User requests SYMBOL_SPECIFIC with 1 symbol, but min_cs=1 makes CS valid,
    so SST resolves to CROSS_SECTIONAL. The bug was that downstream writers used
    caller's view (SYMBOL_SPECIFIC) instead of SST's resolved_mode (CROSS_SECTIONAL),
    resulting in sy_* cohorts under CROSS_SECTIONAL/ paths.
    
    This test verifies:
    1. When requested_mode=SYMBOL_SPECIFIC but resolved_mode=CROSS_SECTIONAL,
       writers should route to CROSS_SECTIONAL and symbol should be stripped.
    2. If symbol sneaks through to _save_to_cohort with CROSS_SECTIONAL view,
       it should hard-error (Patch 3 invariant).
    """
    
    def test_sst_override_strips_symbol_for_cs(self):
        """When SST resolves to CROSS_SECTIONAL, symbol should be stripped."""
        # Simulate the SST override logic from Patch 0
        caller_view = "SYMBOL_SPECIFIC"
        caller_symbol = "AAPL"
        resolved_mode = "CROSS_SECTIONAL"  # SST resolved to CS because min_cs=1
        
        # After Patch 0 SST override:
        view_for_writes = resolved_mode
        symbol_for_writes = caller_symbol if view_for_writes == "SYMBOL_SPECIFIC" else None
        
        # Verify symbol is stripped
        assert view_for_writes == "CROSS_SECTIONAL"
        assert symbol_for_writes is None
        
        # OutputLayout should accept this (no symbol for CS)
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view=view_for_writes,
            universe_sig="abc12345"
        )
        assert layout.view == "CROSS_SECTIONAL"
        assert layout.symbol is None
    
    def test_patch3_invariant_catches_symbol_in_cs(self):
        """Patch 3 invariant: symbol with CROSS_SECTIONAL should fail."""
        # This simulates what happens if Patch 0 doesn't strip symbol
        # Patch 3's invariant should catch it
        with pytest.raises(ValueError, match="cannot have symbol"):
            OutputLayout(
                output_root=Path("/tmp/test"),
                target="fwd_ret_5d",
                view="CROSS_SECTIONAL",
                universe_sig="abc12345",
                symbol="AAPL"  # Bug: symbol shouldn't be here for CS
            )
    
    def test_patch3_invariant_catches_sy_cohort_in_cs(self):
        """Patch 3 invariant: sy_ cohort under CROSS_SECTIONAL should fail."""
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view="CROSS_SECTIONAL",
            universe_sig="abc12345"
        )
        
        # This is the exact bug that was happening
        with pytest.raises(ValueError, match="Cohort ID scope violation"):
            layout.validate_cohort_id("sy_2024Q1_abc12345")
    
    def test_view_resolved_differently_from_requested(self):
        """Test that different resolved_mode from requested is handled correctly."""
        # Simulate: User requested SYMBOL_SPECIFIC for single symbol
        # But SST resolved to CROSS_SECTIONAL because min_cs=1
        requested_mode = "SYMBOL_SPECIFIC"
        resolved_mode = "CROSS_SECTIONAL"
        n_symbols_loaded = 1
        symbol = "AAPL"
        
        # The Patch 0 SST override should:
        view_for_writes = resolved_mode
        symbol_for_writes = symbol if view_for_writes == "SYMBOL_SPECIFIC" else None
        
        # Assertions
        assert view_for_writes == "CROSS_SECTIONAL"
        assert symbol_for_writes is None
        
        # And the resulting layout should be valid
        layout = OutputLayout(
            output_root=Path("/tmp/test"),
            target="fwd_ret_5d",
            view=view_for_writes,
            universe_sig="abc12345"
        )
        # cohort_id should be cs_ (not sy_)
        layout.validate_cohort_id("cs_2024Q1_abc12345")  # Should not raise


class TestDiskLayoutIntegrity:
    """
    Test that verifies no scope violations can exist on disk.
    
    These tests can be run against real output directories to verify
    no contamination exists.
    """
    
    @staticmethod
    def find_scope_violations(output_dir: Path) -> dict:
        """
        Find all scope violations in an output directory.
        
        Returns dict with:
        - sy_under_cs: List of sy_* cohorts under CROSS_SECTIONAL/
        - cs_under_sy: List of cs_* cohorts under SYMBOL_SPECIFIC/
        - unscoped_feature_importances: List of feature_importances at target root
        """
        violations = {
            'sy_under_cs': [],
            'cs_under_sy': [],
            'unscoped_feature_importances': []
        }
        
        if not output_dir.exists():
            return violations
        
        # Find all cohort directories
        for cohort_dir in output_dir.rglob("cohort=*"):
            if not cohort_dir.is_dir():
                continue
            
            cohort_id = cohort_dir.name.replace("cohort=", "")
            parts = cohort_dir.parts
            
            # Check for sy_* under CROSS_SECTIONAL
            if "CROSS_SECTIONAL" in parts and cohort_id.startswith("sy_"):
                violations['sy_under_cs'].append(str(cohort_dir))
            
            # Check for cs_* under SYMBOL_SPECIFIC
            if "SYMBOL_SPECIFIC" in parts and cohort_id.startswith("cs_"):
                violations['cs_under_sy'].append(str(cohort_dir))
        
        # Find unscoped feature_importances
        for fi_dir in output_dir.rglob("feature_importances"):
            if not fi_dir.is_dir():
                continue
            
            parts = fi_dir.parts
            # If feature_importances is directly under reproducibility/ (no view/universe)
            # that's a violation
            try:
                repro_idx = parts.index("reproducibility")
                if repro_idx + 1 < len(parts) and parts[repro_idx + 1] == "feature_importances":
                    violations['unscoped_feature_importances'].append(str(fi_dir))
            except ValueError:
                pass
        
        return violations
    
    def test_no_violations_helper_works(self, tmp_path):
        """Test that the violation finder works on clean directory."""
        violations = self.find_scope_violations(tmp_path)
        assert violations['sy_under_cs'] == []
        assert violations['cs_under_sy'] == []
        assert violations['unscoped_feature_importances'] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

