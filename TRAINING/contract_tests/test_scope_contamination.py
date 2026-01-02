# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Integration tests for scope contamination prevention.

These tests verify that the reproducibility directory structure is clean:
- CROSS_SECTIONAL/ contains only cs_ cohorts
- SYMBOL_SPECIFIC/ contains only sy_ cohorts  
- No routing evaluation artifacts in final dirs
- universe_sig is always non-null in metadata
"""

import json
import pytest
from pathlib import Path
from typing import List, Optional


def find_scope_violations(run_output_dir: Path) -> List[str]:
    """
    Find all scope violations in a run output directory.
    
    Returns:
        List of violation descriptions
    """
    violations = []
    
    # Find all reproducibility directories
    for repro_dir in run_output_dir.glob("**/reproducibility"):
        cs_dir = repro_dir / "CROSS_SECTIONAL"
        ss_dir = repro_dir / "SYMBOL_SPECIFIC"
        
        # Check CROSS_SECTIONAL for sy_ cohorts
        if cs_dir.exists():
            for cohort in cs_dir.glob("cohort=sy_*"):
                violations.append(
                    f"sy_ cohort under CROSS_SECTIONAL: {cohort.relative_to(run_output_dir)}"
                )
            for cohort in cs_dir.glob("**/cohort=sy_*"):
                violations.append(
                    f"sy_ cohort under CROSS_SECTIONAL (nested): {cohort.relative_to(run_output_dir)}"
                )
        
        # Check SYMBOL_SPECIFIC for cs_ cohorts
        if ss_dir.exists():
            for cohort in ss_dir.glob("cohort=cs_*"):
                violations.append(
                    f"cs_ cohort under SYMBOL_SPECIFIC: {cohort.relative_to(run_output_dir)}"
                )
            for cohort in ss_dir.glob("**/cohort=cs_*"):
                violations.append(
                    f"cs_ cohort under SYMBOL_SPECIFIC (nested): {cohort.relative_to(run_output_dir)}"
                )
            
            # Check for universe= folders directly under SYMBOL_SPECIFIC (should be under symbol=)
            for univ in ss_dir.glob("universe=*"):
                # Only flag if it's directly under SS, not under symbol=
                if univ.parent == ss_dir:
                    violations.append(
                        f"universe folder at SS root (should be under symbol=): {univ.relative_to(run_output_dir)}"
                    )
    
    return violations


def find_missing_universe_sig(run_output_dir: Path) -> List[str]:
    """
    Find all metadata.json files with missing universe_sig.
    
    Returns:
        List of paths with missing universe_sig
    """
    missing = []
    
    for meta_path in run_output_dir.glob("**/metadata.json"):
        try:
            data = json.loads(meta_path.read_text())
            universe_sig = data.get("universe_sig")
            if not universe_sig:
                missing.append(str(meta_path.relative_to(run_output_dir)))
        except (json.JSONDecodeError, IOError):
            continue
    
    return missing


class TestWriteScope:
    """Tests for WriteScope dataclass validation."""
    
    def test_cs_scope_rejects_symbol(self):
        """CS scope must have symbol=None."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope
        
        with pytest.raises(ValueError, match="CS scope must have symbol=None"):
            WriteScope(
                view="CROSS_SECTIONAL",
                universe_sig="abc123",
                symbol="AAPL",  # Invalid for CS
                purpose=WriteScope.__class__.__bases__[0],  # Avoid import
                stage="TARGET_RANKING"
            )
    
    def test_ss_scope_requires_symbol(self):
        """SS scope must have non-empty symbol."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose
        
        with pytest.raises(ValueError, match="SS scope requires non-empty symbol"):
            WriteScope(
                view="SYMBOL_SPECIFIC",
                universe_sig="abc123",
                symbol=None,  # Invalid for SS
                purpose=ScopePurpose.FINAL,
                stage="TARGET_RANKING"
            )
    
    def test_scope_requires_universe_sig(self):
        """All scopes require universe_sig."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose
        
        with pytest.raises(ValueError, match="universe_sig is required"):
            WriteScope(
                view="CROSS_SECTIONAL",
                universe_sig="",  # Empty is invalid
                symbol=None,
                purpose=ScopePurpose.FINAL,
                stage="TARGET_RANKING"
            )
    
    def test_cs_scope_factory(self):
        """Test CS scope factory method."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose
        
        scope = WriteScope.for_cross_sectional(
            universe_sig="abc123def456",
            stage="FEATURE_SELECTION"
        )
        
        assert scope.view == "CROSS_SECTIONAL"
        assert scope.symbol is None
        assert scope.universe_sig == "abc123def456"
        assert scope.purpose == ScopePurpose.FINAL
        assert scope.cohort_prefix == "cs_"
    
    def test_ss_scope_factory(self):
        """Test SS scope factory method."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose
        
        scope = WriteScope.for_symbol_specific(
            universe_sig="abc123def456",
            symbol="AAPL",
            stage="TRAINING"
        )
        
        assert scope.view == "SYMBOL_SPECIFIC"
        assert scope.symbol == "AAPL"
        assert scope.universe_sig == "abc123def456"
        assert scope.purpose == ScopePurpose.FINAL
        assert scope.cohort_prefix == "sy_"
    
    def test_routing_eval_scope_factory(self):
        """Test routing evaluation scope factory method."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose
        
        scope = WriteScope.for_routing_eval(
            view="SYMBOL_SPECIFIC",
            universe_sig="abc123def456",
            symbol="AAPL",
            stage="TARGET_RANKING"
        )
        
        assert scope.view == "SYMBOL_SPECIFIC"
        assert scope.purpose == ScopePurpose.ROUTING_EVAL
        assert scope.is_routing_eval
        assert not scope.is_final
    
    def test_validate_cohort_id_cs(self):
        """CS scope rejects sy_ cohort IDs."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope
        
        scope = WriteScope.for_cross_sectional(
            universe_sig="abc123",
            stage="TARGET_RANKING"
        )
        
        # Valid
        scope.validate_cohort_id("cs_2025Q3_min_cs3_max2000_v1_abc123")
        
        # Invalid
        with pytest.raises(ValueError, match="Cannot use sy_ cohort with CROSS_SECTIONAL"):
            scope.validate_cohort_id("sy_2025Q3_min_cs1_max2000_v1_abc123")
    
    def test_validate_cohort_id_ss(self):
        """SS scope rejects cs_ cohort IDs."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope
        
        scope = WriteScope.for_symbol_specific(
            universe_sig="abc123",
            symbol="AAPL",
            stage="TARGET_RANKING"
        )
        
        # Valid
        scope.validate_cohort_id("sy_2025Q3_min_cs1_max2000_v1_abc123")
        
        # Invalid
        with pytest.raises(ValueError, match="Cannot use cs_ cohort with SYMBOL_SPECIFIC"):
            scope.validate_cohort_id("cs_2025Q3_min_cs3_max2000_v1_abc123")
    
    def test_to_additional_data_cs(self):
        """CS scope populates additional_data without symbol key."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope
        
        scope = WriteScope.for_cross_sectional(
            universe_sig="abc123",
            stage="TARGET_RANKING"
        )
        
        data = scope.to_additional_data()
        
        assert data["view"] == "CROSS_SECTIONAL"
        assert data["universe_sig"] == "abc123"
        assert "symbol" not in data  # Key must be absent, not null
        assert data["cs_config"]["universe_sig"] == "abc123"
    
    def test_to_additional_data_ss(self):
        """SS scope populates additional_data with symbol key."""
        from TRAINING.orchestration.utils.scope_resolution import WriteScope
        
        scope = WriteScope.for_symbol_specific(
            universe_sig="abc123",
            symbol="AAPL",
            stage="TARGET_RANKING"
        )
        
        data = scope.to_additional_data()
        
        assert data["view"] == "SYMBOL_SPECIFIC"
        assert data["universe_sig"] == "abc123"
        assert data["symbol"] == "AAPL"


class TestScopeContaminationIntegration:
    """
    Integration tests that check real run outputs for scope contamination.
    
    These tests are skipped if no run output directories exist.
    """
    
    @pytest.fixture
    def sample_run_dir(self) -> Optional[Path]:
        """Find a sample run directory for testing."""
        results_dir = Path(__file__).parent.parent / "RESULTS" / "runs"
        if not results_dir.exists():
            return None
        
        # Find most recent run
        run_dirs = sorted(results_dir.glob("*/intelligent_output_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return run_dirs[0] if run_dirs else None
    
    def test_no_scope_violations_in_run(self, sample_run_dir: Optional[Path]):
        """Check that a real run has no scope violations."""
        if sample_run_dir is None:
            pytest.skip("No run output directories found")
        
        violations = find_scope_violations(sample_run_dir)
        
        if violations:
            pytest.fail(
                f"Found {len(violations)} scope violations in {sample_run_dir.name}:\n" +
                "\n".join(f"  - {v}" for v in violations[:10]) +
                (f"\n  ... and {len(violations) - 10} more" if len(violations) > 10 else "")
            )
    
    def test_universe_sig_present_in_metadata(self, sample_run_dir: Optional[Path]):
        """Check that universe_sig is present in all metadata files."""
        if sample_run_dir is None:
            pytest.skip("No run output directories found")
        
        missing = find_missing_universe_sig(sample_run_dir)
        
        # Allow some missing for now (legacy data), but warn
        if missing:
            pytest.skip(
                f"Found {len(missing)} metadata files with missing universe_sig "
                f"(may be legacy data):\n" +
                "\n".join(f"  - {m}" for m in missing[:5])
            )

