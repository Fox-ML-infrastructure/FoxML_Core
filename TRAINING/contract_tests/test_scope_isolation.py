# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
Scope Isolation Acceptance Tests

Tests to verify:
1. All JSON under reproducibility/routing_evaluation/ has purpose=ROUTING_EVAL
2. All JSON outside routing_evaluation/ has purpose=FINAL
3. Cohort prefix matches containing view directory (even in routing_evaluation)
4. No mixed scope artifacts
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple

# Add project root to path for standalone execution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import pytest only if available (not required for standalone run)
try:
    import pytest
except ImportError:
    pytest = None


def find_all_metadata_files(run_output_dir: Path) -> List[Path]:
    """Find all metadata.json files under reproducibility structure."""
    return list(run_output_dir.glob("**/reproducibility/**/*.json"))


def scan_purpose_metadata_consistency(run_output_dir: Path) -> List[str]:
    """
    Verify path routing matches metadata purpose.
    
    Returns list of violations.
    """
    violations = []
    
    for json_path in run_output_dir.glob("**/reproducibility/**/*.json"):
        path_str = str(json_path)
        try:
            data = json.loads(json_path.read_text())
            # Skip files without purpose field
            if "purpose" not in data:
                continue
            purpose = data["purpose"]
        except (json.JSONDecodeError, IOError):
            continue
        
        # Path has routing_evaluation but metadata says FINAL
        if "routing_evaluation" in path_str and purpose == "FINAL":
            violations.append(f"ROUTING_EVAL path with FINAL metadata: {json_path}")
        
        # Path is FINAL but metadata says ROUTING_EVAL
        if "routing_evaluation" not in path_str and purpose == "ROUTING_EVAL":
            violations.append(f"FINAL path with ROUTING_EVAL metadata: {json_path}")
    
    return violations


def scan_cohort_prefix_consistency(run_output_dir: Path) -> List[str]:
    """
    Verify cohort prefixes match view in ALL dirs (including routing_evaluation).
    
    Returns list of violations.
    """
    violations = []
    
    for cohort_dir in run_output_dir.glob("**/reproducibility/**/cohort=*"):
        path_str = str(cohort_dir)
        cohort_name = cohort_dir.name.replace("cohort=", "")
        
        # Apply same rule everywhere (including routing_evaluation)
        if "/CROSS_SECTIONAL/" in path_str and cohort_name.startswith("sy_"):
            violations.append(f"sy_ cohort under CROSS_SECTIONAL: {cohort_dir}")
        if "/SYMBOL_SPECIFIC/" in path_str and cohort_name.startswith("cs_"):
            violations.append(f"cs_ cohort under SYMBOL_SPECIFIC: {cohort_dir}")
    
    return violations


def scan_missing_universe_sig(run_output_dir: Path) -> List[str]:
    """
    Find metadata files missing universe_sig.
    
    Returns list of violations.
    """
    violations = []
    
    for json_path in run_output_dir.glob("**/reproducibility/**/metadata.json"):
        try:
            data = json.loads(json_path.read_text())
            universe_sig = data.get("universe_sig")
            if not universe_sig:
                # Also check cs_config
                cs_config = data.get("cs_config", {})
                universe_sig = cs_config.get("universe_sig")
            
            if not universe_sig:
                violations.append(f"Missing universe_sig: {json_path}")
        except (json.JSONDecodeError, IOError):
            continue
    
    return violations


def run_all_scans(run_output_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    """Run all scans and return (purpose_violations, cohort_violations, universe_sig_violations)."""
    return (
        scan_purpose_metadata_consistency(run_output_dir),
        scan_cohort_prefix_consistency(run_output_dir),
        scan_missing_universe_sig(run_output_dir)
    )


# --- Unit tests for WriteScope ---

def test_writescope_cs_valid():
    """Test CROSS_SECTIONAL scope creation."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose, Stage
    
    scope = WriteScope.for_cross_sectional(
        universe_sig="abc123def456",
        stage=Stage.TARGET_RANKING
    )
    assert scope.view.value == "CROSS_SECTIONAL"
    assert scope.symbol is None
    assert scope.universe_sig == "abc123def456"
    assert scope.purpose is ScopePurpose.FINAL


def test_writescope_ss_valid():
    """Test SYMBOL_SPECIFIC scope creation."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose, Stage
    
    scope = WriteScope.for_symbol_specific(
        universe_sig="abc123def456",
        symbol="AAPL",
        stage=Stage.TRAINING
    )
    assert scope.view.value == "SYMBOL_SPECIFIC"
    assert scope.symbol == "AAPL"
    assert scope.universe_sig == "abc123def456"
    assert scope.purpose is ScopePurpose.FINAL


def test_writescope_cs_rejects_symbol():
    """Test CROSS_SECTIONAL rejects symbol."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, Stage, View
    
    try:
        WriteScope(
            view=View.CROSS_SECTIONAL,
            universe_sig="abc123",
            symbol="AAPL",  # Should fail
            purpose=None,
            stage=Stage.TRAINING
        )
        raise AssertionError("Expected ValueError for CS with symbol")
    except ValueError as e:
        assert "CS scope must have symbol=None" in str(e)


def test_writescope_ss_requires_symbol():
    """Test SYMBOL_SPECIFIC requires symbol."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose, Stage
    
    try:
        WriteScope.for_symbol_specific(
            universe_sig="abc123def456",
            symbol=None,  # Should fail
            stage=Stage.TRAINING
        )
        raise AssertionError("Expected ValueError for SS without symbol")
    except ValueError as e:
        assert "SS scope requires non-empty symbol" in str(e)


def test_writescope_requires_universe_sig():
    """Test universe_sig is required."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose, Stage
    
    try:
        WriteScope.for_cross_sectional(
            universe_sig=None,  # Should fail
            stage=Stage.TARGET_RANKING
        )
        raise AssertionError("Expected ValueError for missing universe_sig")
    except ValueError as e:
        assert "universe_sig is required" in str(e)


def test_writescope_routing_eval_purpose():
    """Test ROUTING_EVAL purpose."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose, Stage, View
    
    scope = WriteScope.for_routing_eval(
        view=View.SYMBOL_SPECIFIC,
        universe_sig="eval_sig",
        symbol="TSLA",
        stage=Stage.TARGET_RANKING
    )
    assert scope.purpose is ScopePurpose.ROUTING_EVAL
    assert scope.view.value == "SYMBOL_SPECIFIC"
    assert scope.symbol == "TSLA"
    assert scope.is_routing_eval


def test_writescope_cohort_validation():
    """Test cohort_id validation matches view."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, Stage
    
    cs_scope = WriteScope.for_cross_sectional(
        universe_sig="abc123",
        stage=Stage.TARGET_RANKING
    )
    
    # Should not raise for correct prefix
    cs_scope.validate_cohort_id("cs_test123")
    
    # Should raise for wrong prefix
    try:
        cs_scope.validate_cohort_id("sy_test123")
        raise AssertionError("Expected ValueError for wrong cohort prefix")
    except ValueError as e:
        assert "Cannot use sy_ cohort" in str(e)


def test_to_additional_data_includes_purpose():
    """Test to_additional_data includes purpose and stage."""
    from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose, Stage
    
    scope = WriteScope.for_cross_sectional(
        universe_sig="abc123",
        stage=Stage.FEATURE_SELECTION,
        purpose=ScopePurpose.ROUTING_EVAL
    )
    
    data = {}
    scope.to_additional_data(data)
    
    assert data["view"] == "CROSS_SECTIONAL"
    assert data["purpose"] == "ROUTING_EVAL"
    assert data["stage"] == "FEATURE_SELECTION"
    assert data["universe_sig"] == "abc123"
    assert "symbol" not in data  # CS should not have symbol key


if __name__ == "__main__":
    # Run basic tests
    import sys
    
    print("Running WriteScope unit tests...")
    test_writescope_cs_valid()
    print("  ✅ test_writescope_cs_valid")
    test_writescope_ss_valid()
    print("  ✅ test_writescope_ss_valid")
    test_writescope_routing_eval_purpose()
    print("  ✅ test_writescope_routing_eval_purpose")
    test_writescope_cohort_validation()
    print("  ✅ test_writescope_cohort_validation")
    test_to_additional_data_includes_purpose()
    print("  ✅ test_to_additional_data_includes_purpose")
    
    print("\nAll unit tests passed!")
    
    # If a path is provided, scan it
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        if run_dir.exists():
            print(f"\nScanning {run_dir}...")
            purpose_v, cohort_v, universe_v = run_all_scans(run_dir)
            
            if purpose_v:
                print(f"\n❌ Purpose violations ({len(purpose_v)}):")
                for v in purpose_v[:10]:
                    print(f"  - {v}")
            else:
                print("\n✅ No purpose violations")
            
            if cohort_v:
                print(f"\n❌ Cohort prefix violations ({len(cohort_v)}):")
                for v in cohort_v[:10]:
                    print(f"  - {v}")
            else:
                print("\n✅ No cohort prefix violations")
            
            if universe_v:
                print(f"\n⚠️ Missing universe_sig ({len(universe_v)}):")
                for v in universe_v[:10]:
                    print(f"  - {v}")
            else:
                print("\n✅ All metadata has universe_sig")

