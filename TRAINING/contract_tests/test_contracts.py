"""
Contract Tests for Pipeline Legibility

These tests verify that the core contracts (WriteScope, View, Stage, etc.)
enforce their invariants correctly.

Run with: pytest TRAINING/tests/test_contracts.py -v
"""

import pytest
from typing import Optional


class TestViewEnum:
    """Tests for View enum usage and invariants."""

    def test_view_enum_values(self):
        """View enum has exactly two values."""
        from TRAINING.orchestration.utils.scope_resolution import View

        assert hasattr(View, "CROSS_SECTIONAL")
        assert hasattr(View, "SYMBOL_SPECIFIC")
        assert len(View) == 2

    def test_view_from_string_valid(self):
        """View.from_string handles valid inputs."""
        from TRAINING.orchestration.utils.scope_resolution import View

        assert View.from_string("CROSS_SECTIONAL") is View.CROSS_SECTIONAL
        assert View.from_string("SYMBOL_SPECIFIC") is View.SYMBOL_SPECIFIC
        # Aliases
        assert View.from_string("INDIVIDUAL") is View.SYMBOL_SPECIFIC
        assert View.from_string("LOSO") is View.SYMBOL_SPECIFIC

    def test_view_from_string_invalid(self):
        """View.from_string rejects invalid inputs."""
        from TRAINING.orchestration.utils.scope_resolution import View

        with pytest.raises(ValueError, match="Unknown view"):
            View.from_string("INVALID")

        with pytest.raises(ValueError, match="cannot be None"):
            View.from_string(None)

    def test_view_string_representation(self):
        """View enum converts to string correctly."""
        from TRAINING.orchestration.utils.scope_resolution import View

        assert str(View.CROSS_SECTIONAL) == "CROSS_SECTIONAL"
        assert str(View.SYMBOL_SPECIFIC) == "SYMBOL_SPECIFIC"


class TestStageEnum:
    """Tests for Stage enum usage and invariants."""

    def test_stage_enum_values(self):
        """Stage enum has exactly three values."""
        from TRAINING.orchestration.utils.scope_resolution import Stage

        assert hasattr(Stage, "TARGET_RANKING")
        assert hasattr(Stage, "FEATURE_SELECTION")
        assert hasattr(Stage, "TRAINING")
        assert len(Stage) == 3

    def test_stage_from_string_valid(self):
        """Stage.from_string handles valid inputs."""
        from TRAINING.orchestration.utils.scope_resolution import Stage

        assert Stage.from_string("TARGET_RANKING") is Stage.TARGET_RANKING
        assert Stage.from_string("FEATURE_SELECTION") is Stage.FEATURE_SELECTION
        assert Stage.from_string("TRAINING") is Stage.TRAINING

    def test_stage_from_string_invalid(self):
        """Stage.from_string rejects invalid inputs."""
        from TRAINING.orchestration.utils.scope_resolution import Stage

        with pytest.raises(ValueError, match="Unknown stage"):
            Stage.from_string("INVALID")


class TestWriteScope:
    """Tests for WriteScope invariant enforcement."""

    def test_cs_scope_requires_no_symbol(self):
        """Cross-sectional scope must have symbol=None."""
        from TRAINING.orchestration.utils.scope_resolution import (
            WriteScope,
            View,
            Stage,
            ScopePurpose,
        )

        # Valid: CS with no symbol
        scope = WriteScope(
            view=View.CROSS_SECTIONAL,
            universe_sig="abc123",
            symbol=None,
            purpose=ScopePurpose.FINAL,
            stage=Stage.TARGET_RANKING,
        )
        assert scope.symbol is None

        # Invalid: CS with symbol
        with pytest.raises(ValueError, match="CS scope must have symbol=None"):
            WriteScope(
                view=View.CROSS_SECTIONAL,
                universe_sig="abc123",
                symbol="AAPL",  # Invalid for CS
                purpose=ScopePurpose.FINAL,
                stage=Stage.TARGET_RANKING,
            )

    def test_ss_scope_requires_symbol(self):
        """Symbol-specific scope must have non-empty symbol."""
        from TRAINING.orchestration.utils.scope_resolution import (
            WriteScope,
            View,
            Stage,
            ScopePurpose,
        )

        # Valid: SS with symbol
        scope = WriteScope(
            view=View.SYMBOL_SPECIFIC,
            universe_sig="abc123",
            symbol="AAPL",
            purpose=ScopePurpose.FINAL,
            stage=Stage.TRAINING,
        )
        assert scope.symbol == "AAPL"

        # Invalid: SS without symbol
        with pytest.raises(ValueError, match="SS scope requires non-empty symbol"):
            WriteScope(
                view=View.SYMBOL_SPECIFIC,
                universe_sig="abc123",
                symbol=None,  # Invalid for SS
                purpose=ScopePurpose.FINAL,
                stage=Stage.TRAINING,
            )

    def test_universe_sig_required(self):
        """WriteScope requires universe_sig."""
        from TRAINING.orchestration.utils.scope_resolution import (
            WriteScope,
            View,
            Stage,
            ScopePurpose,
        )

        with pytest.raises(ValueError, match="universe_sig is required"):
            WriteScope(
                view=View.CROSS_SECTIONAL,
                universe_sig=None,  # Invalid
                symbol=None,
                purpose=ScopePurpose.FINAL,
                stage=Stage.TARGET_RANKING,
            )

        with pytest.raises(ValueError, match="universe_sig is required"):
            WriteScope(
                view=View.CROSS_SECTIONAL,
                universe_sig="",  # Invalid
                symbol=None,
                purpose=ScopePurpose.FINAL,
                stage=Stage.TARGET_RANKING,
            )

    def test_view_must_be_enum(self):
        """WriteScope rejects string view (must be View enum)."""
        from TRAINING.orchestration.utils.scope_resolution import (
            WriteScope,
            Stage,
            ScopePurpose,
        )

        with pytest.raises(ValueError, match="view must be View enum"):
            WriteScope(
                view="CROSS_SECTIONAL",  # String, not enum
                universe_sig="abc123",
                symbol=None,
                purpose=ScopePurpose.FINAL,
                stage=Stage.TARGET_RANKING,
            )

    def test_factory_for_cross_sectional(self):
        """WriteScope.for_cross_sectional factory works correctly."""
        from TRAINING.orchestration.utils.scope_resolution import (
            WriteScope,
            View,
            Stage,
        )

        scope = WriteScope.for_cross_sectional(
            universe_sig="abc123", stage=Stage.TARGET_RANKING
        )

        assert scope.view is View.CROSS_SECTIONAL
        assert scope.symbol is None
        assert scope.universe_sig == "abc123"
        assert scope.stage is Stage.TARGET_RANKING

    def test_factory_for_symbol_specific(self):
        """WriteScope.for_symbol_specific factory works correctly."""
        from TRAINING.orchestration.utils.scope_resolution import (
            WriteScope,
            View,
            Stage,
        )

        scope = WriteScope.for_symbol_specific(
            universe_sig="abc123", symbol="AAPL", stage=Stage.FEATURE_SELECTION
        )

        assert scope.view is View.SYMBOL_SPECIFIC
        assert scope.symbol == "AAPL"
        assert scope.universe_sig == "abc123"
        assert scope.stage is Stage.FEATURE_SELECTION


class TestRunContextSync:
    """Tests for RunContext field values."""

    def test_view_cross_sectional(self):
        """Setting view to CROSS_SECTIONAL should work."""
        from TRAINING.orchestration.utils.run_context import RunContext

        ctx = RunContext(view="CROSS_SECTIONAL")
        assert ctx.view == "CROSS_SECTIONAL"

    def test_view_symbol_specific(self):
        """Setting view to SYMBOL_SPECIFIC should work."""
        from TRAINING.orchestration.utils.run_context import RunContext

        ctx = RunContext(view="SYMBOL_SPECIFIC")
        assert ctx.view == "SYMBOL_SPECIFIC"

    def test_requested_view(self):
        """Setting requested_view should work."""
        from TRAINING.orchestration.utils.run_context import RunContext

        ctx = RunContext(requested_view="CROSS_SECTIONAL")
        assert ctx.requested_view == "CROSS_SECTIONAL"

    def test_view_reason(self):
        """Setting view_reason should work."""
        from TRAINING.orchestration.utils.run_context import RunContext

        ctx = RunContext(view_reason="test reason")
        assert ctx.view_reason == "test reason"


class TestValidateViewContract:
    """Tests for view contract validation."""

    def test_force_policy_requires_match(self):
        """Force policy requires resolved == requested."""
        from TRAINING.orchestration.utils.run_context import validate_view_contract

        # Valid: matches
        assert validate_view_contract("CROSS_SECTIONAL", "CROSS_SECTIONAL", "force")

        # Invalid: mismatch
        with pytest.raises(ValueError, match="View contract violation"):
            validate_view_contract("SYMBOL_SPECIFIC", "CROSS_SECTIONAL", "force")

    def test_force_policy_requires_requested(self):
        """Force policy requires requested_view to be set."""
        from TRAINING.orchestration.utils.run_context import validate_view_contract

        with pytest.raises(ValueError, match="requires requested_view to be set"):
            validate_view_contract("CROSS_SECTIONAL", None, "force")

    def test_auto_policy_allows_flip(self):
        """Auto policy allows resolver to flip view."""
        from TRAINING.orchestration.utils.run_context import validate_view_contract

        # Both valid with auto
        assert validate_view_contract("SYMBOL_SPECIFIC", "CROSS_SECTIONAL", "auto")
        assert validate_view_contract("CROSS_SECTIONAL", None, "auto")

    def test_legacy_alias_works(self):
        """validate_mode_contract alias works for backward compat."""
        from TRAINING.orchestration.utils.run_context import validate_mode_contract

        assert validate_mode_contract("CROSS_SECTIONAL", "CROSS_SECTIONAL", "force")


class TestUniverseSignature:
    """Tests for universe signature computation."""

    def test_signature_is_deterministic(self):
        """Same symbols produce same signature."""
        from TRAINING.orchestration.utils.run_context import compute_universe_signature

        sig1 = compute_universe_signature(["AAPL", "MSFT", "GOOGL"])
        sig2 = compute_universe_signature(["AAPL", "MSFT", "GOOGL"])
        assert sig1 == sig2

    def test_signature_is_order_independent(self):
        """Order doesn't affect signature."""
        from TRAINING.orchestration.utils.run_context import compute_universe_signature

        sig1 = compute_universe_signature(["AAPL", "MSFT", "GOOGL"])
        sig2 = compute_universe_signature(["GOOGL", "AAPL", "MSFT"])
        assert sig1 == sig2

    def test_signature_is_duplicate_invariant(self):
        """Duplicates don't affect signature."""
        from TRAINING.orchestration.utils.run_context import compute_universe_signature

        sig1 = compute_universe_signature(["AAPL", "MSFT"])
        sig2 = compute_universe_signature(["AAPL", "MSFT", "AAPL", "MSFT"])
        assert sig1 == sig2

    def test_signature_format(self):
        """Signature is 12-character hex string."""
        from TRAINING.orchestration.utils.run_context import compute_universe_signature

        sig = compute_universe_signature(["AAPL"])
        assert len(sig) == 12
        assert all(c in "0123456789abcdef" for c in sig)
