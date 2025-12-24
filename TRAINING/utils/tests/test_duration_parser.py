"""
Comprehensive unit tests for duration parsing and audit rule enforcement.

Tests cover:
- Edge cases (negative, zero, empty strings)
- Compound strings ("1h30m", "90s")
- Bar-based parsing
- Interval-aware strictness
- Domain constraints (fail fast)
- Non-auditable status markers
"""

import pytest
from datetime import timedelta

from TRAINING.common.utils.duration_parser import (
    Duration,
    parse_duration,
    parse_duration_bars,
    enforce_purge_audit_rule,
    format_duration,
    ceil_to_interval
)


class TestParseDuration:
    """Test basic duration parsing."""
    
    def test_parse_string_minutes(self):
        """Parse simple minute strings."""
        d = parse_duration("85.0m")
        assert d.to_minutes() == 85.0
    
    def test_parse_string_hours(self):
        """Parse hour strings."""
        d = parse_duration("1h")
        assert d.to_minutes() == 60.0
    
    def test_parse_compound_string(self):
        """Parse compound duration strings."""
        d1 = parse_duration("1h30m")
        assert d1.to_minutes() == 90.0
        
        d2 = parse_duration("90s")
        assert d2.to_minutes() == 1.5
        
        d3 = parse_duration("1h30m15s")
        assert d3.to_minutes() == pytest.approx(90.25, rel=1e-3)
    
    def test_parse_decimal_string(self):
        """Parse decimal duration strings."""
        d = parse_duration("0.5m")
        assert d.to_minutes() == 0.5
    
    def test_parse_timedelta(self):
        """Parse timedelta objects."""
        td = timedelta(hours=2, minutes=30)
        d = parse_duration(td)
        assert d.to_minutes() == 150.0
    
    def test_parse_float_seconds(self):
        """Parse float as seconds."""
        d = parse_duration(3600.0)  # 1 hour in seconds
        assert d.to_minutes() == 60.0
    
    def test_parse_duration_object(self):
        """Parse Duration object (should return unchanged)."""
        d1 = Duration.from_seconds(120.0)
        d2 = parse_duration(d1)
        assert d1 == d2
    
    def test_negative_duration_rejected(self):
        """Negative durations should be rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            parse_duration("-5m")
        
        with pytest.raises(ValueError, match="cannot be negative"):
            parse_duration(-100.0)
    
    def test_empty_string_rejected(self):
        """Empty strings should be rejected."""
        with pytest.raises(ValueError, match="Empty duration string"):
            parse_duration("")
        
        with pytest.raises(ValueError, match="Empty duration string"):
            parse_duration("   ")
    
    def test_unknown_unit_rejected(self):
        """Unknown units should be rejected."""
        with pytest.raises(ValueError, match="Unknown duration unit"):
            parse_duration("5xyz")
    
    def test_unparseable_string_rejected(self):
        """Unparseable strings should be rejected."""
        with pytest.raises(ValueError):
            parse_duration("abc123")


class TestParseDurationBars:
    """Test explicit bar-based parsing."""
    
    def test_parse_bars_integer(self):
        """Parse integer bars."""
        d = parse_duration_bars(20, "5m")
        assert d.to_minutes() == 100.0  # 20 bars * 5m = 100m
    
    def test_parse_bars_string_number(self):
        """Parse string number as bars."""
        d = parse_duration_bars("20", "5m")
        assert d.to_minutes() == 100.0
    
    def test_parse_bars_with_suffix(self):
        """Parse bars with 'b' or 'bars' suffix."""
        d1 = parse_duration_bars("20b", "5m")
        d2 = parse_duration_bars("20bars", "5m")
        d3 = parse_duration_bars(20, "5m")
        assert d1 == d2 == d3
        assert d1.to_minutes() == 100.0
    
    def test_parse_bars_negative_rejected(self):
        """Negative bars should be rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            parse_duration_bars(-5, "5m")
    
    def test_parse_bars_zero_interval_rejected(self):
        """Zero interval should be rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_duration_bars(20, "0m")


class TestCeilToInterval:
    """Test interval-aware rounding."""
    
    def test_ceil_aligned_duration(self):
        """Duration already aligned to interval should be unchanged."""
        d = parse_duration("100m")
        interval = parse_duration("5m")
        result = ceil_to_interval(d, interval)
        assert result == d
    
    def test_ceil_unaligned_duration(self):
        """Unaligned duration should be rounded up."""
        d = parse_duration("102m")  # Not aligned to 5m
        interval = parse_duration("5m")
        result = ceil_to_interval(d, interval)
        assert result.to_minutes() == 105.0  # Rounded up to next 5m boundary
    
    def test_ceil_with_none_interval(self):
        """None interval should return duration unchanged."""
        d = parse_duration("102m")
        result = ceil_to_interval(d, None)
        assert result == d
    
    def test_ceil_with_zero_interval(self):
        """Zero interval should return duration unchanged."""
        d = parse_duration("102m")
        result = ceil_to_interval(d, "0m")
        assert result == d


class TestEnforcePurgeAuditRule:
    """Test purge audit rule enforcement."""
    
    def test_interval_aware_strictness(self):
        """Interval-aware strictness should use ceil + interval."""
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "85.0m",
            "100.0m",
            interval="5m",
            strict_greater=True
        )
        # Rule: ceil_to_interval(100m, 5m) + 5m = 100m + 5m = 105m
        assert purge_out.to_minutes() == 105.0
        assert min_purge.to_minutes() == 105.0
        assert changed is True
    
    def test_interval_aware_unaligned_lookback(self):
        """Unaligned lookback should be properly ceiled."""
        # 102m lookback at 5m interval: ceil to 105m, then +5m = 110m
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "85.0m",
            "102.0m",
            interval="5m",
            strict_greater=True
        )
        assert purge_out.to_minutes() == 110.0
        assert min_purge.to_minutes() == 110.0
        assert changed is True
    
    def test_strictness_at_equality(self):
        """When purge == lookback, it must be bumped."""
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "100.0m",
            "100.0m",
            interval="5m",
            strict_greater=True
        )
        # Should be bumped: ceil(100m, 5m) + 5m = 105m
        assert purge_out.to_minutes() == 105.0
        assert changed is True
    
    def test_fallback_to_buffer_when_no_interval(self):
        """When interval is None, should fall back to buffer."""
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "85.0m",
            "100.0m",
            interval=None,
            buffer_frac=0.01,
            strict_greater=True
        )
        # Rule: 100m * 1.01 = 101m
        assert purge_out.to_minutes() == pytest.approx(101.0, rel=1e-3)
        assert changed is True
    
    def test_no_change_when_purge_sufficient(self):
        """When purge already exceeds requirement, no change."""
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "110.0m",
            "100.0m",
            interval="5m",
            strict_greater=True
        )
        # 110m > 105m (min required), so should stay at 110m
        assert purge_out.to_minutes() == 110.0
        assert changed is False
    
    def test_negative_duration_rejected(self):
        """Negative durations should be rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            enforce_purge_audit_rule("-5m", "100.0m", interval="5m")
        
        with pytest.raises(ValueError, match="cannot be negative"):
            enforce_purge_audit_rule("100.0m", "-5m", interval="5m")
    
    def test_zero_interval_rejected(self):
        """Zero interval should be rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            enforce_purge_audit_rule("85.0m", "100.0m", interval="0m", strict_greater=True)
    
    def test_strict_greater_requires_interval(self):
        """strict_greater=True requires interval to be provided."""
        with pytest.raises(ValueError, match="strict_greater=True requires interval"):
            enforce_purge_audit_rule(
                "85.0m",
                "100.0m",
                interval=None,
                strict_greater=True
            )
    
    def test_compound_strings(self):
        """Test with compound duration strings."""
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "1h25m",  # 85m
            "1h40m",  # 100m
            interval="5m",
            strict_greater=True
        )
        assert purge_out.to_minutes() == 105.0
        assert changed is True
    
    def test_bar_based_lookback(self):
        """Test with bar-based lookback."""
        # 20 bars at 5m = 100m
        lookback = parse_duration_bars(20, "5m")
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "85.0m",
            lookback,
            interval="5m",
            strict_greater=True
        )
        assert purge_out.to_minutes() == 105.0
        assert changed is True


class TestFormatDuration:
    """Test duration formatting."""
    
    def test_format_minutes(self):
        """Format minutes."""
        d = parse_duration("85.0m")
        assert format_duration(d) == "85.0m"
    
    def test_format_hours(self):
        """Format hours."""
        d = parse_duration("2h")
        assert format_duration(d) == "2h"
    
    def test_format_days(self):
        """Format days."""
        d = parse_duration("2d")
        assert format_duration(d) == "2d"
    
    def test_format_seconds(self):
        """Format seconds."""
        d = parse_duration("90s")
        assert format_duration(d) == "90.0s"


class TestDomainConstraints:
    """Test domain constraint enforcement."""
    
    def test_negative_purge_rejected(self):
        """Negative purge should fail fast."""
        with pytest.raises(ValueError, match="cannot be negative"):
            enforce_purge_audit_rule("-10m", "100.0m", interval="5m")
    
    def test_negative_lookback_rejected(self):
        """Negative lookback should fail fast."""
        with pytest.raises(ValueError, match="cannot be negative"):
            enforce_purge_audit_rule("100.0m", "-10m", interval="5m")
    
    def test_negative_interval_rejected(self):
        """Negative interval should fail fast."""
        with pytest.raises(ValueError, match="must be positive"):
            enforce_purge_audit_rule("85.0m", "100.0m", interval="-5m", strict_greater=True)
    
    def test_zero_interval_rejected(self):
        """Zero interval should fail fast."""
        with pytest.raises(ValueError, match="must be positive"):
            enforce_purge_audit_rule("85.0m", "100.0m", interval="0m", strict_greater=True)


class TestIrregularIntervalHandling:
    """Test handling of irregular/variable intervals."""
    
    def test_unknown_interval_uses_fallback(self):
        """Unknown interval should use buffer fallback."""
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            "85.0m",
            "100.0m",
            interval=None,  # Unknown/variable interval
            buffer_frac=0.01,
            strict_greater=False  # Can't enforce strict without interval
        )
        # Should use buffer fallback
        assert purge_out.to_minutes() == pytest.approx(101.0, rel=1e-3)
        assert changed is True
