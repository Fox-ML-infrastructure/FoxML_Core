# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Tests for Run Identity and Canonicalization SST.

Covers:
- canonicalize() determinism
- Set ordering by canonical_json
- Numpy type handling
- Missing signature rejection in strict/replicate modes
- RunIdentity key computation
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path


class TestCanonicalize:
    """Tests for canonicalize() function."""
    
    def test_primitive_types(self):
        """Primitive types are preserved."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        assert canonicalize(None) is None
        assert canonicalize(True) is True
        assert canonicalize(False) is False
        assert canonicalize(42) == 42
        assert canonicalize("hello") == "hello"
    
    def test_float_precision_normalized(self):
        """Float precision is normalized to avoid representation diffs."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        # Very small precision differences should normalize to same value
        val1 = canonicalize(0.1 + 0.2)
        val2 = canonicalize(0.3)
        assert val1 == val2, "Float precision should be normalized"
    
    def test_float_special_values(self):
        """Special float values are handled deterministically."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        assert canonicalize(float('nan')) == "__NaN__"
        assert canonicalize(float('inf')) == "__Infinity__"
        assert canonicalize(float('-inf')) == "__NegInfinity__"
    
    def test_dict_none_dropped_after_canonicalization(self):
        """None values in dicts are dropped AFTER canonicalization."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        # Direct None values should be dropped
        result = canonicalize({"a": 1, "b": None, "c": 3})
        assert result == {"a": 1, "c": 3}
        
        # Nested None values should also be dropped
        result = canonicalize({"outer": {"inner": None, "keep": 1}})
        assert result == {"outer": {"keep": 1}}
    
    def test_dict_sorted_keys(self):
        """Dict keys are always sorted for deterministic output."""
        from TRAINING.common.utils.config_hashing import canonical_json
        
        d1 = {"z": 1, "a": 2, "m": 3}
        d2 = {"a": 2, "m": 3, "z": 1}
        
        assert canonical_json(d1) == canonical_json(d2)
    
    def test_list_order_preserved(self):
        """List order is preserved (not sorted)."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        result = canonicalize([3, 1, 2])
        assert result == [3, 1, 2]
    
    def test_set_sorted_by_canonical_json(self):
        """Sets are sorted by canonical_json of each element."""
        from TRAINING.common.utils.config_hashing import canonicalize, canonical_json
        
        # Simple set
        result = canonicalize({3, 1, 2})
        assert result == [1, 2, 3]  # Sorted
        
        # Set with mixed types (all canonicalizable to comparable types)
        result = canonicalize({"b", "a", "c"})
        assert result == ["a", "b", "c"]
    
    def test_datetime_utc_normalized(self):
        """Datetimes are normalized to UTC ISO format."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        # Naive datetime (assumed UTC)
        dt_naive = datetime(2025, 1, 1, 12, 0, 0)
        result = canonicalize(dt_naive)
        assert "2025-01-01" in result
        assert result.endswith("+00:00")
        
        # Timezone-aware datetime
        dt_aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result2 = canonicalize(dt_aware)
        assert result == result2  # Should be identical
    
    def test_path_to_string(self):
        """Paths are converted to strings."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        result = canonicalize(Path("/home/user/file.txt"))
        assert result == "/home/user/file.txt"
    
    def test_unknown_type_raises(self):
        """Unknown types raise TypeError (no silent str() fallback)."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        class CustomClass:
            pass
        
        with pytest.raises(TypeError, match="does not support"):
            canonicalize(CustomClass())
    
    def test_numpy_scalar(self):
        """Numpy scalars are converted to Python types."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        # Integer types
        assert canonicalize(np.int64(42)) == 42
        assert canonicalize(np.int32(42)) == 42
        
        # Float types  
        assert canonicalize(np.float64(3.14)) == pytest.approx(3.14, rel=1e-9)
        
        # Boolean
        assert canonicalize(np.bool_(True)) is True
    
    def test_numpy_array(self):
        """Numpy arrays are converted to lists."""
        from TRAINING.common.utils.config_hashing import canonicalize
        
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        arr = np.array([1, 2, 3])
        result = canonicalize(arr)
        assert result == [1, 2, 3]
        
        # 2D array
        arr2d = np.array([[1, 2], [3, 4]])
        result2d = canonicalize(arr2d)
        assert result2d == [[1, 2], [3, 4]]
    
    def test_dataclass(self):
        """Dataclasses are converted to dicts."""
        from TRAINING.common.utils.config_hashing import canonicalize
        from dataclasses import dataclass
        
        @dataclass
        class Config:
            name: str
            value: int
        
        cfg = Config(name="test", value=42)
        result = canonicalize(cfg)
        assert result == {"name": "test", "value": 42}


class TestCanonicalJson:
    """Tests for canonical_json() determinism."""
    
    def test_deterministic_output(self):
        """Same input always produces same output."""
        from TRAINING.common.utils.config_hashing import canonical_json
        
        data = {"b": 2, "a": 1, "nested": {"z": 26, "y": 25}}
        
        # Call multiple times
        results = [canonical_json(data) for _ in range(100)]
        
        # All should be identical
        assert len(set(results)) == 1
    
    def test_no_whitespace(self):
        """Output has no unnecessary whitespace."""
        from TRAINING.common.utils.config_hashing import canonical_json
        
        result = canonical_json({"a": 1, "b": 2})
        assert " " not in result  # No spaces
        assert "\n" not in result  # No newlines


class TestSha256:
    """Tests for SHA256 hashing functions."""
    
    def test_full_hash_length(self):
        """sha256_full returns 64-character hash."""
        from TRAINING.common.utils.config_hashing import sha256_full
        
        result = sha256_full("test string")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
    
    def test_short_hash_length(self):
        """sha256_short returns truncated hash."""
        from TRAINING.common.utils.config_hashing import sha256_short
        
        result = sha256_short("test string", n=16)
        assert len(result) == 16
        
        result8 = sha256_short("test string", n=8)
        assert len(result8) == 8
    
    def test_same_input_same_hash(self):
        """Same input produces same hash."""
        from TRAINING.common.utils.config_hashing import sha256_full
        
        hash1 = sha256_full("identical content")
        hash2 = sha256_full("identical content")
        assert hash1 == hash2


class TestRunIdentity:
    """Tests for RunIdentity dataclass."""
    
    def test_strict_key_includes_seed(self):
        """strict_key includes train_seed."""
        from TRAINING.common.utils.fingerprinting import RunIdentity
        
        id1 = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="pqr678",
            train_seed=42,
        )
        
        id2 = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="pqr678",
            train_seed=99,  # Different seed
        )
        
        # strict_key should differ
        assert id1.strict_key != id2.strict_key
        
        # replicate_key should be same
        assert id1.replicate_key == id2.replicate_key
    
    def test_replicate_key_excludes_seed(self):
        """replicate_key excludes train_seed."""
        from TRAINING.common.utils.fingerprinting import RunIdentity
        
        id_with_seed = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="pqr678",
            train_seed=42,
        )
        
        id_no_seed = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="pqr678",
            train_seed=None,
        )
        
        # replicate_key should be same regardless of seed
        assert id_with_seed.replicate_key == id_no_seed.replicate_key
    
    def test_keys_are_64_chars(self):
        """Identity keys are full 64-character SHA256."""
        from TRAINING.common.utils.fingerprinting import RunIdentity
        
        identity = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="pqr678",
            train_seed=42,
        )
        
        assert len(identity.strict_key) == 64
        assert len(identity.replicate_key) == 64
    
    def test_is_complete(self):
        """is_complete() checks all required signatures."""
        from TRAINING.common.utils.fingerprinting import RunIdentity
        
        # Complete identity
        complete = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="pqr678",
        )
        assert complete.is_complete()
        
        # Incomplete identity (missing routing)
        incomplete = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="",  # Empty
        )
        assert not incomplete.is_complete()
    
    def test_to_dict_from_dict_roundtrip(self):
        """to_dict() and from_dict() roundtrip correctly."""
        from TRAINING.common.utils.fingerprinting import RunIdentity
        
        original = RunIdentity(
            dataset_signature="abc123",
            split_signature="def456",
            target_signature="ghi789",
            feature_signature="jkl012",
            hparams_signature="mno345",
            routing_signature="pqr678",
            train_seed=42,
        )
        
        # Roundtrip
        as_dict = original.to_dict()
        restored = RunIdentity.from_dict(as_dict)
        
        # Check all fields match
        assert restored.dataset_signature == original.dataset_signature
        assert restored.split_signature == original.split_signature
        assert restored.target_signature == original.target_signature
        assert restored.feature_signature == original.feature_signature
        assert restored.hparams_signature == original.hparams_signature
        assert restored.routing_signature == original.routing_signature
        assert restored.train_seed == original.train_seed
        
        # Keys should recompute correctly
        assert restored.strict_key == original.strict_key
        assert restored.replicate_key == original.replicate_key


class TestValidateGroupSignatures:
    """Tests for validate_group_signatures() in analysis.py."""
    
    def test_legacy_mode_always_passes(self):
        """Legacy mode always returns ok=True."""
        from TRAINING.stability.feature_importance.analysis import validate_group_signatures
        from TRAINING.stability.feature_importance.schema import FeatureImportanceSnapshot
        from datetime import datetime
        
        # Create snapshots without signatures
        snapshots = [
            FeatureImportanceSnapshot(
                target="test",
                method="lightgbm",
                universe_sig="sym1",
                run_id=f"run{i}",
                created_at=datetime.now(),
                features=["f1", "f2"],
                importances=[0.5, 0.3],
            )
            for i in range(3)
        ]
        
        result = validate_group_signatures(snapshots, mode="legacy")
        assert result.ok
    
    def test_strict_mode_rejects_missing_signatures(self):
        """Strict mode rejects snapshots with missing signatures."""
        from TRAINING.stability.feature_importance.analysis import validate_group_signatures
        from TRAINING.stability.feature_importance.schema import FeatureImportanceSnapshot
        from datetime import datetime
        
        # Create snapshots without signatures
        snapshots = [
            FeatureImportanceSnapshot(
                target="test",
                method="lightgbm",
                universe_sig="sym1",
                run_id=f"run{i}",
                created_at=datetime.now(),
                features=["f1", "f2"],
                importances=[0.5, 0.3],
            )
            for i in range(3)
        ]
        
        result = validate_group_signatures(snapshots, mode="strict")
        assert not result.ok
        assert "missing required" in result.reason
    
    def test_strict_mode_rejects_mismatched_signatures(self):
        """Strict mode rejects groups with different signatures."""
        from TRAINING.stability.feature_importance.analysis import validate_group_signatures
        from TRAINING.stability.feature_importance.schema import FeatureImportanceSnapshot
        from datetime import datetime
        
        # Create snapshots with different feature signatures
        s1 = FeatureImportanceSnapshot(
            target="test",
            method="lightgbm",
            universe_sig="sym1",
            run_id="run1",
            created_at=datetime.now(),
            features=["f1", "f2"],
            importances=[0.5, 0.3],
            dataset_signature="data1",
            split_signature="split1",
            target_signature="target1",
            feature_signature="features_A",  # Different
            hparams_signature="hparams1",
            routing_signature="routing1",
        )
        s2 = FeatureImportanceSnapshot(
            target="test",
            method="lightgbm",
            universe_sig="sym1",
            run_id="run2",
            created_at=datetime.now(),
            features=["f1", "f2"],
            importances=[0.5, 0.3],
            dataset_signature="data1",
            split_signature="split1",
            target_signature="target1",
            feature_signature="features_B",  # Different
            hparams_signature="hparams1",
            routing_signature="routing1",
        )
        
        result = validate_group_signatures([s1, s2], mode="strict")
        assert not result.ok
        assert "Multiple" in result.reason
    
    def test_strict_mode_passes_identical_signatures(self):
        """Strict mode passes groups with identical signatures."""
        from TRAINING.stability.feature_importance.analysis import validate_group_signatures
        from TRAINING.stability.feature_importance.schema import FeatureImportanceSnapshot
        from datetime import datetime
        
        # Create snapshots with identical signatures
        common_sigs = dict(
            dataset_signature="data1",
            split_signature="split1",
            target_signature="target1",
            feature_signature="features1",
            hparams_signature="hparams1",
            routing_signature="routing1",
            train_seed=42,
        )
        
        snapshots = [
            FeatureImportanceSnapshot(
                target="test",
                method="lightgbm",
                universe_sig="sym1",
                run_id=f"run{i}",
                created_at=datetime.now(),
                features=["f1", "f2"],
                importances=[0.5, 0.3],
                **common_sigs,
            )
            for i in range(3)
        ]
        
        result = validate_group_signatures(snapshots, mode="strict")
        assert result.ok
    
    def test_replicate_mode_allows_different_seeds(self):
        """Replicate mode allows different train_seeds."""
        from TRAINING.stability.feature_importance.analysis import validate_group_signatures
        from TRAINING.stability.feature_importance.schema import FeatureImportanceSnapshot
        from datetime import datetime
        
        # Create snapshots with same signatures but different seeds
        base_sigs = dict(
            dataset_signature="data1",
            split_signature="split1",
            target_signature="target1",
            feature_signature="features1",
            hparams_signature="hparams1",
            routing_signature="routing1",
        )
        
        s1 = FeatureImportanceSnapshot(
            target="test",
            method="lightgbm",
            universe_sig="sym1",
            run_id="run1",
            created_at=datetime.now(),
            features=["f1", "f2"],
            importances=[0.5, 0.3],
            train_seed=42,
            **base_sigs,
        )
        s2 = FeatureImportanceSnapshot(
            target="test",
            method="lightgbm",
            universe_sig="sym1",
            run_id="run2",
            created_at=datetime.now(),
            features=["f1", "f2"],
            importances=[0.5, 0.3],
            train_seed=99,  # Different seed
            **base_sigs,
        )
        
        result = validate_group_signatures([s1, s2], mode="replicate")
        assert result.ok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
