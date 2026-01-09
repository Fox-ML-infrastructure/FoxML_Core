# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Prediction Hashing Utilities

Provides stable, reproducible hashing of model predictions for:
- Strict determinism verification (bitwise identical)
- Live drift detection (tolerance for float jitter)
- Audit trail (tamper-evident)

Usage:
    from TRAINING.common.utils.prediction_hashing import prediction_fingerprint
    
    fingerprint = prediction_fingerprint(
        preds=model.predict(X),
        row_ids=df[["symbol", "ts"]].apply(lambda r: f"{r.symbol}_{r.ts}", axis=1).values,
        kind="regression",
        quantize=None,  # or 1e-6 for live drift detection
    )
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def _hash_bytes(chunks: List[bytes]) -> str:
    """Hash multiple byte chunks into a single SHA256 hex digest."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def canonicalize_preds(
    preds: np.ndarray,
    *,
    kind: str,
    quantize: Optional[float] = None,
) -> np.ndarray:
    """
    Canonicalize predictions for stable hashing.
    
    Args:
        preds: Raw predictions array
        kind: "regression", "binary_proba", "multiclass_proba", or "label"
        quantize: Optional quantization for float tolerance (e.g., 1e-6)
    
    Returns:
        Canonicalized contiguous array with stable dtype
    """
    a = np.asarray(preds)
    
    # Standardize dtype
    if kind == "label":
        a = a.astype(np.int64, copy=False)
    else:
        a = a.astype(np.float32, copy=False)
        
        # Optional quantization for live drift detection
        if quantize is not None:
            a = np.rint(a / quantize) * quantize
        
        # Canonicalize NaNs (avoid platform-specific NaN payloads)
        nan_mask = np.isnan(a)
        if nan_mask.any():
            a = a.copy()
            # Set all NaNs to a canonical quiet NaN
            a[nan_mask] = np.float32('nan')
    
    return np.ascontiguousarray(a)


@dataclass
class PredictionFingerprint:
    """Container for prediction fingerprint data."""
    prediction_hash: str
    prediction_hash_live: Optional[str]
    row_ids_hash: str
    classes_hash: Optional[str]
    kind: str
    dtype: str
    shape: List[int]
    quantize: Optional[float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "prediction_hash": self.prediction_hash,
            "row_ids_hash": self.row_ids_hash,
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": self.shape,
        }
        if self.prediction_hash_live:
            result["prediction_hash_live"] = self.prediction_hash_live
        if self.classes_hash:
            result["classes_hash"] = self.classes_hash
        if self.quantize is not None:
            result["quantize"] = self.quantize
        return result


def prediction_fingerprint(
    preds: np.ndarray,
    row_ids: np.ndarray,
    kind: str,
    quantize: Optional[float] = None,
    class_order: Optional[np.ndarray] = None,
    compute_live_hash: bool = True,
    live_quantize: float = 1e-6,
) -> PredictionFingerprint:
    """
    Compute a stable fingerprint for predictions.
    
    Args:
        preds: Model predictions (1D for regression/binary, 2D for multiclass proba)
        row_ids: Row identifiers aligned with predictions (e.g., "AAPL_2024-01-01T09:30:00")
        kind: Prediction type:
            - "regression": Continuous values
            - "binary_proba": Probability of class 1 (0-1)
            - "multiclass_proba": Probability matrix (n_samples, n_classes)
            - "label": Hard class labels (integers)
        quantize: Quantization for strict hash (None = bitwise exact)
        class_order: Required for *_proba kinds - the order of classes (e.g., model.classes_)
        compute_live_hash: Whether to also compute a quantized hash for live drift detection
        live_quantize: Quantization level for live hash (default: 1e-6)
    
    Returns:
        PredictionFingerprint with hash data
    
    Example:
        fingerprint = prediction_fingerprint(
            preds=model.predict(X),
            row_ids=df.apply(lambda r: f"{r.symbol}_{r.ts}", axis=1).values,
            kind="regression",
        )
    """
    # Validate inputs
    if kind.endswith("_proba") and class_order is None:
        raise ValueError(f"class_order is required for kind='{kind}'")
    
    preds = np.asarray(preds)
    row_ids = np.asarray(row_ids)
    
    if len(row_ids) != len(preds):
        raise ValueError(f"row_ids length ({len(row_ids)}) must match preds length ({len(preds)})")
    
    # Hash row IDs (binding to inputs)
    row_ids_bytes = np.ascontiguousarray(row_ids.astype(str).astype("S"))
    row_ids_hash = _hash_bytes([row_ids_bytes.tobytes()])
    
    # Hash class order if provided
    classes_hash = None
    if class_order is not None:
        class_order_bytes = np.ascontiguousarray(np.asarray(class_order).astype(str).astype("S"))
        classes_hash = _hash_bytes([class_order_bytes.tobytes()])
    
    # Canonicalize predictions for strict hash
    a = canonicalize_preds(preds, kind=kind, quantize=quantize)
    
    # Build strict hash
    chunks = [
        kind.encode("utf-8"),
        str(a.shape).encode("utf-8"),
        str(a.dtype).encode("utf-8"),
        row_ids_hash.encode("utf-8"),
    ]
    if classes_hash:
        chunks.append(classes_hash.encode("utf-8"))
    chunks.append(a.tobytes())
    
    prediction_hash = _hash_bytes(chunks)
    
    # Compute live hash with quantization (for drift detection)
    prediction_hash_live = None
    if compute_live_hash and kind != "label":
        a_live = canonicalize_preds(preds, kind=kind, quantize=live_quantize)
        chunks_live = [
            kind.encode("utf-8"),
            str(a_live.shape).encode("utf-8"),
            str(a_live.dtype).encode("utf-8"),
            row_ids_hash.encode("utf-8"),
        ]
        if classes_hash:
            chunks_live.append(classes_hash.encode("utf-8"))
        chunks_live.append(a_live.tobytes())
        prediction_hash_live = _hash_bytes(chunks_live)
    
    return PredictionFingerprint(
        prediction_hash=prediction_hash,
        prediction_hash_live=prediction_hash_live,
        row_ids_hash=row_ids_hash,
        classes_hash=classes_hash,
        kind=kind,
        dtype=str(a.dtype),
        shape=list(a.shape),
        quantize=quantize,
    )


def compute_prediction_fingerprint_for_model(
    preds: np.ndarray,
    proba: Optional[np.ndarray],
    model: Any,
    task_type: str,
    X: Any,
    strict_mode: bool = False,
) -> Optional[dict]:
    """
    Compute prediction fingerprint for a trained model.
    
    This is the main helper for wiring prediction hashing into model evaluation.
    Handles row ID extraction, binary proba normalization, and strict mode.
    
    Args:
        preds: Model predictions (y_pred)
        proba: Model probabilities (y_proba) - required for classification
        model: Trained model (for classes_ extraction)
        task_type: "REGRESSION", "BINARY_CLASSIFICATION", or "MULTICLASS_CLASSIFICATION"
        X: Feature matrix (DataFrame or ndarray) - used for row ID extraction
        strict_mode: If True, raise on failure; if False, return None
    
    Returns:
        Dictionary with fingerprint data, or None if failed (non-strict mode)
    
    Example:
        fp_dict = compute_prediction_fingerprint_for_model(
            preds=y_pred,
            proba=y_proba,
            model=model,
            task_type="REGRESSION",
            X=X_val,
            strict_mode=False,
        )
        if fp_dict:
            model_metrics[model_name]['prediction_fingerprint'] = fp_dict
    """
    try:
        # Extract row IDs: prefer DataFrame index, fallback to arange
        if hasattr(X, 'index'):
            # DataFrame - use index
            row_ids = X.index.astype(str).values
        else:
            # ndarray - use positional indices
            n_samples = len(preds) if preds is not None else len(proba)
            row_ids = np.arange(n_samples).astype(str)
        
        # Normalize task_type string (handle both enum and string)
        task_str = str(task_type).upper()
        if "REGRESSION" in task_str:
            kind = "regression"
            to_hash = preds
            class_order = None
        elif "BINARY" in task_str:
            kind = "binary_proba"
            # CRITICAL: Always extract positive-class proba as (n,) to avoid shape ambiguity
            if proba is not None:
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    to_hash = proba[:, 1]  # Positive class probability
                elif proba.ndim == 1:
                    to_hash = proba  # Already (n,)
                else:
                    to_hash = proba.ravel()  # Flatten
            else:
                to_hash = preds  # Fallback to predictions
            class_order = getattr(model, 'classes_', None)
        else:  # MULTICLASS
            kind = "multiclass_proba"
            to_hash = proba if proba is not None else preds
            class_order = getattr(model, 'classes_', None)
        
        fp = prediction_fingerprint(
            preds=to_hash,
            row_ids=row_ids,
            kind=kind,
            class_order=class_order,
        )
        return fp.to_dict()
        
    except Exception as e:
        if strict_mode:
            raise RuntimeError(f"Prediction fingerprint failed in strict mode: {e}") from e
        logger.debug(f"Prediction fingerprint computation failed: {e}")
        return None


def compare_prediction_fingerprints(
    fp1: PredictionFingerprint,
    fp2: PredictionFingerprint,
    strict: bool = True,
) -> dict:
    """
    Compare two prediction fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        strict: If True, compare strict hashes; if False, compare live hashes
    
    Returns:
        Dictionary with comparison results
    """
    result = {
        "comparable": True,
        "match": False,
        "reason": None,
    }
    
    # Check if inputs are the same
    if fp1.row_ids_hash != fp2.row_ids_hash:
        result["comparable"] = False
        result["reason"] = "Different row_ids (inputs not identical)"
        return result
    
    # Check if prediction types are the same
    if fp1.kind != fp2.kind:
        result["comparable"] = False
        result["reason"] = f"Different prediction kinds: {fp1.kind} vs {fp2.kind}"
        return result
    
    # Check class order for classification
    if fp1.classes_hash != fp2.classes_hash:
        result["comparable"] = False
        result["reason"] = "Different class order"
        return result
    
    # Compare hashes
    if strict:
        result["match"] = fp1.prediction_hash == fp2.prediction_hash
        if not result["match"]:
            result["reason"] = f"Strict hash mismatch: {fp1.prediction_hash[:16]}... vs {fp2.prediction_hash[:16]}..."
    else:
        if fp1.prediction_hash_live and fp2.prediction_hash_live:
            result["match"] = fp1.prediction_hash_live == fp2.prediction_hash_live
            if not result["match"]:
                result["reason"] = f"Live hash mismatch (drift detected): {fp1.prediction_hash_live[:16]}... vs {fp2.prediction_hash_live[:16]}..."
        else:
            result["comparable"] = False
            result["reason"] = "Live hash not available for comparison"
    
    return result
