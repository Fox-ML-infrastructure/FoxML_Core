"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import contextlib
import random

import numpy as np
import pandas as pd

from .model_interface import Model


def set_seeds(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def build_features(prices: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    df = prices.copy()
    feats = {}
    if "ret_1d" in feature_list:
        feats["ret_1d"] = df["Close"].pct_change()
    if "ret_5d" in feature_list:
        feats["ret_5d"] = df["Close"].pct_change(5)
    if "vol_10d" in feature_list:
        feats["vol_10d"] = df["Close"].pct_change().rolling(10).std()
    F = pd.DataFrame(feats, index=df.index).dropna()
    return F


def _map_scores_to_weights(scores: np.ndarray, map_name: str, max_abs: float) -> np.ndarray:
    if map_name == "linear":
        w = scores
    elif map_name == "softmax":
        ex = np.exp(scores - scores.max())
        w = ex / max(ex.sum(), 1e-9)
    else:  # tanh
        w = np.tanh(scores)
    return np.clip(w, -max_abs, max_abs)


def infer_weights(
    model: Model,
    feat_df: pd.DataFrame,
    feature_order: list[str],
    map_name: str,
    max_abs: float,
    min_bars: int,
) -> dict[str, float] | dict:
    if len(feat_df) < min_bars:
        return {"status": "HOLD", "reason": "insufficient_history"}
    # Feature alignment; require columns to exist
    for col in feature_order:
        if col not in feat_df.columns:
            return {"status": "FAIL", "reason": "feature_mismatch"}
    X = feat_df[feature_order].to_numpy(dtype="float64")
    scores = model.predict(X[-1:].copy())
    arr = np.asarray(scores, dtype="float64").reshape(-1)
    if not np.isfinite(arr).all():
        return {"status": "HOLD", "reason": "nan_in_scores"}
    w = _map_scores_to_weights(arr, map_name, max_abs)
    if not np.isfinite(w).all():
        return {"status": "HOLD", "reason": "nan_in_scores"}
    return {i: float(w[i]) for i in range(len(w))}


def detect_weight_spikes(
    previous: dict[str, float] | None, current: dict[str, float], max_delta: float
) -> list[str]:
    if not previous:
        return []
    spikes: list[str] = []
    for symbol, weight in current.items():
        prev = float(previous.get(symbol, 0.0))
        if abs(weight - prev) > max_delta:
            spikes.append(symbol)
    return spikes


def compute_turnover(previous: dict[str, float] | None, current: dict[str, float]) -> float:
    if not previous:
        return float(sum(abs(float(w)) for w in current.values()))
    all_symbols = set(previous.keys()) | set(current.keys())
    t = 0.0
    for s in all_symbols:
        t += abs(float(current.get(s, 0.0)) - float(previous.get(s, 0.0)))
    return float(t)
