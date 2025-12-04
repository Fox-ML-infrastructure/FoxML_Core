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

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import contextlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

from brokers.paper import PaperBroker
from ml.model_interface import ModelSpec
from ml.registry import load_model
from ml.runtime import (
    build_features,
    compute_turnover,
    detect_weight_spikes,
    infer_weights,
    set_seeds,
)
from utils.ops_runtime import kill_switch, notify_ntfy

try:
    from tools.provenance import write_provenance
except Exception:
    import os
    import sys

    tools_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    try:
        from provenance import write_provenance  # type: ignore
    except Exception:

        def write_provenance(*_a, **_k):
            return {}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,TSLA")
    ap.add_argument("--poll-sec", type=float, default=5.0)
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--profile", default="risk_balanced")
    ap.add_argument("--ntfy", action="store_true")
    args = ap.parse_args(argv)

    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    Path("reports").mkdir(parents=True, exist_ok=True)
    broker = PaperBroker()
    meta = {
        "run_id": run_id,
        "profile": args.profile,
        "symbols": args.symbols.split(","),
        "start": run_id,
    }
    (Path("reports") / "paper_run.meta.json").write_text(json.dumps(meta, indent=2))
    write_provenance("reports/paper_provenance.json", ["config/base.yaml"])

    # Optional model runtime (feature-flagged)
    STATE = Path("reports/runner_state.json")

    def _load_prev_weights() -> dict:
        try:
            if STATE.exists():
                return json.loads(STATE.read_text()).get("prev_weights", {})
        except Exception:
            pass
        return {}

    def _save_prev_weights(prev_weights: dict) -> None:
        try:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps({"prev_weights": prev_weights}, indent=2))
        except Exception:
            pass

    try:
        cfg = yaml.safe_load(Path("config/base.yaml").read_text())
        models_cfg = (cfg or {}).get("models", {}) or {}
        if models_cfg.get("enable", False):
            set_seeds(1337)
            reg = yaml.safe_load(Path("config/models.yaml").read_text())["registry"]
            m_id = models_cfg["selected"]
            spec_cfg = reg[m_id]
            spec = ModelSpec(
                kind=spec_cfg["kind"], path=spec_cfg["path"], metadata=spec_cfg.get("metadata", {})
            )
            model, art_sha = load_model(spec)
            feats_list = models_cfg.get("input_features", [])
            feat_order = (spec.metadata or {}).get("feature_order", feats_list)
            min_bars = int(models_cfg.get("min_history_bars", 120))
            # Load cached CI data if present; fallback to synthetic drift
            weights_by_symbol: dict[str, float] = {}
            prev_weights: dict[str, float] | None = _load_prev_weights()
            model_fallbacks = 0
            for sym in meta["symbols"]:
                cache_pq = Path("data/smoke_cache") / f"{sym}.parquet"
                if cache_pq.exists():
                    df = pd.read_parquet(cache_pq)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                else:
                    # synthetic small series
                    idx = pd.date_range("2020-01-01", periods=180, tz="UTC")
                    # deterministic slight trend
                    close = pd.Series(100.0, index=idx)
                    close = close * (1.0 + 0.0005) ** np.arange(len(idx))
                    df = pd.DataFrame({"Close": close.values}, index=idx)
                if "Close" not in df.columns and df.shape[1] > 0:
                    # pick first as close-like
                    df = pd.DataFrame({"Close": df.iloc[:, 0]}, index=df.index)
                F = build_features(df, feats_list)
                w = infer_weights(
                    model,
                    F,
                    feat_order,
                    models_cfg.get("score_to_weight", "tanh"),
                    float(models_cfg.get("max_abs_weight", 0.5)),
                    min_bars,
                )
                # map index 0 weight to symbol for now
                if isinstance(w, dict) and "status" not in w:
                    # use first weight
                    weights_by_symbol[sym] = float(next(iter(w.values())))
                else:
                    model_fallbacks += 1
            # Tripwires (only evaluate if we have a previous snapshot)
            MAX_DW = 0.25
            spikes = detect_weight_spikes(prev_weights, weights_by_symbol, MAX_DW)
            if spikes:
                meta["model_tripwire"] = {"reason": "weight_spike", "spikes": spikes}
                model_fallbacks += 1
            TURNOVER_CAP = 1.0
            if prev_weights:
                turnover = compute_turnover(prev_weights, weights_by_symbol)
                if turnover > TURNOVER_CAP:
                    meta["model_tripwire_turnover"] = {"turnover": turnover}
                    model_fallbacks += 1
            meta.update(
                {
                    "model_id": m_id,
                    "model_kind": spec.kind,
                    "artifact_sha256": art_sha,
                    "feature_order": feat_order,
                    "model_enabled": True,
                    "model_fallbacks": model_fallbacks,
                }
            )
            # weights_by_symbol is ready for router integration (future step)
            prev_weights = weights_by_symbol.copy()
    except Exception:
        # Non-fatal: log via meta note and continue fallback
        meta["model_runtime"] = "fallback"

    for stopped in kill_switch(interval_s=args.poll_sec):
        if stopped:
            break
        # integrate signal + routing here in next iteration

    meta["stop"] = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    (Path("reports") / "paper_run.meta.json").write_text(json.dumps(meta, indent=2))
    # Persist prev weights across restarts
    with contextlib.suppress(Exception):
        _save_prev_weights(locals().get("prev_weights") or {})
    # End-of-run notification with summary if anomalies occurred
    if args.ntfy:
        fallbacks = int(meta.get("model_fallbacks", 0))
        abnormal = fallbacks > 0 or ("model_tripwire" in meta or "model_tripwire_turnover" in meta)
        title = "Aurora: paper OK" if not abnormal else "Aurora: paper WARN"
        body = {
            "run_id": run_id,
            "fallbacks": fallbacks,
            "tripwire": meta.get("model_tripwire"),
            "turnover": meta.get("model_tripwire_turnover"),
        }
        notify_ntfy(title, body)
        # Auto-issue on anomalies if configured
        if abnormal:
            repo = os.getenv("GITHUB_REPOSITORY", "")
            token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or ""
            if shutil.which("python") and (repo and token):
                with contextlib.suppress(Exception):
                    subprocess.run(["python", "tools/gh_issue.py", repo, token], check=False)


if __name__ == "__main__":
    main()
