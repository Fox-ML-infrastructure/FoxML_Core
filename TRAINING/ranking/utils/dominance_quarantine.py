"""
Dominance Quarantine: Auto-suspect → confirm → quarantine workflow

Detects features with dominant importance (potential leakage indicators),
confirms them via a rerun with suspects removed, and only escalates to
blocking target/view if leakage persists after quarantine.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DominanceConfig:
    """Configuration for dominance quarantine system."""
    enabled: bool
    top1_share: float
    top1_over_top2: float
    hard_top1_share: float
    max_features: int
    confirm_enabled: bool
    rerun_once: bool
    min_samples: int
    min_symbols: int
    auc_drop_abs: float
    auc_drop_rel: float
    
    @classmethod
    def from_config(cls) -> DominanceConfig:
        """Load configuration from safety.yaml."""
        try:
            from CONFIG.config_loader import get_cfg
            cfg = get_cfg("safety.leakage_detection.dominance_quarantine", default={}, config_name="pipeline/training/safety")
            
            if not cfg.get("enabled", False):
                # Return disabled config
                return cls(
                    enabled=False,
                    top1_share=0.30,
                    top1_over_top2=3.0,
                    hard_top1_share=0.40,
                    max_features=3,
                    confirm_enabled=False,
                    rerun_once=False,
                    min_samples=500,
                    min_symbols=3,
                    auc_drop_abs=0.15,
                    auc_drop_rel=0.25
                )
            
            soft = cfg.get("soft", {})
            confirm = cfg.get("confirm", {})
            
            return cls(
                enabled=True,
                top1_share=float(soft.get("top1_share", 0.30)),
                top1_over_top2=float(soft.get("top1_over_top2", 3.0)),
                hard_top1_share=float(soft.get("hard_top1_share", 0.40)),
                max_features=int(soft.get("max_features", 3)),
                confirm_enabled=bool(confirm.get("enabled", True)),
                rerun_once=bool(confirm.get("rerun_once", True)),
                min_samples=int(confirm.get("min_samples", 500)),
                min_symbols=int(confirm.get("min_symbols", 3)),
                auc_drop_abs=float(confirm.get("auc_drop_abs", 0.15)),
                auc_drop_rel=float(confirm.get("auc_drop_rel", 0.25))
            )
        except Exception as e:
            logger.warning(f"Failed to load dominance_quarantine config: {e}, using defaults (disabled)")
            return cls(
                enabled=False,
                top1_share=0.30,
                top1_over_top2=3.0,
                hard_top1_share=0.40,
                max_features=3,
                confirm_enabled=False,
                rerun_once=False,
                min_samples=500,
                min_symbols=3,
                auc_drop_abs=0.15,
                auc_drop_rel=0.25
            )


@dataclass(frozen=True)
class Suspect:
    """A feature suspected of being leaky based on dominance metrics."""
    model_name: str
    feature: str
    top1_share: float  # Top feature's share of total importance (0-1)
    top1_over_top2: float  # Ratio of top1 to top2 importance


@dataclass(frozen=True)
class ConfirmResult:
    """Result of confirm pass (rerun with suspects removed)."""
    confirmed: bool
    reason: str
    pre_auc: float
    post_auc: float
    drop_abs: float
    drop_rel: float
    suspects: List[Suspect]


def _dominance_metrics(importance_pct: Dict[str, float]) -> Optional[Tuple[str, float, float]]:
    """
    Compute dominance metrics from importance percentages.
    
    Args:
        importance_pct: Dict of feature -> importance percentage (sums to ~100)
    
    Returns:
        Tuple of (top_feature, top1_share, top1_over_top2) or None if insufficient data
    """
    if not importance_pct or len(importance_pct) < 2:
        return None
    
    items = sorted(importance_pct.items(), key=lambda kv: kv[1], reverse=True)
    (f1, v1), (f2, v2) = items[0], items[1]
    share = v1 / 100.0  # Convert percentage to 0-1 scale
    ratio = (v1 / v2) if v2 > 0 else float("inf")
    return f1, share, ratio


def detect_suspects(
    per_model_importance_pct: Dict[str, Dict[str, float]],
    cfg: DominanceConfig,
) -> List[Suspect]:
    """
    Detect features with dominant importance across models.
    
    Args:
        per_model_importance_pct: Dict of model_name -> feature -> importance percentage
        cfg: Dominance configuration
    
    Returns:
        List of Suspect objects (unique features, prioritized by strongest dominance)
    """
    if not cfg.enabled:
        return []
    
    suspects: List[Suspect] = []
    
    for model_name, imp_pct in per_model_importance_pct.items():
        m = _dominance_metrics(imp_pct)
        if m is None:
            continue
        
        f1, share, ratio = m
        
        # Check if feature meets suspect criteria
        if share >= cfg.hard_top1_share or (share >= cfg.top1_share and ratio >= cfg.top1_over_top2):
            suspects.append(Suspect(
                model_name=model_name,
                feature=f1,
                top1_share=share,
                top1_over_top2=ratio
            ))
    
    # Keep only top-N unique features, prioritizing strongest dominance
    suspects.sort(key=lambda s: (s.top1_share, s.top1_over_top2), reverse=True)
    uniq: List[Suspect] = []
    seen = set()
    for s in suspects:
        if s.feature in seen:
            continue
        uniq.append(s)
        seen.add(s.feature)
        if len(uniq) >= cfg.max_features:
            break
    
    return uniq


def write_suspects_artifact(
    output_dir: Path,
    target: str,
    view: str,
    symbol: Optional[str] = None
) -> Path:
    """
    Write suspects artifact to disk.
    
    Args:
        output_dir: Base output directory (run root)
        target: Target name
        view: View type (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)
        symbol: Symbol name (for SYMBOL_SPECIFIC) or None
    
    Returns:
        Path to written artifact
    """
    from TRAINING.orchestration.utils.target_first_paths import target_repro_dir
    
    symbol_str = symbol if symbol else "ALL"
    repro_dir = target_repro_dir(output_dir, target, view, symbol=symbol, stage="TARGET_RANKING")
    qdir = repro_dir / "feature_quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    
    # Note: suspects list should be passed in, but for now we'll write empty and update later
    # This function signature will be updated when integrated
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "target": target,
        "view": view,
        "symbol": symbol_str,
        "suspects": []  # Will be populated by caller
    }
    
    p = qdir / f"suspects_{view}_{symbol_str}.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


def write_suspects_artifact_with_data(
    output_dir: Path,
    target: str,
    view: str,
    suspects: List[Suspect],
    symbol: Optional[str] = None
) -> Path:
    """
    Write suspects artifact with actual suspect data.
    
    Args:
        output_dir: Base output directory (run root)
        target: Target name
        view: View type
        suspects: List of Suspect objects
        symbol: Symbol name or None
    
    Returns:
        Path to written artifact
    """
    from TRAINING.orchestration.utils.target_first_paths import target_repro_dir
    
    symbol_str = symbol if symbol else "ALL"
    repro_dir = target_repro_dir(output_dir, target, view, symbol=symbol, stage="TARGET_RANKING")
    qdir = repro_dir / "feature_quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "target": target,
        "view": view,
        "symbol": symbol_str,
        "suspects": [s.__dict__ for s in suspects]
    }
    
    p = qdir / f"suspects_{view}_{symbol_str}.json"
    p.write_text(json.dumps(payload, indent=2))
    logger.info(f"✅ Wrote suspects artifact: {p} ({len(suspects)} suspects)")
    return p


def confirm_quarantine(
    pre_auc: float,
    post_auc: float,
    suspects: List[Suspect],
    n_samples: int,
    n_symbols: int,
    cfg: DominanceConfig,
) -> ConfirmResult:
    """
    Evaluate confirm result based on score drops.
    
    Args:
        pre_auc: Mean score before removing suspects
        post_auc: Mean score after removing suspects
        suspects: List of suspects that were removed
        n_samples: Number of samples
        n_symbols: Number of symbols
        cfg: Dominance configuration
    
    Returns:
        ConfirmResult with confirmed status and metrics
    """
    if not cfg.confirm_enabled:
        return ConfirmResult(
            confirmed=False,
            reason="confirm_disabled",
            pre_auc=pre_auc,
            post_auc=post_auc,
            drop_abs=0.0,
            drop_rel=0.0,
            suspects=suspects
        )
    
    if n_samples < cfg.min_samples or n_symbols < cfg.min_symbols:
        return ConfirmResult(
            confirmed=False,
            reason="insufficient_data_for_confirm",
            pre_auc=pre_auc,
            post_auc=post_auc,
            drop_abs=0.0,
            drop_rel=0.0,
            suspects=suspects
        )
    
    drop_abs = pre_auc - post_auc
    drop_rel = (drop_abs / abs(pre_auc)) if pre_auc != 0 else 0.0
    
    confirmed = (drop_abs >= cfg.auc_drop_abs) or (drop_rel >= cfg.auc_drop_rel)
    reason = "score_collapse" if confirmed else "no_collapse"
    
    return ConfirmResult(
        confirmed=confirmed,
        reason=reason,
        pre_auc=pre_auc,
        post_auc=post_auc,
        drop_abs=drop_abs,
        drop_rel=drop_rel,
        suspects=suspects
    )


def persist_confirmed_quarantine(
    output_dir: Path,
    target: str,
    suspects: List[Suspect],
    view: str,
    symbol: Optional[str] = None
) -> Path:
    """
    Persist confirmed quarantine to disk.
    
    Args:
        output_dir: Base output directory (run root)
        target: Target name
        suspects: List of confirmed suspects
        view: View type
        symbol: Symbol name or None
    
    Returns:
        Path to written artifact
    """
    from TRAINING.orchestration.utils.target_first_paths import target_repro_dir
    
    symbol_str = symbol if symbol else "ALL"
    repro_dir = target_repro_dir(output_dir, target, view, symbol=symbol, stage="TARGET_RANKING")
    qdir = repro_dir / "feature_quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "target": target,
        "view": view,
        "symbol": symbol_str,
        "confirmed_features": sorted({s.feature for s in suspects}),
        "evidence": [s.__dict__ for s in suspects]
    }
    
    p = qdir / "confirmed_quarantine.json"
    p.write_text(json.dumps(payload, indent=2))
    logger.info(f"✅ Persisted confirmed quarantine: {p} ({len(suspects)} features)")
    return p


def load_confirmed_quarantine(
    output_dir: Path,
    target: str,
    view: str,
    symbol: Optional[str] = None
) -> set:
    """
    Load confirmed quarantine features from disk.
    
    Args:
        output_dir: Base output directory (run root)
        target: Target name
        view: View type
        symbol: Symbol name or None
    
    Returns:
        Set of confirmed quarantined feature names
    """
    from TRAINING.orchestration.utils.target_first_paths import target_repro_dir
    
    try:
        repro_dir = target_repro_dir(output_dir, target, view, symbol=symbol, stage="TARGET_RANKING")
        qdir = repro_dir / "feature_quarantine"
        p = qdir / "confirmed_quarantine.json"
        
        if p.exists():
            with open(p, 'r') as f:
                payload = json.load(f)
            confirmed_features = set(payload.get("confirmed_features", []))
            if confirmed_features:
                logger.debug(f"Loaded {len(confirmed_features)} confirmed quarantine features for {target}/{view}/{symbol or 'ALL'}")
            return confirmed_features
    except Exception as e:
        logger.debug(f"Could not load confirmed quarantine for {target}/{view}/{symbol or 'ALL'}: {e}")
    
    return set()

