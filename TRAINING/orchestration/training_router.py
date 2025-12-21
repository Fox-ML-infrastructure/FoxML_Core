"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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

"""
Training Router

Determines training strategy (cross-sectional, symbol-specific, both, experimental, or blocked)
for each (target, symbol) pair based on metrics from feature selection, stability analysis,
and leakage detection.

This is the "quant infra brain" that makes reproducible, config-driven decisions about
where to train models.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import yaml
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RouteState(str, Enum):
    """Training route states."""
    ROUTE_CROSS_SECTIONAL = "ROUTE_CROSS_SECTIONAL"
    ROUTE_SYMBOL_SPECIFIC = "ROUTE_SYMBOL_SPECIFIC"
    ROUTE_BOTH = "ROUTE_BOTH"
    ROUTE_EXPERIMENTAL_ONLY = "ROUTE_EXPERIMENTAL_ONLY"
    ROUTE_BLOCKED = "ROUTE_BLOCKED"


class SignalState(str, Enum):
    """Signal quality states."""
    STRONG = "STRONG"
    WEAK_BUT_OK = "WEAK_BUT_OK"
    EXPERIMENTAL = "EXPERIMENTAL"
    DISALLOWED = "DISALLOWED"


class StabilityCategory(str, Enum):
    """Stability categories."""
    STABLE = "STABLE"
    DRIFTING = "DRIFTING"
    DIVERGED = "DIVERGED"
    UNKNOWN = "UNKNOWN"


class LeakageStatus(str, Enum):
    """Leakage detection status."""
    SAFE = "SAFE"
    SUSPECT = "SUSPECT"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


@dataclass
class CrossSectionalMetrics:
    """Cross-sectional metrics for a target."""
    target: str
    score: float
    score_ci_low: Optional[float] = None
    score_ci_high: Optional[float] = None
    stability: StabilityCategory = StabilityCategory.UNKNOWN
    sample_size: int = 0
    leakage_status: LeakageStatus = LeakageStatus.UNKNOWN
    feature_set_id: Optional[str] = None
    failed_model_families: List[str] = field(default_factory=list)
    stability_metrics: Optional[Dict[str, float]] = None  # mean_overlap, mean_tau, etc.


@dataclass
class SymbolMetrics:
    """Symbol-specific metrics for a (target, symbol) pair."""
    target: str
    symbol: str
    score: float
    score_ci_low: Optional[float] = None
    score_ci_high: Optional[float] = None
    stability: StabilityCategory = StabilityCategory.UNKNOWN
    sample_size: int = 0
    leakage_status: LeakageStatus = LeakageStatus.UNKNOWN
    feature_set_id: Optional[str] = None
    failed_model_families: List[str] = field(default_factory=list)
    model_status: str = "UNKNOWN"  # OK, FAILED, SKIPPED
    stability_metrics: Optional[Dict[str, float]] = None


@dataclass
class RoutingDecision:
    """Routing decision for a (target, symbol) pair."""
    target: str
    symbol: str
    route: RouteState
    cs_state: SignalState
    local_state: SignalState
    reasons: List[str]
    cs_metrics: Optional[CrossSectionalMetrics] = None
    local_metrics: Optional[SymbolMetrics] = None


class TrainingRouter:
    """
    Training router that makes routing decisions based on metrics and config.
    """
    
    def __init__(self, routing_config: Dict[str, Any]):
        """
        Initialize router with config.
        
        Args:
            routing_config: Routing configuration dict (from routing_config.yaml)
        """
        self.config = routing_config.get("routing", {})
        self._validate_config()
    
    def _validate_config(self):
        """Validate routing config has required keys."""
        required = [
            "min_sample_size",
            "cross_sectional",
            "symbol",
            "stability_allowlist",
            "both_strong_behavior"
        ]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required routing config key: {key}")
    
    def classify_stability(
        self,
        stability_metrics: Optional[Dict[str, float]]
    ) -> StabilityCategory:
        """
        Classify stability from metrics.
        
        Args:
            stability_metrics: Dict with mean_overlap, std_overlap, mean_tau, std_tau
        
        Returns:
            StabilityCategory
        """
        if stability_metrics is None:
            return StabilityCategory.UNKNOWN
        
        classification_rules = self.config.get("stability_classification", {})
        
        mean_overlap = stability_metrics.get("mean_overlap", np.nan)
        std_overlap = stability_metrics.get("std_overlap", np.nan)
        mean_tau = stability_metrics.get("mean_tau", np.nan)
        std_tau = stability_metrics.get("std_tau", np.nan)
        
        # Check for divergence (high variance)
        max_std_overlap = classification_rules.get("max_std_overlap", 0.20)
        max_std_tau = classification_rules.get("max_std_tau", 0.25)
        
        if not np.isnan(std_overlap) and std_overlap > max_std_overlap:
            return StabilityCategory.DIVERGED
        if not np.isnan(std_tau) and std_tau > max_std_tau:
            return StabilityCategory.DIVERGED
        
        # Check for stability
        stable_overlap_min = classification_rules.get("stable_overlap_min", 0.70)
        stable_tau_min = classification_rules.get("stable_tau_min", 0.60)
        
        if (not np.isnan(mean_overlap) and mean_overlap >= stable_overlap_min and
            not np.isnan(mean_tau) and mean_tau >= stable_tau_min):
            return StabilityCategory.STABLE
        
        # Check for drifting
        drifting_overlap_min = classification_rules.get("drifting_overlap_min", 0.50)
        drifting_tau_min = classification_rules.get("drifting_tau_min", 0.40)
        
        if (not np.isnan(mean_overlap) and mean_overlap >= drifting_overlap_min and
            not np.isnan(mean_tau) and mean_tau >= drifting_tau_min):
            return StabilityCategory.DRIFTING
        
        return StabilityCategory.UNKNOWN
    
    def evaluate_cross_sectional_eligibility(
        self,
        cs_metrics: CrossSectionalMetrics
    ) -> SignalState:
        """
        Evaluate cross-sectional eligibility.
        
        Args:
            cs_metrics: Cross-sectional metrics
        
        Returns:
            SignalState
        """
        min_sample_size = self.config["min_sample_size"]["cross_sectional"]
        block_on_leakage = self.config.get("block_on_leakage", True)
        cs_config = self.config["cross_sectional"]
        stability_allowlist = self.config["stability_allowlist"]["cross_sectional"]
        experimental_config = self.config.get("experimental", {})
        enable_experimental = self.config.get("enable_experimental_lane", False)
        
        # Hard blocks
        if cs_metrics.sample_size < min_sample_size:
            return SignalState.DISALLOWED
        
        if block_on_leakage and cs_metrics.leakage_status == LeakageStatus.BLOCKED:
            return SignalState.DISALLOWED
        
        # Check feature safety (if required)
        if self.config.get("require_safe_features_only", False):
            allowed_statuses = self.config.get("allowed_feature_leakage_status", ["SAFE"])
            # This would need feature-level leakage status - for now, assume CS metrics
            # already account for this
        
        # Check model family failures
        require_min = self.config.get("require_min_successful_families", 1)
        if len(cs_metrics.failed_model_families) >= require_min and len(cs_metrics.failed_model_families) > 0:
            # This is a soft check - we'd need to know total families attempted
            pass
        
        # Scoring / stability rules
        strong_score = cs_config["strong_score"]
        min_score = cs_config["min_score"]
        
        if (cs_metrics.score >= strong_score and
            cs_metrics.stability.value in stability_allowlist):
            return SignalState.STRONG
        
        if (cs_metrics.score >= min_score and
            cs_metrics.stability.value in stability_allowlist):
            return SignalState.WEAK_BUT_OK
        
        # Experimental lane
        if enable_experimental:
            exp_min_score = experimental_config.get("min_score", 0.52)
            exp_allowed_stabilities = experimental_config.get("allowed_stabilities", ["DRIFTING", "UNKNOWN"])
            if (cs_metrics.score >= exp_min_score and
                cs_metrics.stability.value in exp_allowed_stabilities):
                return SignalState.EXPERIMENTAL
        
        return SignalState.DISALLOWED
    
    def evaluate_symbol_eligibility(
        self,
        symbol_metrics: SymbolMetrics
    ) -> SignalState:
        """
        Evaluate symbol-specific eligibility.
        
        Args:
            symbol_metrics: Symbol metrics
        
        Returns:
            SignalState
        """
        min_sample_size = self.config["min_sample_size"]["symbol"]
        block_on_leakage = self.config.get("block_on_leakage", True)
        symbol_config = self.config["symbol"]
        stability_allowlist = self.config["stability_allowlist"]["symbol"]
        experimental_config = self.config.get("experimental", {})
        enable_experimental = self.config.get("enable_experimental_lane", False)
        
        # Hard blocks
        if symbol_metrics.sample_size < min_sample_size:
            return SignalState.DISALLOWED
        
        if block_on_leakage and symbol_metrics.leakage_status == LeakageStatus.BLOCKED:
            return SignalState.DISALLOWED
        
        # Model status check
        if symbol_metrics.model_status == "FAILED":
            # Check if all families failed
            if len(symbol_metrics.failed_model_families) > 0:
                # This is a soft check - would need total families attempted
                pass
        
        # Scoring / stability rules
        strong_score = symbol_config["strong_score"]
        min_score = symbol_config["min_score"]
        
        if (symbol_metrics.score >= strong_score and
            symbol_metrics.stability.value in stability_allowlist):
            return SignalState.STRONG
        
        if (symbol_metrics.score >= min_score and
            symbol_metrics.stability.value in stability_allowlist):
            return SignalState.WEAK_BUT_OK
        
        # Experimental lane
        if enable_experimental:
            exp_min_score = experimental_config.get("min_score", 0.52)
            exp_allowed_stabilities = experimental_config.get("allowed_stabilities", ["DRIFTING", "UNKNOWN"])
            if (symbol_metrics.score >= exp_min_score and
                symbol_metrics.stability.value in exp_allowed_stabilities):
                return SignalState.EXPERIMENTAL
        
        return SignalState.DISALLOWED
    
    def route_target_symbol(
        self,
        target: str,
        symbol: str,
        cs_metrics: Optional[CrossSectionalMetrics],
        symbol_metrics: Optional[SymbolMetrics]
    ) -> RoutingDecision:
        """
        Route a (target, symbol) pair.
        
        Args:
            target: Target name
            symbol: Symbol name
            cs_metrics: Cross-sectional metrics (None if not available)
            symbol_metrics: Symbol metrics (None if not available)
        
        Returns:
            RoutingDecision
        """
        reasons = []
        
        # Evaluate CS eligibility
        if cs_metrics is None:
            cs_state = SignalState.DISALLOWED
            reasons.append("CS: No metrics available")
        else:
            cs_state = self.evaluate_cross_sectional_eligibility(cs_metrics)
            reasons.append(f"CS: {cs_state.value} (score={cs_metrics.score:.3f}, stability={cs_metrics.stability.value})")
        
        # Evaluate symbol eligibility
        if symbol_metrics is None:
            local_state = SignalState.DISALLOWED
            reasons.append("LOCAL: No metrics available")
        else:
            local_state = self.evaluate_symbol_eligibility(symbol_metrics)
            reasons.append(f"LOCAL: {local_state.value} (score={symbol_metrics.score:.3f}, stability={symbol_metrics.stability.value})")
        
        # Combine into route decision (priority-ordered rules)
        route, route_reasons = self._combine_states(cs_state, local_state, cs_metrics, symbol_metrics)
        reasons.extend(route_reasons)
        
        return RoutingDecision(
            target=target,
            symbol=symbol,
            route=route,
            cs_state=cs_state,
            local_state=local_state,
            reasons=reasons,
            cs_metrics=cs_metrics,
            local_metrics=symbol_metrics
        )
    
    def _combine_states(
        self,
        cs_state: SignalState,
        local_state: SignalState,
        cs_metrics: Optional[CrossSectionalMetrics],
        symbol_metrics: Optional[SymbolMetrics]
    ) -> Tuple[RouteState, List[str]]:
        """
        Combine CS and local states into route decision.
        
        Returns:
            (RouteState, list of reason strings)
        """
        reasons = []
        
        # Rule 1: Hard blocks
        cs_blocked = cs_state == SignalState.DISALLOWED
        local_blocked = local_state == SignalState.DISALLOWED
        
        if cs_blocked and local_blocked:
            reasons.append("Both CS and local disallowed")
            return RouteState.ROUTE_BLOCKED, reasons
        
        if cs_blocked and not local_blocked:
            reasons.append("CS disallowed, falling back to local-only")
            return RouteState.ROUTE_SYMBOL_SPECIFIC, reasons
        
        if local_blocked and not cs_blocked:
            reasons.append("Local disallowed, falling back to CS-only")
            return RouteState.ROUTE_CROSS_SECTIONAL, reasons
        
        # Rule 2: CS strong, local disallowed
        if cs_state in [SignalState.STRONG, SignalState.WEAK_BUT_OK] and local_blocked:
            reasons.append("CS available, local not available")
            return RouteState.ROUTE_CROSS_SECTIONAL, reasons
        
        # Rule 3: Local strong, CS disallowed
        if local_state in [SignalState.STRONG, SignalState.WEAK_BUT_OK] and cs_blocked:
            reasons.append("Local strong, CS not available")
            return RouteState.ROUTE_SYMBOL_SPECIFIC, reasons
        
        # Rule 4: Both strong
        if (cs_state in [SignalState.STRONG, SignalState.WEAK_BUT_OK] and
            local_state in [SignalState.STRONG, SignalState.WEAK_BUT_OK]):
            both_behavior = self.config["both_strong_behavior"]
            if both_behavior == "ROUTE_BOTH":
                reasons.append("Both CS and local strong → ROUTE_BOTH")
                return RouteState.ROUTE_BOTH, reasons
            elif both_behavior == "PREFER_CS":
                reasons.append("Both strong, preferring CS")
                return RouteState.ROUTE_CROSS_SECTIONAL, reasons
            elif both_behavior == "PREFER_SYMBOL":
                reasons.append("Both strong, preferring local")
                return RouteState.ROUTE_SYMBOL_SPECIFIC, reasons
        
        # Rule 5: Experimental lane
        enable_experimental = self.config.get("enable_experimental_lane", False)
        if enable_experimental:
            exp_config = self.config.get("experimental", {})
            max_fraction = exp_config.get("max_fraction_symbols_per_target", 0.2)
            
            # Check if either side is experimental
            if (cs_state == SignalState.EXPERIMENTAL or
                local_state == SignalState.EXPERIMENTAL):
                # Note: We'd need to know total symbols per target to enforce max_fraction
                # For now, allow if either is experimental
                reasons.append("Experimental lane enabled")
                return RouteState.ROUTE_EXPERIMENTAL_ONLY, reasons
        
        # Fallback: blocked
        reasons.append("NO_RULE_MATCH")
        return RouteState.ROUTE_BLOCKED, reasons
    
    def generate_routing_plan(
        self,
        routing_candidates: pd.DataFrame,
        output_dir: Path,
        git_commit: Optional[str] = None,
        config_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate routing plan from routing candidates DataFrame.
        
        Args:
            routing_candidates: DataFrame with columns: target, symbol (nullable), mode, score, etc.
            output_dir: Output directory for plan artifacts
            git_commit: Git commit hash
            config_hash: Config hash
        
        Returns:
            Routing plan dict
        """
        # Group by target
        targets = routing_candidates["target"].unique()
        
        plan = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "git_commit": git_commit or "unknown",
                "config_hash": config_hash or "unknown",
                "metrics_snapshot": "routing_candidates.parquet"
            },
            "targets": {}
        }
        
        for target in targets:
            target_rows = routing_candidates[routing_candidates["target"] == target]
            
            # Extract CS metrics
            cs_rows = target_rows[target_rows["mode"] == "CROSS_SECTIONAL"]
            cs_metrics = None
            if len(cs_rows) > 0:
                cs_row = cs_rows.iloc[0]
                cs_metrics = CrossSectionalMetrics(
                    target=target,
                    score=cs_row.get("score", 0.0),
                    score_ci_low=cs_row.get("score_ci_low"),
                    score_ci_high=cs_row.get("score_ci_high"),
                    stability=StabilityCategory(cs_row.get("stability", "UNKNOWN")),
                    sample_size=int(cs_row.get("sample_size", 0)),
                    leakage_status=LeakageStatus(cs_row.get("leakage_status", "UNKNOWN")),
                    feature_set_id=cs_row.get("feature_set_id"),
                    failed_model_families=cs_row.get("failed_model_families", []),
                    stability_metrics=cs_row.get("stability_metrics")
                )
                # Classify stability from metrics if needed
                if cs_metrics.stability == StabilityCategory.UNKNOWN and cs_metrics.stability_metrics:
                    cs_metrics.stability = self.classify_stability(cs_metrics.stability_metrics)
            
            # Extract symbol metrics
            symbol_rows = target_rows[target_rows["mode"] == "SYMBOL"]
            symbols = symbol_rows["symbol"].unique() if "symbol" in symbol_rows.columns else []
            
            if len(symbols) == 0:
                logger.warning(f"  [{target}]: No symbol metrics found in routing candidates (expected for SYMBOL mode rows)")
            
            # Evaluate CS eligibility and get detailed reasons if disabled
            cs_state_eval = None
            if cs_metrics:
                cs_state_eval = self.evaluate_cross_sectional_eligibility(cs_metrics)
            
            cs_route = "ENABLED" if cs_metrics and cs_state_eval != SignalState.DISALLOWED else "DISABLED"
            
            # Build detailed reason if disabled
            if cs_route == "DISABLED":
                reason_parts = []
                if not cs_metrics:
                    reason_parts.append("no_metrics")
                else:
                    min_sample_size = self.config["min_sample_size"]["cross_sectional"]
                    cs_config = self.config["cross_sectional"]
                    stability_allowlist = self.config["stability_allowlist"]["cross_sectional"]
                    
                    if cs_metrics.stability.value not in stability_allowlist:
                        reason_parts.append(f"stability={cs_metrics.stability.value} not in {stability_allowlist}")
                    if cs_metrics.score < cs_config["min_score"]:
                        reason_parts.append(f"score={cs_metrics.score:.3f} < {cs_config['min_score']}")
                    if cs_metrics.sample_size < min_sample_size:
                        reason_parts.append(f"sample_size={cs_metrics.sample_size} < {min_sample_size}")
                    if self.config.get("block_on_leakage", True) and cs_metrics.leakage_status == LeakageStatus.BLOCKED:
                        reason_parts.append(f"leakage_status=BLOCKED")
                
                detailed_reason = "; ".join(reason_parts) if reason_parts else "unknown_reason"
                logger.info(f"    CS DISABLED: {detailed_reason}")
            else:
                logger.info(f"    CS ENABLED: score={cs_metrics.score:.3f}, stability={cs_metrics.stability.value}")
            
            target_plan = {
                "cross_sectional": {
                    "state": cs_metrics.stability.value if cs_metrics else "DISALLOWED",
                    "route": cs_route,
                    "reason": f"score={cs_metrics.score:.3f}, stability={cs_metrics.stability.value}, sample_size={cs_metrics.sample_size}" if cs_metrics else "No CS metrics"
                },
                "symbols": {}
            }
            
            for symbol in symbols:
                sym_rows = symbol_rows[symbol_rows["symbol"] == symbol]
                if len(sym_rows) > 0:
                    sym_row = sym_rows.iloc[0]
                    sym_metrics = SymbolMetrics(
                        target=target,
                        symbol=symbol,
                        score=sym_row.get("score", 0.0),
                        score_ci_low=sym_row.get("score_ci_low"),
                        score_ci_high=sym_row.get("score_ci_high"),
                        stability=StabilityCategory(sym_row.get("stability", "UNKNOWN")),
                        sample_size=int(sym_row.get("sample_size", 0)),
                        leakage_status=LeakageStatus(sym_row.get("leakage_status", "UNKNOWN")),
                        feature_set_id=sym_row.get("feature_set_id"),
                        failed_model_families=sym_row.get("failed_model_families", []),
                        model_status=sym_row.get("model_status", "UNKNOWN"),
                        stability_metrics=sym_row.get("stability_metrics")
                    )
                    # Classify stability from metrics if needed
                    if sym_metrics.stability == StabilityCategory.UNKNOWN and sym_metrics.stability_metrics:
                        sym_metrics.stability = self.classify_stability(sym_metrics.stability_metrics)
                    
                    # Route this (target, symbol)
                    decision = self.route_target_symbol(target, symbol, cs_metrics, sym_metrics)
                    
                    target_plan["symbols"][symbol] = {
                        "route": decision.route.value,
                        "cs_state": decision.cs_state.value,
                        "local_state": decision.local_state.value,
                        "reason": decision.reasons
                    }
            
            plan["targets"][target] = target_plan
        
        # Save plan
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dump_formats = self.config.get("dump_plan_as", ["JSON"])
        if "JSON" in dump_formats:
            json_path = output_dir / "routing_plan.json"
            with open(json_path, "w") as f:
                json.dump(plan, f, indent=2)
            logger.info(f"✅ Saved routing plan JSON: {json_path}")
        
        if "YAML" in dump_formats:
            yaml_path = output_dir / "routing_plan.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(plan, f, default_flow_style=False)
            logger.info(f"✅ Saved routing plan YAML: {yaml_path}")
        
        if "MARKDOWN" in dump_formats:
            md_path = output_dir / "routing_plan.md"
            self._write_markdown_report(plan, md_path)
            logger.info(f"✅ Saved routing plan Markdown: {md_path}")
        
        return plan
    
    def _write_markdown_report(self, plan: Dict[str, Any], output_path: Path):
        """Write human-readable Markdown report."""
        with open(output_path, "w") as f:
            f.write("# Training Routing Plan\n\n")
            f.write(f"**Generated:** {plan['metadata']['generated_at']}\n")
            f.write(f"**Git Commit:** {plan['metadata']['git_commit']}\n")
            f.write(f"**Config Hash:** {plan['metadata']['config_hash']}\n\n")
            
            # Overall summary
            total_targets = len(plan["targets"])
            total_symbol_decisions = sum(len(t.get("symbols", {})) for t in plan["targets"].values())
            
            route_counts_all = {}
            for target_data in plan["targets"].values():
                symbols = target_data.get("symbols", {})
                for sym_data in symbols.values():
                    route = sym_data['route']
                    route_counts_all[route] = route_counts_all.get(route, 0) + 1
            
            f.write("## Overall Summary\n\n")
            f.write(f"- **Total Targets:** {total_targets}\n")
            f.write(f"- **Total Symbol Decisions:** {total_symbol_decisions}\n\n")
            f.write("**Route Distribution:**\n")
            for route, count in sorted(route_counts_all.items(), key=lambda x: -x[1]):
                f.write(f"- {route}: {count} symbols\n")
            f.write("\n---\n\n")
            
            f.write("## Summary by Target\n\n")
            
            for target, target_data in plan["targets"].items():
                f.write(f"### {target}\n\n")
                
                cs_info = target_data["cross_sectional"]
                f.write(f"**Cross-Sectional:** {cs_info['route']} ({cs_info['state']})\n")
                f.write(f"- {cs_info['reason']}\n\n")
                
                symbols = target_data.get("symbols", {})
                if symbols:
                    f.write("**Symbol Routing:**\n\n")
                    f.write("| Symbol | Route | CS State | Local State | Reasons |\n")
                    f.write("|--------|-------|----------|-------------|----------|\n")
                    
                    for symbol, sym_data in symbols.items():
                        reasons_str = "; ".join(sym_data.get('reason', []))[:100]  # Truncate long reasons
                        f.write(f"| {symbol} | {sym_data['route']} | {sym_data['cs_state']} | {sym_data['local_state']} | {reasons_str} |\n")
                    
                    f.write("\n")
                    
                    # Count routes
                    route_counts = {}
                    for sym_data in symbols.values():
                        route = sym_data['route']
                        route_counts[route] = route_counts.get(route, 0) + 1
                    
                    f.write("**Route Distribution:**\n")
                    for route, count in sorted(route_counts.items()):
                        f.write(f"- {route}: {count} symbols\n")
                    f.write("\n")
                
                f.write("\n")
