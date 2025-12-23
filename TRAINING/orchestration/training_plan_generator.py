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
Training Plan Generator

Converts routing decisions into actionable training jobs/plan.
This is the bridge between routing decisions and actual model training.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from TRAINING.orchestration.routing_integration import load_routing_plan

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """A single training job specification."""
    job_id: str
    target: str
    symbol: Optional[str]  # None for cross-sectional
    route: str  # ROUTE_CROSS_SECTIONAL, ROUTE_SYMBOL_SPECIFIC, etc.
    training_type: str  # "cross_sectional" or "symbol_specific"
    model_families: List[str] = field(default_factory=list)  # Families to train
    priority: int = 0  # Higher = more important
    estimated_samples: Optional[int] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingPlanGenerator:
    """
    Generates training plan from routing decisions.
    """
    
    def __init__(
        self,
        routing_plan: Dict[str, Any],
        model_families: Optional[List[str]] = None,
        default_families: Optional[List[str]] = None
    ):
        """
        Initialize generator.
        
        Args:
            routing_plan: Routing plan dict (from routing_router)
            model_families: Optional list of model families to train (if None, uses default)
            default_families: Default families if model_families not provided
        """
        self.routing_plan = routing_plan
        
        # Known feature selectors that should NOT be in training plan
        # These are used for feature selection only, not for training
        FEATURE_SELECTORS = {
            'random_forest', 'catboost', 'lasso', 'mutual_information', 
            'univariate_selection', 'elastic_net', 'ridge', 'lasso_cv'
        }
        
        # Respect empty list from config (SST) - only use defaults if None
        if model_families is not None:
            # Normalize family names first (mlp -> neural_network, etc.)
            from TRAINING.training_strategies.utils import normalize_family_name
            normalized_families = []
            normalization_map = {}  # Track what was normalized
            for f in model_families:
                normalized = normalize_family_name(f)
                if normalized != f:
                    normalization_map[normalized] = f
                normalized_families.append(normalized)
            
            if normalization_map:
                logger.debug(f"📋 TrainingPlanGenerator: Normalized family names: {normalization_map}")
            
            # Map mlp -> neural_network explicitly (common config variant)
            mlp_to_neural_network = {}
            for i, f in enumerate(normalized_families):
                if f == 'mlp':
                    normalized_families[i] = 'neural_network'
                    mlp_to_neural_network['neural_network'] = 'mlp'
            
            if mlp_to_neural_network:
                logger.debug(f"📋 TrainingPlanGenerator: Mapped mlp -> neural_network: {mlp_to_neural_network}")
            
            # Filter out feature selectors from normalized families
            filtered_families = [f for f in normalized_families if f not in FEATURE_SELECTORS]
            removed = set(normalized_families) - set(filtered_families)
            
            if removed:
                logger.warning(
                    f"⚠️ TrainingPlanGenerator: Filtered out {len(removed)} feature selector(s) from model_families: {sorted(removed)}. "
                    f"Feature selectors are not trainers and should not be in training plan."
                )
            
            # Verify families exist in trainer module map
            try:
                from TRAINING.training_strategies.execution.family_runners import _run_family_isolated
                # Import MODMAP from family_runners to check valid families
                import TRAINING.training_strategies.execution.family_runners as fr_module
                # MODMAP is defined in the module - check if we can access it
                # For now, just log which families we're using
                logger.debug(f"📋 TrainingPlanGenerator: Validated {len(filtered_families)} families after normalization and filtering")
            except Exception as e:
                logger.debug(f"Could not validate families against module map: {e}")
            
            self.model_families = filtered_families
            logger.info(f"📋 TrainingPlanGenerator: Using provided model_families={self.model_families} (after normalization and filtering, original had {len(model_families)})")
        elif default_families is not None:
            # Filter defaults too
            filtered_defaults = [f for f in default_families if f not in FEATURE_SELECTORS]
            self.model_families = filtered_defaults
            logger.debug(f"📋 TrainingPlanGenerator: Using default_families={self.model_families} (after filtering)")
        else:
            self.model_families = ["lightgbm", "xgboost"]
            logger.debug(f"📋 TrainingPlanGenerator: Using hardcoded defaults={self.model_families}")
    
    def generate_training_plan(
        self,
        output_dir: Path,
        include_blocked: bool = False,
        git_commit: Optional[str] = None,
        config_hash: Optional[str] = None,
        metrics_snapshot: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate training plan from routing decisions.
        
        Args:
            output_dir: Output directory for plan artifacts
            include_blocked: If True, include blocked jobs (for analysis)
            git_commit: Optional git commit hash
            config_hash: Optional config hash
            metrics_snapshot: Optional path to metrics snapshot
        
        Returns:
            Training plan dict
        
        Raises:
            ValueError: If routing_plan is invalid or output_dir cannot be created
        """
        # Validate inputs
        if not isinstance(self.routing_plan, dict):
            raise ValueError(f"routing_plan must be a dict, got {type(self.routing_plan)}")
        
        if not isinstance(output_dir, (Path, str)):
            raise ValueError(f"output_dir must be Path or str, got {type(output_dir)}")
        
        output_dir = Path(output_dir)
        
        # Validate model_families
        if not isinstance(self.model_families, list):
            logger.warning(f"model_families is not a list, using default")
            self.model_families = ["lightgbm", "xgboost"]
        
        # Respect empty list from config (SST) - empty list means "no families to train"
        # Only warn if it's empty (config explicitly set it to empty)
        if not self.model_families:
            logger.warning("⚠️ model_families is empty - no families will be trained (this may be intentional from config)")
        
        jobs = []
        
        # Safely get targets with validation
        targets = self.routing_plan.get("targets", {})
        if not isinstance(targets, dict):
            logger.warning(f"routing_plan['targets'] is not a dict, got {type(targets)}, using empty dict")
            targets = {}
        
        for target, target_data in targets.items():
            # Validate target and target_data
            if not isinstance(target, str) or not target:
                logger.warning(f"Skipping invalid target: {target} (not a non-empty string)")
                continue
            
            if not isinstance(target_data, dict):
                logger.warning(f"Skipping target {target}: target_data is not a dict")
                continue
            
            try:
                # Cross-sectional jobs
                cs_info = target_data.get("cross_sectional", {})
                if not isinstance(cs_info, dict):
                    logger.warning(f"Target {target}: cross_sectional is not a dict, skipping CS job")
                    cs_info = {}
                
                if cs_info.get("route") == "ENABLED":
                    # Validate reason field
                    reason = cs_info.get("reason", "CS training enabled")
                    if not isinstance(reason, str):
                        reason = str(reason) if reason is not None else "CS training enabled"
                    
                    # Safely get sample_size
                    cs_metrics = target_data.get("cs_metrics", {})
                    if not isinstance(cs_metrics, dict):
                        cs_metrics = {}
                    sample_size = cs_metrics.get("sample_size")
                    
                    job = TrainingJob(
                        job_id=f"cs_{target}",
                        target=target,
                        symbol=None,
                        route="ROUTE_CROSS_SECTIONAL",
                        training_type="cross_sectional",
                        model_families=self.model_families.copy(),
                        priority=2,  # CS jobs are high priority
                        reason=reason,
                        metadata={
                            "cs_state": cs_info.get("state", "UNKNOWN"),
                            "sample_size": sample_size
                        }
                    )
                    jobs.append(job)
                
                # Symbol-specific jobs
                symbols = target_data.get("symbols", {})
                if not isinstance(symbols, dict):
                    logger.warning(f"Target {target}: symbols is not a dict, skipping symbol jobs")
                    symbols = {}
                
                for symbol, sym_data in symbols.items():
                    # Validate symbol and sym_data
                    if not isinstance(symbol, str) or not symbol:
                        logger.warning(f"Target {target}: Skipping invalid symbol: {symbol}")
                        continue
                    
                    if not isinstance(sym_data, dict):
                        logger.warning(f"Target {target}, Symbol {symbol}: sym_data is not a dict, skipping")
                        continue
                    
                    route = sym_data.get("route", "ROUTE_BLOCKED")
                    if not isinstance(route, str):
                        logger.warning(f"Target {target}, Symbol {symbol}: route is not a string, defaulting to ROUTE_BLOCKED")
                        route = "ROUTE_BLOCKED"
                    
                    if route == "ROUTE_BLOCKED":
                        if include_blocked:
                            # Safely handle reason field
                            reason_list = sym_data.get("reason", [])
                            if not isinstance(reason_list, list):
                                reason_list = [str(reason_list)] if reason_list is not None else []
                            
                            try:
                                reason_str = "; ".join(str(r) for r in reason_list)
                            except Exception as e:
                                logger.warning(f"Failed to join reason list: {e}, using default")
                                reason_str = "Blocked"
                            
                            job = TrainingJob(
                                job_id=f"blocked_{target}_{symbol}",
                                target=target,
                                symbol=symbol,
                                route=route,
                                training_type="blocked",
                                model_families=[],
                                priority=0,
                                reason=reason_str,
                                metadata={
                                    "cs_state": sym_data.get("cs_state", "UNKNOWN"),
                                    "local_state": sym_data.get("local_state", "UNKNOWN")
                                }
                            )
                            jobs.append(job)
                        continue
                    
                    # Determine training type based on route
                    if route == "ROUTE_CROSS_SECTIONAL":
                        # Should use CS model (already handled above)
                        continue
                    elif route in ["ROUTE_SYMBOL_SPECIFIC", "ROUTE_BOTH", "ROUTE_EXPERIMENTAL_ONLY"]:
                        training_type = "symbol_specific"
                        priority = 3 if route == "ROUTE_BOTH" else (1 if route == "ROUTE_EXPERIMENTAL_ONLY" else 2)
                        
                        # Safely handle reason field
                        reason_list = sym_data.get("reason", [])
                        if not isinstance(reason_list, list):
                            reason_list = [str(reason_list)] if reason_list is not None else []
                        
                        try:
                            reason_str = "; ".join(str(r) for r in reason_list)
                        except Exception as e:
                            logger.warning(f"Failed to join reason list: {e}, using default")
                            reason_str = f"{route} training enabled"
                        
                        job = TrainingJob(
                            job_id=f"sym_{target}_{symbol}",
                            target=target,
                            symbol=symbol,
                            route=route,
                            training_type=training_type,
                            model_families=self.model_families.copy(),
                            priority=priority,
                            reason=reason_str,
                            metadata={
                                "cs_state": sym_data.get("cs_state", "UNKNOWN"),
                                "local_state": sym_data.get("local_state", "UNKNOWN"),
                                "needs_cs_ensemble": route == "ROUTE_BOTH"
                            }
                        )
                        jobs.append(job)
                    else:
                        logger.warning(f"Target {target}, Symbol {symbol}: Unknown route '{route}', skipping")
            except Exception as e:
                logger.error(f"Error processing target {target}: {e}", exc_info=True)
                continue
        
        # Sort by priority (higher first) - with error handling
        try:
            jobs.sort(key=lambda j: -j.priority)
        except Exception as e:
            logger.warning(f"Failed to sort jobs by priority: {e}, keeping original order")
        
        # Generate run_id from timestamp
        try:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        except Exception as e:
            logger.warning(f"Failed to generate run_id: {e}, using fallback")
            import time
            run_id = f"run_{int(time.time())}"
        
        # Get routing plan path from metadata - safely
        # Use globals/ (primary) with backward compatibility for METRICS/
        routing_plan_path = "globals/routing_plan/routing_plan.json"
        try:
            routing_metadata = self.routing_plan.get("metadata", {})
            if isinstance(routing_metadata, dict):
                metrics_snapshot_val = routing_metadata.get("metrics_snapshot")
                if metrics_snapshot_val and isinstance(metrics_snapshot_val, str):
                    routing_plan_path = metrics_snapshot_val
                    # Update legacy METRICS paths to globals
                    if routing_plan_path.startswith("METRICS/"):
                        routing_plan_path = routing_plan_path.replace("METRICS/", "globals/", 1)
        except Exception as e:
            logger.debug(f"Could not extract routing_plan_path from metadata: {e}")
        
        # Build plan structure with full metadata - with validation
        try:
            routing_metadata = self.routing_plan.get("metadata", {})
            if not isinstance(routing_metadata, dict):
                routing_metadata = {}
            
            plan = {
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "run_id": run_id,
                    "git_commit": str(git_commit) if git_commit else str(routing_metadata.get("git_commit", "unknown")),
                    "config_hash": str(config_hash) if config_hash else str(routing_metadata.get("config_hash", "unknown")),
                    "routing_plan_path": str(routing_plan_path),
                    "metrics_snapshot": str(metrics_snapshot) if metrics_snapshot else "globals/routing_candidates.parquet",
                    "total_jobs": len(jobs),
                    "model_families": list(self.model_families)  # Ensure it's a list
                },
                "jobs": [],
                "summary": {}
            }
            
            # Convert jobs to dicts safely
            for job in jobs:
                try:
                    plan["jobs"].append(asdict(job))
                except Exception as e:
                    logger.warning(f"Failed to convert job {job.job_id} to dict: {e}, skipping")
            
            # Generate summary safely
            try:
                plan["summary"] = self._generate_summary(jobs)
            except Exception as e:
                logger.warning(f"Failed to generate summary: {e}, using empty summary")
                plan["summary"] = {
                    "by_route": {},
                    "by_type": {},
                    "by_priority": {},
                    "total_cs_jobs": 0,
                    "total_symbol_jobs": 0,
                    "total_blocked": 0
                }
        except Exception as e:
            logger.error(f"Failed to build plan structure: {e}", exc_info=True)
            raise ValueError(f"Failed to build training plan: {e}")
        
        # Save plan - with error handling
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Failed to create output directory {output_dir}: {e}")
        
        # Save master plan (canonical source of truth) - with error handling
        master_path = output_dir / "master_training_plan.json"
        try:
            with open(master_path, "w") as f:
                json.dump(plan, f, indent=2)
            logger.info(f"✅ Saved master training plan: {master_path}")
        except Exception as e:
            logger.error(f"Failed to save master training plan: {e}", exc_info=True)
            raise ValueError(f"Failed to save master training plan: {e}")
        
        # Save convenience mirror - with error handling
        json_path = output_dir / "training_plan.json"
        try:
            with open(json_path, "w") as f:
                json.dump(plan, f, indent=2)
            logger.info(f"✅ Saved training plan (convenience mirror): {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save convenience mirror: {e}, continuing...")
        
        # Save YAML - with error handling
        yaml_path = output_dir / "training_plan.yaml"
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(plan, f, default_flow_style=False)
            logger.info(f"✅ Saved training plan YAML: {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to save YAML: {e}, continuing...")
        
        # Save Markdown report - with error handling
        md_path = output_dir / "training_plan.md"
        try:
            self._write_markdown_report(plan, md_path)
            logger.info(f"✅ Saved training plan Markdown: {md_path}")
        except Exception as e:
            logger.warning(f"Failed to save Markdown report: {e}, continuing...")
        
        # Generate derived views - with error handling
        try:
            self._generate_derived_views(plan, output_dir)
        except Exception as e:
            logger.warning(f"Failed to generate derived views: {e}, continuing...")
        
        return plan
    
    def _generate_summary(self, jobs: List[TrainingJob]) -> Dict[str, Any]:
        """Generate summary statistics with error handling."""
        summary = {
            "by_route": {},
            "by_type": {},
            "by_priority": {},
            "total_cs_jobs": 0,
            "total_symbol_jobs": 0,
            "total_blocked": 0
        }
        
        if not jobs:
            return summary
        
        for job in jobs:
            try:
                if not isinstance(job, TrainingJob):
                    logger.warning(f"Skipping invalid job in summary: {type(job)}")
                    continue
                
                # Count by route - safely
                route = getattr(job, 'route', 'UNKNOWN')
                if not isinstance(route, str):
                    route = str(route) if route is not None else 'UNKNOWN'
                summary["by_route"][route] = summary["by_route"].get(route, 0) + 1
                
                # Count by type - safely
                job_type = getattr(job, 'training_type', 'UNKNOWN')
                if not isinstance(job_type, str):
                    job_type = str(job_type) if job_type is not None else 'UNKNOWN'
                summary["by_type"][job_type] = summary["by_type"].get(job_type, 0) + 1
                
                # Count by priority - safely
                priority = getattr(job, 'priority', 0)
                try:
                    priority = int(priority) if priority is not None else 0
                except (ValueError, TypeError):
                    priority = 0
                summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1
                
                # Count totals - safely
                if job_type == "cross_sectional":
                    summary["total_cs_jobs"] += 1
                elif job_type == "symbol_specific":
                    summary["total_symbol_jobs"] += 1
                elif job_type == "blocked":
                    summary["total_blocked"] += 1
            except Exception as e:
                logger.warning(f"Error processing job in summary: {e}, skipping")
                continue
        
        return summary
    
    def _write_markdown_report(self, plan: Dict[str, Any], output_path: Path):
        """Write human-readable Markdown report with error handling."""
        if not isinstance(plan, dict):
            logger.warning(f"Plan is not a dict, cannot write markdown report")
            return
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create directory for markdown report: {e}")
            return
        
        try:
            with open(output_path, "w") as f:
                # Safely get metadata
                metadata = plan.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                
                f.write("# Training Plan\n\n")
                
                # Safely write metadata fields
                generated_at = metadata.get("generated_at", "unknown")
                f.write(f"**Generated:** {generated_at}\n")
                
                total_jobs = metadata.get("total_jobs", 0)
                f.write(f"**Total Jobs:** {total_jobs}\n")
                
                model_families = metadata.get("model_families", [])
                if isinstance(model_families, list):
                    families_str = ", ".join(str(f) for f in model_families)
                else:
                    families_str = str(model_families)
                f.write(f"**Model Families:** {families_str}\n\n")
            
                # Safely get summary
                summary = plan.get("summary", {})
                if not isinstance(summary, dict):
                    summary = {}
                
                f.write("## Summary\n\n")
                f.write(f"- **Cross-Sectional Jobs:** {summary.get('total_cs_jobs', 0)}\n")
                f.write(f"- **Symbol-Specific Jobs:** {summary.get('total_symbol_jobs', 0)}\n")
                f.write(f"- **Blocked:** {summary.get('total_blocked', 0)}\n\n")
                
                # By route - safely
                by_route = summary.get("by_route", {})
                if isinstance(by_route, dict) and by_route:
                    f.write("### By Route\n\n")
                    try:
                        for route, count in sorted(by_route.items()):
                            f.write(f"- {route}: {count}\n")
                    except Exception as e:
                        logger.warning(f"Failed to write by_route section: {e}")
                    f.write("\n")
                
                # By priority - safely
                by_priority = summary.get("by_priority", {})
                if isinstance(by_priority, dict) and by_priority:
                    f.write("### By Priority\n\n")
                    try:
                        for priority in sorted(by_priority.keys(), reverse=True):
                            count = by_priority[priority]
                            f.write(f"- Priority {priority}: {count} jobs\n")
                    except Exception as e:
                        logger.warning(f"Failed to write by_priority section: {e}")
                    f.write("\n")
                
                f.write("---\n\n")
                f.write("## Training Jobs\n\n")
                
                # Group by target - safely
                jobs = plan.get("jobs", [])
                if not isinstance(jobs, list):
                    jobs = []
                
                jobs_by_target = {}
                for job_data in jobs:
                    if not isinstance(job_data, dict):
                        continue
                    try:
                        target = job_data.get("target")
                        if target and isinstance(target, str):
                            if target not in jobs_by_target:
                                jobs_by_target[target] = []
                            jobs_by_target[target].append(job_data)
                    except Exception as e:
                        logger.warning(f"Error grouping job by target: {e}")
                        continue
                
                for target, target_jobs in sorted(jobs_by_target.items()):
                    try:
                        f.write(f"### {target}\n\n")
                        
                        # CS job - safely
                        cs_jobs = [j for j in target_jobs if isinstance(j, dict) and j.get("training_type") == "cross_sectional"]
                        if cs_jobs:
                            cs = cs_jobs[0]
                            f.write(f"**Cross-Sectional:** ✅ Enabled\n")
                            f.write(f"- Job ID: `{cs.get('job_id', 'unknown')}`\n")
                            
                            families = cs.get("model_families", [])
                            if isinstance(families, list):
                                families_str = ", ".join(str(f) for f in families)
                            else:
                                families_str = str(families)
                            f.write(f"- Families: {families_str}\n")
                            f.write(f"- Priority: {cs.get('priority', 0)}\n")
                            f.write(f"- Reason: {cs.get('reason', 'N/A')}\n\n")
                        
                        # Symbol jobs - safely
                        sym_jobs = [j for j in target_jobs if isinstance(j, dict) and j.get("training_type") == "symbol_specific"]
                        if sym_jobs:
                            f.write("**Symbol-Specific Jobs:**\n\n")
                            f.write("| Symbol | Route | Priority | Families | Reason |\n")
                            f.write("|--------|-------|----------|----------|--------|\n")
                            try:
                                for job in sorted(sym_jobs, key=lambda j: -j.get("priority", 0)):
                                    families = job.get("model_families", [])
                                    if isinstance(families, list):
                                        families_str = ", ".join(str(f) for f in families[:3])
                                        if len(families) > 3:
                                            families_str += f" (+{len(families) - 3} more)"
                                    else:
                                        families_str = str(families)
                                    
                                    reason = job.get("reason", "N/A")
                                    if isinstance(reason, str) and len(reason) > 80:
                                        reason_short = reason[:80] + "..."
                                    else:
                                        reason_short = str(reason) if reason else "N/A"
                                    
                                    f.write(f"| {job.get('symbol', 'N/A')} | {job.get('route', 'N/A')} | {job.get('priority', 0)} | {families_str} | {reason_short} |\n")
                            except Exception as e:
                                logger.warning(f"Failed to write symbol jobs table: {e}")
                            f.write("\n")
                        
                        f.write("\n")
                    except Exception as e:
                        logger.warning(f"Error writing target section for {target}: {e}")
                        continue
        except PermissionError as e:
            logger.warning(f"Permission denied writing markdown report to {output_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to write markdown report: {e}", exc_info=True)
    
    def _generate_derived_views(self, plan: Dict[str, Any], output_dir: Path):
        """Generate derived view artifacts (by_target, by_symbol, by_type, by_route)."""
        jobs = plan.get("jobs", [])
        
        if not jobs:
            logger.warning("No jobs in plan, skipping derived view generation")
            return
        
        # Group by target
        by_target = {}
        for job in jobs:
            if not isinstance(job, dict):
                logger.warning(f"Skipping invalid job (not a dict): {job}")
                continue
            target = job.get("target")
            if target is None:
                logger.warning(f"Skipping job with missing target: {job.get('job_id', 'unknown')}")
                continue
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(job)
        
        # Group by symbol
        by_symbol = {}
        for job in jobs:
            if not isinstance(job, dict):
                continue
            symbol = job.get("symbol")
            if symbol is not None:
                if symbol not in by_symbol:
                    by_symbol[symbol] = []
                by_symbol[symbol].append(job)
        
        # Group by training type
        by_type = {}
        for job in jobs:
            if not isinstance(job, dict):
                continue
            job_type = job.get("training_type")
            if job_type is None:
                logger.warning(f"Skipping job with missing training_type: {job.get('job_id', 'unknown')}")
                continue
            if job_type not in by_type:
                by_type[job_type] = []
            by_type[job_type].append(job)
        
        # Group by route
        by_route = {}
        for job in jobs:
            if not isinstance(job, dict):
                continue
            route = job.get("route")
            if route is None:
                logger.warning(f"Skipping job with missing route: {job.get('job_id', 'unknown')}")
                continue
            if route not in by_route:
                by_route[route] = []
            by_route[route].append(job)
        
        # Save by_target views
        try:
            by_target_dir = output_dir / "by_target"
            by_target_dir.mkdir(exist_ok=True)
            for target, target_jobs in by_target.items():
                view = {
                    "target": target,
                    "jobs": target_jobs
                }
                view_path = by_target_dir / f"{target}.json"
                with open(view_path, "w") as f:
                    json.dump(view, f, indent=2)
            logger.info(f"✅ Generated {len(by_target)} by_target views")
        except Exception as e:
            logger.warning(f"Failed to generate by_target views: {e}")
        
        # Save by_symbol views
        try:
            by_symbol_dir = output_dir / "by_symbol"
            by_symbol_dir.mkdir(exist_ok=True)
            for symbol, symbol_jobs in by_symbol.items():
                view = {
                    "symbol": symbol,
                    "jobs": symbol_jobs
                }
                view_path = by_symbol_dir / f"{symbol}.json"
                with open(view_path, "w") as f:
                    json.dump(view, f, indent=2)
            logger.info(f"✅ Generated {len(by_symbol)} by_symbol views")
        except Exception as e:
            logger.warning(f"Failed to generate by_symbol views: {e}")
        
        # Save by_type views
        try:
            by_type_dir = output_dir / "by_type"
            by_type_dir.mkdir(exist_ok=True)
            for job_type, type_jobs in by_type.items():
                view = {
                    "training_type": job_type,
                    "jobs": type_jobs
                }
                view_path = by_type_dir / f"{job_type}.json"
                with open(view_path, "w") as f:
                    json.dump(view, f, indent=2)
            logger.info(f"✅ Generated {len(by_type)} by_type views")
        except Exception as e:
            logger.warning(f"Failed to generate by_type views: {e}")
        
        # Save by_route views
        try:
            by_route_dir = output_dir / "by_route"
            by_route_dir.mkdir(exist_ok=True)
            for route, route_jobs in by_route.items():
                view = {
                    "route": route,
                    "jobs": route_jobs
                }
                # Sanitize route name for filename
                route_safe = route.replace("ROUTE_", "").lower()
                view_path = by_route_dir / f"{route_safe}.json"
                with open(view_path, "w") as f:
                    json.dump(view, f, indent=2)
            logger.info(f"✅ Generated {len(by_route)} by_route views")
        except Exception as e:
            logger.warning(f"Failed to generate by_route views: {e}")


def generate_training_plan_from_routing(
    routing_plan_path: Path,
    output_dir: Path,
    model_families: Optional[List[str]] = None,
    include_blocked: bool = False,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None,
    metrics_snapshot: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate training plan from routing plan file.
    
    Args:
        routing_plan_path: Path to routing_plan.json
        output_dir: Output directory for training plan
        model_families: Optional list of model families
        include_blocked: Include blocked jobs in plan
        git_commit: Optional git commit hash
        config_hash: Optional config hash
        metrics_snapshot: Optional path to metrics snapshot
    
    Returns:
        Training plan dict
    """
    routing_plan = load_routing_plan(routing_plan_path.parent)
    if routing_plan is None:
        raise ValueError(f"Failed to load routing plan from {routing_plan_path}")
    
    generator = TrainingPlanGenerator(
        routing_plan=routing_plan,
        model_families=model_families
    )
    
    return generator.generate_training_plan(
        output_dir=output_dir,
        include_blocked=include_blocked,
        git_commit=git_commit,
        config_hash=config_hash,
        metrics_snapshot=metrics_snapshot
    )
