# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Training Plan Utilities

Helper functions for working with training plans.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def print_training_plan_summary(training_plan_dir: Path) -> None:
    """
    Print a human-readable summary of the training plan.
    
    Args:
        training_plan_dir: Directory containing master_training_plan.json
    """
    from TRAINING.orchestration.training_plan_consumer import (
        load_training_plan,
        get_training_plan_summary
    )
    
    plan = load_training_plan(training_plan_dir)
    if plan is None:
        print(f"❌ No training plan found at {training_plan_dir}")
        return
    
    summary = get_training_plan_summary(plan)
    
    print("\n" + "="*80)
    print("TRAINING PLAN SUMMARY")
    print("="*80)
    print(f"Generated: {summary.get('generated_at', 'unknown')}")
    print(f"Run ID: {summary.get('run_id', 'unknown')}")
    print(f"\nTotal Jobs: {summary['total_jobs']}")
    print(f"  - Cross-Sectional: {summary['cs_jobs']}")
    print(f"  - Symbol-Specific: {summary['symbol_jobs']}")
    
    print("\nBy Route:")
    for route, count in sorted(summary.get('by_route', {}).items()):
        print(f"  - {route}: {count}")
    
    print("\nBy Type:")
    for job_type, count in sorted(summary.get('by_type', {}).items()):
        print(f"  - {job_type}: {count}")
    
    print("="*80 + "\n")


def compare_training_plans(plan1_path: Path, plan2_path: Path) -> Dict[str, Any]:
    """
    Compare two training plans and return differences.
    
    Args:
        plan1_path: Path to first training plan directory
        plan2_path: Path to second training plan directory
    
    Returns:
        Dict with comparison results
    """
    from TRAINING.orchestration.training_plan_consumer import load_training_plan
    
    plan1 = load_training_plan(plan1_path)
    plan2 = load_training_plan(plan2_path)
    
    if plan1 is None or plan2 is None:
        return {"error": "One or both plans not found"}
    
    jobs1 = {job["job_id"]: job for job in plan1.get("jobs", [])}
    jobs2 = {job["job_id"]: job for job in plan2.get("jobs", [])}
    
    job_ids1 = set(jobs1.keys())
    job_ids2 = set(jobs2.keys())
    
    added = job_ids2 - job_ids1
    removed = job_ids1 - job_ids2
    common = job_ids1 & job_ids2
    
    changed = []
    for job_id in common:
        if jobs1[job_id] != jobs2[job_id]:
            changed.append(job_id)
    
    return {
        "plan1_total": len(jobs1),
        "plan2_total": len(jobs2),
        "added_jobs": sorted(added),
        "removed_jobs": sorted(removed),
        "changed_jobs": sorted(changed),
        "common_jobs": len(common)
    }
