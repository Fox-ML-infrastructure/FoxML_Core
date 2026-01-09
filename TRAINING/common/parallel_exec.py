# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Callable, Iterable, TypeVar, List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

logger = logging.getLogger(__name__)

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using hardcoded defaults")

# Import threading utilities for thread-safe execution
try:
    from TRAINING.common.threads import default_threads, effective_threads
except ImportError:
    def default_threads():
        return max(1, (os.cpu_count() or 8) - 1)
    def effective_threads(requested: int | None = None) -> int:
        return requested or default_threads()

T = TypeVar('T')
R = TypeVar('R')


def get_max_workers(task_type: str = "process", requested: Optional[int] = None) -> int:
    """
    Get max_workers for parallel execution, respecting config and system limits.
    
    Args:
        task_type: "process" for CPU-bound tasks, "thread" for I/O-bound tasks
        requested: Requested number of workers (None = use config/default)
    
    Returns:
        Max workers (capped by available CPUs and config)
    """
    if requested is not None and requested > 0:
        return min(requested, effective_threads(requested))
    
    # Load from config
    if _CONFIG_AVAILABLE:
        try:
            if task_type == "process":
                max_workers = get_cfg("threading.parallel.max_workers_process", default=None, config_name="threading_config")
            else:
                max_workers = get_cfg("threading.parallel.max_workers_thread", default=None, config_name="threading_config")
            
            if max_workers is not None and max_workers > 0:
                return min(max_workers, effective_threads(max_workers))
        except Exception:
            pass
    
    # Default: use available CPUs (minus 1 for system)
    return effective_threads(default_threads())


def execute_parallel(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: Optional[int] = None,
    task_type: str = "process",
    desc: str = "Processing",
    show_progress: bool = True
) -> List[tuple[T, R]]:
    """
    Execute function on items in parallel, returning results in completion order.
    
    Args:
        func: Function to execute on each item (must be picklable for ProcessPoolExecutor)
        items: Iterable of items to process
        max_workers: Max parallel workers (None = auto-detect from config)
        task_type: "process" for CPU-bound, "thread" for I/O-bound
        desc: Description for logging
        show_progress: Whether to log progress
    
    Returns:
        List of (item, result) tuples in completion order
    
    Example:
        def evaluate_target(target, target_config):
            return evaluate_target_predictability(...)
        
        results = execute_parallel(
            lambda t: evaluate_target(t[0], t[1]),
            targets.items(),
            max_workers=4,
            task_type="process",
            desc="Evaluating targets"
        )
    """
    items_list = list(items)
    if not items_list:
        return []
    
    n_items = len(items_list)
    max_workers = get_max_workers(task_type, max_workers)
    
    # If only one item or max_workers=1, run sequentially
    if n_items == 1 or max_workers == 1:
        if show_progress:
            logger.info(f"{desc}: Running sequentially (1 item or max_workers=1)")
        results = []
        for item in items_list:
            try:
                result = func(item)
                results.append((item, result))
            except Exception as e:
                logger.error(f"{desc}: Failed for {item}: {e}")
                results.append((item, None))
        return results
    
    if show_progress:
        logger.info(f"{desc}: Running in parallel ({n_items} items, {max_workers} workers, {task_type})")
    
    # Choose executor based on task type
    Executor = ProcessPoolExecutor if task_type == "process" else ThreadPoolExecutor
    
    results = []
    completed = 0
    
    with Executor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items_list}
        
        # Process results as they complete
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            completed += 1
            
            try:
                result = future.result()
                results.append((item, result))
                if show_progress and completed % max(1, n_items // 10) == 0:
                    logger.info(f"{desc}: Completed {completed}/{n_items} ({100*completed//n_items}%)")
            except Exception as e:
                logger.error(f"{desc}: Failed for {item}: {e}", exc_info=True)
                results.append((item, None))
    
    if show_progress:
        logger.info(f"{desc}: Completed all {n_items} items")
    
    return results


def execute_parallel_with_context(
    func: Callable[[T, Dict[str, Any]], R],
    items: Iterable[T],
    context: Dict[str, Any],
    max_workers: Optional[int] = None,
    task_type: str = "process",
    desc: str = "Processing",
    show_progress: bool = True
) -> List[tuple[T, R]]:
    """
    Execute function on items in parallel with shared context.
    
    Args:
        func: Function that takes (item, context) and returns result
        items: Iterable of items to process
        context: Shared context dict passed to each function call
        max_workers: Max parallel workers (None = auto-detect)
        task_type: "process" for CPU-bound, "thread" for I/O-bound
        desc: Description for logging
        show_progress: Whether to log progress
    
    Returns:
        List of (item, result) tuples in completion order
    
    Example:
        def evaluate_target_with_context(item, context):
            target, target_config = item
            return evaluate_target_predictability(
                target=target,
                target_config=target_config,
                symbols=context['symbols'],
                data_dir=context['data_dir'],
                ...
            )
        
        results = execute_parallel_with_context(
            evaluate_target_with_context,
            targets.items(),
            context={'symbols': symbols, 'data_dir': data_dir, ...},
            max_workers=4
        )
    """
    # Create partial function with context
    func_with_context = partial(func, context=context)
    return execute_parallel(
        func_with_context,
        items,
        max_workers=max_workers,
        task_type=task_type,
        desc=desc,
        show_progress=show_progress
    )
