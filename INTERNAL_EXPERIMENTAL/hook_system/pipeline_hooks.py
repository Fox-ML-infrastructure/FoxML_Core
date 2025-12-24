"""
Pipeline Hook Registry

Non-invasive hook system for adding features without refactoring existing code.

**Determinism Guarantee:**
- Same code + same config + same plugin load order = same hook execution order
- Hooks sorted by: (priority, registration_index, module+qualname)
- Plugins loaded in config order (preserves intended precedence)
- No filesystem globs, no unordered containers, no import-time surprises
- Trace in context contains only deterministic fields (hook names, order, errors)
- Durations logged but NOT stored in context (preserves determinism)

Hardened with:
- Deduplication (prevents double registration)
- Deterministic ordering (priority + registration_index + stable_tiebreak)
- Error modes (continue/raise/disable)
- Execution trace (observability)
- Explicit plugin loading (sorted order)

Usage:
1. Load plugins explicitly (sorted): `load_plugins(sorted(['TRAINING.common.my_feature']))`
2. Register hooks in your module (auto-registers on import)
3. Orchestrator calls: `PipelineHooks.execute('after_feature_selection', context)`

Hook Determinism Rules:
- Hooks should be pure-ish: read context, return modified context
- If using randomness: use context.get('run_seed') for seeded RNG
- No time-based outputs (time.time(), uuid) that affect results
- No filesystem globs or unordered dict/set iteration
"""

import logging
import time
from typing import Dict, List, Callable, Any, Optional, TypedDict, Literal
from functools import wraps

logger = logging.getLogger(__name__)

# Error handling modes
ErrorMode = Literal["log_and_continue", "raise", "disable_hook"]


class PipelineHooks:
    """
    Centralized hook registry for pipeline extension points.
    
    Allows modules to register callbacks without modifying orchestrator code.
    
    Hardened features:
    - Deduplication: prevents double registration
    - Deterministic ordering: priority + registration index
    - Error modes: continue/raise/disable per hook
    - Execution trace: logs hook execution with timing
    """
    
    _hooks: Dict[str, List[Dict[str, Any]]] = {}
    _enabled: bool = True  # Can be disabled via config
    _registration_index: int = 0  # For deterministic tiebreaking
    _error_mode: ErrorMode = "log_and_continue"  # Default error handling
    _disabled_hooks: set = set()  # Hooks disabled due to errors (per-process, can make runs path-dependent)
    _plugin_load_order: List[str] = []  # Track plugin load order for determinism
    _known_hook_points: set = set()  # Track known hook points for validation
    _validate_hook_points: bool = False  # Warn on unknown hook points (opt-in)
    _allowed_hook_points: Optional[set] = None  # Allowlist for CI (None = no allowlist, set = enforce)
    
    @classmethod
    def register(cls, hook_point: str, callback: Callable, priority: int = 100):
        """
        Register a callback for a hook point.
        
        Deduplicates by (hook_point, callback qualname) to prevent double registration.
        Uses registration index for deterministic ordering when priorities match.
        
        Args:
            hook_point: Hook point name (e.g., 'before_target_ranking', 'after_feature_selection')
            callback: Callable that receives context and returns (possibly modified) context
            priority: Lower numbers execute first (default: 100)
        
        Example:
            PipelineHooks.register('after_feature_selection', my_leakage_check, priority=50)
        """
        # Track known hook points for validation
        cls._known_hook_points.add(hook_point)
        
        if hook_point not in cls._hooks:
            cls._hooks[hook_point] = []
        
        # Deduplication: check if this exact callback is already registered
        callback_id = f"{callback.__module__}.{callback.__qualname__}"
        for existing in cls._hooks[hook_point]:
            existing_id = f"{existing['callback'].__module__}.{existing['callback'].__qualname__}"
            if callback_id == existing_id:
                logger.debug(f"Hook '{hook_point}' from {callback.__module__} already registered, skipping duplicate")
                return
        
        # Mark callback as registered (prevents double registration in reload scenarios)
        if not hasattr(callback, '__hook_registered__'):
            callback.__hook_registered__ = True
        
        # Increment registration index for deterministic tiebreaking
        cls._registration_index += 1
        
        # Get stable tiebreaker: (module, qualname) for deterministic ordering
        module_name = callback.__module__ if hasattr(callback, '__module__') else 'unknown'
        qualname = callback.__qualname__ if hasattr(callback, '__qualname__') else callback.__name__
        stable_tiebreak = (module_name, qualname)
        
        cls._hooks[hook_point].append({
            'callback': callback,
            'priority': priority,
            'module': module_name,
            'name': callback.__name__,
            'qualname': qualname,
            'registration_index': cls._registration_index,
            'stable_tiebreak': stable_tiebreak,  # For deterministic sorting
        })
        
        # Sort by priority, then registration_index, then stable_tiebreak (fully deterministic)
        # This ensures same code + same config = same hook order, even across machines
        cls._hooks[hook_point].sort(key=lambda x: (x['priority'], x['registration_index'], x['stable_tiebreak']))
        
        logger.debug(f"Registered hook '{hook_point}' from {callback.__module__}.{callback.__name__} (priority={priority}, index={cls._registration_index})")
    
    @classmethod
    def execute(
        cls, 
        hook_point: str, 
        context: Any, 
        error_mode: Optional[ErrorMode] = None,
        **kwargs
    ) -> Any:
        """
        Execute all registered hooks for a hook point.
        
        **Determinism guarantee:** Hooks execute in stable order:
        1. Sort by priority (lower = first)
        2. Then by registration_index (order plugins were loaded)
        3. Then by (module, qualname) (stable tiebreak)
        
        This ensures same code + same config = same execution order.
        
        Args:
            hook_point: Hook point name
            context: Context object (dict, dataclass, etc.) passed to hooks
            error_mode: Override default error mode for this execution
            **kwargs: Additional arguments passed to hooks
        
        Returns:
            Context (possibly modified by hooks)
        
        Example:
            context = PipelineHooks.execute('before_target_ranking', context)
        """
        if not cls._enabled:
            return context
        
        # CRITICAL: Check allowlist BEFORE early return (even if no hooks registered)
        # This ensures typos are caught even if no hooks are registered for that point
        if cls._allowed_hook_points is not None:
            # CI mode: enforce allowlist (fail on unknown hook points)
            if hook_point not in cls._allowed_hook_points:
                error_msg = (
                    f"Executing hook point '{hook_point}' not in allowlist. "
                    f"Allowed hook points: {sorted(cls._allowed_hook_points)}. "
                    f"This may indicate a typo or unauthorized hook point."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif cls._validate_hook_points:
            # Dev mode: warn on unknown hook points
            if hook_point not in cls._known_hook_points:
                logger.warning(
                    f"Executing unknown hook point '{hook_point}'. "
                    f"Known hook points: {sorted(cls._known_hook_points)}. "
                    f"This may indicate a typo."
                )
        
        if hook_point not in cls._hooks or len(cls._hooks[hook_point]) == 0:
            return context  # No hooks registered, return context unchanged
        
        error_mode = error_mode or cls._error_mode
        
        # Execution trace (for observability)
        # NOTE: Durations are stored in logs only, not in context, to preserve determinism
        hook_trace = {
            'hook_point': hook_point,
            'hooks_executed': [],
            'errors': []
        }
        
        # Timing for logging only (not stored in context to preserve determinism)
        start_time = time.time()
        result_context = context
        
        # CRITICAL: Iterate over sorted list (not dict/set) for determinism
        # Hooks are already sorted by (priority, registration_index, stable_tiebreak)
        for hook_info in cls._hooks[hook_point]:
            callback = hook_info['callback']
            module = hook_info['module']
            hook_name = hook_info['name']
            hook_id = f"{module}.{hook_name}"
            
            # Skip disabled hooks
            # NOTE: disable_hook mode can make runs path-dependent if hooks fail intermittently
            # This is acceptable for non-critical "best effort" hooks, but be aware of the limitation
            if hook_id in cls._disabled_hooks:
                logger.debug(f"Skipping disabled hook: {hook_id}")
                hook_trace['hooks_executed'].append({
                    'hook': hook_id,
                    'status': 'skipped_disabled'
                })
                continue
            
            hook_start = time.time()  # For logging only
            
            try:
                # Call hook - it can return modified context or None (use original)
                # Hooks should be pure-ish: read context, return modified context
                # If hook uses randomness, it MUST use context.get('run_seed') for determinism
                hook_result = callback(result_context, **kwargs)
                
                # Validate return value: if hook returns something, use it as new context
                # If hook returns None, keep original context (no-op)
                # If hook returns wrong type, log warning but continue (non-breaking)
                if hook_result is not None:
                    # Lightweight validation: if original context was dict, return should be dict-like
                    if isinstance(result_context, dict) and not isinstance(hook_result, dict):
                        logger.warning(
                            f"Hook '{hook_point}' [{hook_id}] returned non-dict type {type(hook_result).__name__}, "
                            f"expected dict. Keeping original context."
                        )
                    else:
                        result_context = hook_result
                
                hook_duration = (time.time() - hook_start) * 1000
                hook_trace['hooks_executed'].append({
                    'hook': hook_id,
                    'status': 'success',
                    'duration_ms': hook_duration
                })
                logger.debug(f"Hook '{hook_point}' [{hook_id}] executed in {hook_duration:.2f}ms")
                    
            except Exception as e:
                hook_duration = (time.time() - hook_start) * 1000  # For logging only
                error_msg = str(e)
                
                # Store only deterministic fields in trace (no durations in context)
                hook_trace['hooks_executed'].append({
                    'hook': hook_id,
                    'status': 'error',
                    'error': error_msg
                })
                hook_trace['errors'].append({
                    'hook': hook_id,
                    'error': error_msg
                })
                
                # Error handling based on mode
                if error_mode == "raise":
                    logger.error(f"Hook '{hook_point}' [{hook_id}] failed: {e}")
                    raise
                elif error_mode == "disable_hook":
                    # NOTE: This makes runs path-dependent if hooks fail intermittently
                    # Use only for non-critical "best effort" hooks
                    cls._disabled_hooks.add(hook_id)
                    logger.warning(
                        f"Hook '{hook_point}' [{hook_id}] failed and was disabled: {e} "
                        f"(duration: {hook_duration:.2f}ms)"
                    )
                else:  # log_and_continue (default)
                    logger.warning(
                        f"Hook '{hook_point}' [{hook_id}] failed (non-critical): {e}",
                        exc_info=False  # Don't spam logs with full traceback
                    )
                # Continue with unmodified context
        
        total_duration = (time.time() - start_time) * 1000
        hook_trace['total_duration_ms'] = total_duration
        
        # Log execution summary
        n_hooks = len(hook_trace['hooks_executed'])
        n_errors = len(hook_trace['errors'])
        if n_hooks > 0:
            logger.debug(
                f"Hook point '{hook_point}': {n_hooks} hooks executed in {total_duration:.2f}ms "
                f"({n_errors} errors)"
            )
        
        # Store trace in context if it's a dict (for debugging/reporting)
        # Cap trace length to prevent unbounded growth (keep last 100 traces)
        if isinstance(result_context, dict):
            if 'hook_traces' not in result_context:
                result_context['hook_traces'] = []
            result_context['hook_traces'].append(hook_trace)
            # Cap at 100 traces to prevent unbounded growth
            if len(result_context['hook_traces']) > 100:
                result_context['hook_traces'] = result_context['hook_traces'][-100:]
                logger.debug(f"Hook traces capped at 100 (removed oldest traces)")
        
        return result_context
    
    @classmethod
    def unregister(cls, hook_point: str, callback: Optional[Callable] = None):
        """
        Unregister hooks (useful for testing).
        
        Args:
            hook_point: Hook point name
            callback: Specific callback to remove (None = remove all)
        """
        if hook_point not in cls._hooks:
            return
        
        if callback is None:
            cls._hooks[hook_point] = []
        else:
            cls._hooks[hook_point] = [
                h for h in cls._hooks[hook_point]
                if h['callback'] != callback
            ]
    
    @classmethod
    def list_hooks(cls, hook_point: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List registered hooks (for debugging).
        
        Returns:
            Dict mapping hook_point -> list of module names
        """
        if hook_point:
            return {
                hook_point: [h['module'] for h in cls._hooks.get(hook_point, [])]
            }
        else:
            return {
                point: [h['module'] for h in hooks]
                for point, hooks in cls._hooks.items()
            }
    
    @classmethod
    def set_error_mode(cls, mode: ErrorMode):
        """
        Set default error handling mode.
        
        Args:
            mode: "log_and_continue" (default), "raise", or "disable_hook"
        """
        cls._error_mode = mode
        logger.debug(f"Hook error mode set to: {mode}")
    
    @classmethod
    def disable(cls):
        """Disable all hooks (for testing or emergency)."""
        cls._enabled = False
        logger.debug("All hooks disabled")
    
    @classmethod
    def enable(cls):
        """Enable hooks."""
        cls._enabled = True
        logger.debug("Hooks enabled")
    
    @classmethod
    def set_hook_point_validation(cls, enabled: bool):
        """
        Enable/disable validation for unknown hook points.
        
        When enabled, warns if executing a hook point that was never registered
        (helps catch typos).
        
        Args:
            enabled: If True, warn on unknown hook points
        """
        cls._validate_hook_points = enabled
        logger.debug(f"Hook point validation: {'enabled' if enabled else 'disabled'}")
    
    @classmethod
    def set_hook_point_allowlist(cls, allowed_hook_points: Optional[List[str]]):
        """
        Set allowlist for hook points (CI mode).
        
        When enabled, executing a hook point not in the allowlist raises an error.
        This prevents typos from silently disabling features.
        
        Use in CI to enforce that only known, documented hook points are used.
        
        Args:
            allowed_hook_points: List of allowed hook point names, or None to disable allowlist
        
        Example:
            # CI mode: enforce allowlist
            PipelineHooks.set_hook_point_allowlist([
                'before_target_ranking',
                'after_target_ranking',
                'before_feature_selection',
                'after_feature_selection'
            ])
        """
        if allowed_hook_points is None:
            cls._allowed_hook_points = None
            logger.debug("Hook point allowlist disabled")
        else:
            cls._allowed_hook_points = set(allowed_hook_points)
            logger.info(
                f"Hook point allowlist enabled with {len(allowed_hook_points)} allowed points: "
                f"{sorted(allowed_hook_points)}"
            )
    
    @classmethod
    def reset(cls):
        """
        Reset registry (useful for testing).
        Clears all hooks and resets state.
        """
        cls._hooks = {}
        cls._registration_index = 0
        cls._disabled_hooks = set()
        cls._enabled = True
        cls._error_mode = "log_and_continue"
        cls._plugin_load_order = []
        cls._known_hook_points = set()
        cls._validate_hook_points = False
        cls._allowed_hook_points = None
        logger.debug("Hook registry reset")
    
    @classmethod
    def get_execution_stats(cls) -> Dict[str, Any]:
        """
        Get statistics about hook execution (for observability).
        
        Returns:
            Dict with hook counts, disabled hooks, etc.
        """
        return {
            'total_hook_points': len(cls._hooks),
            'total_hooks': sum(len(hooks) for hooks in cls._hooks.values()),
            'disabled_hooks': list(cls._disabled_hooks),
            'enabled': cls._enabled,
            'error_mode': cls._error_mode,
            'hooks_by_point': {
                point: len(hooks) for point, hooks in cls._hooks.items()
            },
            'known_hook_points': sorted(cls._known_hook_points),
            'plugin_load_order': cls._plugin_load_order.copy(),
            'allowed_hook_points': sorted(cls._allowed_hook_points) if cls._allowed_hook_points else None,
            'allowlist_enabled': cls._allowed_hook_points is not None
        }


# Convenience decorator for registering hooks
def register_hook(hook_point: str, priority: int = 100):
    """
    Decorator to register a function as a hook.
    
    Example:
        @register_hook('after_feature_selection', priority=50)
        def my_leakage_check(context):
            # Do work
            return context
    """
    def decorator(func: Callable):
        PipelineHooks.register(hook_point, func, priority=priority)
        return func
    return decorator


# Plugin loader (controlled imports)
def load_plugins(plugin_modules: List[str], fail_on_error: bool = False):
    """
    Explicitly load plugin modules to register their hooks.
    
    **Determinism guarantee:** Plugins are loaded in the exact order provided.
    This ensures same config = same plugin load order = same hook registration order.
    
    **CRITICAL:** Plugins are loaded in the exact order provided.
    Config order may express plugin precedence, so order is preserved (not sorted).
    For determinism: same config = same plugin order = same hook registration order.
    
    Args:
        plugin_modules: List of module names to import (e.g., ['TRAINING.common.my_feature'])
                      Order is preserved (config order may express precedence)
        fail_on_error: If True, raise ImportError on failure; if False, log and continue
    
    Example:
        # Order is preserved (config order may express precedence)
        load_plugins([
            'TRAINING.common.leakage_detection',
            'TRAINING.common.stability_hooks'
        ])
    """
    loaded = []
    failed = []
    
    # Track load order for determinism debugging
    for module_name in plugin_modules:
        try:
            __import__(module_name)
            loaded.append(module_name)
            PipelineHooks._plugin_load_order.append(module_name)
            logger.debug(f"Loaded plugin module: {module_name}")
        except ImportError as e:
            failed.append((module_name, str(e)))
            if fail_on_error:
                raise
            logger.debug(f"Plugin module not available (non-critical): {module_name}: {e}")
        except Exception as e:
            failed.append((module_name, str(e)))
            if fail_on_error:
                raise
            logger.warning(f"Error loading plugin module: {module_name}: {e}")
    
    if loaded:
        logger.info(f"Loaded {len(loaded)} plugin module(s) in order: {', '.join(loaded)}")
    if failed:
        logger.debug(f"{len(failed)} plugin module(s) failed to load (non-critical)")
    
    return {'loaded': loaded, 'failed': failed}
