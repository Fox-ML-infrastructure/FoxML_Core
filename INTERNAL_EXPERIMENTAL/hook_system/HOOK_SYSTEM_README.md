# Pipeline Hook System

**Purpose:** Add new features (like leakage detection/fixing) without refactoring orchestrator code.

## The Problem

Currently, adding a new feature like leakage detection requires:
1. Modifying `intelligent_trainer.py` to import the module
2. Adding calls at specific points
3. Handling errors and edge cases
4. Extensive testing to ensure nothing breaks

**This is fragile and time-consuming.**

## The Solution: Hook Registry (Hardened)

A centralized hook system where:
- **Orchestrator** defines hook points (one line each)
- **New modules** register themselves (no orchestrator changes)
- **Failures are graceful** (hooks never break the pipeline)
- **Hardened features:**
  - Deduplication (prevents double registration)
  - Deterministic ordering (priority + registration index)
  - Error modes (continue/raise/disable)
  - Execution trace (observability)
  - Controlled plugin loading

## Quick Start

### 1. Add Hook Points in Orchestrator (One-Time Setup)

In `intelligent_trainer.py`, add hook calls at key points:

```python
from TRAINING.common.pipeline_hooks import PipelineHooks

# Before target ranking
context = PipelineHooks.execute('before_target_ranking', {
    'stage': 'target_ranking',
    'output_dir': self.output_dir,
    'symbols': self.symbols
})

rankings = rank_targets(...)

# After target ranking
context = PipelineHooks.execute('after_target_ranking', {
    'stage': 'target_ranking',
    'rankings': rankings,
    'output_dir': self.output_dir
})
```

**That's it for the orchestrator** - just add one line before/after each major stage.

### 2. Create Your New Feature Module

Create `TRAINING/common/my_feature.py`:

```python
from TRAINING.common.pipeline_hooks import register_hook
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@register_hook('after_feature_selection', priority=50)
def my_feature_hook(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your feature logic here.
    
    Receives context dict, can modify it and return it.
    Failures are caught automatically - won't break pipeline.
    """
    target = context.get('target')
    features = context.get('selected_features', [])
    
    # Do your work
    modified_features = do_something(features)
    
    # Update context
    context['selected_features'] = modified_features
    return context
```

### 3. Load Plugins Explicitly (Recommended)

**Option A: Config-driven (recommended)**
In `CONFIG/base.yaml`:
```yaml
pipeline:
  enabled_plugins:
    - TRAINING.common.my_feature
    - TRAINING.common.leakage_detection
```

In orchestrator startup:
```python
from TRAINING.common.plugin_loader import load_plugins_from_config
load_plugins_from_config()  # Loads from config
```

**Option B: Explicit list**
```python
from TRAINING.common.pipeline_hooks import load_plugins
load_plugins(['TRAINING.common.my_feature'])
```

**Option C: Import-based (less reliable)**
In `TRAINING/common/__init__.py`:
```python
try:
    from TRAINING.common import my_feature
except ImportError:
    pass
```

**Done!** Your feature now runs automatically without any orchestrator changes.

## Hook Points Available

See `pipeline_hook_points.md` for complete list. Common ones:

- `before_target_ranking` - Before target ranking starts
- `after_target_ranking` - After rankings complete
- `before_feature_selection` - Before feature selection
- `after_feature_selection` - After features selected (most common)
- `before_training` - Before training starts
- `after_training` - After training completes

## Priority System

Lower priority = executes first:
- **0-50**: Critical hooks (validation, safety)
- **51-100**: Standard hooks
- **101+**: Post-processing (logging, reporting)

## Context Object

Hooks receive a dict with:
- `stage`: Current stage name
- `output_dir`: Output directory
- `target`: Target name (if applicable)
- `selected_features`: Feature list (if applicable)
- `rankings`: Rankings (if applicable)
- `results`: Stage results
- `metadata`: Additional data

**Modify and return context** to pass data to next hooks or stages.

## Error Handling

Hooks support three error modes:

1. **`log_and_continue`** (default): Log error, continue pipeline
2. **`raise`**: Fail fast (useful for CI/strict mode)
3. **`disable_hook`**: Auto-disable failing hooks after first error

Set mode globally:
```python
PipelineHooks.set_error_mode("raise")  # Strict mode
```

Or per execution:
```python
context = PipelineHooks.execute('after_feature_selection', context, error_mode="raise")
```

**Default behavior:** Hooks never break the pipeline (failures are logged and ignored).

## Example: Leakage Detection Hook

```python
from TRAINING.common.pipeline_hooks import register_hook

@register_hook('after_feature_selection', priority=30)  # High priority = runs early
def leakage_detection_hook(context):
    """Detect and fix leakage in selected features."""
    try:
        from TRAINING.common.leakage_auto_fixer import LeakageAutoFixer
        
        target = context['target']
        features = context['selected_features']
        
        fixer = LeakageAutoFixer()
        fixed_features, fix_info = fixer.auto_fix_leakage(
            target_name=target,
            feature_list=features,
            data_dir=context['data_dir'],
            output_dir=context['output_dir']
        )
        
        # Update context with fixed features
        context['selected_features'] = fixed_features
        context['leakage_fix_info'] = fix_info
        
        return context
    except Exception as e:
        logger.warning(f"Leakage detection failed (non-critical): {e}")
        return context  # Return original context on failure
```

## Hardened Features

✅ **Deduplication** - Prevents double registration (reloads, tests)  
✅ **Deterministic ordering** - Priority + registration index (no randomness)  
✅ **Error modes** - Continue/raise/disable (configurable)  
✅ **Execution trace** - Logs hook execution with timing (observability)  
✅ **Controlled loading** - Explicit plugin loading (no import-time surprises)  
✅ **Context contract** - Documented context structure per hook point

## Benefits

✅ **No orchestrator refactoring** - just add hook points once  
✅ **No breaking changes** - hooks are optional  
✅ **Easy to test** - test hooks independently  
✅ **Multiple hooks** - many modules can hook into same point  
✅ **Priority control** - control execution order  
✅ **Graceful degradation** - failures don't break pipeline  
✅ **Observable** - execution traces show what ran and how long  

## Migration Path

1. **Add hook points** to orchestrator (one-time, ~10 lines total)
2. **Migrate existing optional features** to hooks (leakage detection, stability, etc.)
3. **New features** use hooks from the start

**No need to refactor everything at once** - migrate incrementally.
