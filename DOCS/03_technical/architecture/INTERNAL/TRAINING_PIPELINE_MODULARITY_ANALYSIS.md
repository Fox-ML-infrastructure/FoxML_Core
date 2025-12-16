# TRAINING Pipeline Modularity Analysis

**Date:** 2025-12-13  
**Purpose:** Assess current architecture and recommend improvements for easier module hook-in

---

## Current Architecture Assessment

### ✅ **What's Working Well**

1. **Hook System (Stability Module)**
   - Non-invasive hooks: `save_snapshot_hook`, `analyze_all_stability_hook`
   - Optional integration (try/except pattern)
   - Clean separation: hooks don't break pipeline if unavailable
   - **Pattern:** `TRAINING/stability/feature_importance/hooks.py`

2. **Abstract Base Classes**
   - `BaseModelTrainer` (ABC) - defines model training interface
   - `BaseTrainingStrategy` (ABC) - defines strategy interface
   - **Pattern:** Clear contracts for extensions

3. **Config-Driven System**
   - ExperimentConfig, FeatureSelectionConfig, TargetRankingConfig
   - Typed configs enable validation and IDE support
   - **Pattern:** Configuration as extension point

4. **Modular Directory Structure**
   - Clear separation: `orchestration/`, `ranking/`, `model_fun/`, `strategies/`
   - Each module has clear responsibilities

---

## Current Limitations

### ❌ **Tight Coupling Issues**

1. **Direct Imports in Orchestrator**
   ```python
   # intelligent_trainer.py
   from TRAINING.ranking import rank_targets, select_features_for_target
   from TRAINING.train_with_strategies import train_models_for_interval_comprehensive
   ```
   - **Problem:** Hard to swap implementations or add alternatives
   - **Impact:** Adding new ranking/selection methods requires modifying orchestrator

2. **Monolithic Orchestrator**
   - `IntelligentTrainer` class handles:
     - Target ranking
     - Feature selection
     - Training execution
     - Caching
     - Directory organization
     - Metadata extraction
   - **Problem:** Single class does too much, hard to extend individual stages

3. **No Pipeline Stage Abstraction**
   - Stages are hardcoded method calls:
     ```python
     rankings = rank_targets(...)
     features = select_features_for_target(...)
     train_models_for_interval_comprehensive(...)
     ```
   - **Problem:** Can't easily insert new stages, reorder, or conditionally skip

4. **Limited Extension Points**
   - Only stability hooks are pluggable
   - Other stages (ranking, selection, training) are direct calls
   - **Problem:** Hard to add pre/post-processing, validation, or alternative implementations

---

## Recommended Refactoring Approach

### **Option A: Pipeline Stage Pattern (Recommended)**

**Concept:** Define pipeline as a sequence of stages, each with a standard interface.

**Benefits:**
- Easy to add/remove/reorder stages
- Each stage is independently testable
- Can conditionally enable/disable stages
- Clear extension points

**Structure:**
```python
# TRAINING/orchestration/pipeline/stages.py

from typing import Protocol, Dict, Any, Optional
from abc import ABC, abstractmethod

class PipelineStage(Protocol):
    """Protocol for pipeline stages"""
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute stage and return updated context"""
        ...
    
    def can_skip(self, context: PipelineContext) -> bool:
        """Check if stage can be skipped"""
        ...

class PipelineContext:
    """Shared context passed between stages"""
    data_dir: Path
    symbols: List[str]
    output_dir: Path
    config: Dict[str, Any]
    results: Dict[str, Any]  # Stage results
    metadata: Dict[str, Any]

# Example stages
class TargetRankingStage(PipelineStage):
    def execute(self, context):
        rankings = rank_targets(...)
        context.results['target_rankings'] = rankings
        return context

class FeatureSelectionStage(PipelineStage):
    def execute(self, context):
        for target in context.results['target_rankings']:
            features = select_features_for_target(...)
            context.results['features'][target] = features
        return context

class TrainingStage(PipelineStage):
    def execute(self, context):
        train_models_for_interval_comprehensive(...)
        return context
```

**Usage:**
```python
pipeline = Pipeline([
    TargetRankingStage(),
    FeatureSelectionStage(),
    TrainingStage(),
    # Easy to add: CustomValidationStage()
])

context = PipelineContext(...)
result = pipeline.run(context)
```

---

### **Option B: Event-Driven Architecture**

**Concept:** Use event emitter pattern for stage communication.

**Benefits:**
- Loose coupling between stages
- Multiple listeners per event
- Easy to add cross-cutting concerns (logging, validation, monitoring)

**Structure:**
```python
# TRAINING/orchestration/pipeline/events.py

class PipelineEventEmitter:
    def emit(self, event: str, data: Dict[str, Any]):
        """Emit event to all registered listeners"""
        ...

class PipelineStage:
    def __init__(self, emitter: PipelineEventEmitter):
        self.emitter = emitter
    
    def execute(self, context):
        self.emitter.emit('stage.started', {'stage': self.name})
        result = self._do_work(context)
        self.emitter.emit('stage.completed', {'stage': self.name, 'result': result})
        return result

# Listeners can hook in
emitter.on('target_ranking.completed', lambda data: save_metrics(data))
emitter.on('feature_selection.completed', lambda data: validate_features(data))
```

---

### **Option C: Plugin Registry Pattern**

**Concept:** Register stages/processors via decorator or config.

**Benefits:**
- Discoverable extensions
- No need to modify orchestrator to add stages
- Can load plugins dynamically

**Structure:**
```python
# TRAINING/orchestration/pipeline/registry.py

class StageRegistry:
    _stages: Dict[str, Type[PipelineStage]] = {}
    
    @classmethod
    def register(cls, name: str, stage: Type[PipelineStage]):
        cls._stages[name] = stage
    
    @classmethod
    def get(cls, name: str) -> PipelineStage:
        return cls._stages[name]()

# Usage
@StageRegistry.register('target_ranking')
class TargetRankingStage(PipelineStage):
    ...

# Load from config
pipeline_config = {
    'stages': ['target_ranking', 'feature_selection', 'training']
}
stages = [StageRegistry.get(name) for name in pipeline_config['stages']]
```

---

## Hybrid Approach (Best of All Worlds)

**Recommended:** Combine Pipeline Stage Pattern + Hook System + Plugin Registry

### **Core Pipeline**
```python
# TRAINING/orchestration/pipeline/core.py

class Pipeline:
    def __init__(self, stages: List[PipelineStage], hooks: Optional[HookRegistry] = None):
        self.stages = stages
        self.hooks = hooks or HookRegistry()
    
    def run(self, context: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            # Pre-stage hook
            context = self.hooks.execute('before_stage', stage, context)
            
            # Execute stage
            context = stage.execute(context)
            
            # Post-stage hook
            context = self.hooks.execute('after_stage', stage, context)
        
        return context
```

### **Hook Registry**
```python
# TRAINING/orchestration/pipeline/hooks.py

class HookRegistry:
    _hooks: Dict[str, List[Callable]] = {}
    
    def register(self, event: str, callback: Callable):
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def execute(self, event: str, *args, **kwargs):
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {event} failed: {e}")
```

### **Stage Registration**
```python
# TRAINING/orchestration/pipeline/stages.py

# Built-in stages
@register_stage('target_ranking')
class TargetRankingStage(PipelineStage):
    ...

@register_stage('feature_selection')
class FeatureSelectionStage(PipelineStage):
    ...

# External modules can register their own
# TRAINING/custom/my_stage.py
@register_stage('custom_validation')
class CustomValidationStage(PipelineStage):
    ...
```

---

## Migration Strategy

### **Phase 1: Extract Stages (Low Risk)**
1. Create `PipelineStage` protocol/ABC
2. Extract existing logic into stage classes
3. Keep orchestrator as thin wrapper
4. **No breaking changes** - orchestrator still works

### **Phase 2: Add Hook System (Medium Risk)**
1. Add `HookRegistry` to orchestrator
2. Add hook points at key stages
3. Migrate existing hooks (stability) to new system
4. **Backward compatible** - hooks are optional

### **Phase 3: Plugin Registry (Low Risk)**
1. Add stage registry
2. Load stages from config
3. Enable dynamic stage discovery
4. **Optional** - can still use direct instantiation

---

## Current Architecture: Is It Fine?

### **✅ Current Architecture is Acceptable If:**
- You don't need to frequently add new pipeline stages
- Stages are relatively stable (ranking → selection → training)
- You're okay with modifying orchestrator for new stages
- Hook system (stability) is sufficient for your extension needs

### **⚠️ Refactor If:**
- You need to add many new stages (validation, monitoring, custom processing)
- You want to conditionally enable/disable stages
- You want to support alternative implementations (e.g., different ranking methods)
- You need better testability (isolate stages)
- You want to support pipeline composition (different pipelines for different use cases)

---

## Recommendation

**For your current needs:** The current architecture is **fine** for now. The hook system (stability) shows the pattern works, and you can extend it incrementally.

**For future extensibility:** Consider **Option A (Pipeline Stage Pattern)** when you need to:
- Add 3+ new stages
- Support alternative implementations
- Enable conditional stage execution

**Implementation effort:** Medium (2-3 days to extract stages, add hooks, maintain backward compatibility)

**Risk:** Low (can be done incrementally, backward compatible)

---

## Quick Wins (No Refactoring Needed)

1. **Standardize Hook Pattern**
   - Use same pattern as stability hooks for other extensions
   - Create `TRAINING/orchestration/hooks.py` for common hooks

2. **Add Extension Points**
   - Add hook calls at key points in orchestrator:
     ```python
     # Before/after target ranking
     _execute_hook('before_target_ranking', context)
     rankings = rank_targets(...)
     _execute_hook('after_target_ranking', rankings, context)
     ```

3. **Create Stage Interfaces**
   - Define protocols for ranking/selection/training
   - Allow swapping implementations via config

4. **Config-Driven Stage Selection**
   - Add config option to enable/disable stages
   - Allow custom stage ordering

---

## Conclusion

**Current state:** Functional but tightly coupled. Hook system shows extensibility is possible.

**Recommended path:** 
- **Short term:** Use hook pattern for new extensions (like stability)
- **Medium term:** Extract stages when you need 3+ new stages or alternative implementations
- **Long term:** Full pipeline pattern if you need complex workflows

**The current way is fine** for now, but the pipeline stage pattern would make future extensions much easier.
