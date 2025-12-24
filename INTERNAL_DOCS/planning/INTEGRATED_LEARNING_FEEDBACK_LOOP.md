# Integrated Learning Feedback Loop: CILS ↔ ARPO ↔ Training Pipeline

**Status**: Design Phase  
**Date**: 2025-12-08  
**Classification**: Internal Planning Document

## Executive Summary

This document describes the **Integrated Learning Feedback Loop** that connects:
1. **Continuous Integrated Learning System (CILS)**: Learns from training runs, leakage patterns, feature performance
2. **Adaptive Real-Time Portfolio Optimization (ARPO)**: Learns from live trading P&L, position performance, market regimes
3. **Training Pipeline**: Automatically scheduled and optimized based on learned insights from both systems

**Core Value Proposition**: Create a self-improving system where:
- Portfolio learnings inform which models/targets to retrain
- Training learnings inform portfolio allocation strategies
- Automated scheduling optimizes compute resources based on expected value
- Continuous improvement cycle: Train → Deploy → Learn → Retrain

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Integrated Learning Feedback Loop                    │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│     CILS      │   │     ARPO       │   │   Training    │
│  (Training    │   │  (Portfolio    │   │   Pipeline    │
│   Learning)   │   │   Learning)   │   │               │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Learning Orchestrator │
                │  (Scheduling & Routing)│
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Shared Learning DB   │
                │  (Unified Knowledge)  │
                └───────────────────────┘
```

### 1.2 Data Flow

```
Training Pipeline
    │
    ├─→ Generates: Model predictions, feature importance, target rankings
    │
    └─→ Feeds CILS: Training metadata, leakage events, performance metrics
                    │
                    └─→ CILS learns: Optimal thresholds, feature patterns, strategies
                                    │
                                    └─→ Updates: Training configs, feature selection, target ranking
                                                                    │
                                                                    └─→ Feeds back to Training Pipeline

Live Trading (ARPO)
    │
    ├─→ Generates: Real-time P&L, position performance, market regime
    │
    └─→ Feeds ARPO: P&L signals, drawdowns, risk metrics
                    │
                    └─→ ARPO learns: Optimal position sizes, allocation strategies, risk limits
                                    │
                                    └─→ Generates: Portfolio optimization signals
                                                                    │
                                                                    └─→ Feeds back to Training Pipeline:
                                                                        - Which models to retrain
                                                                        - Which targets need updates
                                                                        - Priority scheduling

Learning Orchestrator
    │
    ├─→ Receives: Insights from CILS and ARPO
    │
    ├─→ Computes: Training priorities, expected value, resource allocation
    │
    └─→ Schedules: Automated training runs based on learned data
```

---

## 2. Learning Data Integration

### 2.1 Unified Learning Database Schema

```python
@dataclass
class UnifiedLearningState:
    """Unified learning state combining CILS and ARPO insights."""
    
    # From CILS (Training Learning)
    cils_state: CILSLearningState
    optimal_thresholds: Dict[str, float]
    feature_patterns: Dict[str, FeaturePattern]
    target_characteristics: Dict[str, TargetProfile]
    training_strategies: Dict[str, Strategy]
    
    # From ARPO (Portfolio Learning)
    arpo_state: ARPOLearningState
    position_adjustments: Dict[str, float]  # symbol -> adjustment factor
    regime_strategies: Dict[str, RegimeStrategy]  # regime -> optimal strategy
    risk_limits: Dict[str, float]  # risk metric -> limit
    portfolio_performance: PortfolioMetrics
    
    # Cross-System Insights
    model_performance_map: Dict[str, ModelPerformance]  # model -> live performance
    target_priority_map: Dict[str, float]  # target -> priority score
    feature_importance_map: Dict[str, float]  # feature -> live importance
    retraining_signals: List[RetrainingSignal]
    
    # Metadata
    last_updated: datetime
    version: int
    confidence_scores: Dict[str, float]  # confidence in learned insights
```

### 2.2 Retraining Signal Generation

```python
@dataclass
class RetrainingSignal:
    """Signal to retrain a model based on learned insights."""
    
    signal_id: str
    timestamp: datetime
    source: str  # "CILS", "ARPO", or "ORCHESTRATOR"
    priority: float  # 0.0 to 1.0
    urgency: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # What to retrain
    target_name: Optional[str]  # Specific target, or None for all
    model_family: Optional[str]  # Specific model, or None for all
    feature_set: Optional[List[str]]  # Specific features, or None for all
    
    # Why retrain
    reason: str  # Human-readable reason
    metrics: Dict[str, float]  # Supporting metrics
    
    # Expected value
    expected_improvement: float  # Expected performance improvement
    expected_cost: float  # Expected compute cost
    roi: float  # Return on investment (improvement / cost)
    
    # Scheduling
    recommended_schedule: datetime
    estimated_duration: timedelta
    resource_requirements: Dict[str, Any]
```

### 2.3 Priority Scoring Algorithm

```python
def compute_training_priority(
    target_name: str,
    cils_insights: CILSInsights,
    arpo_insights: ARPOInsights,
    current_performance: ModelPerformance
) -> float:
    """
    Compute priority score for retraining a target/model.
    
    Factors:
    1. CILS: Model performance degradation, new feature patterns
    2. ARPO: Live trading performance, position sizing issues
    3. Expected value: Improvement potential vs. cost
    4. Urgency: How quickly performance is degrading
    """
    
    # Factor 1: CILS insights (training-side)
    cils_score = 0.0
    if target_name in cils_insights.degrading_targets:
        cils_score += 0.3  # Performance degrading
    if target_name in cils_insights.new_patterns:
        cils_score += 0.2  # New patterns discovered
    if target_name in cils_insights.optimal_strategies:
        cils_score += 0.2  # Better strategy available
    
    # Factor 2: ARPO insights (live trading)
    arpo_score = 0.0
    if target_name in arpo_insights.underperforming_positions:
        arpo_score += 0.4  # Live performance poor
    if target_name in arpo_insights.high_priority_targets:
        arpo_score += 0.3  # High portfolio impact
    if target_name in arpo_insights.regime_mismatch:
        arpo_score += 0.2  # Not adapting to current regime
    
    # Factor 3: Expected value
    expected_improvement = estimate_improvement(target_name, cils_insights, arpo_insights)
    expected_cost = estimate_training_cost(target_name)
    value_score = expected_improvement / (expected_cost + 1e-6)  # ROI
    
    # Factor 4: Urgency (rate of degradation)
    urgency_score = compute_urgency(current_performance, arpo_insights)
    
    # Weighted combination
    priority = (
        0.25 * cils_score +
        0.35 * arpo_score +  # Portfolio learnings weighted higher (live money)
        0.25 * value_score +
        0.15 * urgency_score
    )
    
    return min(1.0, priority)  # Clamp to [0, 1]
```

---

## 3. Automated Scheduling System

### 3.1 Scheduling Architecture

```python
class AutomatedTrainingScheduler:
    """Automatically schedule training based on learned insights."""
    
    def __init__(
        self,
        cils: CILS,
        arpo: ARPO,
        training_pipeline: TrainingPipeline,
        learning_db: UnifiedLearningDB
    ):
        self.cils = cils
        self.arpo = arpo
        self.training_pipeline = training_pipeline
        self.learning_db = learning_db
        
        # Scheduling parameters
        self.max_concurrent_training = 3
        self.min_priority_threshold = 0.3
        self.resource_budget = ResourceBudget()
    
    def run_scheduling_loop(self):
        """Main scheduling loop."""
        while True:
            # 1. Collect latest insights
            cils_insights = self.cils.get_latest_insights()
            arpo_insights = self.arpo.get_latest_insights()
            
            # 2. Generate retraining signals
            signals = self._generate_retraining_signals(cils_insights, arpo_insights)
            
            # 3. Prioritize signals
            prioritized = self._prioritize_signals(signals)
            
            # 4. Schedule training jobs
            scheduled = self._schedule_training_jobs(prioritized)
            
            # 5. Update learning database
            self.learning_db.update_with_scheduled_jobs(scheduled)
            
            # 6. Wait before next cycle
            time.sleep(self.scheduling_interval)
    
    def _generate_retraining_signals(
        self,
        cils_insights: CILSInsights,
        arpo_insights: ARPOInsights
    ) -> List[RetrainingSignal]:
        """Generate retraining signals from CILS and ARPO insights."""
        signals = []
        
        # Signals from CILS
        for target_name in cils_insights.targets_needing_retrain:
            signal = RetrainingSignal(
                signal_id=f"cils_{target_name}_{timestamp()}",
                timestamp=datetime.now(),
                source="CILS",
                priority=compute_training_priority(target_name, cils_insights, arpo_insights, ...),
                urgency=self._determine_urgency(cils_insights, target_name),
                target_name=target_name,
                reason=f"Performance degradation detected: {cils_insights.get_degradation_reason(target_name)}",
                metrics=cils_insights.get_metrics(target_name),
                expected_improvement=cils_insights.estimate_improvement(target_name),
                expected_cost=self._estimate_cost(target_name),
                roi=self._compute_roi(target_name, cils_insights),
                recommended_schedule=self._recommend_schedule(target_name),
                estimated_duration=self._estimate_duration(target_name),
                resource_requirements=self._get_resource_requirements(target_name)
            )
            signals.append(signal)
        
        # Signals from ARPO
        for target_name in arpo_insights.targets_needing_retrain:
            signal = RetrainingSignal(
                signal_id=f"arpo_{target_name}_{timestamp()}",
                timestamp=datetime.now(),
                source="ARPO",
                priority=compute_training_priority(target_name, cils_insights, arpo_insights, ...),
                urgency=self._determine_urgency_arpo(arpo_insights, target_name),
                target_name=target_name,
                reason=f"Live trading underperformance: {arpo_insights.get_performance_issue(target_name)}",
                metrics=arpo_insights.get_metrics(target_name),
                expected_improvement=arpo_insights.estimate_improvement(target_name),
                expected_cost=self._estimate_cost(target_name),
                roi=self._compute_roi(target_name, arpo_insights),
                recommended_schedule=self._recommend_schedule(target_name),
                estimated_duration=self._estimate_duration(target_name),
                resource_requirements=self._get_resource_requirements(target_name)
            )
            signals.append(signal)
        
        # Cross-system signals (orchestrator-generated)
        orchestrator_signals = self._generate_orchestrator_signals(cils_insights, arpo_insights)
        signals.extend(orchestrator_signals)
        
        return signals
    
    def _prioritize_signals(
        self,
        signals: List[RetrainingSignal]
    ) -> List[RetrainingSignal]:
        """Prioritize signals by ROI, urgency, and resource availability."""
        # Filter by minimum priority
        filtered = [s for s in signals if s.priority >= self.min_priority_threshold]
        
        # Sort by composite score
        def score(signal: RetrainingSignal) -> float:
            return (
                0.4 * signal.priority +
                0.3 * signal.roi +
                0.2 * urgency_weight(signal.urgency) +
                0.1 * (1.0 / signal.expected_cost)  # Prefer cheaper jobs
            )
        
        sorted_signals = sorted(filtered, key=score, reverse=True)
        return sorted_signals
    
    def _schedule_training_jobs(
        self,
        prioritized_signals: List[RetrainingSignal]
    ) -> List[ScheduledJob]:
        """Schedule training jobs based on priority and resource availability."""
        scheduled = []
        active_jobs = self.training_pipeline.get_active_jobs()
        
        for signal in prioritized_signals:
            # Check if we have capacity
            if len(active_jobs) >= self.max_concurrent_training:
                break
            
            # Check if resources are available
            if not self.resource_budget.can_allocate(signal.resource_requirements):
                continue
            
            # Check if this target is already being retrained
            if self._is_already_scheduled(signal.target_name):
                continue
            
            # Schedule the job
            job = self.training_pipeline.schedule_training(
                target_name=signal.target_name,
                model_family=signal.model_family,
                feature_set=signal.feature_set,
                config_updates=self._get_config_updates(signal),
                priority=signal.priority,
                estimated_duration=signal.estimated_duration
            )
            
            scheduled.append(job)
            active_jobs.append(job)
        
        return scheduled
```

### 3.2 Config Update Generation

```python
def generate_config_updates(
    signal: RetrainingSignal,
    cils_insights: CILSInsights,
    arpo_insights: ARPOInsights
) -> Dict[str, Any]:
    """
    Generate config updates for retraining based on learned insights.
    
    Combines:
    - CILS: Optimal thresholds, feature patterns, strategies
    - ARPO: Position sizing preferences, regime strategies
    """
    updates = {}
    
    # From CILS
    if signal.target_name in cils_insights.optimal_thresholds:
        updates['leakage_detection'] = {
            'thresholds': cils_insights.optimal_thresholds[signal.target_name]
        }
    
    if signal.target_name in cils_insights.optimal_strategies:
        strategy = cils_insights.optimal_strategies[signal.target_name]
        updates['feature_selection'] = {
            'top_n': strategy.feature_count,
            'aggregation_method': strategy.aggregation_method,
            'model_weights': strategy.model_weights
        }
    
    # From ARPO
    if signal.target_name in arpo_insights.position_adjustments:
        # If position sizing suggests model is underperforming, adjust training
        adjustment = arpo_insights.position_adjustments[signal.target_name]
        if adjustment < 0.8:  # Position size reduced (underperforming)
            updates['training'] = {
                'focus_on_validation': True,  # Emphasize validation
                'early_stopping_patience': 10,  # More patience
                'cross_validation_folds': 5  # More CV folds
            }
    
    if signal.target_name in arpo_insights.regime_strategies:
        regime_strategy = arpo_insights.regime_strategies[signal.target_name]
        # Adapt training to current regime
        updates['data_sampling'] = {
            'regime_filter': regime_strategy.current_regime,
            'regime_weight': regime_strategy.regime_weight
        }
    
    return updates
```

---

## 4. Feedback Loops

### 4.1 CILS → Training Pipeline

**Flow**:
1. CILS learns optimal thresholds, feature patterns, strategies from training runs
2. Learning Orchestrator generates retraining signals
3. Training Pipeline receives config updates with learned insights
4. New training runs use optimized configs
5. Results feed back to CILS for further learning

**Example**:
- CILS learns: "For barrier targets, top 50 features with weighted consensus works best"
- Signal generated: Retrain barrier targets with new strategy
- Training scheduled: Barrier targets retrained with top_n=50, weighted consensus
- Results: Better validation scores
- Feedback: CILS confirms strategy effectiveness

### 4.2 ARPO → Training Pipeline

**Flow**:
1. ARPO learns optimal position sizes, allocation strategies from live P&L
2. Identifies underperforming models/targets
3. Learning Orchestrator generates high-priority retraining signals
4. Training Pipeline prioritizes retraining underperforming models
5. New models deployed to live trading
6. Performance feeds back to ARPO

**Example**:
- ARPO learns: "Target X position size reduced to 0.6x (underperforming)"
- Signal generated: HIGH priority retrain for target X
- Training scheduled: Target X retrained with emphasis on validation
- Results: Improved model performance
- Deployment: New model deployed to live trading
- Feedback: ARPO observes improved P&L, increases position size back to 1.0x

### 4.3 Training Pipeline → CILS

**Flow**:
1. Training Pipeline generates metadata (scores, leakage events, feature importance)
2. Metadata feeds into CILS
3. CILS learns patterns and optimizes thresholds/strategies
4. Updated insights stored in learning database

### 4.4 Training Pipeline → ARPO

**Flow**:
1. Training Pipeline generates model predictions
2. Predictions feed into ARPO for portfolio allocation
3. ARPO uses predictions to inform position sizing
4. Live trading performance feeds back to ARPO learning

### 4.5 Cross-System Learning

**CILS ↔ ARPO**:
- CILS learns which features work best → ARPO can prioritize models using those features
- ARPO learns which targets perform best live → CILS can prioritize those targets for training
- Shared learning database enables cross-pollination of insights

---

## 5. Implementation Architecture

### 5.1 Module Structure

```python
# LEARNING/integrated_feedback/
├── __init__.py
├── orchestrator.py          # Learning Orchestrator (scheduling & routing)
├── unified_db.py            # Unified Learning Database
├── signal_generator.py      # Generate retraining signals
├── priority_scorer.py       # Compute training priorities
├── scheduler.py             # Automated training scheduler
├── config_updater.py        # Generate config updates from insights
├── feedback_loops.py        # Manage feedback loops
└── utils/
    ├── metrics.py           # Cross-system metrics
    ├── validation.py        # Validate learned insights
    └── monitoring.py        # Monitor feedback loop health
```

### 5.2 Learning Orchestrator Implementation

```python
class LearningOrchestrator:
    """Orchestrates learning feedback between CILS, ARPO, and Training Pipeline."""
    
    def __init__(
        self,
        cils: CILS,
        arpo: ARPO,
        training_pipeline: TrainingPipeline,
        learning_db: UnifiedLearningDB
    ):
        self.cils = cils
        self.arpo = arpo
        self.training_pipeline = training_pipeline
        self.learning_db = learning_db
        
        self.scheduler = AutomatedTrainingScheduler(
            cils, arpo, training_pipeline, learning_db
        )
        self.signal_generator = RetrainingSignalGenerator(cils, arpo)
        self.config_updater = ConfigUpdateGenerator(cils, arpo)
        self.feedback_manager = FeedbackLoopManager()
    
    def run_orchestration_loop(self):
        """Main orchestration loop."""
        while True:
            # 1. Collect insights from both systems
            cils_insights = self.cils.get_latest_insights()
            arpo_insights = self.arpo.get_latest_insights()
            
            # 2. Update unified learning database
            self.learning_db.update(cils_insights, arpo_insights)
            
            # 3. Generate retraining signals
            signals = self.signal_generator.generate(cils_insights, arpo_insights)
            
            # 4. Schedule training jobs
            scheduled = self.scheduler.schedule(signals)
            
            # 5. Update training pipeline with learned configs
            for job in scheduled:
                config_updates = self.config_updater.generate(
                    job.signal, cils_insights, arpo_insights
                )
                self.training_pipeline.update_config(job.target_name, config_updates)
            
            # 6. Manage feedback loops
            self.feedback_manager.process_feedback(
                cils_insights, arpo_insights, scheduled
            )
            
            # 7. Wait before next cycle
            time.sleep(self.orchestration_interval)
    
    def get_training_priorities(self) -> Dict[str, float]:
        """Get current training priorities for all targets."""
        cils_insights = self.cils.get_latest_insights()
        arpo_insights = self.arpo.get_latest_insights()
        
        priorities = {}
        for target_name in self._get_all_targets():
            priority = compute_training_priority(
                target_name, cils_insights, arpo_insights, ...
            )
            priorities[target_name] = priority
        
        return priorities
```

### 5.3 Unified Learning Database

```python
class UnifiedLearningDB:
    """Unified database for CILS and ARPO learning state."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_schema()
    
    def update(
        self,
        cils_insights: CILSInsights,
        arpo_insights: ARPOInsights
    ):
        """Update database with latest insights from both systems."""
        # Store CILS insights
        self._store_cils_insights(cils_insights)
        
        # Store ARPO insights
        self._store_arpo_insights(arpo_insights)
        
        # Compute cross-system insights
        cross_insights = self._compute_cross_insights(cils_insights, arpo_insights)
        self._store_cross_insights(cross_insights)
        
        # Update unified learning state
        self._update_unified_state()
    
    def get_training_recommendations(
        self,
        target_name: Optional[str] = None
    ) -> List[TrainingRecommendation]:
        """Get training recommendations based on learned insights."""
        # Query unified learning state
        # Generate recommendations
        # Return prioritized list
        pass
    
    def get_config_updates(
        self,
        target_name: str
    ) -> Dict[str, Any]:
        """Get config updates for a target based on learned insights."""
        # Query CILS insights for target
        # Query ARPO insights for target
        # Combine into config updates
        # Return updates
        pass
```

---

## 6. Automated Scheduling Logic

### 6.1 Scheduling Criteria

**When to Schedule Training**:

1. **CILS Signals**:
   - Performance degradation detected (> 10% drop in validation score)
   - New optimal strategy discovered (expected improvement > 5%)
   - Feature pattern changes (new leakage patterns, new important features)
   - Threshold optimization (new optimal thresholds found)

2. **ARPO Signals**:
   - Position size reduced (indicates model underperformance)
   - Live P&L below expected (model predictions inaccurate)
   - Regime mismatch (model not adapting to current regime)
   - Risk limit violations (model risk predictions inaccurate)

3. **Orchestrator Signals**:
   - Cross-system insights (CILS + ARPO both suggest retraining)
   - Scheduled maintenance (periodic retraining for model freshness)
   - Data drift (new data patterns detected)

### 6.2 Priority Calculation

```python
def compute_composite_priority(
    target_name: str,
    cils_priority: float,
    arpo_priority: float,
    expected_improvement: float,
    expected_cost: float,
    urgency: float
) -> float:
    """
    Compute composite priority from multiple factors.
    
    NOTE: This is a simplified version of compute_training_priority() (Section 2.3).
    Use compute_training_priority() for full implementation with CILS/ARPO insights.
    
    Formula:
    priority = w1 * cils_priority + 
               w2 * arpo_priority + 
               w3 * (improvement / cost) + 
               w4 * urgency
    
    Weights (consistent with Section 2.3):
    - w1 = 0.25 (training-side insights)
    - w2 = 0.35 (live trading insights - weighted higher)
    - w3 = 0.25 (expected value)
    - w4 = 0.15 (urgency)
    """
    roi = expected_improvement / (expected_cost + 1e-6)
    
    composite = (
        0.25 * cils_priority +
        0.35 * arpo_priority +
        0.25 * min(1.0, roi / 10.0) +  # Normalize ROI
        0.15 * urgency
    )
    
    return min(1.0, composite)
```

### 6.3 Resource-Aware Scheduling

```python
class ResourceAwareScheduler:
    """Schedule training with resource constraints."""
    
    def __init__(
        self,
        max_concurrent: int = 3,
        gpu_memory_limit: float = 32.0,  # GB
        cpu_limit: int = 16,
        budget_per_day: float = 100.0  # Compute budget
    ):
        self.max_concurrent = max_concurrent
        self.gpu_memory_limit = gpu_memory_limit
        self.cpu_limit = cpu_limit
        self.budget_per_day = budget_per_day
        self.daily_spent = 0.0
    
    def can_schedule(
        self,
        job: RetrainingSignal,
        active_jobs: List[TrainingJob]
    ) -> bool:
        """Check if job can be scheduled given resource constraints."""
        # Check concurrent limit
        if len(active_jobs) >= self.max_concurrent:
            return False
        
        # Check GPU memory
        total_gpu_memory = sum(j.gpu_memory for j in active_jobs)
        if total_gpu_memory + job.resource_requirements['gpu_memory'] > self.gpu_memory_limit:
            return False
        
        # Check CPU
        total_cpu = sum(j.cpu_cores for j in active_jobs)
        if total_cpu + job.resource_requirements['cpu_cores'] > self.cpu_limit:
            return False
        
        # Check daily budget
        if self.daily_spent + job.expected_cost > self.budget_per_day:
            return False
        
        return True
```

---

## 7. Config Update Propagation

### 7.1 Config Update Flow

```
CILS/ARPO Insights
    │
    ├─→ Learning Orchestrator
    │       │
    │       └─→ Generate Config Updates
    │               │
    │               ├─→ Leakage Detection Thresholds (from CILS)
    │               ├─→ Feature Selection Strategy (from CILS)
    │               ├─→ Position Sizing Preferences (from ARPO)
    │               ├─→ Regime-Specific Configs (from ARPO)
    │               └─→ Training Hyperparameters (from both)
    │
    └─→ Training Pipeline
            │
            ├─→ Update Config Files
            │       ├─→ CONFIG/leakage_detection.yaml
            │       ├─→ CONFIG/feature_selection.yaml
            │       ├─→ CONFIG/training/target_ranking_config.yaml
            │       └─→ CONFIG/training/models.yaml
            │
            └─→ Apply Updates to Next Training Run
```

### 7.2 Config Update Implementation

```python
class ConfigUpdatePropagator:
    """Propagate learned insights to training configs."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.backup_manager = ConfigBackupManager(config_dir)
    
    def apply_updates(
        self,
        target_name: str,
        updates: Dict[str, Any],
        source: str  # "CILS" or "ARPO"
    ):
        """Apply config updates for a target."""
        # Backup current configs
        self.backup_manager.backup(f"before_{source}_update_{timestamp()}")
        
        # Apply updates
        if 'leakage_detection' in updates:
            self._update_leakage_config(target_name, updates['leakage_detection'])
        
        if 'feature_selection' in updates:
            self._update_feature_selection_config(target_name, updates['feature_selection'])
        
        if 'training' in updates:
            self._update_training_config(target_name, updates['training'])
        
        if 'data_sampling' in updates:
            self._update_data_sampling_config(target_name, updates['data_sampling'])
        
        # Validate updates
        self._validate_updates(target_name, updates)
        
        # Log update
        self._log_update(target_name, updates, source)
    
    def _update_leakage_config(
        self,
        target_name: str,
        thresholds: Dict[str, float]
    ):
        """Update leakage detection thresholds."""
        config_path = self.config_dir / "leakage_detection.yaml"
        config = yaml.safe_load(open(config_path))
        
        # Update thresholds for target (or global if target-specific not supported)
        if 'target_specific' not in config:
            config['target_specific'] = {}
        
        config['target_specific'][target_name] = thresholds
        
        yaml.dump(config, open(config_path, 'w'))
    
    def _update_feature_selection_config(
        self,
        target_name: str,
        strategy: Dict[str, Any]
    ):
        """Update feature selection strategy."""
        config_path = self.config_dir / "feature_selection" / f"{target_name}.yaml"
        
        config = {
            'target': target_name,
            'top_n': strategy['top_n'],
            'aggregation_method': strategy['aggregation_method'],
            'model_weights': strategy['model_weights']
        }
        
        yaml.dump(config, open(config_path, 'w'))
```

---

## 8. Monitoring & Observability

### 8.1 Feedback Loop Health Metrics

```python
@dataclass
class FeedbackLoopHealth:
    """Health metrics for feedback loops."""
    
    # CILS → Training
    cils_to_training_latency: float  # Time from CILS insight to training scheduled
    cils_signal_accuracy: float  # % of CILS signals that improve performance
    cils_config_adoption_rate: float  # % of learned configs actually used
    
    # ARPO → Training
    arpo_to_training_latency: float  # Time from ARPO insight to training scheduled
    arpo_signal_accuracy: float  # % of ARPO signals that improve live performance
    arpo_priority_correlation: float  # Correlation between ARPO priority and actual improvement
    
    # Training → CILS/ARPO
    training_to_cils_latency: float  # Time from training completion to CILS update
    training_to_arpo_latency: float  # Time from training completion to ARPO update
    
    # Overall
    feedback_loop_effectiveness: float  # Overall improvement from feedback loops
    scheduling_efficiency: float  # % of scheduled jobs that complete successfully
    resource_utilization: float  # % of compute resources utilized
```

### 8.2 Dashboard Metrics

**Real-Time Dashboard**:
- Current training priorities (top 10 targets)
- Active training jobs (with progress)
- Latest CILS insights (thresholds, patterns, strategies)
- Latest ARPO insights (position adjustments, regime strategies)
- Feedback loop health (latency, accuracy, effectiveness)
- Scheduled vs. completed jobs
- Resource utilization

---

## 9. Implementation Roadmap

### Phase 1: Unified Learning Database (Weeks 1-2)
- Design unified schema
- Implement database
- Create APIs for CILS and ARPO to store insights
- Create APIs for Training Pipeline to query insights
- **Deliverable**: Unified learning database operational

### Phase 2: Signal Generation (Weeks 3-4)
- Implement retraining signal generation from CILS
- Implement retraining signal generation from ARPO
- Implement priority scoring algorithm
- Test signal generation on historical data
- **Deliverable**: Signal generation working

### Phase 3: Automated Scheduler (Weeks 5-6)
- Implement automated training scheduler
- Implement resource-aware scheduling
- Implement config update propagation
- Integrate with training pipeline
- **Deliverable**: Automated scheduling operational

### Phase 4: Feedback Loops (Weeks 7-8)
- Implement CILS → Training feedback loop
- Implement ARPO → Training feedback loop
- Implement Training → CILS feedback loop
- Implement Training → ARPO feedback loop
- Test end-to-end feedback loops
- **Deliverable**: All feedback loops operational

### Phase 5: Integration & Production (Weeks 9-10)
- Integrate all components
- Add monitoring and observability
- Add safety mechanisms and rollback
- Production testing
- Documentation
- **Deliverable**: Production-ready integrated system

---

## 10. Success Metrics

### 10.1 System Effectiveness

- **Training Efficiency**: 30% reduction in wasted training runs
- **Model Performance**: 10-15% improvement in live trading performance
- **Resource Utilization**: 80%+ compute resource utilization
- **Feedback Latency**: < 1 hour from insight to training scheduled

### 10.2 Learning Effectiveness

- **CILS Signal Accuracy**: > 70% of signals lead to improvement
- **ARPO Signal Accuracy**: > 80% of signals lead to improvement (live money)
- **Config Adoption**: > 90% of learned configs adopted
- **Priority Correlation**: > 0.7 correlation between priority and actual improvement

### 10.3 Operational Metrics

- **Scheduling Efficiency**: > 95% of scheduled jobs complete successfully
- **System Uptime**: 99.9% availability
- **Feedback Loop Health**: All loops healthy (latency < thresholds)

---

## 11. Safety & Rollback

### 11.1 Safety Mechanisms

- **Config Backups**: All config updates backed up before application
- **Gradual Rollout**: Learned configs applied gradually, not all at once
- **Validation**: Config updates validated before application
- **Human Override**: Manual configs always respected
- **Rollback**: Can revert to baseline configs at any time

### 11.2 Monitoring

- **Feedback Loop Health**: Monitor latency, accuracy, effectiveness
- **Signal Quality**: Track signal accuracy over time
- **Resource Usage**: Monitor compute resource utilization
- **Performance Impact**: Track improvement from feedback loops

---

## 12. References

- **CILS**: `DOCS/internal/planning/CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md`
- **ARPO**: `DOCS/internal/planning/ADAPTIVE_REALTIME_PORTFOLIO_OPTIMIZATION.md`
- **Training Pipeline**: `TRAINING/orchestration/intelligent_trainer.py`
- **Config System**: `CONFIG/config_builder.py`

---

**End of Document**

