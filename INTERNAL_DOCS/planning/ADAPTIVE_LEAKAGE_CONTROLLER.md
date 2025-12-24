# Adaptive Leakage Detection & Feature/Target Controller

**Status:** Design Phase  
**Owner:** Maintainer (Jennifer)  
**Scope:** FoxML Core training & ranking pipeline  
**Last Updated:** 2025-12-07

---

## 1. Objective

Move from **hard-coded leakage heuristics** (e.g. "ROC-AUC > 0.95 → suspect leakage") to a **model-based controller** that:

1. **Estimates leakage risk** for each target and feature
2. **Ranks targets & features** by *utility* (predictive power, stability, robustness)
3. **Proposes or enforces policies** (keep / down-rank / quarantine / block)
4. **Learns over time** from:
   * Historical training runs
   * Explicit human decisions
   * Production performance (where available)

The end state is a **meta-model / controller** that drives target/feature selection, while **rule-based safeguards remain as hard constraints**.

---

## 2. Current State

### What We Have Now

* Feature/target ranking pipeline computes:
  * ROC-AUC, accuracy, R², etc.
  * Simple importance metrics (SHAP, feature importance)
  * Cross-validation stability

* Leakage detection is based on:
  * Hand-tuned thresholds (e.g., ROC-AUC > 0.95 → warning)
  * Pattern-based heuristics (e.g., same-period lookahead features)
  * Pre-training leak scan (near-copy detection)
  * Auto-fixer with config updates

* Decisions are mostly:
  * Manual pruning by maintainer/users
  * Static config knobs with no memory of past outcomes
  * Auto-fixer applies fixes but doesn't learn from patterns

### Pain Points

* Thresholds are **global**, not context-aware (symbol, horizon, dataset size)
* No "learning" from what the maintainer accepts/rejects
* No central place where "this feature was leaky in 3 other runs" gets remembered
* Auto-fixer fixes individual runs but doesn't build institutional knowledge
* High false positive rate (e.g., RF overfitting flagged as leakage)

---

## 3. Design Principles

1. **Safety first:** Rule-based constraints are never removed, only complemented
2. **Explainability > cleverness:** Every auto-decision must be logged and explainable
3. **Incremental rollout:** Start as a **recommender**, promote to **enforcer** only after validation
4. **Config-first, model-second:** Human-authored configs always win; model proposes defaults/overrides, not vice versa
5. **Reproducibility:** Decisions must be reproducible given the same inputs and controller model version
6. **Conservative by default:** When in doubt, err on the side of caution (quarantine, not block)

---

## 4. System Overview

```text
Raw Training Runs + Metadata
       ↓
Signal Extraction (metrics, descriptors, histories)
       ↓
Controller Model(s)
  - Leakage Risk Model
  - Utility / Priority Model
       ↓
Policy Engine
  - Decide: keep / warn / quarantine / block
  - Emit ranked lists + reasons
       ↓
TRAINING Pipeline
  - Uses approved targets/features
  - Logs feedback and outcomes back into history
```

The **controller** becomes a separate internal subsystem (`TRAINING/controller/`) with its own configs and model artifacts.

---

## 5. Data & Signals the Controller Sees

### 5.1 Target Descriptors

For each target in each run:

**Basic Metadata:**
- Name, type (classification/regression), horizon (e.g., `peak_60m_0.8`)
- Target construction method (barrier, forward return, etc.)

**Dataset Characteristics:**
- Number of samples (train/val/test)
- Class balance (for classification: positive class ratio)
- Symbol count and distribution
- Time range covered

**Predictability Metrics:**
- ROC-AUC, PR-AUC, accuracy (classification)
- R², MAPE, RMSE (regression)
- Train vs validation gap
- Stability across folds / walk-forwards (CV std)
- Consistency score

**Temporal Information:**
- Lead/lag structure relative to features
- Data latency constraints from config
- Horizon in bars/minutes

**Leakage Signals:**
- Pre-training leak scan results (near-copy features found)
- Perfect correlation models detected
- Auto-fixer actions taken
- Historical flags from previous runs

### 5.2 Feature Descriptors

For each feature in each run:

**Basic Metadata:**
- Name and family (lag, rolling, future, cross-sectional aggregate, etc.)
- Source (price, derived, metadata, target_proxy)
- Lag bars (if applicable)

**Access Pattern:**
- Uses same-period future info? (from dependency analysis)
- Uses labels or target-derived signals?
- Matches excluded patterns?
- Registry status (allowed/rejected for this horizon)

**Statistical Descriptors:**
- Mean, variance, correlation with target (train vs val)
- Correlation drift across time slices
- Missing value ratio
- Distribution characteristics (skew, kurtosis)

**Model Contributions:**
- Average SHAP/importance across models
- Coefficients / gain (per model family)
- Ablation deltas (performance change if removed)
- Importance stability across folds

**Historical Context:**
- How often has this feature been flagged/leaky?
- How often has it been quarantined?
- Past performance in other runs/symbols

### 5.3 Global Run Context

**Dataset:**
- Symbol universe, timeframe, data vendor
- Data interval (5m, 1h, etc.)
- Total samples, train/val/test splits

**Training Configuration:**
- Training window size & structure (walk-forward, etc.)
- Model families used and their respective scores
- Cross-validation strategy
- Feature selection method

**Leakage Context:**
- Existing leakage warnings triggered by rules
- Auto-fixer actions taken
- Config modifications made

All of this becomes **training data for the controller**.

---

## 6. Controller Components

### 6.1 Leakage Risk Model

**Goal:** For each {target, feature}, estimate `P(leaky | descriptors)` or a continuous **Leakage Risk Score** in `[0, 1]`.

**Input Features:**

**Extreme Predictability Signals:**
- Very high ROC-AUC (> 0.95) + low variance
- Perfect training accuracy (100%) in tree models
- High train/val gap (overfitting indicator)

**Temporal Leakage Indicators:**
- Strong contemporaneous correlation with future labels
- Uses known "forbidden" columns (future close, PnL, realized outcomes)
- Zero or negative lag_bars for forward-looking features

**Pattern-Based Signals:**
- Matches excluded patterns (y_*, fwd_ret_*, barrier_*, etc.)
- Registry marked as rejected
- Pre-training scan detected near-copy

**Historical Evidence:**
- Was this feature previously flagged/leaky elsewhere?
- How often has it been quarantined?
- Past auto-fixer actions on this feature

**Output:**
- Risk score (0.0 = safe, 1.0 = definitely leaky)
- Top contributing reasons (derived from feature importance / rules overlay)
- Confidence interval (if using probabilistic model)

**Model Architecture:**
- Start with: LightGBM / XGBoost (tree-based, interpretable)
- Later: Neural network if we have enough data
- Ensemble: Combine multiple models for robustness

### 6.2 Utility / Priority Model

**Goal:** Rank **safe** targets/features by *expected value* to the pipeline.

**Signals:**

**Predictive Power:**
- Marginal lift over baseline model when added
- Average importance across model families
- Cross-validation performance

**Stability:**
- Importance stability across folds and time windows
- Robustness across symbol subsets
- Low sensitivity to hyperparameters

**Efficiency:**
- Computational cost (if available)
- Feature engineering complexity
- Data availability (missing value ratio)

**Output:**
- Utility score (higher = more valuable)
- Used to sort the shortlist after leakage filtering

**Model Architecture:**
- Start with: Simple scoring function (weighted combination)
- Later: Learned model if we have enough labeled data

### 6.3 Policy Engine

Combines:
- Rule-based constraints (hard)
- Leakage Risk Model (soft)
- Utility Model (soft)
- User-defined policy preferences (YAML)

**Example Policy Config:**

```yaml
# CONFIG/training_config/adaptive_controller.yaml

adaptive_controller:
  enabled: true
  enforcement_mode: "recommend"  # "recommend" | "enforce" | "audit-only"
  
  leakage:
    # Hard rules (never overridden)
    hard_max_roc_auc: 0.995
    hard_max_r2: 0.995
    hard_block_future_features: true
    hard_block_target_derived: true
    
    # Soft thresholds (controller can adjust)
    quarantine_threshold: 0.5   # risk score
    block_threshold: 0.8
    warning_threshold: 0.3
    
    # Context-aware adjustments
    adjust_by_horizon: true
    adjust_by_symbol_count: true
    adjust_by_sample_size: true
  
  features:
    max_features: 200
    max_quarantine_fraction: 0.2
    min_utility_score: 0.1
    
  targets:
    max_targets: 20
    min_predictability_score: 0.3
    require_manual_approval_for_suspicious: true
  
  learning:
    update_frequency: "weekly"  # "daily" | "weekly" | "monthly" | "manual"
    min_runs_for_training: 10
    validation_split: 0.2
    retrain_on_override: true  # Retrain when human overrides controller
```

**Decision Grid:**

| Risk Score | Utility | Decision | Action |
|------------|---------|----------|--------|
| > block_threshold | Any | **BLOCK** | Exclude from pipeline, log reason |
| [quarantine, block) | Any | **QUARANTINE** | Flag for human review, don't use by default |
| [warning, quarantine) | High | **KEEP (WARN)** | Use but log warning, monitor closely |
| [warning, quarantine) | Low | **QUARANTINE** | Low utility + risk = quarantine |
| < warning | High | **KEEP** | Normal operation |
| < warning | Low | **KEEP (LOW PRIORITY)** | Include but deprioritize |

**Hard Rules (Always Applied):**
- Any feature with `lag_bars < 0` → BLOCK
- Any feature matching excluded patterns → BLOCK
- Any feature using future labels → BLOCK
- Any target with perfect train accuracy + zero val improvement → QUARANTINE

---

## 7. Training the Controller

### 7.1 Label Sources

**Historical Human Decisions:**
- Features/targets manually removed due to leakage
- Marked via explicit overrides in config or CLI
- Auto-fixer actions (what was fixed and why)

**Synthetic Leakage Injection:**
- Intentionally create variants with known leakage
  - Duplicate future returns as "features"
  - Create target-derived features with zero lag
  - Inject perfect correlation features
- Label these as leaky (ground truth)

**Weak Supervision:**
- Use existing hard rules as initial labelers:
  - "ROC-AUC > 0.98 + small train/val gap" → high probability leaky
  - "Feature uses target column with zero lag" → leaky by definition
  - "Pre-training scan found near-copy" → leaky

**Production Failures (Later):**
- If a target/feature causes severe live/forward performance collapse
- Record that as negative evidence

**Manual Corrections:**
- Maintainer can explicitly label features/targets as:
  - `leaky: true` (confirmed leaky)
  - `safe: true` (confirmed safe, override controller)
  - `quarantine: true` (needs review)

### 7.2 Modeling Approach

**Start Simple:**
- Tree-based model (LightGBM / XGBoost) for leakage risk
- Simple scoring function for utility (weighted combination)

**Training Process:**
- Train/val splits over runs/time
- Evaluate with:
  - Precision/recall at high risk thresholds
  - "False negatives" measured by known leaky features slipping through
  - Calibration (predicted risk vs actual leak rate)

**Feature Engineering for Controller:**
- Extract all descriptors from training runs
- Create aggregated features (rolling averages, counts, etc.)
- Handle missing data (some features may not exist in all runs)

---

## 8. Learning Over Time (Adaptation)

### 8.1 Data Collection

Each training run logs:
- Controller inputs (descriptors)
- Controller outputs (risk, decision)
- Human overrides (if any)
- Downstream outcomes (validation performance, later live performance)

**Storage:**
- `data/controller_history/` - JSON files per run
- `data/controller_models/` - Trained model artifacts
- `data/controller_reports/` - Evaluation reports

### 8.2 Periodic Re-training

**Schedule:**
- Nightly/weekly job retrains controller on accumulated history
- Model versions tracked (`controller_v1.0.0`, `controller_v1.1.0`, etc.)
- Training & evaluation reports stored in `docs/internal/adaptive_controller/`

**Versioning:**
- Semantic versioning for controller models
- Track which runs used which controller version
- A/B testing: compare new vs old controller on held-out runs

### 8.3 Online Adaptation (Optional, Later)

**Bandit-Style Learning:**
- Adjust thresholds/policies based on:
  - How often quarantined items are released/kept
  - How often blocked items turn out to be false positives
  - Performance of controller-recommended features vs manually selected

**Bayesian Updating:**
- Update risk estimates as new evidence arrives
- Maintain uncertainty estimates (confidence intervals)

---

## 9. Safety Rails & Non-Goals

### Hard Constraints (Controller Can't Override)

**Temporal Rules:**
- No future-label access: any feature that reads future labels/outcomes is auto-blocked
- No same-bar lookahead beyond allowed market microstructure
- Config flags like `allow_future_features: false` are absolute

**Pattern-Based Rules:**
- Features matching excluded patterns are always blocked
- Registry-rejected features are always blocked
- Metadata columns are always excluded

**Config Overrides:**
- Human-authored configs always win
- Explicit `rejected: true` in registry → always blocked
- Explicit `allowed: true` in config → always allowed (with warning if risky)

### Non-Goals (For Now)

**Not Doing:**
- Fully unsupervised "magic leakage detector" with no rules
- Silently rewriting user configs
- Auto-modifying label definitions without explicit maintainer policy
- Removing features without logging/explanation

**The controller should NEVER:**
- Introduce new features
- Change the label function
- Modify target construction logic

Without either:
- An explicit policy, or
- A separate manual config update

---

## 10. Rollout Plan

### Phase 0 — Instrumentation (Current State)

**Status:** ✅ Partially Complete

**Completed:**
- Logging of predictability metrics (ROC-AUC, R², etc.)
- Leakage detection with hard-coded thresholds
- Auto-fixer with config updates
- Pre-training leak scan

**Remaining:**
- Standardize descriptor schema
- Create controller history storage
- Add controller metadata to run logs

**Deliverables:**
- `TRAINING/controller/schema.py` - Descriptor schema definitions
- `TRAINING/controller/history.py` - History collection and storage
- Update `rank_target_predictability.py` to emit descriptors

### Phase 1 — Offline Controller Prototype

**Timeline:** 2-4 weeks

**Tasks:**
1. Build leakage risk model offline against:
   - Historical runs (if available)
   - Synthetic leakage injection
   - Current auto-fixer decisions

2. Run in **shadow mode**:
   - Emits recommendations into logs/reports
   - Does NOT influence actual pipeline
   - Parallel execution with current system

3. Evaluation:
   - Compare model suggestions vs existing hard-coded rules
   - Measure: how many issues caught earlier/better
   - False positive/negative rates
   - Precision/recall at various thresholds

**Deliverables:**
- `TRAINING/controller/leakage_risk_model.py`
- `TRAINING/controller/utility_model.py`
- `TRAINING/controller/policy_engine.py`
- `TRAINING/controller/train_controller.py` - Training script
- `docs/internal/adaptive_controller/evaluation_report_v1.md`

### Phase 2 — Recommend Mode in Production

**Timeline:** 1-2 weeks after Phase 1

**Tasks:**
1. Enable controller in `enforcement_mode: "recommend"`
2. Generate outputs:
   - `targets_ranked.yaml` and `features_ranked.yaml` as artifacts
   - HTML/Markdown reports highlighting risky items and rationale
3. Integration:
   - Controller recommendations appear in ranking output
   - Maintainer can accept/reject via config
   - Track acceptance rate

**Deliverables:**
- `TRAINING/controller/report_generator.py`
- `CONFIG/training_config/adaptive_controller.yaml` (default config)
- Updated `target_ranker.py` to use controller recommendations
- Report templates (HTML/Markdown)

### Phase 3 — Enforced Policies (with Escape Hatches)

**Timeline:** 2-4 weeks after Phase 2 (after validation)

**Tasks:**
1. Switch some policies to `"enforce"`:
   - Auto-block features above very high risk (e.g., > 0.9)
   - Auto-quarantine suspicious targets
2. Keep others as recommend/quarantine
3. Provide escape hatches:
   - CLI flag: `--disable-adaptive-controller`
   - Config to pin specific features/targets regardless of model
   - Override mechanism in config

**Deliverables:**
- Updated policy engine with enforcement logic
- Override mechanism in config
- Documentation for escape hatches
- Monitoring dashboard for controller decisions

### Phase 4 — Tight Integration with Adaptive Intelligence Layer

**Timeline:** Future (after core adaptive intelligence is stable)

**Tasks:**
1. Use controller outputs as part of larger adaptive intelligence layer:
   - Control which targets/horizons are active per symbol/regime
   - Dynamically adjust feature families as market regime shifts
   - Adaptive target selection based on current market conditions

2. Advanced features:
   - Multi-symbol feature importance aggregation
   - Regime-aware feature selection
   - Adaptive horizon selection

**Deliverables:**
- Integration with adaptive intelligence orchestrator
- Regime detection and adaptation
- Advanced policy configurations

---

## 11. Module Structure

```
TRAINING/
  controller/
    __init__.py
    schema.py              # Descriptor schemas
    history.py             # History collection and storage
    leakage_risk_model.py  # Leakage risk prediction
    utility_model.py       # Utility/priority scoring
    policy_engine.py       # Decision engine
    report_generator.py    # HTML/Markdown reports
    train_controller.py    # Training script
    evaluate_controller.py # Evaluation script
    config.py              # Config loading and validation
    
  ranking/
    target_ranker.py       # Updated to use controller
    rank_target_predictability.py  # Emits descriptors

CONFIG/
  training_config/
    adaptive_controller.yaml  # Controller configuration

data/
  controller_history/      # Run history (JSON)
  controller_models/       # Trained models
  controller_reports/      # Evaluation reports

docs/
  internal/
    adaptive_controller/
      evaluation_report_v1.md
      design_decisions.md
      model_versions.md
```

---

## 12. Example Usage

### Config Example

```yaml
# CONFIG/training_config/adaptive_controller.yaml

adaptive_controller:
  enabled: true
  enforcement_mode: "recommend"  # Start conservative
  
  leakage:
    hard_max_roc_auc: 0.995
    quarantine_threshold: 0.5
    block_threshold: 0.8
    warning_threshold: 0.3
  
  features:
    max_features: 200
    max_quarantine_fraction: 0.2
  
  learning:
    update_frequency: "weekly"
    min_runs_for_training: 10
```

### Code Integration Example

```python
# In target_ranker.py

from TRAINING.controller.policy_engine import PolicyEngine
from TRAINING.controller.history import collect_run_descriptors

# Collect descriptors from current run
descriptors = collect_run_descriptors(
    targets=targets,
    features=features,
    metrics=model_metrics,
    run_metadata=run_metadata
)

# Initialize controller
controller = PolicyEngine.from_config()

# Get recommendations
recommendations = controller.evaluate(descriptors)

# Apply recommendations (based on enforcement_mode)
for target_name, rec in recommendations.targets.items():
    if rec.decision == "BLOCK":
        logger.warning(f"Controller BLOCKS {target_name}: {rec.reason}")
        # Skip this target
    elif rec.decision == "QUARANTINE":
        logger.warning(f"Controller QUARANTINES {target_name}: {rec.reason}")
        # Flag for review, don't use by default
    # ... etc

# Log feedback for learning
controller.log_feedback(descriptors, recommendations, human_overrides=None)
```

---

## 13. Success Metrics

**Phase 1 (Offline Prototype):**
- Controller catches > 90% of known leaky features
- False positive rate < 10% at block threshold
- Recommendations align with maintainer decisions > 80% of the time

**Phase 2 (Recommend Mode):**
- Maintainer acceptance rate > 70%
- Reduction in manual leakage review time
- Clear, explainable recommendations

**Phase 3 (Enforced Policies):**
- Zero false positives at block threshold (conservative)
- Reduction in leakage incidents
- Maintainer confidence in controller decisions

**Phase 4 (Full Integration):**
- Adaptive feature/target selection improves model performance
- Controller learns and improves over time
- Reduced manual intervention needed

---

## 14. Risks & Mitigations

**Risk: Controller makes bad decisions**
- **Mitigation:** Start in recommend mode, conservative thresholds, escape hatches

**Risk: Overfitting to historical data**
- **Mitigation:** Regular retraining, validation on held-out runs, monitor drift

**Risk: Black box decisions**
- **Mitigation:** Explainable models (tree-based), detailed logging, report generation

**Risk: Performance overhead**
- **Mitigation:** Efficient feature extraction, caching, optional execution

**Risk: Config complexity**
- **Mitigation:** Sensible defaults, clear documentation, validation

---

## 15. Next Steps

1. **Immediate (This Week):**
   - Create `TRAINING/controller/` directory structure
   - Define descriptor schema (`schema.py`)
   - Create history collection system (`history.py`)

2. **Short-term (Next 2 Weeks):**
   - Build offline leakage risk model prototype
   - Create synthetic leakage injection for training data
   - Run shadow mode evaluation

3. **Medium-term (Next Month):**
   - Integrate into ranking pipeline (recommend mode)
   - Generate reports and documentation
   - Collect feedback and iterate

---

## 16. References

- Current leakage detection: `TRAINING/common/leakage_auto_fixer.py`
- Target ranking: `TRAINING/ranking/rank_target_predictability.py`
- Feature registry: `CONFIG/feature_registry.yaml`
- Safety config: `CONFIG/training_config/safety_config.yaml`

---

**Status:** Ready for implementation  
**Priority:** High (enables adaptive intelligence layer)  
**Dependencies:** Current leakage detection system, target ranking pipeline

