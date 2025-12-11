# FEATURE IMPORTANCE STABILITY TRACKING SYSTEM

**COMPREHENSIVE DOCUMENTATION**

---

## OVERVIEW

THE FEATURE IMPORTANCE STABILITY TRACKING SYSTEM PROVIDES NON-INVASIVE HOOKS FOR TRACKING AND ANALYZING FEATURE IMPORTANCE CONSISTENCY ACROSS PIPELINE RUNS. IT AUTOMATICALLY CAPTURES SNAPSHOTS OF FEATURE IMPORTANCES FROM VARIOUS METHODS (MODEL TRAINING, FEATURE SELECTION, PRUNING) AND COMPUTES STABILITY METRICS TO DETECT INSTABILITY OR DRIFT.

**KEY BENEFITS:**
- AUTOMATIC SNAPSHOT CAPTURE AT ALL INTEGRATION POINTS
- CONFIG-DRIVEN AUTOMATION (NO CODE CHANGES NEEDED TO ENABLE/DISABLE)
- STANDARDIZED JSON FORMAT FOR EASY ANALYSIS
- MULTIPLE STABILITY METRICS (OVERLAP, RANK CORRELATION, SELECTION FREQUENCY)
- CLI TOOL FOR MANUAL ANALYSIS
- NON-INVASIVE DESIGN (FAILURES ARE LOGGED BUT DON'T BREAK PIPELINE)

---

## ARCHITECTURE

### MODULE STRUCTURE

```
TRAINING/stability/feature_importance/
├── __init__.py          # PUBLIC API EXPORTS
├── schema.py            # FeatureImportanceSnapshot DATACLASS
├── io.py                # SAVE/LOAD SNAPSHOTS TO/FROM DISK
├── analysis.py          # STABILITY METRICS COMPUTATION
└── hooks.py             # PIPELINE INTEGRATION HOOKS
```

### CORE COMPONENTS

1. **SCHEMA** (`schema.py`)
   - `FeatureImportanceSnapshot`: STANDARDIZED DATACLASS FOR SNAPSHOT DATA
   - FACTORY METHODS: `from_dict_series()`, `from_series()`
   - SERIALIZATION: `to_dict()`, `from_dict()`

2. **IO** (`io.py`)
   - `save_importance_snapshot()`: SAVE SNAPSHOT AS JSON
   - `load_snapshots()`: LOAD ALL SNAPSHOTS FOR TARGET+METHOD
   - `get_snapshot_base_dir()`: DETERMINE ARTIFACT LOCATION

3. **ANALYSIS** (`analysis.py`)
   - `top_k_overlap()`: JACCARD SIMILARITY OF TOP-K FEATURES
   - `rank_correlation()`: KENDALL TAU RANK CORRELATION
   - `selection_frequency()`: FREQUENCY OF FEATURE APPEARANCE IN TOP-K
   - `compute_stability_metrics()`: AGGREGATE METRICS ACROSS ALL PAIRS
   - `analyze_stability_auto()`: AUTOMATED ANALYSIS WITH LOGGING

4. **HOOKS** (`hooks.py`)
   - `save_snapshot_hook()`: MAIN ENTRY POINT FOR SAVING SNAPSHOTS
   - `save_snapshot_from_series_hook()`: CONVENIENCE WRAPPER FOR PANDAS SERIES
   - `analyze_all_stability_hook()`: COMPREHENSIVE ANALYSIS FOR ALL TARGETS/METHODS

---

## SNAPSHOT STORAGE

### DIRECTORY STRUCTURE

```
artifacts/feature_importance/
  {target_name}/
    {method}/
      {run_id}.json
```

**EXAMPLE:**
```
artifacts/feature_importance/
  peak_60m_0.8/
    lightgbm/
      20251210_143022_abc123.json
      20251210_150045_def456.json
    quick_pruner/
      20251210_143022_abc123.json
  valley_60m_0.8/
    rfe/
      20251210_143022_abc123.json
```

### SNAPSHOT JSON FORMAT

```json
{
  "target_name": "peak_60m_0.8",
  "method": "lightgbm",
  "universe_id": "CROSS_SECTIONAL",
  "run_id": "20251210_143022_abc123",
  "created_at": "2025-12-10T14:30:22.123456",
  "features": ["feature_1", "feature_2", "feature_3", ...],
  "importances": [0.45, 0.32, 0.18, ...]
}
```

**NOTES:**
- FEATURES ARE SORTED BY IMPORTANCE (DESCENDING)
- IMPORTANCES ARE IN SAME ORDER AS FEATURES
- RUN_ID IS UUID OR TIMESTAMP-BASED IDENTIFIER
- CREATED_AT IS ISO 8601 FORMAT

---

## INTEGRATION POINTS

### 1. TARGET RANKING (`TRAINING/ranking/predictability/model_evaluation.py`)

**INTEGRATION POINT A: AFTER QUICK PRUNING**
- LOCATION: LINE ~347-360
- METHOD: `"quick_pruner"`
- UNIVERSE_ID: `"CROSS_SECTIONAL"`
- TRIGGERS: AFTER `quick_importance_prune()` COMPLETES

**INTEGRATION POINT B: AFTER MODEL TRAINING**
- LOCATION: LINE ~1827
- METHOD: MODEL NAME (E.G., `"lightgbm"`, `"random_forest"`, `"neural_network"`)
- UNIVERSE_ID: `"CROSS_SECTIONAL"`
- TRIGGERS: AFTER EACH MODEL TRAINS AND COMPUTES IMPORTANCES

**INTEGRATION POINT C: END-OF-RUN ANALYSIS**
- LOCATION: `TRAINING/ranking/predictability/main.py` LINE ~322-328
- FUNCTION: `analyze_all_stability_hook()`
- TRIGGERS: AFTER ALL TARGETS ARE EVALUATED

### 2. FEATURE SELECTION (`TRAINING/ranking/multi_model_feature_selection.py`)

**INTEGRATION POINT: AFTER EACH METHOD**
- LOCATION: LINE ~1370-1382
- METHODS: `"rfe"`, `"boruta"`, `"stability_selection"`, `"mutual_information"`
- UNIVERSE_ID: SYMBOL NAME OR `"ALL"`
- TRIGGERS: AFTER EACH `train_model_and_get_importance()` CALL

### 3. FEATURE SELECTION AGGREGATION (`TRAINING/ranking/feature_selector.py`)

**INTEGRATION POINT: AFTER AGGREGATION**
- LOCATION: LINE ~212-226
- METHOD: `"multi_model_aggregated"`
- UNIVERSE_ID: COMMA-SEPARATED SYMBOL LIST OR `"ALL"`
- TRIGGERS: AFTER `aggregate_multi_model_importance()` COMPLETES

---

## CONFIGURATION

### SAFETY CONFIG (`CONFIG/training_config/safety_config.yaml`)

```yaml
safety:
  feature_importance:
    # AUTOMATICALLY ANALYZE STABILITY AFTER SAVING SNAPSHOTS
    auto_analyze_stability: true  # SET TO FALSE TO DISABLE
    
    # STABILITY THRESHOLDS FOR WARNINGS
    stability_thresholds:
      min_top_k_overlap: 0.7      # WARN IF OVERLAP < 0.7 (JACCARD SIMILARITY)
      min_kendall_tau: 0.6        # WARN IF TAU < 0.6 (RANK CORRELATION)
      top_k: 20                   # NUMBER OF TOP FEATURES TO ANALYZE
      min_snapshots: 2            # MINIMUM SNAPSHOTS REQUIRED FOR ANALYSIS
```

### CONFIG-DRIVEN BEHAVIOR

- **AUTO_ANALYZE_STABILITY**: IF `TRUE`, AUTOMATICALLY RUNS STABILITY ANALYSIS AFTER EACH SNAPSHOT SAVE
- **THRESHOLDS**: USED TO DETERMINE WHEN TO LOG WARNINGS ABOUT LOW STABILITY
- **TOP_K**: NUMBER OF TOP FEATURES TO COMPARE ACROSS RUNS
- **MIN_SNAPSHOTS**: MINIMUM NUMBER OF SNAPSHOTS REQUIRED BEFORE ANALYSIS RUNS

**NOTE:** IF `auto_analyze_stability` IS `FALSE`, SNAPSHOTS ARE STILL SAVED BUT ANALYSIS IS SKIPPED.

---

## USAGE

### FROM PIPELINE CODE

#### BASIC USAGE (DICT)

```python
from TRAINING.stability.feature_importance import save_snapshot_hook

# AFTER COMPUTING FEATURE IMPORTANCE
importance_dict = {
    "feature_1": 0.45,
    "feature_2": 0.32,
    "feature_3": 0.18,
    # ...
}

save_snapshot_hook(
    target_name="peak_60m_0.8",
    method="lightgbm",
    importance_dict=importance_dict,
    universe_id="CROSS_SECTIONAL",  # OR "AAPL", "ALL", ETC.
    output_dir=output_dir,  # OPTIONAL
    auto_analyze=None,  # NONE = LOAD FROM CONFIG
)
```

#### PANDAS SERIES USAGE

```python
from TRAINING.stability.feature_importance import save_snapshot_from_series_hook
import pandas as pd

# IF YOU HAVE A PANDAS SERIES
importance_series = pd.Series({
    "feature_1": 0.45,
    "feature_2": 0.32,
    # ...
})

save_snapshot_from_series_hook(
    target_name="peak_60m_0.8",
    method="quick_pruner",
    importance_series=importance_series,
    universe_id="CROSS_SECTIONAL",
    output_dir=output_dir,
    auto_analyze=None,  # LOADS FROM CONFIG
)
```

#### END-OF-RUN ANALYSIS

```python
from TRAINING.stability.feature_importance import analyze_all_stability_hook

# AT END OF PIPELINE RUN
analyze_all_stability_hook(output_dir=output_dir)
# AUTOMATICALLY LOGS STABILITY FOR ALL TARGETS/METHODS
```

### CLI ANALYSIS

#### BASIC USAGE

```bash
# ANALYZE SPECIFIC TARGET/METHOD
python scripts/analyze_importance_stability.py \
    --target peak_60m_0.8 \
    --method lightgbm \
    --top-k 20
```

#### CUSTOM DIRECTORY

```bash
# USE CUSTOM SNAPSHOT DIRECTORY
python scripts/analyze_importance_stability.py \
    --target peak_60m_0.8 \
    --method quick_pruner \
    --base-dir artifacts/feature_importance
```

#### OUTPUT DIRECTORY

```bash
# USE OUTPUT DIRECTORY (SNAPSHOTS IN {output_dir}/feature_importance_snapshots)
python scripts/analyze_importance_stability.py \
    --target peak_60m_0.8 \
    --method rfe \
    --output-dir results/run_20251210
```

#### FULL OPTIONS

```bash
python scripts/analyze_importance_stability.py \
    --target peak_60m_0.8 \
    --method lightgbm \
    --top-k 30 \
    --min-snapshots 3 \
    --base-dir artifacts/feature_importance
```

---

## STABILITY METRICS

### 1. TOP-K OVERLAP (JACCARD SIMILARITY)

**DEFINITION:** INTERSECTION / UNION OF TOP-K FEATURES BETWEEN TWO SNAPSHOTS

**FORMULA:**
```
overlap = |top_k_1 ∩ top_k_2| / |top_k_1 ∪ top_k_2|
```

**INTERPRETATION:**
- `1.0`: PERFECT MATCH (SAME TOP-K FEATURES)
- `0.7-1.0`: GOOD STABILITY
- `0.5-0.7`: MODERATE STABILITY (WARNING THRESHOLD)
- `< 0.5`: LOW STABILITY (POTENTIAL INSTABILITY)

**EXAMPLE:**
- RUN 1 TOP-10: `[A, B, C, D, E, F, G, H, I, J]`
- RUN 2 TOP-10: `[A, B, C, D, E, F, G, H, K, L]`
- OVERLAP: `8 / 12 = 0.667` (8 COMMON, 4 UNIQUE)

### 2. KENDALL TAU (RANK CORRELATION)

**DEFINITION:** RANK CORRELATION COEFFICIENT OF FEATURE IMPORTANCE ORDERING

**FORMULA:**
```
tau = (concordant_pairs - discordant_pairs) / total_pairs
```

**INTERPRETATION:**
- `1.0`: PERFECT RANK AGREEMENT
- `0.6-1.0`: GOOD RANK STABILITY
- `0.4-0.6`: MODERATE RANK STABILITY (WARNING THRESHOLD)
- `< 0.4`: LOW RANK STABILITY (RANKS ARE INCONSISTENT)

**NOTE:** REQUIRES `scipy` TO BE INSTALLED. RETURNS `NaN` IF UNAVAILABLE.

### 3. SELECTION FREQUENCY

**DEFINITION:** PERCENTAGE OF RUNS WHERE EACH FEATURE APPEARS IN TOP-K

**FORMULA:**
```
frequency(feature) = count(feature in top_k) / total_runs
```

**INTERPRETATION:**
- `1.0`: FEATURE ALWAYS IN TOP-K (HIGHLY STABLE)
- `0.7-1.0`: FEATURE USUALLY IN TOP-K (STABLE)
- `0.3-0.7`: FEATURE SOMETIMES IN TOP-K (MODERATE)
- `< 0.3`: FEATURE RARELY IN TOP-K (UNSTABLE)

**USE CASE:** IDENTIFY FEATURES THAT ARE CONSISTENTLY IMPORTANT VS. NOISY

### 4. AGGREGATE METRICS

WHEN MULTIPLE SNAPSHOTS EXIST, METRICS ARE COMPUTED FOR ALL PAIRS:

- **MEAN_OVERLAP**: AVERAGE TOP-K OVERLAP ACROSS ALL PAIRS
- **STD_OVERLAP**: STANDARD DEVIATION OF OVERLAP
- **MEAN_TAU**: AVERAGE KENDALL TAU ACROSS ALL PAIRS
- **STD_TAU**: STANDARD DEVIATION OF TAU
- **N_COMPARISONS**: NUMBER OF PAIRWISE COMPARISONS

---

## AUTOMATED ANALYSIS

### AUTOMATIC TRIGGERING

WHEN `auto_analyze_stability: true` IN CONFIG:

1. **AFTER EACH SNAPSHOT SAVE**: IF 2+ SNAPSHOTS EXIST FOR TARGET+METHOD, RUNS ANALYSIS
2. **LOGS TO CONSOLE**: STABILITY METRICS ARE LOGGED IMMEDIATELY
3. **WARNINGS**: IF METRICS BELOW THRESHOLDS, LOGS WARNING
4. **SAVES REPORT**: OPTIONALLY SAVES TEXT REPORT TO `{base_dir}/stability_reports/`

### EXAMPLE OUTPUT

```
📊 Stability for peak_60m_0.8/lightgbm:
   Top-20 overlap: 0.852 ± 0.042
   Kendall tau: 0.734 ± 0.056
   Comparisons: 15
✅ Stability is good

📊 Stability for peak_60m_0.8/quick_pruner:
   Top-20 overlap: 0.623 ± 0.089
   Kendall tau: 0.512 ± 0.071
   Comparisons: 10
⚠️  Low stability detected (overlap < 0.7) - feature importance may be unstable
```

### END-OF-RUN ANALYSIS

AT THE END OF TARGET RANKING RUNS, `analyze_all_stability_hook()` IS CALLED:

- SCANS ALL TARGETS/METHODS IN OUTPUT DIRECTORY
- COMPUTES STABILITY FOR EACH COMBINATION WITH 2+ SNAPSHOTS
- LOGS COMPREHENSIVE SUMMARY
- SAVES REPORTS TO `{output_dir}/stability_reports/`

---

## METHOD NAMES

### TARGET RANKING METHODS

- `"quick_pruner"`: QUICK IMPORTANCE-BASED PRUNING (LIGHTGBM)
- `"lightgbm"`: LIGHTGBM MODEL IMPORTANCES
- `"random_forest"`: RANDOM FOREST MODEL IMPORTANCES
- `"neural_network"`: NEURAL NETWORK MODEL IMPORTANCES
- `"xgboost"`: XGBOOST MODEL IMPORTANCES
- `"linear"`: LINEAR MODEL COEFFICIENTS

### FEATURE SELECTION METHODS

- `"rfe"`: RECURSIVE FEATURE ELIMINATION
- `"boruta"`: BORUTA FEATURE SELECTION
- `"stability_selection"`: STABILITY SELECTION
- `"mutual_information"`: MUTUAL INFORMATION FILTER
- `"multi_model_aggregated"`: FINAL AGGREGATED CONSENSUS

### UNIVERSE IDS

- `"CROSS_SECTIONAL"`: CROSS-SECTIONAL RANKING (ALL SYMBOLS)
- `"ALL"`: ALL SYMBOLS IN UNIVERSE
- `"AAPL"`, `"MSFT"`, ETC.: SINGLE SYMBOL
- `"AAPL,MSFT,GOOGL"`: COMMA-SEPARATED SYMBOL LIST (IF ≤10 SYMBOLS)

---

## TROUBLESHOOTING

### NO SNAPSHOTS BEING SAVED

**CHECK:**
1. VERIFY HOOKS ARE CALLED: CHECK LOGS FOR `"Saved importance snapshot"`
2. CHECK OUTPUT DIRECTORY: SNAPSHOTS SAVE TO `{output_dir}/feature_importance_snapshots/` OR `artifacts/feature_importance/`
3. CHECK EXCEPTIONS: HOOKS CATCH EXCEPTIONS AND LOG AS DEBUG (NON-CRITICAL)

### ANALYSIS NOT RUNNING

**CHECK:**
1. VERIFY CONFIG: `safety.feature_importance.auto_analyze_stability: true`
2. CHECK SNAPSHOT COUNT: NEED AT LEAST 2 SNAPSHOTS (CONFIGURED VIA `min_snapshots`)
3. CHECK LOGS: ANALYSIS FAILURES ARE LOGGED AS DEBUG (NON-CRITICAL)

### LOW STABILITY METRICS

**POSSIBLE CAUSES:**
1. **NON-DETERMINISTIC RANDOM SEEDS**: CHECK THAT `random_state` IS SET CONSISTENTLY
2. **DATA DRIFT**: FEATURE IMPORTANCES MAY CHANGE IF UNDERLYING DATA CHANGES
3. **SMALL SAMPLE SIZE**: WITH FEW SAMPLES, IMPORTANCES CAN BE NOISY
4. **FEATURE INTERACTIONS**: COMPLEX INTERACTIONS CAN CAUSE INSTABILITY

**ACTIONS:**
- REVIEW CONFIG FOR DETERMINISM SETTINGS
- CHECK DATA QUALITY AND CONSISTENCY
- INCREASE SAMPLE SIZE IF POSSIBLE
- CONSIDER FEATURE ENGINEERING TO REDUCE INTERACTIONS

### KENDALL TAU IS NaN

**CAUSE:** `scipy` IS NOT INSTALLED OR IMPORT FAILED

**FIX:**
```bash
pip install scipy
```

**NOTE:** OVERLAP METRIC DOES NOT REQUIRE SCIPY AND WILL STILL WORK.

### SNAPSHOTS ACCUMULATING

**ISSUE:** MANY SNAPSHOTS CAN CONSUME DISK SPACE

**SOLUTIONS:**
1. **MANUAL CLEANUP**: DELETE OLD SNAPSHOTS PERIODICALLY
2. **RETENTION POLICY**: IMPLEMENT SCRIPT TO KEEP ONLY LAST N SNAPSHOTS
3. **COMPRESSION**: SNAPSHOTS ARE JSON (CAN BE COMPRESSED)

**FUTURE ENHANCEMENT:** ADD RETENTION POLICY TO CONFIG.

---

## API REFERENCE

### `save_snapshot_hook()`

```python
def save_snapshot_hook(
    target_name: str,
    method: str,
    importance_dict: Dict[str, float],
    universe_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    auto_analyze: Optional[bool] = None,  # NONE = LOAD FROM CONFIG
) -> Optional[Path]:
    """
    SAVE FEATURE IMPORTANCE SNAPSHOT.
    
    ARGS:
        target_name: TARGET NAME (E.G., "peak_60m_0.8")
        method: METHOD NAME (E.G., "lightgbm", "quick_pruner")
        importance_dict: DICT MAPPING FEATURE NAMES TO IMPORTANCE VALUES
        universe_id: OPTIONAL UNIVERSE IDENTIFIER
        output_dir: OPTIONAL OUTPUT DIRECTORY
        run_id: OPTIONAL RUN ID (GENERATES UUID IF NOT PROVIDED)
        auto_analyze: IF NONE, LOADS FROM CONFIG
    
    RETURNS:
        PATH TO SAVED SNAPSHOT, OR NONE IF SAVING FAILED
    """
```

### `save_snapshot_from_series_hook()`

```python
def save_snapshot_from_series_hook(
    target_name: str,
    method: str,
    importance_series,  # pd.Series
    universe_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    auto_analyze: Optional[bool] = None,
) -> Optional[Path]:
    """
    SAVE SNAPSHOT FROM PANDAS SERIES.
    
    CONVENIENCE WRAPPER FOR SERIES-BASED IMPORTANCE DATA.
    """
```

### `analyze_all_stability_hook()`

```python
def analyze_all_stability_hook(
    output_dir: Optional[Path] = None,
    log_to_console: bool = True,
    save_report: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    ANALYZE STABILITY FOR ALL TARGETS/METHODS.
    
    SCANS OUTPUT DIRECTORY FOR SNAPSHOTS AND COMPUTES STABILITY
    METRICS FOR EACH TARGET+METHOD COMBINATION.
    
    ARGS:
        output_dir: OUTPUT DIRECTORY (DEFAULTS TO artifacts/)
        log_to_console: IF TRUE, LOGS METRICS TO CONSOLE
        save_report: IF TRUE, SAVES TEXT REPORTS
    
    RETURNS:
        DICT MAPPING "{target}/{method}" TO METRICS DICT
    """
```

### `load_snapshots()`

```python
def load_snapshots(
    base_dir: Path,
    target_name: str,
    method: str,
) -> List[FeatureImportanceSnapshot]:
    """
    LOAD ALL SNAPSHOTS FOR TARGET+METHOD.
    
    RETURNS:
        LIST OF SNAPSHOTS, SORTED BY CREATED_AT (OLDEST FIRST)
    """
```

### `compute_stability_metrics()`

```python
def compute_stability_metrics(
    snapshots: List[FeatureImportanceSnapshot],
    top_k: int = 20,
) -> Dict[str, float]:
    """
    COMPUTE STABILITY METRICS FOR LIST OF SNAPSHOTS.
    
    RETURNS:
        DICT WITH:
        - mean_overlap: AVERAGE TOP-K OVERLAP
        - std_overlap: STANDARD DEVIATION OF OVERLAP
        - mean_tau: AVERAGE KENDALL TAU
        - std_tau: STANDARD DEVIATION OF TAU
        - n_comparisons: NUMBER OF PAIRWISE COMPARISONS
    """
```

---

## EXAMPLES

### EXAMPLE 1: MANUAL SNAPSHOT SAVE

```python
from TRAINING.stability.feature_importance import save_snapshot_hook
from pathlib import Path

# AFTER TRAINING A MODEL
importance_dict = model.feature_importances_
feature_names = X.columns.tolist()
importance_dict = dict(zip(feature_names, importance_dict))

snapshot_path = save_snapshot_hook(
    target_name="peak_60m_0.8",
    method="lightgbm",
    importance_dict=importance_dict,
    universe_id="CROSS_SECTIONAL",
    output_dir=Path("results/run_20251210"),
    auto_analyze=None,  # LOAD FROM CONFIG
)

if snapshot_path:
    print(f"✅ Saved snapshot: {snapshot_path}")
```

### EXAMPLE 2: CLI ANALYSIS

```bash
# ANALYZE STABILITY FOR LIGHTGBM ON PEAK_60M_0.8
python scripts/analyze_importance_stability.py \
    --target peak_60m_0.8 \
    --method lightgbm \
    --top-k 20 \
    --min-snapshots 2

# OUTPUT:
# ============================================================
# Feature Importance Stability Analysis
# ============================================================
# Target: peak_60m_0.8
# Method: lightgbm
# Snapshots: 5
# Top-K: 20
#
# Stability Metrics:
#   Top-20 overlap: 0.852 ± 0.042
#   Kendall tau:        0.734 ± 0.056
#   Comparisons:        10
```

### EXAMPLE 3: CUSTOM ANALYSIS

```python
from TRAINING.stability.feature_importance import (
    load_snapshots,
    compute_stability_metrics,
    selection_frequency,
    get_snapshot_base_dir,
)

# LOAD SNAPSHOTS
base_dir = get_snapshot_base_dir()
snapshots = load_snapshots(base_dir, "peak_60m_0.8", "lightgbm")

# COMPUTE METRICS
metrics = compute_stability_metrics(snapshots, top_k=20)
print(f"Mean overlap: {metrics['mean_overlap']:.3f}")
print(f"Mean tau: {metrics['mean_tau']:.3f}")

# SELECTION FREQUENCY
freq = selection_frequency(snapshots, top_k=20)
print("\nTop features by frequency:")
for feat, p in sorted(freq.items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat}: {p:.2%}")
```

---

## BEST PRACTICES

1. **ENABLE AUTO-ANALYSIS**: SET `auto_analyze_stability: true` IN CONFIG FOR AUTOMATIC MONITORING
2. **REVIEW WARNINGS**: PAY ATTENTION TO LOW STABILITY WARNINGS IN LOGS
3. **REGULAR CLI CHECKS**: RUN CLI ANALYSIS PERIODICALLY TO TRACK TRENDS
4. **CONSISTENT METHOD NAMES**: USE STANDARD METHOD NAMES FOR EASIER AGGREGATION
5. **CLEANUP OLD SNAPSHOTS**: PERIODICALLY CLEAN UP OLD SNAPSHOTS TO SAVE SPACE
6. **DOCUMENT UNIVERSE IDS**: USE CONSISTENT UNIVERSE ID NAMING FOR EASIER FILTERING

---

## FUTURE ENHANCEMENTS

- **RETENTION POLICY**: CONFIG-DRIVEN RETENTION (KEEP LAST N SNAPSHOTS)
- **COMPRESSION**: OPTIONAL GZIP COMPRESSION FOR SNAPSHOTS
- **DATABASE BACKEND**: OPTIONAL DATABASE STORAGE FOR LARGE-SCALE DEPLOYMENTS
- **VISUALIZATION**: OPTIONAL PLOTS (HEATMAPS, TREND CHARTS)
- **ALERTING**: INTEGRATION WITH ALERTING SYSTEM FOR LOW STABILITY
- **BATCH ANALYSIS**: BACKGROUND JOB FOR PERIODIC COMPREHENSIVE ANALYSIS

---

## RELATED DOCUMENTATION

- `INTERNAL_DOCS/FEATURE_IMPORTANCE_STABILITY_INTEGRATION_PLAN.md`: DETAILED INTEGRATION PLAN
- `TRAINING/stability/README.md`: QUICK START GUIDE
- `CONFIG/training_config/safety_config.yaml`: CONFIGURATION REFERENCE

---

**LAST UPDATED:** 2025-12-10  
**VERSION:** 1.0.0
