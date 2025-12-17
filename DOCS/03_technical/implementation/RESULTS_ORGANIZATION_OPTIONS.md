# RESULTS Directory Organization

## Current Implementation (2025-12-16)

**Structure: Comparison Group Organization**

```
RESULTS/
  runs/                      # All runs organized by comparison group metadata
    cg-abc123def456_n-5000_fam-lightgbm/    # Comparison group directory
      test_run_20251216_012021/
      test_run_20251216_020000/
      ...
    cg-def456ghi789_n-25000_fam-xgboost/
      ...
```

**Why This Structure?**

Runs are organized by **all outcome-influencing metadata** (dataset, task, routing, model family, feature set, split protocol) using a comparison group key. This ensures that only truly comparable runs are grouped together, enabling strict audit-grade comparisons.

**Benefits:**
- ✅ **Strict comparability**: Only runs with identical outcome-influencing metadata are grouped together
- ✅ **Audit-grade**: Comparison group key includes fingerprints for config, data, features, targets, and split protocol
- ✅ **Prevents fold drift**: Split protocol signature includes `split_seed` and `fold_assignment_hash`
- ✅ **Human navigable**: Directory names include `n-{sample_size}` and `fam-{model_family}` for quick identification
- ✅ **Clean structure**: All runs under `RESULTS/runs/` keeps root directory clean

**Bins:**
- `sample_0-5k`: 0 <= N < 5,000
- `sample_5k-10k`: 5,000 <= N < 10,000
- `sample_10k-25k`: 10,000 <= N < 25,000
- `sample_25k-50k`: 25,000 <= N < 50,000
- `sample_50k-100k`: 50,000 <= N < 100,000
- `sample_100k-250k`: 100,000 <= N < 250,000
- `sample_250k-500k`: 250,000 <= N < 500,000
- `sample_500k-1M`: 500,000 <= N < 1,000,000
- `sample_1M+`: N >= 1,000,000

**Boundary Rules:**
- Boundaries are **EXCLUSIVE upper bounds**: `bin_min <= N_effective < bin_max`
- Example: `sample_25k-50k` means `25000 <= N_effective < 50000`
- This ensures unambiguous binning (50,000 always goes to `sample_50k-100k`, never `sample_25k-50k`)

**Metadata:**
Bin information is stored in `metadata.json`:
```json
{
  "N_effective": 943943,
  "sample_size_bin": {
    "bin_name": "sample_500k-1M",
    "bin_min": 500000,
    "bin_max": 1000000,
    "binning_scheme_version": "sample_bin_v1"
  }
}
```

**Early Estimation:**
The system automatically estimates `N_effective` during initialization by:
1. Checking existing metadata from previous runs with same symbols
2. Sampling data files (using parquet metadata for fast row counts)
3. Extrapolating to all symbols

This allows the run directory to be created directly in the correct bin, avoiding `_pending/` directories in most cases.

**Trend Analysis:**
- Bin is **NOT** included in trend series keys (which use stable identity: cohort_id, stage, target, data_fingerprint)
- This prevents series fragmentation when binning scheme changes
- Exact `N_effective` remains first-class in metadata for precise comparisons

---

## Alternative Options (Not Implemented)

### Option 1: Date-Based

```
RESULTS/
  {YYYY-MM-DD}/               # e.g., 2025-12-12
    {run_name}/               # e.g., test_e2e_ranking_unified_20251212_012021
      target_rankings/
      REPRODUCIBILITY/
      metadata.json           # Contains N_effective for filtering
      ...
```

**Pros:**
- ✅ Easy to find recent runs
- ✅ Natural chronological organization
- ✅ Cleaner top-level structure
- ✅ N_effective still queryable via `index.parquet` or metadata

**Cons:**
- Runs with same N_effective but different dates are separated
- Need to query metadata/index to find by sample size

**Implementation:**
- Use `datetime.now().strftime("%Y-%m-%d")` for date folder
- Store N_effective in `metadata.json` (already done)
- Use `index.parquet` for filtering by N_effective

---

## Option 2: Flatter Structure with Queryable Index

```
RESULTS/
  {run_name}/                 # e.g., test_e2e_ranking_unified_20251212_012021
    target_rankings/
    REPRODUCIBILITY/
    metadata.json             # Contains N_effective, date, experiment_type
    ...
```

**Pros:**
- ✅ Simplest structure
- ✅ All metadata in one place
- ✅ Easy to query via `index.parquet`
- ✅ No arbitrary grouping

**Cons:**
- All runs at same level (can get cluttered)
- Need tooling to filter/find runs

**Implementation:**
- Store all metadata in `metadata.json`
- Use `index.parquet` for all filtering
- Add CLI tool: `python tools/list_runs.py --by-sample-size 943943`

---

## Option 3: Hybrid (Date + Sample Size)

```
RESULTS/
  {YYYY-MM-DD}/               # e.g., 2025-12-12
    {run_name}/               # e.g., test_e2e_ranking_unified_20251212_012021
      ...
```

But also maintain symlinks or tags:
```
RESULTS/
  by_sample_size/
    {N_effective}/            # e.g., 943943 -> symlink to actual run
      {run_name} -> ../../2025-12-12/{run_name}
```

**Pros:**
- ✅ Best of both worlds
- ✅ Find by date OR by sample size

**Cons:**
- More complex to maintain
- Symlinks can break

---

## Option 4: Experiment Type + Date

```
RESULTS/
  {experiment_type}/          # e.g., test, production, experiment
    {YYYY-MM-DD}/            # e.g., 2025-12-12
      {run_name}/
        ...
```

**Pros:**
- ✅ Separates test vs production runs
- ✅ Date-based within each type
- ✅ Clear organization

**Cons:**
- Need to detect experiment type (already done via "test" in output-dir)
- More nesting

**Implementation:**
- Detect from `output_dir` name (already checks for "test")
- Structure: `RESULTS/{experiment_type}/{date}/{run_name}/`

---

## Option 5: Sample Size Ranges (Binned)

```
RESULTS/
  sample_900k-1M/            # Binned ranges
    {run_name}/
      ...
  sample_500k-600k/
    {run_name}/
      ...
```

**Pros:**
- ✅ Groups similar sample sizes
- ✅ Fewer top-level directories

**Cons:**
- Arbitrary binning
- Less precise than exact N_effective

---

## Why Sample Size Bins Were Chosen

**Primary use case**: Compare runs with similar cross-sectional sample sizes (e.g., "show me all ~25k runs")

**Benefits over alternatives:**
- **Better than exact N_effective**: Groups similar runs together (25,000 vs 25,100 both in `sample_25k-50k`)
- **Better than date-based**: Keeps comparable runs together even if run on different days
- **Better than flat**: Provides natural grouping without needing query tools
- **Trend analysis friendly**: Prevents series fragmentation from small N_effective variations

**Trade-offs:**
- Less precise than exact N_effective (but exact value still in metadata)
- Can't easily find "runs from today" (but can query `index.parquet` or sort by timestamp)
- Fixed bins may not fit all use cases (but versioned for future changes)

---

## Finding Runs

**By sample size bin:**
```bash
ls RESULTS/sample_25k-50k/
```

**By exact N_effective:**
```python
# Query index.parquet
import pandas as pd
index = pd.read_parquet("RESULTS/*/REPRODUCIBILITY/index.parquet")
runs = index[index['N_effective'] == 25000]
```

**By date:**
```python
# Query index.parquet
runs = index[index['created_at'] >= '2025-12-12']
```

**By cohort:**
```python
runs = index[index['cohort_id'] == 'cs_2025Q2_min_cs3_max1000_v1_abc123']
```
