# Internal Handbook

**Your complete guide to the trader codebase, development processes, and internal documentation.**

---

## 🚀 Quick Start

### Main Entry Points

**Training Pipeline:**
```bash
# Intelligent training (recommended)
python TRAINING/train.py --experiment-config my_experiment --auto-targets --auto-features

# Manual training
python TRAINING/train.py --data-dir data/data_labeled/interval=5m --symbols AAPL MSFT
```

**Target Ranking:**
```bash
python TRAINING/ranking/predictability/rank_target_predictability.py --symbols AAPL MSFT
```

**Feature Selection:**
```bash
python TRAINING/feature_selection/multi_model_feature_selection.py --target y_will_swing_high_30m_0.10
```

### Key Config Files

- `CONFIG/training_config/safety_config.yaml` - Safety thresholds, leakage detection
- `CONFIG/training_config/pipeline_config.yaml` - Pipeline settings
- `CONFIG/experiments/` - Experiment configs (preferred)
- `CONFIG/model_config/` - Model-specific hyperparameters

### Environment Variables

- `FOXML_STRICT_MODE=1` - Enable strict mode (fail-fast on config errors)
- `BASE_SEED=42` - Set random seed for reproducibility

---

## 📚 Documentation Navigation

### 🧹 [cleanup/](cleanup/) - Code Hardening & Fixes (28 files)
**Bug fixes, leakage fixes, verification, and code quality improvements.**

**Critical Fixes:**
- `FINAL_LEAKAGE_SUMMARY.md` - Complete leakage fix summary
- `DEEPER_LEAK_FIX.md` - Deep leakage fixes
- `TARGET_LEAKAGE_CLARIFICATION.md` - Target leakage analysis
- `PRODUCTION_VERIFICATION_COMPLETE.md` - Production readiness verification
- `HARDENING_SUMMARY.md` - Code hardening summary

**Recent Work:**
- `AUTO_FIXER_FIXES.md` - Auto-fixer improvements
- `SILENT_FAILURES_FIXES.md` - Silent failure fixes
- `F821_FIX_STRATEGY.md` - Undefined variable fixes

### ⚙️ [config/](config/) - Configuration System (11 files)
**Config audits, centralization, and system configuration.**

**Key Documents:**
- `CONFIG_FINAL_CENTRALIZATION_REPORT.md` - Config centralization summary
- `CONFIG_LOCATIONS_AUDIT.md` - Config file locations
- `SST_COMPLIANCE_CHECKLIST.md` - Single Source of Truth compliance

### 🔍 [audits/](audits/) - Audits & Verification (8 files)
**Code audits, reproducibility checks, and validation reports.**

**Key Audits:**
- `VALIDATION_LEAK_AUDIT.md` - Validation and leakage audits
- `REPRODUCIBILITY_AUDIT.md` - Reproducibility verification
- `IMPORT_AUDIT_AND_STRUCTURE.md` - Import structure audit

### 📊 [analysis/](analysis/) - System Analysis (4 files)
**Architecture analysis, optimization analysis, and integration docs.**

**Key Analysis:**
- `OPTIMIZATION_ARCHITECTURE.md` - Optimization architecture
- `OPTIMIZATION_ENGINE_ANALYSIS.md` - Engine analysis
- `INTRADAY_TRADING_ANALYSIS.md` - Intraday trading analysis
- `YAHOO_FINANCE_INTEGRATION.md` - Data integration

### 📋 [planning/](planning/) - Planning & Roadmaps (25 files)
**Feature planning, architecture plans, and implementation roadmaps.**

**Feature Registry:**
- `FEATURE_REGISTRY_COMPLETE.md` - Feature registry completion
- `FEATURE_REGISTRY_DESIGN.md` - Registry design
- `FEATURE_REGISTRY_PHASE1_COMPLETE.md` through `PHASE4_COMPLETE.md`

**Architecture Plans:**
- `ADAPTIVE_INTELLIGENCE_ARCHITECTURE.md` - Adaptive intelligence
- `DISTRIBUTED_TRAINING_ARCHITECTURE.md` - Distributed training
- `NVLINK_READY_ARCHITECTURE.md` - GPU architecture
- `SYSTEMD_DEPLOYMENT_PLAN.md` - Deployment planning

**Optimization:**
- `PERFORMANCE_OPTIMIZATION_PLAN.md` - Performance optimization
- `PRESSURE_TEST_IMPLEMENTATION_ROADMAP.md` - Testing roadmap
- `ALPHA_ENHANCEMENT_ROADMAP.md` - Alpha enhancement

### 🔬 [research/](research/) - Research & Development (14 files)
**Model research, feature selection, and experimental work.**

**Model Research:**
- `ALL_MODELS_ENABLED.md` - Model enabling status
- `MODEL_ENABLING_RECOMMENDATIONS.md` - Recommendations
- `ADDITIONAL_MODELS_QUICKSTART.md` - Quick start guides

**Feature Selection:**
- `COMPLETE_FEATURE_SELECTION_MODELS.md` - Feature selection models
- `FEATURE_IMPORTANCE_FIX.md` - Importance fixes
- `IMPORTANCE_SCORE_INTERPRETATION.md` - Score interpretation

**Targets:**
- `TARGET_DISCOVERY_UPDATE.md` - Target discovery
- `TARGET_RECOMMENDATIONS.md` - Recommendations
- `TARGET_TO_FEATURE_WORKFLOW.md` - Workflow documentation

### 🎯 [features/](features/) - Feature Documentation (2 files)
**Feature implementation and stability documentation.**

- `FEATURE_IMPORTANCE_STABILITY_IMPLEMENTATION.md`
- `FEATURE_IMPORTANCE_STABILITY_INTEGRATION_PLAN.md`

### ⚡ [optimization/](optimization/) - Performance (3 files)
**HPC, optimization engine, and observability.**

- `HPC_OPTIMIZATION_STATUS.md`
- `OPTIMIZATION_ENGINE.md`
- `OBSERVABILITY_LOGGING_ADDED.md`

### 🧮 [foundations/](foundations/) - Core Foundations (2 files)
**Mathematical foundations and core technical docs.**

- `MATHEMATICAL_FOUNDATIONS.md`
- `C++_INTEGRATION.md`

### 🛠️ [tools/](tools/) - Internal Tools (1 file)
**Internal tools and scripts.**

- `tools/README.md`
- `tools/regenerate_enterprise_pdfs.py`

---

## 🏗️ Codebase Structure

### Core Modules

**TRAINING/** - Main training infrastructure
- `train.py` - Main entry point (intelligent training)
- `orchestration/intelligent_trainer.py` - Orchestrator
- `ranking/predictability/` - Target ranking system
- `feature_selection/` - Feature selection
- `model_fun/` - Model trainers (17 models)
- `common/` - Common utilities (safety, threading, strict_mode)
- `utils/` - Data loading, leakage filtering

**CONFIG/** - Configuration system
- `config_loader.py` - Central config loader
- `config_schemas.py` - Config validation schemas
- `training_config/` - Training configs (safety, pipeline, etc.)
- `model_config/` - Model hyperparameters
- `experiments/` - Experiment configs

**DOCS/** - Public documentation
- `01_tutorials/` - Tutorials and guides
- `02_reference/` - API reference
- `03_technical/` - Technical deep-dives

**SCRIPTS/** - Utility scripts (untracked)
- `tests/` - Test scripts
- Various utility scripts

### Key Systems

**1. Configuration System**
- Single Source of Truth (SST) - all params from config
- Schema validation via `config_schemas.py`
- Strict mode: `FOXML_STRICT_MODE=1`
- Config loader: `CONFIG/config_loader.py`

**2. Leakage Detection**
- Pre-training scans
- Auto-fixer (LeakageAutoFixer)
- Feature registry filtering
- Config-driven thresholds

**3. Reproducibility**
- `BASE_SEED` for all random operations
- Config-driven hyperparameters
- Deterministic data splits
- Validation via `REPRODUCIBILITY_AUDIT.md`

**4. Safety System**
- `safety_config.yaml` - Safety thresholds
- Runtime limits (max_runtime_minutes, max_symbols)
- Risk limits
- Leakage detection thresholds

---

## 🚀 Common Workflows

### Starting a New Feature
1. Check `planning/` for existing plans
2. Review `research/` for related work
3. Check `features/` for implementation patterns
4. Create experiment config in `CONFIG/experiments/`
5. Update `journallog.md` with progress

### Fixing a Bug
1. Check `cleanup/` for similar fixes
2. Review `audits/` for related issues
3. Enable strict mode: `FOXML_STRICT_MODE=1`
4. Document fix in `cleanup/`
5. Update verification if needed

### Config Changes
1. Review `config/CONFIG_LOCATIONS_AUDIT.md`
2. Check `config/SST_COMPLIANCE_CHECKLIST.md`
3. Update schema in `CONFIG/config_schemas.py` if needed
4. Follow centralization guidelines
5. Test with strict mode enabled

### Performance Work
1. Check `optimization/` for existing work
2. Review `analysis/OPTIMIZATION_ARCHITECTURE.md`
3. See `planning/PERFORMANCE_OPTIMIZATION_PLAN.md`
4. Profile with appropriate tools
5. Document in `optimization/`

### Research & Experimentation
1. Check `research/` for related work
2. Review model docs in `research/`
3. Create experiment config
4. Document findings
5. Update planning docs if needed

### Running Training
1. **Intelligent Training (Recommended):**
   ```bash
   python TRAINING/train.py --experiment-config my_experiment --auto-targets --auto-features
   ```

2. **Manual Training:**
   ```bash
   python TRAINING/train.py --data-dir data/data_labeled/interval=5m --symbols AAPL MSFT
   ```

3. **With Strict Mode:**
   ```bash
   FOXML_STRICT_MODE=1 python TRAINING/train.py --experiment-config my_experiment
   ```

### Debugging Issues

**Config Issues:**
- Enable strict mode: `FOXML_STRICT_MODE=1`
- Check `CONFIG/config_schemas.py` for schema
- Review `config/CONFIG_LOCATIONS_AUDIT.md`

**Leakage Issues:**
- Check `cleanup/FINAL_LEAKAGE_SUMMARY.md`
- Review `audits/VALIDATION_LEAK_AUDIT.md`
- Enable auto-fixer in config

**Import Errors:**
- Check `audits/IMPORT_AUDIT_AND_STRUCTURE.md`
- Verify module structure
- Check Python path

**Performance Issues:**
- Review `optimization/` docs
- Check `analysis/OPTIMIZATION_ARCHITECTURE.md`
- Profile with appropriate tools

---

## 🔧 Development Practices

### Code Quality

**Static Analysis:**
```bash
# Ruff (fast)
ruff check TRAINING CONFIG

# mypy (type checking)
mypy TRAINING --ignore-missing-imports
```

**Testing:**
```bash
# Config integrity
pytest tests/test_config_integrity.py

# Smoke tests
pytest tests/test_smoke_imports.py
```

### Config Management

**Always:**
- Use experiment configs when possible
- Validate with schema (`config_schemas.py`)
- Test with strict mode
- Document new config keys

**Never:**
- Hardcode parameters
- Access config without validation
- Skip schema updates

### Git Workflow

**Before Committing:**
1. Run static analysis: `ruff check .`
2. Test config loading
3. Check for undefined variables
4. Update relevant docs

**Commit Messages:**
```
feat(<area>): <concise change>

- Summary of change (why)
- Touched: paths (+N/−M) with highlights
- Tests: <paths> (pass)
- Audit: docs/audit/YYYY-MM-DD/HH-MM-SS_<topic>/
```

---

## 📝 Development Journal

See `journallog.md` for chronological development notes and decisions.

---

## 🔗 Related Documentation

- **Public Docs:** See `DOCS/` for user-facing documentation
- **Enterprise Docs:** See `ENTERPRISE_BUNDLE/` for enterprise documentation
- **Config:** See `CONFIG/` for configuration files
- **Scripts:** See `SCRIPTS/` for utility scripts

---

## 📊 Statistics

- **Total Files:** ~106 organized documents
- **Categories:** 10 organized categories
- **Last Updated:** See individual file timestamps

---

## 🎯 Quick Reference

**Need to find...**

- **A bug fix?** → `cleanup/`
- **Config info?** → `config/`
- **Architecture plans?** → `planning/`
- **Model research?** → `research/`
- **Performance work?** → `optimization/`
- **System analysis?** → `analysis/`
- **Feature work?** → `features/`
- **Core foundations?** → `foundations/`

**Need to do...**

- **Start training?** → `python TRAINING/train.py --help`
- **Fix config?** → Check `config/` + enable strict mode
- **Debug leakage?** → Check `cleanup/FINAL_LEAKAGE_SUMMARY.md`
- **Add feature?** → Check `planning/` + `features/`
- **Optimize?** → Check `optimization/` + `analysis/`

---

## 🚨 Important Notes

### Strict Mode
Always test with `FOXML_STRICT_MODE=1` before committing config changes. This catches silent failures early.

### Config Schema
Always update `CONFIG/config_schemas.py` when adding new config keys. This prevents silent fallbacks.

### Leakage Detection
The auto-fixer is enabled by default. Check `cleanup/` for known issues and fixes.

### Reproducibility
All random operations use `BASE_SEED`. Same config → same results. Verify with `REPRODUCIBILITY_AUDIT.md`.

---

*This handbook is a living document. Update it as the codebase evolves.*
