# Hook System (Experimental)

**Status:** Design complete, ready for integration when needed

This directory contains the complete hook system design for extending the TRAINING pipeline without refactoring orchestrator code.

## Files

### Core Implementation
- `pipeline_hooks.py` - Core hook registry with determinism guarantees
- `plugin_loader.py` - Controlled plugin loading
- `pipeline_hook_points.md` - Hook point documentation
- `pipeline_hook_points_list.py` - Allowlist for CI validation

### Documentation
- `HOOK_SYSTEM_README.md` - Complete usage guide
- `HOOK_SYSTEM_FINAL_SUMMARY.md` - Production-ready summary
- `DETERMINISM_GUIDE.md` - Determinism rules and patterns
- `DETERMINISM_FIXES.md` - Fixes applied for determinism
- `HOOK_POINT_ALLOWLIST.md` - CI allowlist usage

### Design & Audit
- `HOOK_SYSTEM_HARDENING.md` - Hardening checklist
- `HOOK_SYSTEM_AUDIT.md` - Initial audit findings
- `HOOK_SYSTEM_FINAL_AUDIT.md` - Final audit with tweaks

### Examples & Tests
- `pipeline_hooks_example.py` - Usage examples
- `test_pipeline_hooks.py` - Integration tests

## Key Features

1. **Deterministic ordering** - Same config = same hook execution order
2. **Explicit plugin loading** - Config-driven, preserves order
3. **Error modes** - continue/raise/disable
4. **Hook point allowlist** - CI enforcement
5. **Execution trace** - Observability without breaking determinism

## Integration Plan

When ready to integrate:

1. Move core files to `TRAINING/common/`
2. Add hook points to orchestrator (one-time setup)
3. Migrate existing optional features to hooks
4. Enable allowlist in CI
5. Update documentation

## Status

✅ Design complete
✅ Determinism hardened
✅ All footguns fixed
✅ Production-ready pattern
⏸️ Awaiting integration decision
