# Changelog Entry Template

**Internal template for creating changelog entries. This file is not pushed to the repository.**

---

## Entry Format

```markdown
## [Unreleased]

### Added
- New feature or capability
- Another new feature

### Changed
- Change to existing functionality
- Another change

### Commercial
- Business/pricing/licensing change
- Another commercial update

### Deprecated
- Feature that will be removed in a future version

### Removed
- Removed feature

### Fixed
- Bug fix
- Another bug fix

### Security
- Security improvement

### Documentation
- Documentation update
```

---

## ⚠️ CRITICAL: What NOT to Include

**NEVER include in public changelog entries:**

1. **Internal Documentation**
   - ❌ References to `DOCS/internal/` files or changes
   - ❌ Internal planning documents (CILS, ARPO, Integrated Feedback Loop, etc.)
   - ❌ Internal research, analysis, or fixes documentation
   - ❌ Internal audit trails or journal logs
   - ✅ Only include public-facing documentation changes in `DOCS/` (outside `internal/`)

2. **Special Projects & Research**
   - ❌ Experimental features or research projects not yet public
   - ❌ Internal architecture discussions or design documents
   - ❌ Proprietary algorithms or methodologies under development
   - ❌ Internal performance optimizations or tuning details
   - ✅ Only include completed, public-facing features

3. **Internal Processes**
   - ❌ Internal workflow changes or process improvements
   - ❌ Development tooling or infrastructure changes (unless user-facing)
   - ❌ Internal testing or validation procedures
   - ✅ Only include user-visible improvements or changes

**Remember:** The changelog is public-facing. If it's in `DOCS/internal/` or marked as internal/special project, it does NOT belong in the changelog.

---

## Guidelines

1. **Group changes by category** (Added, Changed, Commercial, Fixed, Security, Documentation, etc.)
2. **Use present tense** ("Add feature" not "Added feature")
3. **Be specific** – Include what changed and why (if relevant)
4. **Link to issues/PRs** – Use `[#123]` format if applicable
5. **Include breaking changes** – Clearly mark with `[BREAKING]` prefix
6. **Version format** – Use `[Unreleased]` for ongoing work, or `[1.2.3] - YYYY-MM-DD` for tagged releases
7. **Commercial changes** – Use Commercial category for pricing, licensing, or business policy updates (not technical changes)
8. **Use em dashes** – Use em dashes (–) for ranges (e.g., "1–10 employees")
9. **Exclude internal work** – Never mention internal docs, special projects, or research in public changelog

---

## Example Entry

```markdown
## [Unreleased]

### Added
- Centralized configuration system with YAML-based configs
- New compliance documentation suite
- Base trainer scaffolding for 2D and 3D models

### Changed
- All model trainers now use centralized configs
- Pipeline settings integrated into centralized config system

### Commercial
- Commercial license pricing recalibrated to align with enterprise market norms:
  - 1–10 employees: $18,000 → $25,200/year
  - 11–50 employees: $36,000 → $60,000/year

### Fixed
- VAE serialization issues resolved
- Sequential models 3D preprocessing corrected
- Type conversion issues in callback configs

### Security
- Enhanced compliance documentation for production use

### Documentation
- Added LICENSE_ENFORCEMENT.md
- Updated ROADMAP.md with UI integration approach
```

---

## When to Update

- After completing a significant feature
- After fixing critical bugs
- After major refactoring
- Before tagging a release
- When documentation is significantly updated

---

**Note:** Copy this template structure when creating changelog entries. Only the filled-out `CHANGELOG.md` in the root directory is pushed to the repository.

**⚠️ Final Check Before Publishing:**
- [ ] No references to `DOCS/internal/` or internal documentation
- [ ] No mentions of special projects or research initiatives
- [ ] No internal-only changes or processes
- [ ] All entries are user-facing or public-appropriate

