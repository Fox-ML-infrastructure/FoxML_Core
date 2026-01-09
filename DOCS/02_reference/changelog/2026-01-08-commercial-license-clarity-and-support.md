# Commercial License Clarity and Root Support Documentation

**Date:** 2026-01-08

## Summary

Clarified commercial license requirements in README and added root-level support documentation for easier discovery.

## Changes

### README Licensing Wording Updates

**File:** `README.md`

Updated licensing section to make commercial license requirements crystal clear:

- **Before:** "Commercial License (for proprietary or for-profit use)"
- **After:** "Commercial License required for proprietary/closed deployments or to avoid AGPL obligations (especially SaaS/network use)"

**Rationale:** "For-profit use" is ambiguous. The clear trigger is: "Can't comply with AGPL source disclosure → need commercial license." This makes it immediately obvious to legal teams when a commercial license is required, especially for SaaS/network deployments.

### Root-Level Support Documentation

**File:** `SUPPORT.md` (new)

Created a simple, scannable support file at the repository root that:
- States who gets support (commercial licensees vs free users)
- Provides contact information
- Lists support tiers briefly
- Links to detailed support policy in LEGAL folder

**Rationale:** People scan the root directory first. A simple SUPPORT.md makes it immediately clear what's available without navigating the LEGAL folder, which contains many documents.

## Impact

- **Legal clarity:** Commercial license trigger is now explicit and unambiguous
- **Discoverability:** Support information is easily found at root level
- **Conversion:** Clearer messaging should improve commercial license inquiries

## Related Files

- `README.md` - Updated licensing wording
- `SUPPORT.md` - New root-level support documentation
- `LEGAL/SUPPORT_POLICY.md` - Detailed support policy (unchanged)
- `LEGAL/TRADEMARK_POLICY.md` - Trademark policy (unchanged)
