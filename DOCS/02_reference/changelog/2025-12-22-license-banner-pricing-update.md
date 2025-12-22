# 2025-12-22: License Banner Pricing Structure Update

## Summary

Updated the license banner with a more realistic and approachable pricing structure that better matches FoxML's current stage. Split License vs Support, added entry-level tiers, and fixed AGPL wording to be more accurate.

## Problem

The previous pricing structure had a $120k/year floor that was too high for organizations just starting to evaluate FoxML. The pricing was structured as if FoxML already had enterprise deployments, references, and full support infrastructure in place.

## Solution

### Restructured Pricing Ladder

**Before**: Single $120k/year floor with limited entry points

**After**: Multi-tier pricing ladder with clear progression:

1. **Evaluation**: 30-day $0 (unchanged)
   - Non-production, no client deliverables

2. **Commercial Evaluation**: $1,000–$5,000 for 60–90 days (NEW)
   - Time-limited, no SLA, internal use only
   - Entry point for organizations wanting to test in their environment

3. **Commercial License (no SLA)**:
   - Small team (1–10 users): $10,000–$25,000/year (NEW)
   - Team (11–25 users): $25,000–$60,000/year (NEW)
   - License-only, no support SLA

4. **Enterprise (SLA + onboarding)**: $120,000+/year
   - Includes support, response times, on-call, roadmap commitments
   - This is where the previous "Core Desk" tier belongs

5. **Pilot**: $10,000–$20,000 (reduced from $35,000)
   - Includes hands-on integration work
   - More realistic pricing for pilot engagements

### Key Changes

1. **Split License vs Support**:
   - Commercial License = legal right to use (no SLA)
   - Enterprise = License + Support/SLA (justifies higher pricing)

2. **Added Entry-Level Tiers**:
   - Commercial Evaluation ($1k–$5k) for 60–90 days
   - Small team license ($10k–$25k/year)
   - Team license ($25k–$60k/year)

3. **Fixed AGPL Wording**:
   - **Before**: "must publish your modifications if deployed as service"
   - **After**: "must make source available to users who interact over a network (can be internal availability to employees, not necessarily public)"
   - More accurate: AGPL requires source availability to network users, which can be internal-only, not necessarily public

4. **Clearer Messaging**:
   - Replaced hard "$120k floor" with: "Commercial licenses start in the low five figures; enterprise SLAs start at six figures. Contact for a quote."
   - Makes pricing more approachable and less intimidating

## Files Changed

- **`TRAINING/common/license_banner.py`**:
  - Updated pricing section with new tier structure
  - Fixed AGPL wording for accuracy
  - Added Commercial Evaluation tier
  - Split License vs Support clearly
  - Reduced Pilot pricing to $10k–$20k

## Impact

- **More approachable**: Organizations can start with lower-cost evaluation and entry tiers
- **Clearer value proposition**: Enterprise tier clearly includes SLA/support, justifying higher price
- **Better match to reality**: Pricing structure matches FoxML's current stage (not assuming enterprise infrastructure exists)
- **Accurate AGPL description**: Corrects common misconception about AGPL requirements

## Rationale

The previous pricing assumed FoxML had:
- Enterprise deployments and references
- Full onboarding playbook
- Support infrastructure and load
- Urgent legal/compliance needs

The new structure:
- Provides a "wedge tier" that turns interest into usage
- Doesn't scare off potential customers with high floor
- Clearly separates license cost from support cost
- Matches pricing to actual value delivered at each tier

## Related Documentation

- `LEGAL/QUICK_REFERENCE.md` - May need similar updates
- `LEGAL/SUBSCRIPTIONS.md` - May need similar updates
- `DOCS/02_reference/commercial/COMMERCIAL.md` - May need similar updates

