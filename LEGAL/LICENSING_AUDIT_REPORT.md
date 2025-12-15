# Licensing Audit Report — Recommendations

**Date:** 2025-12-15  
**Auditor:** AI Assistant  
**Based on:** Source-available dual licensing best practices

---

## Executive Summary

Your licensing setup is **strong overall** with comprehensive documentation. However, there are **3 critical gaps** and **5 recommended improvements** to harden enforcement and close loopholes.

**Status:** ✅ **85% Complete** — Ready for production with minor hardening needed.

---

## ✅ What You Have (Strengths)

1. **Clear dual-license structure**
   - `LICENSE` (source-available, free use)
   - `COMMERCIAL_LICENSE.md` (commercial terms)
   - `DUAL_LICENSE.md` (overview)

2. **Comprehensive documentation**
   - `LEGAL/COMMERCIAL_USE.md` (usage guide)
   - `LEGAL/FAQ.md` (business license FAQ)
   - `LEGAL/CLA.md` (contributor agreement)

3. **Legal infrastructure**
   - Copyright notices policy (`LEGAL/COPYRIGHT_NOTICE.md`)
   - Trademark policy (`LEGAL/TRADEMARK_POLICY.md`)
   - SPDX headers in 220+ files

4. **Enforcement mechanisms**
   - 30-day evaluation period (trial business license)
   - Clear document hierarchy
   - Audit rights defined

---

## ❌ Critical Gaps (Must Fix)

### Gap 1: Missing "Natural Person" Hard Rule

**Issue:** Your `LICENSE` doesn't include the explicit rule: **"If you are not a natural person acting solely on your own behalf, you need a business license."**

**Why it matters:** This closes the "I'm an individual contractor" loophole and makes enforcement unambiguous.

**Fix:** Add to `LICENSE` after line 10:

```markdown
**HARD RULE: If you are not a natural person acting solely on your own behalf, you need a business license.**

This means:
- ✅ Natural person using Software for personal, non-commercial purposes → Free
- ❌ Natural person using Software for any business, client work, or organizational purpose → Commercial license required
- ❌ Any legal entity (LLC, corporation, partnership, etc.) → Commercial license required
- ❌ Employee, contractor, or agent acting for an organization → Commercial license required
```

**Location:** Insert after line 10 in `LICENSE`.

---

### Gap 2: No Root NOTICE File

**Issue:** You have `LEGAL/COPYRIGHT_NOTICE.md` (guidance) but no root `NOTICE` file that serves as a machine-readable copyright statement.

**Why it matters:** Industry standard for source-available projects. Makes copyright ownership immediately visible.

**Fix:** Create `NOTICE` file in repo root:

```
FoxML Core
Copyright (c) 2025-2026 Fox ML Infrastructure LLC
All rights reserved.

Source-available license. See LICENSE for free use terms.
Commercial license required for organizational use. See COMMERCIAL_LICENSE.md.

This software is provided "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

### Gap 3: "Personal Use" Definition Could Be Stronger

**Issue:** Your definition of "Personal Use" (LICENSE lines 41-44) is good but could explicitly exclude:
- Freelancers/contractors doing client work
- Sole proprietors using it for business
- "Personal" projects that support business activities

**Fix:** Strengthen the Personal Use definition in `LICENSE`:

```markdown
1. **Personal Use**: Individual use in your personal capacity where:
   - You are a natural person acting solely on your own behalf
   - You are NOT using it as an employee, contractor, or agent of any business, organization, or entity
   - You are NOT using it for the benefit of any third party (including clients, employers, or business partners)
   - You are NOT using it in connection with any commercial activity, revenue generation, or business purpose
   - You are NOT a sole proprietor, freelancer, or independent contractor using it for client work or business operations
   
   **Explicitly NOT personal use:**
   - Any use by freelancers, contractors, or consultants for client work
   - Any use by sole proprietors for business operations
   - Any "personal" project that supports or relates to business activities
   - Any use that could reasonably be considered business-related, even if labeled "personal" or "experimental"
```

---

## ⚠️ Recommended Improvements (Should Fix)

### Improvement 1: Add "Business License FAQ" Section

**Current:** You have `LEGAL/FAQ.md` but it could be more explicit about edge cases.

**Recommendation:** Add a dedicated section to `LEGAL/FAQ.md`:

```markdown
## Business License Decision Tree

**Q: Does my hedge fund backtest count?**  
A: Yes. Any use by or for a business requires a commercial license.

**Q: I'm a freelancer doing client work. Do I need a license?**  
A: Yes. Use for client work is business use, not personal use.

**Q: I'm evaluating it for my company. Is that free?**  
A: You have 30 days for evaluation. After that, commercial license required.

**Q: I'm a student doing research. Is that free?**  
A: Only if at a qualifying tax-exempt institution AND research is non-commercial (see LICENSE for full definition).

**Q: I'm a sole proprietor. Is that personal use?**  
A: No. Sole proprietors using Software for business operations require a commercial license.
```

---

### Improvement 2: Strengthen README Licensing Section

**Current:** README has licensing info but could be more explicit about the "natural person" rule.

**Recommendation:** Update README.md lines 5-15 to include:

```markdown
## License

**FoxML Core is source-available. Free for personal use, paid for business.**

- ✅ **Free for:** Natural persons using Software for personal, non-commercial purposes
- 💰 **Paid license required for:** Any organizational, institutional, or commercial use

**Hard rule:** If you are not a natural person acting solely on your own behalf, you need a business license.

**Examples requiring a commercial license:**
- Any use by or for a company, university, lab, or organization
- Freelancers/contractors using it for client work
- Sole proprietors using it for business operations
- Employees using it in the scope of their work
- Any evaluation, pilot, or PoC done by an organization (after 30-day trial)

📧 **jenn.lewis5789@gmail.com** | Subject: `FoxML Core Commercial Licensing`

See [`LICENSE`](LICENSE) for free use terms. See [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for commercial terms.
```

---

### Improvement 3: Add SPDX Identifier to LICENSE

**Current:** LICENSE doesn't have an SPDX identifier.

**Recommendation:** Add to top of `LICENSE`:

```
SPDX-License-Identifier: Proprietary
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Fox ML Infrastructure LLC
```

**Note:** Since this is proprietary/source-available (not OSI), use `Proprietary` or create a custom identifier like `FoxML-Source-Available`.

---

### Improvement 4: Add "Trial Business License" Clarification

**Current:** You have a 30-day evaluation period, but it's not explicitly called a "trial business license."

**Recommendation:** In `COMMERCIAL_LICENSE.md` Section 2, clarify:

```markdown
**Evaluation License (30-Day Trial Business License).**

This is a **trial business license** for organizations to evaluate the Software. 
It is NOT a free license for ongoing business use. After 30 days, continued 
organizational use requires a paid commercial license.
```

---

### Improvement 5: Add "Weasel-Proof" Language to FAQ

**Current:** FAQ covers common questions but could pre-empt more excuses.

**Recommendation:** Add to `LEGAL/FAQ.md`:

```markdown
## Common Excuses (Pre-empted)

**"It's just research."**  
→ If done by or for an organization, commercial license required.

**"It's internal only."**  
→ Internal organizational use requires a commercial license.

**"We're not selling it."**  
→ Revenue generation is not required. Organizational use = commercial license.

**"It's for evaluation."**  
→ 30-day trial available. After that, commercial license required.

**"It's experimental/non-production."**  
→ Environment label doesn't matter. Organizational use = commercial license.

**"I'm a contractor, not an employee."**  
→ Use for client work or in scope of work = commercial license required.
```

---

## 📋 Implementation Checklist

### Critical (Do First)
- [ ] Add "natural person" hard rule to `LICENSE` (after line 10)
- [ ] Create root `NOTICE` file
- [ ] Strengthen "Personal Use" definition in `LICENSE`

### Recommended (Do Soon)
- [ ] Add "Business License Decision Tree" to `LEGAL/FAQ.md`
- [ ] Strengthen README licensing section
- [ ] Add SPDX identifier to `LICENSE`
- [ ] Clarify "trial business license" in `COMMERCIAL_LICENSE.md`
- [ ] Add "weasel-proof" language to FAQ

### Optional (Nice to Have)
- [ ] Consider renaming `LICENSE` → `LICENSE-PERSONAL.md` and `COMMERCIAL_LICENSE.md` → `LICENSE-BUSINESS.md` (more explicit naming)
- [ ] Add machine-readable license metadata (e.g., `license.json`)
- [ ] Create a quick-reference decision matrix (you have `LEGAL/DECISION_MATRIX.md` — verify it's up to date)

---

## 🎯 Summary

**Your licensing is 85% complete and production-ready.** The three critical gaps are easy fixes that will:
1. Close the "natural person" loophole
2. Add industry-standard NOTICE file
3. Strengthen personal use definition

**Estimated time to fix:** 2-3 hours for all critical + recommended items.

**Risk if not fixed:** Medium — Most users will comply, but the gaps could allow some to rationalize unlicensed use.

---

## 📞 Next Steps

1. **Review this audit** with your legal counsel (if you have one)
2. **Implement critical fixes** (Gap 1-3)
3. **Implement recommended improvements** (Improvement 1-5)
4. **Test the updated language** with a few potential customers
5. **Update version numbers** in license files after changes

---

**Questions?** Contact: jenn.lewis5789@gmail.com  
**Subject:** Licensing Audit Follow-up
