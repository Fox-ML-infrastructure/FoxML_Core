# FoxML Core — License Decision Matrix

**Quick reference: Which license applies to your use case?**

> **Important:** This matrix is for convenience only. In case of any conflict, `COMMERCIAL_LICENSE.md` is the authoritative and controlling document. See `COMMERCIAL_LICENSE.md` Section 22 for the complete document hierarchy.

---

## Quick Decision Matrix

| Scenario | AGPL-3.0 | Commercial License | Notes |
|----------|----------|-------------------|-------|
| **Individual personal research** (not for any business purpose) | ✅ | ❌ | Must be in personal capacity, not for any business or organization |
| **Non-commercial academic research** at degree-granting institution | ✅ | ❌ | Results must not support commercial operations. See `LEGAL/SUBSCRIPTIONS.md` for full definition |
| **Internal evaluation at a company** | ❌ | ✅ | **ALWAYS commercial** — no free trial for commercial use |
| **Proof of concept / pilot project** within a business | ❌ | ✅ | **ALWAYS commercial** — even if experimental or non-revenue |
| **Development / testing / staging** environments in a business | ❌ | ✅ | **ALWAYS commercial** — any environment within a business context |
| **Production use** in a business | ❌ | ✅ | **ALWAYS commercial** |
| **Trading, investment analysis, financial decision-making** | ❌ | ✅ | **ALWAYS commercial** — regardless of revenue status |
| **Client services, consulting, freelance work** | ❌ | ✅ | **ALWAYS commercial** — any work for clients or third parties |
| **Use by employees** in scope of their work | ❌ | ✅ | **ALWAYS commercial** — regardless of role or department |
| **Use by contractors / consultants** for a business | ❌ | ✅ | **ALWAYS commercial** — even if temporary or part-time |
| **Use by interns** in scope of their work | ❌ | ✅ | **ALWAYS commercial** — employment status doesn't matter |
| **Pre-revenue startup** or experimental project | ❌ | ✅ | **ALWAYS commercial** — revenue status doesn't matter |
| **Sole proprietor / freelancer** business use | ❌ | ✅ | **ALWAYS commercial** — any business activity requires commercial license |
| **Corporate research lab** | ❌ | ✅ | **ALWAYS commercial** — even if non-profit status |
| **University research with corporate funding** | ❌ | ✅ | Corporate sponsorship = commercial use |
| **For-profit university** | ❌ | ✅ | **ALWAYS commercial** — excluded from academic carve-out |
| **Academic partnership with commercial entity** | ❌ | ✅ | If integrated into commercial operations, requires commercial license |
| **Selling outputs** (signals, analytics, predictions) to clients | ❌ | ✅ | Allowed with commercial license, provided clients don't access Software |
| **Internal tools / dashboards** within a business | ❌ | ✅ | **ALWAYS commercial** — any internal use in business context |
| **Research pipelines** within a business | ❌ | ✅ | **ALWAYS commercial** — even if "research-only" |
| **Mixed use** (some teams AGPL, some commercial) | ❌ | ✅ | **PROHIBITED** — if any part of org uses commercially, all use requires commercial license |
| **Subsidiary / parent company** sharing license | ❌ | ✅ | Each legal entity needs its own license — sharing prohibited |
| **Hosting on AWS / GCP / Azure** for internal use | ❌ | ✅ | Allowed with commercial license for internal use only |
| **Building competing infrastructure service** | ❌ | ❌ | **PROHIBITED** — may not build Competing Infrastructure Service (see `COMMERCIAL_LICENSE.md`) |
| **Reverse engineering** | ❌ | ❌ | **PROHIBITED** — under both licenses |
| **Benchmarking without consent** | ❌ | ❌ | **PROHIBITED** — under commercial license (Material Breach) |
| **Sharing / loaning / renting license** | ❌ | ❌ | **PROHIBITED** — each entity needs its own license |

---

## Key Principles

### ✅ AGPL-3.0 is ONLY for:
1. **Individual personal research** (not for any business purpose)
2. **Non-commercial academic research** at degree-granting institutions (where results don't support commercial operations)

### ❌ Commercial License is REQUIRED for:
1. **Any use within a business, organization, or commercial entity**
2. **Any use that directly or indirectly supports revenue-generating activities**
3. **Any use by employees, contractors, interns, or Affiliates in scope of their work**
4. **Any use in any environment** (development, testing, staging, production) within a business context
5. **Any experimental, proof of concept, or pilot project** within a business context

### 🚫 PROHIBITED under both licenses:
1. **Building a Competing Infrastructure Service**
2. **Reverse engineering**
3. **Benchmarking without consent** (commercial license)
4. **Sharing / loaning / renting licenses**

---

## Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "We're just evaluating internally" | ❌ Still requires commercial license — no free trial for commercial use |
| "It's not in production yet" | ❌ Development/testing/staging still requires commercial license |
| "We're not generating revenue" | ❌ Revenue status doesn't matter — business use requires commercial license |
| "It's just research" | ❌ Research within a business context requires commercial license |
| "Our interns are using it" | ❌ Use by employees/contractors/interns requires commercial license |
| "We're a non-profit" | ❌ Non-profit status doesn't exempt commercial use — still requires commercial license |
| "We're a university" | ⚠️ Only non-commercial academic research at degree-granting institutions is exempt — corporate funding or commercial integration requires commercial license |
| "We'll only use it for 30 days" | ❌ Duration doesn't matter — commercial use requires commercial license from day one |
| "It's open source, so it's free" | ❌ AGPL-3.0 is free only for non-commercial use — commercial use requires commercial license |
| "We'll contribute back to open source" | ❌ Open source contributions don't exempt commercial use from licensing requirements |

---

## Decision Flow

```
Are you using this for personal research (not for any business)?
├─ YES → AGPL-3.0 ✅
└─ NO → Continue

Are you at a degree-granting educational institution doing non-commercial research?
├─ YES → Is it funded by or integrated into commercial operations?
│   ├─ YES → Commercial License ❌
│   └─ NO → AGPL-3.0 ✅
└─ NO → Continue

Are you using this within a business, organization, or commercial entity?
├─ YES → Commercial License ❌
└─ NO → Continue

Are you using this for any business purpose, client work, or revenue-generating activity?
├─ YES → Commercial License ❌
└─ NO → Review definitions in COMMERCIAL_LICENSE.md Section 1
```

**Default rule:** If you're unsure, you almost certainly need a commercial license. When in doubt, contact jenn.lewis5789@gmail.com.

---

## Still Unsure?

1. Review `COMMERCIAL_LICENSE.md` Section 1 ("Commercial Use" definition)
2. Review `LEGAL/SUBSCRIPTIONS.md` for usage scenarios
3. Review `LEGAL/FAQ.md` for common questions
4. Contact: jenn.lewis5789@gmail.com

---

**Last Updated:** 2025-12-09  
**Document Hierarchy:** This matrix is for convenience only. `COMMERCIAL_LICENSE.md` is the authoritative and controlling document.

