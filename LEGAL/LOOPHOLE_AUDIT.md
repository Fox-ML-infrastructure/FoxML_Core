# Loophole Audit Report — 2025-12-16

**Purpose:** Identify and document potential loopholes in licensing that could allow unauthorized commercial use.

---

## 🔴 CRITICAL GAPS FOUND

### 1. **Repeated Evaluation Requests Loophole**

**Issue:** The 30-day evaluation doesn't explicitly prohibit repeated requests. An organization could theoretically request multiple 30-day evaluations to extend free use indefinitely.

**Current Language:**
- "30-Day Evaluation ($0): Organizations may evaluate the Software for 30 days..."
- "Must request in writing (subject to approval)"

**Risk:** Organization requests evaluation → uses for 30 days → requests another evaluation → repeats indefinitely.

**Fix Needed:** Add explicit language: "Each organization is limited to one 30-day evaluation period. Repeated evaluation requests from the same organization or related entities are not permitted."

---

### 2. **API Wrapper / Indirect Access Loophole**

**Issue:** The license prohibits "access to the Software's APIs" but doesn't explicitly prohibit wrapping the Software in a custom API layer and selling access to that wrapper.

**Current Language:**
- Prohibits: "access to the Software's APIs"
- Prohibits: "ability to execute code or workflows within the Software"
- But: What if someone wraps the Software in their own API and sells access to the wrapper?

**Risk:** Company wraps Software → creates "MyAPI" that calls Software internally → sells MyAPI access to clients → clients never directly access Software APIs, but effectively use it.

**Fix Needed:** Clarify that any indirect access pattern that allows third parties to use the Software's functionality (even through wrappers) is prohibited.

---

### 3. **Fork and Rebrand Loophole (Academic License)**

**Issue:** Academic redistribution allows forks, and while the license must be preserved, the product name could be changed, potentially creating confusion or allowing someone to claim it's a "new" product.

**Current Language:**
- "You may create forks or modified versions for non-commercial academic use"
- "You may publish modified versions for non-commercial academic purposes, provided this license is preserved in full"

**Risk:** Academic forks the Software → changes name to "NewML" → publishes as "new" product → organizations use it thinking it's different → but it's still subject to the same license.

**Note:** This is actually protected because the license must be preserved, but the risk is confusion/misrepresentation.

**Fix Needed:** Explicitly state that forks must maintain clear attribution and cannot be presented as a "new" or "independent" product.

---

### 4. **Training Competing Models (Not Infrastructure) Gap**

**Issue:** We prohibit using outputs to build "Competing Infrastructure Services" but what about using outputs to train competing ML models (not infrastructure platforms)?

**Current Language:**
- Prohibits: "use outputs... to build, train, develop, or operate a Competing Infrastructure Service"

**Risk:** Company uses Software outputs to train a competing trading model → sells that model → not technically a "Competing Infrastructure Service" but still competes.

**Fix Needed:** Clarify that using outputs to train competing models/algorithms (even if not infrastructure) is prohibited if those models compete with the Software's purpose.

---

### 5. **Affiliate "For Benefit Of" Ambiguity**

**Issue:** Affiliates can use Software "solely for the benefit of Licensee" but the distinction between "for benefit of" vs "for own benefit" could be exploited.

**Current Language:**
- "Licensee may permit its Affiliates to use the Software solely for the benefit of Licensee"
- "each separate legal entity... that wishes to use the Software for its own benefit must obtain its own license"

**Risk:** Subsidiary claims it's using Software "for benefit of parent" when it's actually using it for its own operations.

**Fix Needed:** Clarify that "for benefit of Licensee" means the affiliate's use directly supports Licensee's operations, not the affiliate's independent operations.

---

## 🟡 MINOR GAPS (Lower Risk)

### 6. **Academic Redistribution to Commercial Entities**

**Status:** ✅ **PROTECTED** — The license explicitly states:
- "Redistribution does NOT grant recipients any organizational rights or commercial use rights"
- "Any redistribution to an organization (even if labeled 'for evaluation') constitutes unauthorized organizational use unless the recipient has a valid commercial license"

**No fix needed.**

---

### 7. **Contractor Multi-Client Loophole**

**Status:** ✅ **PROTECTED** — The license explicitly states:
- "a Contractor may not use a single instance of the Software, or the same credentials or license, to provide services to multiple clients unless each such client is itself a separately licensed customer"

**No fix needed.**

---

### 8. **Older Version Circumvention**

**Status:** ✅ **PROTECTED** — The license explicitly states:
- "Using an older version does not exempt Licensee from any obligations, restrictions, or payment requirements. Licensee may not use an older version to circumvent restrictions that apply to newer versions."

**No fix needed.**

---

## 📋 RECOMMENDED FIXES

### Priority 1 (Critical):
1. Add explicit "one evaluation per organization" limit
2. Clarify API wrapper / indirect access prohibition
3. Clarify training competing models prohibition

### Priority 2 (Important):
4. Clarify affiliate "for benefit of" language
5. Add explicit attribution requirement for forks

---

**Next Steps:** Review and implement fixes for Priority 1 items.

