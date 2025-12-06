# Fox ML Infrastructure — Intellectual Property Ownership Clarification

This document clarifies intellectual property ownership boundaries between Fox ML Infrastructure and commercial licensees.  
This is essential for hedge funds, quantitative firms, and enterprises concerned about IP contamination and ownership clarity.

---

## 1. Executive Summary

**Clear IP boundaries prevent disputes and protect both parties:**

- **Fox ML Infrastructure LLC** retains ownership of all original Fox-v1-infra code and core platform IP
- **Clients** own their additions, configurations, strategies, and proprietary data
- **Custom features** developed under contract belong to the party specified in the Statement of Work (SOW)
- **General patterns** and non-specific code may be reused by Fox ML Infrastructure

---

## 2. Core Platform IP Ownership

### 2.1 Fox ML Infrastructure Owns Core Platform

**Fox ML Infrastructure LLC retains ownership of:**

- **Core codebase** — All original Fox-v1-infra source code, modules, and components
- **Architecture and design** — System architecture, design patterns, and structural decisions
- **Framework components** — Training frameworks, validation systems, pipeline orchestration
- **Generic features** — Features that are not client-specific
- **Documentation** — Core platform documentation and technical specifications
- **Pre-existing IP** — All IP developed prior to any client engagement

**This ownership is perpetual and does not transfer to clients.**

### 2.2 License, Not Sale

**Commercial licenses grant usage rights, not ownership:**

- Clients receive a **license to use** the Software
- Clients do **not** receive ownership of the Software
- Core platform IP remains the property of Fox ML Infrastructure LLC
- Clients may modify the Software for internal use, but modifications do not transfer ownership of the core platform

---

## 3. Client-Owned IP

### 3.1 Client Additions and Customizations

**Clients own:**

- **Client-specific configurations** — Configuration files, parameter settings, and deployment recipes created by the client
- **Client-specific strategies** — Trading strategies, research methodologies, and proprietary algorithms developed by the client
- **Client data** — All client data, datasets, and proprietary information
- **Client integrations** — Integration code written by the client to connect Fox ML Infrastructure with client systems
- **Client modifications** — Modifications made by the client that are not derived from core platform code

### 3.2 Client Developments

**Work created exclusively for the client under a Statement of Work (SOW):**

- **Client-owned deliverables** — If the SOW specifies client ownership, deliverables become client-owned IP
- **Consultant-owned deliverables** — If the SOW does not specify client ownership, deliverables may remain consultant-owned (see Section 4)
- **Custom features** — Ownership of custom features is determined by the SOW terms

---

## 4. Custom Features Developed Under Contract

### 4.1 Default Ownership

**Unless explicitly assigned in the SOW:**

- **Custom features developed by Fox ML Infrastructure** remain the property of Fox ML Infrastructure LLC
- **Generalization rights** — Fox ML Infrastructure may generalize client-specific work and incorporate it into the core platform
- **Reuse rights** — Fox ML Infrastructure may reuse non-proprietary patterns and techniques in future projects

### 4.2 Client Ownership (When Specified)

**If the SOW explicitly assigns ownership to the client:**

- **Client-owned custom features** — Features become client-owned IP
- **Exclusive rights** — If exclusivity is purchased, Fox ML Infrastructure will not provide similar features to other clients
- **License back** — Fox ML Infrastructure may receive a license to use generalized patterns (if negotiated)

### 4.3 Exclusivity Premium

**Clients may purchase exclusivity rights:**

- **Exclusive features** — Features developed exclusively for the client and not provided to others
- **Exclusivity premium** — Additional fee required for exclusive rights
- **SOW terms** — Exclusivity terms are defined in the Statement of Work

**See `legal/consulting/IP_TERMS_ADDENDUM.md` for detailed IP terms for consulting engagements.**

---

## 5. Derivative Works

### 5.1 Derivative Works Based on Core Platform

**If client work incorporates or extends core platform components:**

- **Client receives a license** — Client receives an internal-use license to use the derivative works
- **Core IP remains owned** — Core underlying components remain the property of Fox ML Infrastructure LLC
- **License terms** — License terms (perpetual, internal-use, non-exclusive) are defined in the SOW or commercial license

### 5.2 Client Modifications

**Client modifications to the Software:**

- **Internal use only** — Modifications may be used internally by the client
- **No redistribution** — Client may not redistribute modified versions
- **No ownership transfer** — Modifications do not transfer ownership of the core platform
- **Client owns modifications** — Client owns their specific modifications (but not the underlying platform)

---

## 6. Reuse Rights

### 6.1 General Patterns and Techniques

**Fox ML Infrastructure may reuse:**

- **General engineering patterns** — Common patterns, architectural concepts, and design principles
- **Non-client-specific techniques** — Techniques that are not proprietary to a specific client
- **Generic improvements** — Improvements that are not client-specific
- **Templates and utilities** — Reusable templates and utility functions

### 6.2 What Will Never Be Reused

**Fox ML Infrastructure will never reuse:**

- **Client data** — Client datasets, proprietary data, or confidential information
- **Client-specific algorithms** — Proprietary algorithms or strategies developed for a specific client
- **Client secrets** — Client API keys, credentials, or confidential configurations
- **Client proprietary code** — Code that is clearly proprietary to the client

---

## 7. Open-Source Components

### 7.1 Third-Party Open-Source

**Open-source components follow their upstream licenses:**

- **Upstream licenses** — Components licensed under AGPL, MIT, Apache, etc. follow their respective licenses
- **Client rights** — Clients receive rights under both the commercial license and applicable open-source licenses
- **No ownership transfer** — Open-source components remain under their original licenses

### 7.2 Fox ML Infrastructure Open-Source Core

**The public OSS core (AGPL-3.0):**

- **Open-source license** — Available under AGPL-3.0 for non-commercial use
- **Commercial license required** — Commercial use requires a commercial license
- **Core IP remains owned** — Core platform IP remains the property of Fox ML Infrastructure LLC

---

## 8. Feedback and Improvements

### 8.1 Feedback License

**Client feedback grants Fox ML Infrastructure a license:**

- **Non-exclusive license** — Fox ML Infrastructure receives a non-exclusive, perpetual, irrevocable license to use feedback
- **No obligation** — Fox ML Infrastructure has no obligation to implement feedback or compensate the client
- **No ownership transfer** — Client does not receive ownership of improvements made by Fox ML Infrastructure

**This is standard in commercial licenses and allows Fox ML Infrastructure to improve the platform based on client input.**

---

## 9. No Transfer of Core IP

### 9.1 Core IP Does Not Transfer

**Core platform IP does not transfer to clients under any circumstances:**

- **No purchase option** — Core IP is not available for purchase
- **No assignment** — Core IP cannot be assigned to clients
- **No exclusive licensing** — Core IP is licensed non-exclusively to all commercial licensees
- **Perpetual ownership** — Fox ML Infrastructure LLC retains perpetual ownership of core platform IP

### 9.2 Separate Purchase (If Available)

**If core IP transfer is desired (rare and expensive):**

- **Separate agreement required** — Would require a separate, custom agreement
- **Significant premium** — Would require a substantial premium payment
- **Not standard** — This is not a standard offering and would be negotiated case-by-case

---

## 10. IP Contamination Protection

### 10.1 Clear Boundaries Prevent Contamination

**This clarification document prevents IP contamination concerns:**

- **Clear ownership** — Explicit ownership boundaries prevent disputes
- **No accidental transfer** — Core IP cannot be accidentally transferred
- **Client protection** — Clients own their additions and customizations
- **Vendor protection** — Vendor retains core platform IP

### 10.2 Hedge Fund Considerations

**For hedge funds and quantitative firms:**

- **Strategy protection** — Client strategies remain client-owned and confidential
- **No cross-contamination** — Work done for one client does not contaminate another client's IP
- **Clear separation** — Client-specific repositories keep IP separate
- **NDA protection** — Non-Disclosure Agreements provide additional protection

**See `legal/ENTERPRISE_DELIVERY.md` for repository structure and IP isolation.**

---

## 11. Documentation Ownership

### 11.1 Core Documentation

**Core platform documentation:**

- **Vendor-owned** — Core documentation remains the property of Fox ML Infrastructure LLC
- **Client may use** — Clients may use documentation for internal purposes
- **No redistribution** — Clients may not redistribute documentation

### 11.2 Client-Specific Documentation

**Documentation created for a specific client:**

- **SOW determines ownership** — Ownership is determined by the SOW terms
- **Client-owned (if specified)** — If SOW specifies client ownership, documentation becomes client-owned
- **Vendor-owned (default)** — If not specified, documentation may remain vendor-owned

---

## 12. Summary

**Key Ownership Principles:**

1. **Core platform IP** → Fox ML Infrastructure LLC (perpetual ownership)
2. **Client additions** → Client-owned (configs, strategies, data)
3. **Custom features** → Ownership determined by SOW (default: vendor-owned, may be client-owned if specified)
4. **General patterns** → May be reused by Fox ML Infrastructure
5. **Client data/secrets** → Never reused, always client-owned
6. **No core IP transfer** → Core IP does not transfer under any standard agreement

**This structure protects both parties and prevents IP disputes.**

---

## Contact

For questions about IP ownership or custom IP terms:

**Jennifer Lewis**  
Fox ML Infrastructure LLC  
Email: **jenn.lewis5789@gmail.com**  
Subject: *IP Ownership Inquiry — Fox ML Infrastructure*

---

## Related Documents

- `legal/consulting/IP_TERMS_ADDENDUM.md` — Detailed IP terms for consulting engagements
- `legal/ENTERPRISE_DELIVERY.md` — Repository structure and IP isolation
- `COMMERCIAL_LICENSE.md` — Commercial license terms (Section 3: Ownership)
- `legal/consulting/STATEMENT_OF_WORK_TEMPLATE.md` — SOW template for custom IP terms

