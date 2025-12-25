# Fox ML Infrastructure — Security Statement

> **Document Hierarchy:** This document is provided for guidance only. In case of any conflict, `COMMERCIAL_LICENSE.md` is the authoritative and controlling document for all commercial licensing terms and obligations, and `LICENSE` controls for free use.

> **Commercial Licensees Only:** This document applies only to customers with a valid commercial license. free users are not eligible for the policies, services, or terms described here.

> **No Warranties:** Statements in this document are provided on a best-effort basis. No warranties, express or implied, are provided. This document does not grant any license rights. All rights and restrictions are defined solely in `COMMERCIAL_LICENSE.md`.

This document outlines the security practices and data handling policies for Fox ML Infrastructure.  
This statement is designed for enterprise buyers, legal departments, and security teams evaluating the platform.

---

## 1. Client-Hosted Architecture

**Fox ML Infrastructure is client-hosted software.**  
The platform runs entirely within the client's infrastructure and environment.

**Implications:**
- **No vendor-run runtime** — Fox ML Infrastructure does not operate client systems
- **No vendor-caused downtime** — Since there is no vendor-hosted service, there is no vendor-caused service interruption
- **Client controls all infrastructure** — Clients manage their own hosting, networking, and deployment
- **No external dependencies** — The platform operates independently within the client's environment

---

## 2. Data Handling & Privacy

### 2.0 Data Handling Commitment

**Fox ML Infrastructure's policy: we do not sell user data.**

This policy reflects our current practices and applies to:
- All software usage data (which we do not collect)
- All licensing and support communications
- Any information collected through website interactions

While we intend to maintain this policy, this statement is provided for informational purposes only and does not create any binding obligation, warranty, or guarantee. This is a core privacy principle that we follow to protect our clients' confidentiality and trust.

### 2.1 No Data Collection

**Fox ML Infrastructure does not collect client data.**

- No telemetry or usage analytics
- No outbound calls to external services
- No data transmission to vendor systems
- No client data stored on vendor infrastructure

### 2.2 Client Data Control

**All client data remains fully client-controlled.**

- Data is stored only in client-approved environments
- Data access is limited to client-defined permissions
- No vendor access to production systems or keys
- Client data is never transferred to vendor systems


---

## 3. Code Delivery & Supply Chain

### 3.1 Private Repository Access

**Commercial licenses include access to private GitHub repositories.**

- Code is delivered via private, client-specific repositories
- Access is controlled via GitHub organization or repository-level permissions
- Code is version-tagged and auditable
- No public exposure of client-specific customizations

### 3.2 Supply Chain Integrity

**Fox ML Infrastructure maintains supply chain integrity.**

- **No telemetry** — The platform does not make outbound calls or collect usage data
- **No external dependencies at runtime** — All dependencies are explicit and auditable
- **Source-available transparency** — Core platform is source-available, enabling full code review
- **Version control** — All code is version-controlled and tagged
- **Auditable builds** — Build processes are documented and reproducible

### 3.3 Dependency Management

- Dependencies are explicitly declared and versioned
- Security updates are provided via patch releases
- Clients can audit all dependencies before deployment
- No hidden or obfuscated code

---

## 4. Access Control & Credentials

### 4.1 No Production Access

**Fox ML Infrastructure does not access client production systems.**

- No vendor access to production environments
- No vendor access to production API keys or credentials
- No vendor access to trading accounts or financial systems
- All deployment and operations are client-controlled


---

## 5. Security Practices

### 5.1 Code Security

- **Regular security reviews** — Code is reviewed for security best practices
- **Dependency updates** — Security patches are applied and released via patch versions
- **No hardcoded secrets** — All secrets are externalized to configuration
- **Secure defaults** — Platform ships with secure default configurations

### 5.2 Deployment Security

- **Client-controlled deployment** — Clients control all deployment processes
- **No vendor deployment access** — Vendor does not deploy to client systems
- **Documented security practices** — Deployment security is documented in client onboarding materials

---

## 6. Compliance & Auditing

### 6.1 Auditability

- **Version control** — All code changes are version-controlled and auditable
- **Tagged releases** — All releases are tagged with semantic versioning
- **Change logs** — Enterprise changelog documents all changes (see `CHANGELOG_ENTERPRISE.md`)
- **Documentation** — Security practices are documented and available

### 6.2 Compliance Support

- **NDA support** — Non-Disclosure Agreements are standard for consulting engagements
- **Data handling policies** — Explicit data handling policies are documented
- **Client-specific compliance** — Client-specific compliance requirements can be addressed in Statements of Work

---

## 7. Incident Response

### 7.1 Security Issues

If a security issue is discovered:

- **Immediate notification** — Enterprise and Premium support customers are notified immediately
- **Patch releases** — Security patches are released as patch versions (e.g., `v1.2.0` → `v1.2.1`)
- **Documentation** — Security issues and fixes are documented in release notes

### 7.2 Reporting Security Issues

To report a security issue:

**Email:** jenn.lewis5789@gmail.com  
**Subject:** Security Issue — Fox ML Infrastructure

Please include:
- Description of the issue
- Steps to reproduce (if applicable)
- Potential impact assessment
- Your contact information

---

## 8. Third-Party Services

**Fox ML Infrastructure does not integrate with third-party vendor services.**

- No external API calls (except as configured by the client for data sources)
- No vendor-hosted services
- No third-party analytics or telemetry
- All functionality is self-contained within the client's environment

---

## 9. Financial Services & Regulated Industry Considerations

**Additional security considerations for financial services and regulated industries:**

- **No data exfiltration** — Platform does not transmit data outside client environment
- **Auditable codebase** — Full source code is available for security review
- **No vendor access** — Vendor has no access to trading strategies or proprietary algorithms
- **Client-controlled secrets** — All API keys, credentials, and secrets are client-controlled
- **Isolated deployments** — Client-specific repositories keep proprietary code isolated

---

## 10. Summary

**Key Security Principles:**

1. **Client-hosted** — No vendor-run services, no vendor-caused downtime
2. **No data collection** — No telemetry, no outbound calls, no data transmission
3. **Client-controlled** — All data, credentials, and infrastructure are client-controlled
4. **Private delivery** — Code delivered via private repositories
5. **Supply chain integrity** — No hidden dependencies, fully auditable
6. **No production access** — Vendor does not access client production systems

**This architecture is designed to operate as a client-controlled platform with no vendor access to client data or systems. This statement is provided for informational purposes only and does not create any warranty or guarantee.**

---

## Contact

For security-related questions or to report security issues:

**Jennifer Lewis**  
Fox ML Infrastructure LLC  
Email: **jenn.lewis5789@gmail.com**  
Subject: *Security Inquiry — Fox ML Infrastructure*

---

## Related Documents

- `LEGAL/ENTERPRISE_DELIVERY.md` — Repository structure and delivery model
- `LEGAL/SUPPORT_POLICY.md` — Support tiers and response times
- `LEGAL/SERVICE_LEVEL_AGREEMENT.md` — SLA terms for Enterprise support

