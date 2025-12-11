# Enterprise Bundle — PDF Index

This directory contains PDF versions of all enterprise-ready legal and operational documents for Fox ML Infrastructure.

**Note:** This index is automatically maintained. All PDFs are generated from Markdown source files using `SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py`.

---

## Core Legal Documents

1. **COMMERCIAL_LICENSE.pdf** — Commercial License terms and conditions
2. **DUAL_LICENSE.pdf** — Dual license overview (AGPL-3.0 and Commercial)
3. **LICENSE.pdf** — GNU Affero General Public License v3.0 (AGPL-3.0) full text
4. **CLA.pdf** — Contributor License Agreement (CLA) for organizational use
5. **LICENSING.pdf** — Complete licensing documentation
6. **LICENSE_ENFORCEMENT.pdf** — License enforcement procedures
7. **COPYRIGHT_NOTICE.pdf** — Copyright information

---

## Enterprise Documentation

### Security & Compliance

8. **SECURITY.pdf** — Public-facing security statement
9. **DATA_PROCESSING_ADDENDUM.pdf** — Data Processing Addendum (DPA)
10. **INFOSEC_SELF_ASSESSMENT.pdf** — Information security self-assessment
11. **SECURITY_CONTROLS_MATRIX.pdf** — Security controls matrix (access control, encryption, logging, secrets)
12. **EXPORT_COMPLIANCE.pdf** — Export compliance statement (EAR99)
13. **PRIVACY_POLICY.pdf** — Privacy Policy (public-facing)
14. **DATA_RETENTION_DELETION_POLICY.pdf** — Data retention and deletion policy
15. **INCIDENT_RESPONSE_PLAN.pdf** — Incident Response Plan (IRP)
16. **BUSINESS_CONTINUITY_PLAN.pdf** — Business Continuity Plan (BCP)
17. **RISK_ASSESSMENT_MATRIX.pdf** — Risk assessment matrix
18. **PENETRATION_TESTING_STATEMENT.pdf** — Penetration testing statement

### Legal & Risk Management

19. **INDEMNIFICATION.pdf** — Indemnification clause
20. **WARRANTY_LIABILITY_ADDENDUM.pdf** — Warranty & liability addendum
21. **ACCEPTABLE_USE_POLICY.pdf** — Acceptable Use Policy (AUP)
22. **IP_OWNERSHIP_CLARIFICATION.pdf** — IP ownership clarification
23. **TOS.pdf** — Terms of Service

### Support & Operations

24. **SUPPORT_POLICY.pdf** — Support tiers (Standard, Business, Enterprise, Premium)
25. **SERVICE_LEVEL_AGREEMENT.pdf** — SLA terms for Enterprise support
26. **RELEASE_POLICY.pdf** — Versioning strategy and release cadence
27. **RELEASE_NOTES_TAGGING_STANDARD.pdf** — Release notes and tagging standard

### Delivery & Onboarding

28. **ENTERPRISE_DELIVERY.pdf** — Enterprise delivery model
29. **CLIENT_ONBOARDING.pdf** — Client onboarding guide
30. **ENTERPRISE_CHECKLIST.pdf** — Enterprise readiness checklist
31. **PRODUCTION_USE_NOTIFICATION.pdf** — Production use notification requirements

### Policies & Standards

32. **TRADEMARK_POLICY.pdf** — Trademark and branding policy
33. **COMMERCIAL_USE.pdf** — Commercial use policy and licensing overview

### Architecture & Technical

34. **SYSTEM_ARCHITECTURE_DIAGRAM.pdf** — System architecture diagram

### Reference & FAQ

35. **FAQ.pdf** — Frequently asked questions
36. **COMPLIANCE_FAQ.pdf** — Frequently asked compliance questions
37. **CREDITS.pdf** — Credits and acknowledgments
38. **DECISION_MATRIX.pdf** — Decision matrix for licensing and usage

---


---

## Total: 53 PDF Documents

All documents are ready for enterprise procurement, legal review, and compliance audits.

---

## Usage

These PDFs can be:
- **Printed** for physical documentation
- **Forwarded** to internal teams (legal, procurement, security)
- **Uploaded** to procurement portals or secure file sharing services
- **Archived** for compliance and audit purposes

---

## Source Files

All PDFs are generated from Markdown source files in:
- `../COMMERCIAL_LICENSE.md` (root)
- `../DUAL_LICENSE.md` (root)
- `../LICENSE` (root - AGPL-3.0 text)
- `../LEGAL/*.md` (legal directory, including subdirectories)

Source files are version-controlled in the Git repository.

---

## Regenerating PDFs

PDFs are automatically generated from source Markdown files using the regeneration script:

```bash
# From repository root
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py

# Clean and regenerate (removes old PDFs first)
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --clean

# Verbose output
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --verbose

# Specify LaTeX engine
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --engine pdflatex
```

The script automatically:
- Discovers all `.md` files in the `LEGAL/` directory (including subdirectories)
- Includes root-level legal documents (COMMERCIAL_LICENSE.md, DUAL_LICENSE.md, LICENSE)
- Handles Unicode characters (emojis, special symbols) by converting them to ASCII equivalents
- Generates PDFs with consistent formatting

**Note:** The script excludes `README.md` and `CHANGELOG_ENTERPRISE.md` from PDF generation as these are index/changelog files, not standalone documents.

---

## Contact

For questions about the enterprise bundle:

**Jennifer Lewis**  
Fox ML Infrastructure LLC  
Email: **jenn.lewis5789@gmail.com**  
Subject: *Enterprise Bundle Inquiry — Fox ML Infrastructure*

---

## Document Categories Summary

- **Core Legal**: 7 documents (licensing, copyright, CLA)
- **Security & Compliance**: 11 documents
- **Legal & Risk Management**: 5 documents
- **Support & Operations**: 4 documents
- **Delivery & Onboarding**: 4 documents
- **Policies & Standards**: 2 documents
- **Architecture & Technical**: 1 document
- **Reference & FAQ**: 4 documents

---

*Last updated: Automatically maintained — run regeneration script to update PDFs from source files.*
