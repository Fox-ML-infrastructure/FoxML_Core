# Enterprise Bundle — PDF Index

This directory contains PDF versions of all enterprise-ready legal and operational documents for Fox ML Infrastructure.

---

## Core Legal Documents

1. **COMMERCIAL_LICENSE.pdf** — Commercial License terms and conditions
2. **CLA.pdf** — Commercial License Agreement (CLA) for organizational use

---

## Enterprise Documentation

### Security & Compliance

3. **SECURITY.pdf** — Public-facing security statement
4. **DATA_PROCESSING_ADDENDUM.pdf** — Data Processing Addendum (DPA)
5. **INFOSEC_SELF_ASSESSMENT.pdf** — Information security self-assessment
6. **EXPORT_COMPLIANCE.pdf** — Export compliance statement (EAR99)
7. **PRIVACY_POLICY.pdf** — Privacy Policy (public-facing)
8. **DATA_RETENTION_DELETION_POLICY.pdf** — Data retention and deletion policy
9. **INCIDENT_RESPONSE_PLAN.pdf** — Incident Response Plan (IRP)
10. **BUSINESS_CONTINUITY_PLAN.pdf** — Business Continuity Plan (BCP)
11. **RISK_ASSESSMENT_MATRIX.pdf** — Risk assessment matrix

### Legal & Risk Management

7. **INDEMNIFICATION.pdf** — Indemnification clause
8. **WARRANTY_LIABILITY_ADDENDUM.pdf** — Warranty & liability addendum
9. **ACCEPTABLE_USE_POLICY.pdf** — Acceptable Use Policy (AUP)
10. **IP_OWNERSHIP_CLARIFICATION.pdf** — IP ownership clarification

### Support & Operations

11. **SUPPORT_POLICY.pdf** — Support tiers (Standard, Business, Enterprise, Premium)
12. **SERVICE_LEVEL_AGREEMENT.pdf** — SLA terms for Enterprise support
13. **RELEASE_POLICY.pdf** — Versioning strategy and release cadence

### Delivery & Onboarding

14. **ENTERPRISE_DELIVERY.pdf** — Enterprise delivery model
15. **CLIENT_ONBOARDING.pdf** — Client onboarding guide
16. **ENTERPRISE_CHECKLIST.pdf** — Enterprise readiness checklist
17. **TRADEMARK_POLICY.pdf** — Trademark and branding policy

### Release & Standards

18. **RELEASE_NOTES_TAGGING_STANDARD.pdf** — Release notes and tagging standard

---

## Total: 23 PDF Documents

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
- `../legal/*.md` (legal directory)

Source files are version-controlled in the Git repository.

---

## Updates

PDFs are generated from source Markdown files. To regenerate:

```bash
cd enterprise_bundle
# Regenerate all PDFs
for file in ../legal/*.md; do
    pandoc "$file" -o "$(basename "$file" .md).pdf" \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V fontsize=11pt
done
```

---

## Contact

For questions about the enterprise bundle:

**Jennifer Lewis**  
Fox ML Infrastructure LLC  
Email: **jenn.lewis5789@gmail.com**  
Subject: *Enterprise Bundle Inquiry — Fox ML Infrastructure*

