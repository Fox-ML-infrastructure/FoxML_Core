# Enterprise Bundle — Procurement-Ready Documentation

This directory contains enterprise-ready documentation that can be printed, forwarded to internal teams, or converted to PDF for procurement processes.

---

## Contents

This bundle includes the following documents (available as Markdown, ready for PDF conversion):

### Core Legal Documents

- **Commercial License** — `../COMMERCIAL_LICENSE.md`
- **Commercial License Agreement (CLA)** — `../legal/CLA.md`
- **Terms of Service** — `../legal/TOS.md`

### Enterprise Documentation

- **Support Policy** — `../legal/SUPPORT_POLICY.md`
- **Service Level Agreement (SLA)** — `../legal/SERVICE_LEVEL_AGREEMENT.md`
- **Security Statement** — `../legal/SECURITY.md`
- **Data Processing Addendum (DPA)** — `../legal/DATA_PROCESSING_ADDENDUM.md`
- **Export Compliance** — `../legal/EXPORT_COMPLIANCE.md`
- **Acceptable Use Policy (AUP)** — `../legal/ACCEPTABLE_USE_POLICY.md`
- **Indemnification** — `../legal/INDEMNIFICATION.md`
- **Warranty & Liability Addendum** — `../legal/WARRANTY_LIABILITY_ADDENDUM.md`
- **IP Ownership Clarification** — `../legal/IP_OWNERSHIP_CLARIFICATION.md`
- **InfoSec Self-Assessment** — `../legal/INFOSEC_SELF_ASSESSMENT.md`

### Operational Documents

- **Release Policy** — `../legal/RELEASE_POLICY.md`
- **Enterprise Delivery Model** — `../legal/ENTERPRISE_DELIVERY.md`
- **Client Onboarding Guide** — `../legal/CLIENT_ONBOARDING.md`
- **Enterprise Checklist** — `../legal/ENTERPRISE_CHECKLIST.md`
- **Trademark Policy** — `../legal/TRADEMARK_POLICY.md`

### Pricing and Subscriptions

- **Subscriptions** — `../legal/SUBSCRIPTIONS.md`
- **Licensing Overview** — `../legal/LICENSING.md`

### Consulting Documents (If Applicable)

- **Master Consulting Agreement** — `../legal/consulting/MASTER_CONSULTING_AGREEMENT.md`
- **Statement of Work Template** — `../legal/consulting/STATEMENT_OF_WORK_TEMPLATE.md`
- **Consulting Pricing** — `../legal/consulting/CONSULTING_PRICING.md`
- **IP Terms Addendum** — `../legal/consulting/IP_TERMS_ADDENDUM.md`

---

## PDF Conversion

To convert Markdown files to PDF:

### Option 1: Using Pandoc

```bash
# Install pandoc (if not already installed)
# On macOS: brew install pandoc
# On Linux: sudo apt-get install pandoc

# Convert a single file
pandoc ../legal/SECURITY.md -o SECURITY.pdf

# Convert all files (example script)
for file in ../legal/*.md; do
    pandoc "$file" -o "$(basename "$file" .md).pdf"
done
```

### Option 2: Using Markdown to PDF Tools

- **Marked 2** (macOS) — https://marked2app.com/
- **Typora** — https://typora.io/
- **Online converters** — Various online Markdown to PDF converters

### Option 3: Using GitHub/GitLab

- View files on GitHub/GitLab
- Use browser print-to-PDF functionality
- Ensures consistent formatting

---

## Customization

Documents can be customized for specific clients:

- **Client-specific terms** — Modify terms in SOW or custom agreements
- **Pricing** — Include client-specific pricing in custom proposals
- **Support tiers** — Specify client's support tier and SLA terms
- **Custom features** — Include client-specific features or customizations

---

## Distribution

**Recommended distribution methods:**

1. **Email** — Send PDFs via email to procurement teams
2. **Secure portal** — Upload to client's secure procurement portal
3. **GitHub releases** — Create a GitHub release with PDF bundle
4. **Direct delivery** — Provide via secure file sharing service

---

## Version Control

**All documents are version-controlled:**

- **Git repository** — All source Markdown files are in the Git repository
- **Version tags** — Documents are tagged with version numbers
- **Change tracking** — All changes are tracked via Git history
- **Audit trail** — Complete audit trail of document changes

---

## Contact

For questions about the enterprise bundle or to request custom documentation:

**Jennifer Lewis**  
Fox ML Infrastructure LLC  
Email: **jenn.lewis5789@gmail.com**  
Subject: *Enterprise Bundle Inquiry — Fox ML Infrastructure*

---

## Related Documents

- `../legal/README.md` — Complete legal documentation index
- `../docs/INDEX.md` — Complete documentation navigation
- `../README.md` — Project overview and getting started

