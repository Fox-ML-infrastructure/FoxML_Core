# FOX ML INFRASTRUCTURE — COMMERCIAL LICENSE v1.0
Copyright (c) 2025-2026 Jennifer Lewis

> **Note:** This Commercial License is provided for informational purposes and may be supplemented or modified by a separate signed agreement. Licensor is not providing legal advice; Licensee should consult its own counsel before entering into this License.

---

**Who This License Is For**

This Commercial License is intended for trading firms, hedge funds, financial institutions, and other organizations that wish to use FoxML Core in production or revenue-generating environments without AGPL copyleft obligations.

## 1. DEFINITIONS

"Software" refers to the FoxML Core codebase, modules, configurations, models, documentation, and all included assets, excluding third-party components governed by their own licenses.

"Licensee" refers to the individual or organization obtaining rights under this License.

"Authorized Users" means employees and individual contractors of Licensee who are permitted by Licensee to use the Software on Licensee's behalf, up to any seat limits specified in the Ordering Documents.

"Ordering Documents" means one or more Order Forms or Statements of Work executed by the parties that specify fees, term length, payment schedule, seat limits, and other commercial terms.

"Affiliate" means any entity that directly or indirectly Controls, is Controlled by, or is under common Control with a party, where "Control" means ownership of more than fifty percent (50%) of the voting interests of such entity or the contractual power to direct its management and policies.

"Direct Competitor" means any third party whose primary business includes developing, licensing, or providing machine learning or quantitative trading infrastructure products or services that are substantially similar to the Software (for example, commercial platforms primarily marketed as quantitative research, ML pipeline, or trading infrastructure solutions).

"Major Version" means a release of the Software identified by a change in the first digit of the version number (e.g., 1.x to 2.0).

"Minor Version" means a release identified by a change in the second digit (e.g., 1.1 to 1.2) or subsequent maintenance or patch releases.

"Commercial Use" means any use of the Software by a business, organization, institution, or individual that:
  (a) generates revenue directly or indirectly,
  (b) contributes to a commercial product or service,
  (c) is deployed internally within a company for operational, analytical, modeling, forecasting, optimization, or research purposes,
  (d) is used to provide services to third parties,
  (e) enables, assists, or supports any for-profit activity.

**Business / Internal Use:** Any use of the Software within a company, organization, or other legal entity (including internal tools, proofs of concept, evaluations, pilot projects, or use by employees, contractors, interns, or affiliates) requires a commercial license, regardless of whether the use directly generates revenue. See `LEGAL/SUBSCRIPTIONS.md` for complete definitions.

Non-commercial academic or personal research use of the open-source version remains available under the AGPL-3.0 license, subject to the definitions in `LEGAL/SUBSCRIPTIONS.md`. Academic research conducted for, funded by, or operationally integrated into a commercial organization's activities requires a commercial license. This Commercial License is intended for organizations that wish to:
  (a) avoid AGPL copyleft obligations in commercial / internal systems, or
  (b) obtain additional rights or assurances not provided by AGPL-3.0.

Nothing in this Commercial License limits any rights granted under AGPL-3.0 for copies of the Software obtained and used under that license.

## 2. GRANT OF LICENSE

Subject to payment of applicable fees and execution of a commercial agreement, the Licensor grants the Licensee a non-exclusive, non-transferable, worldwide right to:

- Use the Software for Commercial Use
- Modify the Software for internal commercial use
- Deploy the Software internally without AGPL disclosure obligations
- Integrate the Software into proprietary systems, stacks, or workflows
- Use the Software solely for use by Authorized Users, up to the limits set forth in the applicable Ordering Documents

**Authorized Users and Seat Limits.**

Use of the Software is limited to the number and type of Authorized Users specified in the applicable Ordering Document ("Seats"). Licensee shall ensure that only Authorized Users use the Software and that such use does not exceed the purchased Seats.

If Licensee exceeds the purchased Seats, Licensor may (without limiting its other rights and remedies) invoice Licensee for the excess usage at Licensor's then-current rates, and Licensee shall pay such invoice within thirty (30) days.

**External Contractors.**

Authorized Users may include Licensee's employees and individual contractors (including individual consultants) who are engaged by Licensee and acting solely for Licensee's benefit ("Contractors"), provided that:

(a) each Contractor is bound by written obligations of confidentiality and use restrictions no less protective of Licensor than those set forth in this License;

(b) Licensee ensures that Contractors use the Software solely for Licensee's internal business purposes and not for the benefit of any other person or entity; and

(c) Licensee remains fully responsible and liable for Contractors' compliance with this License.

For the avoidance of doubt, a Contractor may not use a single instance of the Software, or the same credentials or license, to provide services to multiple clients unless each such client is itself a separately licensed customer of Licensor under its own agreement.

No sublicensing rights are granted unless explicitly permitted in a separate SOW.

**Open-Source Use Unaffected.**

Non-commercial use of the Software remains available under the AGPL-3.0 license. This Commercial License does not limit or restrict rights granted under AGPL-3.0 for qualifying non-commercial academic or personal research use.

## 3. OWNERSHIP

The Software is licensed, not sold. As between Licensor and Licensee, Licensor retains all right, title, and interest in and to the Software and all related intellectual property rights, whether registered or unregistered.

**Licensee Developments.** Licensee retains ownership of its own internal configurations, scripts, and integrations created to interface with the Software ("Licensee Developments"), provided that such items do not include source code from the Software other than code allowed to be modified under this License.

**Feedback License.** Licensee grants Licensor a non-exclusive, perpetual, irrevocable, worldwide, royalty-free license to use, modify, and incorporate any feedback, suggestions, error reports, or improvements provided by Licensee regarding the Software, without any obligation to Licensee and without giving Licensee ownership of any improvements or modifications made by Licensor.

## 4. RESTRICTIONS

Unless explicitly permitted in writing by the Licensor, the Licensee may NOT:

- Distribute or sell the Software or derivatives as a standalone product
- Provide the Software, or any substantially similar functionality, as a hosted service to third parties (including SaaS, PaaS, or multi-tenant environments) where external users access functionality primarily provided by the Software
- Build or market a product whose primary purpose is to offer FoxML-like ML/quant infrastructure as a general-purpose platform to third parties
- Use the Software to offer a product or service whose primary value is substantially identical to the Software itself (including "FoxML-as-a-service" or similar hosted offerings)

**No Competing Service.**

Licensee shall not use the Software to develop, host, or provide any software-as-a-service, managed service, platform, or other offering that exposes to third parties functionality that is substantially similar to the Software, including without limitation:

(a) a hosted quantitative research or model-training platform for third parties;

(b) a multi-tenant or shared service that allows third parties to run experiments, training, or feature selection workflows powered by the Software; or

(c) an API, SDK, or other interface that provides third parties with direct or indirect access to the Software's core infrastructure capabilities.

For clarity, Licensee may use the Software to build and operate its own internal trading, research, or analytics systems and to provide outputs (e.g., analytics, reports, trading decisions) to its own clients, provided such clients do not receive direct access to the Software itself.
- Publish or disclose source code derived from the commercial version
- Publish or disclose performance benchmarks or comparative tests involving the Software to any third party without Licensor's prior written consent (see Section 4.1 below)
- Remove or alter copyright notices
- Claim authorship of the Software or its architectural design
- Reverse engineer, decompile, disassemble, or otherwise attempt to derive or access the source code, underlying ideas, algorithms, file formats, or non-public APIs of the Software, or any trade secrets embodied therein, except to the extent expressly permitted by applicable law notwithstanding a contractual restriction
- Use the Software in violation of US export controls or applicable law
- Use the Software in high-risk environments including but not limited to: medical diagnosis, critical infrastructure, weapons systems, life support systems, nuclear systems, or any environment where failure could result in death, serious injury, or significant property damage

**4.1 Benchmarking.**

Licensee shall not publish or disclose to any third party any benchmark, performance, or comparison tests of the Software without Licensor's prior written consent.

Any breach of this Section 4.1 shall be deemed a material breach of this License and may cause irreparable harm to Licensor, for which monetary damages may be an insufficient remedy. In addition to any other rights and remedies, Licensor shall be entitled to seek injunctive or other equitable relief to prevent or curtail any actual or threatened breach of this Section 4.1.

A material breach of Section 4.1 (Benchmarking) shall be grounds for immediate termination by Licensor upon written notice, without a cure period.

## 5. TERM AND TERMINATION

This License is effective upon payment and execution of a commercial agreement.

The Licensor may terminate this License if:
- the Licensee violates any term of this License,
- the Licensee fails to pay required fees,
- the Licensee attempts unauthorized distribution or sublicensing,
- the Licensee materially breaches Section 4.1 (Benchmarking), which shall be grounds for immediate termination upon written notice without a cure period.

Upon termination of this Commercial License:
- all rights granted under this Commercial License revert to Licensor,
- Licensee must cease all Commercial Use and destroy or disable all copies of the Software used under this Commercial License,
- any continued Commercial Use without a valid license will constitute unlicensed use and violation of Licensor's copyright.

Termination of this Commercial License does not terminate any rights Licensee may have under AGPL-3.0 for copies of the Software obtained and used solely under that license.

## 6. NO WARRANTY / LIMITATION OF LIABILITY

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.

THE LICENSOR IS NOT LIABLE FOR ANY DAMAGES, INCLUDING BUT NOT LIMITED TO:
LOSS OF PROFITS, MODELING ERRORS, DATA LOSS, BUSINESS INTERRUPTION,
OR ANY CONSEQUENTIAL DAMAGES ARISING FROM USE OF THE SOFTWARE.

**Liability Cap:** IN NO EVENT WILL LICENSOR'S TOTAL AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS LICENSE EXCEED THE FEES PAID BY LICENSEE TO LICENSOR FOR THE SOFTWARE IN THE TWELVE (12) MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM.

## 7. INDEMNITY

Licensee agrees to indemnify and hold harmless Licensor from and against any third-party claims, damages, or expenses (including reasonable attorneys' fees) arising out of Licensee's misuse of the Software, violation of this License, or violation of applicable laws.

## 8. SUPPORT AND MAINTENANCE

Support, integration, customization, improvements, or feature development require separate consulting agreements or Statements of Work (SOWs).

Unless explicitly set out in a separate written agreement:
- Support is NOT included in this License.
- No uptime, response time, or defect resolution Service Level Agreements (SLAs) are provided.

## 9. UPDATES AND UPGRADES

**Version Definition.** "Version" refers to a release of the Software with a distinct version number (e.g., v1.0, v1.1, v2.0).

**Updates and Upgrades.** Unless explicitly included in a separate agreement:
- Minor updates, patches, and hotfixes are NOT automatically included
- Major version upgrades are NOT automatically included
- New modules, features, or components are NOT automatically included
- Licensee must purchase separate licenses or upgrade agreements for new versions

**No Obligation.** Licensor has no obligation to provide updates, upgrades, or new versions of the Software.

## 10. FEES

Commercial licensing fees are based on Licensee's organization size, usage tier, and selected add-ons.

Specific fees, term length, and payment schedule will be set forth in one or more **Order Forms** or **Statements of Work** executed by the parties (collectively, "Ordering Documents"). If there is any conflict between this License and an Ordering Document regarding fees, the Ordering Document will control.

Standard pricing tiers are published in `LEGAL/SUBSCRIPTIONS.md` for reference, but actual fees are determined by the Ordering Documents.

## 11. AUDIT RIGHTS

Licensor reserves the right to audit Licensee's use of the Software to verify compliance with this License.

**Audit Terms:**
- Licensor may conduct no more than one audit per 12-month period
- Audits shall be conducted during normal business hours
- Licensor must provide at least 30 days' written notice
- Audit scope limited to verification of Software usage and compliance with license restrictions
- Remote audit methods are permitted (e.g., usage logs, deployment verification)
- If an audit reveals that Licensee underpaid fees by more than five percent (5%) for the audited period or materially violated the license restrictions, Licensee shall reimburse Licensor's reasonable audit costs in addition to paying any unpaid fees
- Licensee must provide reasonable cooperation and access to relevant records

**Confidentiality:** All audit information shall be treated as confidential and used solely for compliance verification.

## 12. EXPORT CONTROL COMPLIANCE

Licensee acknowledges that the Software may be subject to export control laws and regulations, including but not limited to:
- United States Export Administration Regulations (EAR)
- European Union export control regulations
- United Kingdom export control regulations

Licensee agrees to comply with all applicable export control laws and regulations. Licensee represents and warrants that:
- Licensee is not located in, under the control of, or a national or resident of any country subject to comprehensive sanctions
- Licensee will not export, re-export, or transfer the Software to any prohibited destination or end-user
- Licensee will not use the Software for any purpose prohibited by applicable export control laws

## 13. INDEPENDENT DEVELOPMENT

Licensor may continue to develop, market, and provide software, products, or services that are similar to or competitive with the Software, regardless of any suggestions, feedback, or feature requests provided by Licensee. This License does not restrict Licensor's right to develop or provide such products or services independently.

## 14. PUBLICITY

Unless Licensee objects in writing, Licensor may use Licensee's name and logo in customary customer lists and marketing materials to identify Licensee as a customer of the Software. Any other publicity will require Licensee's prior written consent.

## 15. USE BY AFFILIATES

Subject to the terms of this License and the applicable Ordering Documents, Licensee may permit its Affiliates to use the Software solely for the benefit of Licensee, provided that:

(a) each such Affiliate is expressly identified in an applicable Ordering Document or other written agreement with Licensor;

(b) Licensee remains fully responsible and liable for such Affiliates' compliance with this License; and

(c) use by any Affiliate constitutes use by Licensee for purposes of any user, seat, environment, or usage limitations.

For the avoidance of doubt, each separate legal entity (including each Affiliate) that wishes to use the Software for its own benefit must obtain its own license (typically via a separate Ordering Document), unless expressly stated otherwise in writing by Licensor.

## 16. ASSIGNMENT

**Licensee Assignment:**
Licensee may not assign this License, by operation of law or otherwise, without Licensor's prior written consent, except that Licensee may assign this License without consent (i) to a successor in interest in connection with a merger, acquisition, corporate reorganization, or sale of all or substantially all of Licensee's assets, or (ii) to an Affiliate that assumes all of Licensee's obligations under this License, provided that in each case the assignee is not a Direct Competitor (as defined in Section 1) and Licensee provides Licensor with written notice of the assignment.

**Licensor Assignment:**
Licensor may freely assign this License, including in connection with a merger, acquisition, or sale of all or substantially all of Licensor's assets. Such assignment shall not affect Licensee's rights under this License.

Any attempted assignment in violation of this section is void.

## 17. DISPUTE RESOLUTION

**Informal Resolution:**
The parties agree to attempt to resolve disputes through good faith negotiation for at least 30 days before initiating formal proceedings.

**Arbitration for Small Claims:**
For claims under $250,000, either party may elect binding arbitration under the Commercial Arbitration Rules of the American Arbitration Association (AAA). Arbitration shall be conducted in Delaware by a single arbitrator with expertise in software licensing disputes.

**Litigation for Large Claims:**
For claims of $250,000 or more, disputes shall be resolved in the state or federal courts located in Delaware. Both parties consent to the exclusive jurisdiction and venue of such courts.

**Class Action Waiver:**
Both parties waive any right to participate in class actions, collective actions, or representative proceedings.

## 18. GOVERNING LAW

This License is governed by the laws of the State of Delaware, United States, without regard to conflicts of law principles. The United Nations Convention on Contracts for the International Sale of Goods does not apply.

## 19. FORCE MAJEURE

Neither party shall be liable for any failure or delay in performance under this License due to circumstances beyond its reasonable control, including but not limited to: acts of God, war, terrorism, riots, embargoes, acts of civil or military authorities, fire, floods, accidents, network or infrastructure failures, strikes, or shortages of transportation facilities, fuel, energy, labor, or materials.

## 20. DATA HANDLING AND CONFIDENTIALITY

**Licensee Confidentiality Obligations:**
Licensee agrees to maintain the confidentiality of:
- Source code and internal architecture of the Software
- Proprietary algorithms and techniques
- Documentation marked as confidential
- Pricing and licensing terms
- Any other information designated as confidential by Licensor

**Security Requirements:**
Licensee must implement reasonable technical and administrative safeguards to protect the Software and prevent unauthorized access, use, or disclosure.

**Breach Notification:**
Licensee must notify Licensor immediately upon discovery of any unauthorized use, disclosure, or breach of security related to the Software.

## 21. SEVERABILITY

If any provision of this License is held to be invalid, illegal, or unenforceable by a court of competent jurisdiction, the remaining provisions shall remain in full force and effect. The invalid, illegal, or unenforceable provision shall be replaced with a valid, legal, and enforceable provision that comes closest to the intent of the original provision.

## 22. ENTIRE AGREEMENT

This License, together with any separate signed commercial agreement or Statement of Work, constitutes the entire agreement between the parties regarding the subject matter hereof and supersedes all prior or contemporaneous agreements, understandings, negotiations, and discussions, whether oral or written, relating to the Software. No modification, amendment, or waiver of any provision of this License shall be effective unless in writing and signed by both parties.

## 23. SURVIVAL

Sections 4 (Restrictions), 6 (No Warranty / Limitation of Liability), 7 (Indemnity), 11 (Audit Rights), 12 (Export Control Compliance), 16 (Dispute Resolution), 17 (Governing Law), 19 (Data Handling and Confidentiality), 20 (Severability), 21 (Entire Agreement), and this Section 22 (Survival) will survive termination or expiration of this License.

## 24. CONTACT

For commercial licensing, enterprise usage, or SOW negotiation, contact:

**Jennifer Lewis**  
Fox ML Infrastructure  
Email: **jenn.lewis5789@gmail.com**  
Subject: *Fox Infrastructure Licensing Inquiry*

