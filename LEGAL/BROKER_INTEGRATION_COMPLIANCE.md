# Broker Integration Compliance

**Fox ML Infrastructure LLC — Broker Integration Legal Framework**

---

## Executive Summary

FoxML Core provides **client-side trading execution software** that connects to **user-owned brokerage accounts** via **user-provided API keys**. This document establishes the legal framework for broker integrations (IBKR, Alpaca, and future brokers).

**Key Principle**: We sell **software infrastructure**, not brokerage services, investment advice, or custodial services.

---

## 1. Legal Model

### What We Provide

FoxML Core is a **user-authorized API integration software platform** that:

- Connects to user-owned brokerage accounts via user-provided API credentials
- Executes trades **on behalf of the user's account** based on **user-defined strategies**
- Provides execution infrastructure, risk controls, and safety rails
- Offers research engines, backtesting, and signal generation tools

### What We Do NOT Provide

- **Brokerage services** — We are not a broker, do not execute trades in our own name, and do not hold customer funds
- **Investment advice** — We do not provide buy/sell recommendations, portfolio management, or financial planning
- **Custodial services** — We do not hold, control, or manage customer funds or securities
- **Data reselling** — We do not resell broker market data or present ourselves as a data provider
- **Regulatory registration** — We are not registered as a broker-dealer, investment advisor, or similar entity

---

## 2. User Responsibilities

### Brokerage Relationship

**Users are solely responsible for:**

- Establishing and maintaining their own brokerage relationship with IBKR, Alpaca, or other brokers
- Providing their own API credentials and maintaining API key security
- Ensuring their brokerage account is properly configured and authorized for API trading
- Complying with their broker's terms of service and API usage policies

### Regulatory Compliance

**Users are solely responsible for:**

- **SEC compliance** — Compliance with Securities and Exchange Commission regulations
- **CFTC compliance** — Compliance with Commodity Futures Trading Commission regulations
- **State regulations** — Compliance with state securities and financial regulations
- **Exchange rules** — Compliance with exchange rules and trading regulations
- **Professional licensing** — Obtaining and maintaining any required professional licenses (if applicable)
- **Regulatory filings** — Making any required regulatory filings or disclosures

### Trading Decisions

**Users are solely responsible for:**

- All trading and investment decisions
- Strategy selection and configuration
- Risk management and position sizing
- Monitoring and oversight of automated trading systems
- Verifying the suitability of strategies for their risk profile and regulatory environment

---

## 3. Broker-Specific Compliance

### Interactive Brokers (IBKR)

#### ✅ Allowed Activities

- Connecting via IBKR TWS / Gateway API using user-provided credentials
- Placing orders on behalf of the user's account
- Reading positions, balances, and fills from the user's account
- Charging for software, infrastructure, execution logic, risk controls, and strategy engines

#### ❌ Prohibited Activities

- Reselling IBKR market data
- Branding as "powered by IBKR" without explicit permission
- Running pooled accounts unless properly registered
- Providing discretionary investment advice unless licensed

#### Key Points

- IBKR explicitly supports third-party trading software
- This integration model is standard in prop shops, funds, and professional quant stacks
- Users must comply with IBKR's API terms of service

### Alpaca Markets

#### ✅ Allowed Activities

- API-based trading using user-provided API keys
- White-label / third-party tools connecting to user accounts
- SaaS platforms connecting user accounts
- Charging for software and infrastructure

#### ❌ Prohibited Activities

- Reselling Alpaca data feeds
- Acting as the broker without explicit agreement
- Misrepresenting regulatory status

#### Key Points

- Alpaca is designed specifically for third-party trading software integration
- This is the intended use case for Alpaca's API
- Users must comply with Alpaca's API terms of service

---

## 4. Product Structure

### Architecture Model

FoxML Core is structured as:

```
FoxML Core (Software Product)
├── Research Engine (offline / backtest)
├── Signal Generation (user-defined strategies)
├── Risk & Compliance Layer (user-configurable)
├── Execution Adapter
│   ├── IBKR Adapter (user-provided API keys)
│   ├── Alpaca Adapter (user-provided API keys)
│   └── (future broker adapters)
└── User Account (API keys provided by user)
```

### What We Sell

- **The execution engine** — Reliable, auditable order execution
- **The infrastructure** — Scalable, production-grade trading infrastructure
- **The reliability** — Safety rails, error handling, disaster recovery
- **The auditability** — Complete audit trails and compliance logging
- **The safety rails** — Risk controls, position limits, kill switches

### What We Do NOT Sell

- Trading advice or recommendations
- Guaranteed returns or performance
- Brokerage services
- Investment management

---

## 5. Language Requirements

### Required Disclaimers

All broker integration documentation, code comments, and user-facing materials must use language that:

- **Emphasizes user control**: "User configures strategies", "User executes at their discretion", "System automates user-defined logic"
- **Avoids advisory language**: Do NOT use "recommends", "suggests", "advises", "guarantees returns"
- **Clarifies non-custodial nature**: "User-owned accounts", "User-provided API keys", "Non-custodial execution"

### Prohibited Language

**DO NOT:**

- Provide "buy/sell recommendations" framed as advice
- Market guaranteed returns or performance
- Suggest the system trades without user authorization
- Imply we control customer funds directly
- Claim regulatory approval or endorsement

### Approved Language Examples

✅ **CORRECT:**
- "User configures trading strategies"
- "System executes user-defined logic"
- "User authorizes trades via API credentials"
- "Non-custodial execution on user-owned accounts"

❌ **INCORRECT:**
- "System recommends trades"
- "Guaranteed returns"
- "Automated trading without oversight"
- "We manage your account"

---

## 6. Commercial Licensing

### License Structure

Commercial licenses explicitly state:

- **User is responsible for:**
  - Brokerage relationship
  - Regulatory compliance
  - API credentials and security
  - Trading decisions and oversight

- **Software is:**
  - Non-custodial
  - Non-advisory
  - Execution-assistive only
  - User-authorized API integration software

### Pricing Model

We charge for:

- Software licenses
- Infrastructure and hosting (if applicable)
- Support and maintenance
- Custom development (enterprise)

We do NOT charge for:

- Trading commissions (handled by broker)
- Investment management fees
- Performance-based fees (unless explicitly structured as software licensing)

---

## 7. Technical Implementation Requirements

### API Key Management

- **User-provided credentials** — All API keys must be provided by the user
- **No credential storage** — We do not store or manage user API keys (user responsibility)
- **Secure handling** — API keys must be handled securely in transit and at rest
- **Access control** — Users control API key permissions and scope

### Execution Model

- **User authorization required** — All trades require explicit user authorization
- **User-defined strategies** — Strategies are configured by the user, not recommended by us
- **Audit trails** — Complete audit trails of all execution decisions
- **User oversight** — Systems must support user monitoring and intervention

### Risk Controls

- **User-configurable** — Risk limits are user-defined, not system-recommended
- **Transparent** — All risk controls are visible and auditable
- **User responsibility** — Users are responsible for setting appropriate risk limits

---

## 8. Documentation Requirements

### Required Disclaimers in Documentation

All broker integration documentation must include:

1. **Non-advisory disclaimer** — Clear statement that we do not provide investment advice
2. **User responsibility** — Explicit statement of user responsibilities for brokerage, compliance, and trading decisions
3. **Non-custodial statement** — Clear statement that we do not hold or control customer funds
4. **Regulatory compliance** — Statement that users are responsible for regulatory compliance
5. **Risk warnings** — Trading involves substantial risk of loss

### Documentation Locations

These disclaimers must appear in:

- Module README files (`ALPACA_trading/README.md`, `IBKR_trading/README.md`)
- Broker adapter code documentation
- User-facing documentation and tutorials
- Commercial license agreements
- Terms of Service

---

## 9. Compliance Verification

### Checklist

Before deploying broker integrations, verify:

- [ ] All documentation includes required disclaimers
- [ ] Code comments use approved language (no advisory language)
- [ ] User-facing materials emphasize user control and responsibility
- [ ] Commercial licenses explicitly state non-custodial, non-advisory nature
- [ ] API key handling is secure and user-controlled
- [ ] Execution model requires user authorization
- [ ] Risk controls are user-configurable
- [ ] Audit trails are complete and accessible

### Regular Review

- Review broker integration compliance quarterly
- Update disclaimers as regulations change
- Verify language remains compliant
- Ensure new features maintain compliance

---

## 10. Industry Context

### This is Standard Practice

This integration model is:

- **Normal** — Standard practice in professional trading systems
- **Industry-standard** — Used by prop shops, funds, and quant firms
- **Broker-supported** — Explicitly supported by IBKR and Alpaca
- **Regulatory-compliant** — When properly structured and documented

### Competitive Advantage

FoxML Core's compliance framework provides:

- **Clear legal structure** — Explicit non-advisory, non-custodial model
- **Professional documentation** — Comprehensive compliance documentation
- **Enterprise-ready** — Suitable for institutional use
- **Risk mitigation** — Reduces regulatory and legal risk

---

## 11. Contact

**For questions about broker integration compliance:**

**Jennifer Lewis**  
Fox ML Infrastructure LLC  
Email: **jenn.lewis5789@gmail.com**  
Subject: *Broker Integration Compliance Inquiry*

**Note:** We cannot provide legal or regulatory advice. Please consult your own legal counsel for compliance questions.

---

## Related Documents

- [`REGULATORY_DISCLAIMERS.md`](REGULATORY_DISCLAIMERS.md) — General regulatory disclaimers
- [`TOS.md`](TOS.md) — Terms of Service (Section 8: No Financial Advice)
- [`COMMERCIAL_LICENSE.md`](../COMMERCIAL_LICENSE.md) — Commercial license terms
- [`WARRANTY_LIABILITY_ADDENDUM.md`](WARRANTY_LIABILITY_ADDENDUM.md) — Liability limitations
- [`INDEMNIFICATION.md`](INDEMNIFICATION.md) — Indemnification obligations

---

**Copyright (c) 2025-2026 Fox ML Infrastructure LLC**

This document is part of the FoxML Core legal documentation. See [`LEGAL/README.md`](README.md) for complete legal documentation index.
