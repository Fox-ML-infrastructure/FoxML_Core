# Broker Integration Compliance Checklist

**Quick Reference Checklist for Broker Integration Compliance**

This checklist ensures all broker integration code, documentation, and user-facing materials comply with the legal framework established in [`BROKER_INTEGRATION_COMPLIANCE.md`](BROKER_INTEGRATION_COMPLIANCE.md).

---

## Code Compliance

### Broker Interface Code

- [ ] Broker interface includes compliance notice in module docstring
- [ ] Code comments use approved language (no advisory language)
- [ ] API key handling is user-provided (no credential storage)
- [ ] Execution model requires user authorization
- [ ] Risk controls are user-configurable (not system-recommended)
- [ ] Audit trails are complete and accessible

### Language in Code

- [ ] Uses "user configures", "user authorizes", "user-defined"
- [ ] Avoids "recommends", "suggests", "advises", "guarantees"
- [ ] Emphasizes non-custodial nature
- [ ] Clarifies user ownership of accounts

---

## Documentation Compliance

### Module READMEs

- [ ] `ALPACA_trading/README.md` includes compliance section
- [ ] `IBKR_trading/README.md` includes compliance section
- [ ] Both include required disclaimers:
  - Non-advisory disclaimer
  - User responsibility statement
  - Non-custodial statement
  - Regulatory compliance statement
  - Risk warnings

### Technical Documentation

- [ ] `DOCS/02_reference/trading/TRADING_MODULES.md` includes compliance notice
- [ ] Broker adapter documentation includes compliance notices
- [ ] User-facing tutorials include disclaimers
- [ ] Architecture docs reflect "user-authorized API integration" model

### Language in Documentation

- [ ] Uses approved language examples
- [ ] Avoids prohibited language
- [ ] Emphasizes user control and responsibility
- [ ] Clarifies what we provide vs. what we don't provide

---

## Legal Documentation

### Compliance Documents

- [ ] `BROKER_INTEGRATION_COMPLIANCE.md` exists and is complete
- [ ] `REGULATORY_DISCLAIMERS.md` is up to date
- [ ] `TOS.md` Section 8 (No Financial Advice) is current
- [ ] Commercial licenses include broker integration language

### Legal Language

- [ ] Commercial licenses state non-custodial, non-advisory nature
- [ ] User responsibilities are explicitly stated
- [ ] Broker-specific compliance is documented
- [ ] Risk warnings are included

---

## Architecture Compliance

### System Structure

- [ ] Architecture clearly separates:
  - Research engine (offline/backtest)
  - Signal generation (user-defined)
  - Risk & compliance layer (user-configurable)
  - Execution adapter (user-authorized)
  - User account (user-provided API keys)

### API Integration

- [ ] API keys are user-provided (not stored/managed by us)
- [ ] Execution requires user authorization
- [ ] Strategies are user-configured (not recommended)
- [ ] Risk limits are user-defined (not system-recommended)

---

## Broker-Specific Compliance

### IBKR Integration

- [ ] Documentation states IBKR explicitly supports third-party software
- [ ] No IBKR data reselling
- [ ] No unauthorized IBKR branding
- [ ] User must comply with IBKR API terms

### Alpaca Integration

- [ ] Documentation states Alpaca is designed for third-party integration
- [ ] No Alpaca data reselling
- [ ] No misrepresentation of regulatory status
- [ ] User must comply with Alpaca API terms

---

## User-Facing Materials

### Marketing & Sales

- [ ] Marketing materials emphasize software, not advice
- [ ] No guaranteed returns or performance claims
- [ ] Clear statements about user responsibilities
- [ ] Non-custodial nature is emphasized

### Support & Training

- [ ] Support materials include compliance disclaimers
- [ ] Training materials emphasize user control
- [ ] Examples use approved language
- [ ] Risk warnings are included

---

## Regular Review

### Quarterly Review

- [ ] Review broker integration compliance quarterly
- [ ] Update disclaimers as regulations change
- [ ] Verify language remains compliant
- [ ] Ensure new features maintain compliance

### Documentation Updates

- [ ] Update compliance docs when adding new brokers
- [ ] Review language when adding new features
- [ ] Verify disclaimers are current
- [ ] Check cross-references are correct

---

## Verification

### Before Deployment

- [ ] All checklist items are verified
- [ ] Legal review completed (if required)
- [ ] Documentation is current
- [ ] Code includes compliance notices

### After Changes

- [ ] New code includes compliance notices
- [ ] Documentation updated with disclaimers
- [ ] Language reviewed for compliance
- [ ] Architecture maintains compliance model

---

## Quick Reference

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

### Required Disclaimers

All broker integration materials must include:

1. **Non-advisory disclaimer** — We do not provide investment advice
2. **User responsibility** — Users responsible for brokerage, compliance, trading decisions
3. **Non-custodial statement** — We do not hold or control customer funds
4. **Regulatory compliance** — Users responsible for regulatory compliance
5. **Risk warnings** — Trading involves substantial risk of loss

---

## Related Documents

- [`BROKER_INTEGRATION_COMPLIANCE.md`](BROKER_INTEGRATION_COMPLIANCE.md) — Complete compliance framework
- [`REGULATORY_DISCLAIMERS.md`](REGULATORY_DISCLAIMERS.md) — General regulatory disclaimers
- [`TOS.md`](TOS.md) — Terms of Service (Section 8: No Financial Advice)

---

**Copyright (c) 2025-2026 Fox ML Infrastructure LLC**
