# FoxML — Version 1.0 Manifest

## Scope

This document defines the guarantees and non-goals of FoxML version 1.0.

---

## Guaranteed Capabilities

- **Deterministic, reproducible training** given an explicit training plan
- **Automatic routing** of targets and features to appropriate model classes
- **Policy-enforced feature eligibility** and leakage constraints
- **Run-to-run telemetry** with historical comparison
- **Aggregate drift tracking** via interpretable linear regression

---

## Explicit Non-Goals

- Real-time inference
- Online learning
- Neural / transformer-based models
- Automated strategy execution
- Black-box optimization or opaque monitoring

---

## Stability Contract

All behaviors described above are guaranteed for all 1.x releases. Breaking changes require a major version increment.

---

## Reproducibility Contract

Given identical inputs, configs, and training plans, FoxML 1.0 will produce identical artifacts and metrics within numerical tolerance.

---

## Ownership

FoxML is governed by explicit policy and documented invariants. Behavioral changes are intentional, documented, and versioned.

---

## The 1.0 Test

FoxML 1.0 is achieved when, given a training plan, the system will always:

1. Route correctly (features → families, targets → models)
2. Train deterministically (same inputs → same outputs)
3. Record comparable telemetry (run-to-run comparison artifacts)
4. Surface drift coherently (interpretable trend signals)

without manual intervention.

---

## Related Documentation

- [README.md](./README.md) - Project overview and getting started
- [ROADMAP.md](DOCS/02_reference/roadmap/ROADMAP.md) - Development roadmap and milestones
- [DOCS/03_technical/README.md](./DOCS/03_technical/README.md) - Technical documentation
- [DOCS/03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md](./DOCS/03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md) - Reproducibility architecture

---

**Last Updated:** 2025-12-14
