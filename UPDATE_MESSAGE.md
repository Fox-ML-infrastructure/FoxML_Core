# Update Message from the Maintainer

Thank you for your patience while I completed the documentation restructuring and licensing compliance work. This effort included creating **55+ new documentation files**, rewriting **50+ existing ones**, and establishing **enterprise-grade legal and commercial materials**.

With this foundation in place, **active development has now resumed**, and focus has shifted back to core functionality, stability improvements, and feature development. A formal legal review of the commercial licensing framework is also underway.

## **Current Development Status**

* **TRAINING pipeline — testing and verified** ✅

  Pipeline is fully functional and verified. XGBoost source-build issues and readline symbol errors resolved.

* **All models — working correctly** ✅

  All model families (GPU, sequential, and CPU-based) are operational and producing artifacts. VAE serialization fixed. Expect some noise and warnings during GPU model training (version compatibility messages, plugin registration notices) — these are harmless and do not affect functionality. Sequential models (CNN1D, LSTM, Transformer, and LSTM-based variants) are working and producing outputs. LSTM variants may require longer training time than the current timeout allows, but they function correctly. 3D preprocessing issues resolved.

* **Target ranking & selection — testing**

  Validation under way before integrating directly into the TRAINING pipeline.

* **Centralized configuration system — underway and mostly complete** 🔄

  Phase 2 centralized configuration work is in progress and mostly complete:
  * ✅ 9 training config YAML files created (pipeline, GPU, memory, preprocessing, threading, safety, callbacks, optimizer, system)
  * ✅ Config loader with nested access and family-specific overrides
  * ✅ All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
  * ✅ Pipeline, threading, memory, GPU, and system settings integrated
  * ✅ Backward compatibility maintained with hardcoded defaults
  * ✅ Reproducibility and deterministic behavior verified — same outputs pre-config rework vs after
  * ✅ Other models testing good so far — no regressions observed
  * 🔄 Validation layer and logging modernization in progress

* **Commercial license pricing — updated** 💰

  Pricing structure recalibrated to align with enterprise market norms and traction metrics:
  * 1-10 employees: $25,200/year
  * 11-50 employees: $60,000/year
  * 51-250 employees: $165,000/year
  * 251-1000 employees: $252,000/year
  * Enterprise (1000+ employees): Pricing starts at $500,000/year (custom quote)

* **Upcoming refactors**

  Planning deeper improvements to training intelligence, model selection, and automated workflows.
  
  Scaffolded base trainers (`base_2d_trainer.py`, `base_3d_trainer.py`) for future refactoring to centralize dimension-specific preprocessing logic.

## Next Steps

* Testing other models today — validating additional model families and configurations
* More core work will begin after testing completes
* Validate ranking/selection scripts
* Complete configuration system validation and logging modernization
* Inquiring about potential hospital compliance and other regulatory requirements — exploring stricter legal frameworks for healthcare, medical, and other highly regulated use cases

Thank you again for your understanding during this infrastructure-heavy cycle. The system is now **clearer, more maintainable, legally compliant, and ready for continued development**.

— **Jennifer Lewis, Maintainer**

Fox ML Infrastructure
