# Update Message from the Maintainer

Thank you for your patience while I completed the documentation restructuring and licensing compliance work. This effort included creating **55+ new documentation files**, rewriting **50+ existing ones**, and establishing **enterprise-grade legal and commercial materials**.

With this foundation in place, **active development has now resumed**, and focus has shifted back to core functionality, stability improvements, and feature development. A formal legal review of the commercial licensing framework is also underway.

## **Current Development Status**

* **TRAINING pipeline — testing and verified** ✅

  Pipeline is fully functional and verified. XGBoost source-build issues and readline symbol errors resolved.

* **GPU families — all working** ✅

  All GPU model families are operational and producing artifacts. Expect some noise and warnings during GPU model training (version compatibility messages, plugin registration notices) — these are harmless and do not affect functionality.

* **Sequential models — mostly working**

  CNN1D, Transformer, and other sequential models are working. LSTM appears functional but is timing out at the 3-hour limit (10800s) — it seems to work but requires longer training time than the current timeout allows. 3D preprocessing issues resolved.

* **Target ranking & selection — testing**

  Validation under way before integrating directly into the TRAINING pipeline.

* **Centralized configuration system — underway and mostly complete** 🔄

  Phase 2 centralized configuration work is in progress and mostly complete:
  * ✅ 9 training config YAML files created (pipeline, GPU, memory, preprocessing, threading, safety, callbacks, optimizer, system)
  * ✅ Config loader with nested access and family-specific overrides
  * ✅ All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
  * ✅ Pipeline, threading, memory, GPU, and system settings integrated
  * ✅ Backward compatibility maintained with hardcoded defaults
  * 🔄 Testing underway for new config refactor — validating all integrations work correctly
  * 🔧 Minor fixes underway — addressing type conversion and edge cases discovered during testing
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

* Complete final testing of remaining sequential models
* Validate ranking/selection scripts
* Complete configuration system validation and logging modernization

Thank you again for your understanding during this infrastructure-heavy cycle. The system is now **clearer, more maintainable, legally compliant, and ready for continued development**.

— **Jennifer Lewis, Maintainer**

Fox ML Infrastructure
