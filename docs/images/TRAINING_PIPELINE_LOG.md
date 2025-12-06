# Training Pipeline Log Visualization

*A picture is worth a thousand words* 📸

This image captures a comprehensive view of the TRAINING pipeline in action, showing:

- **LightGBM Training** - Complete training cycle with CPU affinity management
- **RewardBased Training** - Isolation runner with MKL/OMP configuration
- **XGBoost Training** - GPU-enabled training with proper CUDA setup

The log demonstrates:
- Proper thread management (OMP/MKL isolation)
- Memory cleanup between model families
- GPU allocation and visibility control
- Runtime policy validation
- Model saving and persistence

![Training Pipeline Log](training_pipeline_log.png)

*This screenshot shows the actual output from a multi-model training run, demonstrating the system's ability to handle complex training workflows with proper resource management and isolation.*

