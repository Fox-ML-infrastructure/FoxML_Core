# Training Pipeline in Action

*A picture is worth a thousand words* 📸

This page showcases visualizations of the TRAINING pipeline and data building processes in action.

## Training Pipeline Log

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

---

## Data Building Pipeline

This image shows the data building pipeline in action, demonstrating:

- Data processing and feature engineering workflows
- Pipeline orchestration and data flow
- Feature construction and validation
- Cross-sectional and time-series data handling

![Data Building Pipeline](data_building_pipeline.png)

*This screenshot shows the actual output from the data building pipeline, demonstrating the system's ability to process and construct features at scale.*

