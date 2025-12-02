# Jennifer Lewis — ML & Quantitative Systems Consultant

**Independent Contractor | Machine Learning Engineering | Quantitative Research | High-Performance Systems**

I build ML research pipelines, quantitative modeling systems, and HPC workflows for institutions that need high-performance engineering without hiring a full-time team.

🔗 **Aurora-v2** (50k+ LOC Quant/ML System): This repository demonstrates research-grade ML pipelines, quantitative research frameworks, and high-performance trading infrastructure. It is a reference system for internal research and engineering—not a plug-and-play trading bot.

---

## 📑 Table of Contents

### Consulting & Services
- [Consulting Overview](CONSULTING.md) — Services and engagement process
- [Consulting Policy](CONSULTING_POLICY.md) — Professional terms, rates, and policies
- [Consulting Pricing](CONSULTING_PRICING.md) — Detailed pricing tiers and rate structure
- [Master Consulting Agreement](MASTER_CONSULTING_AGREEMENT.md) — Standard agreement framework
- [Statement of Work Template](STATEMENT_OF_WORK.md) — SOW template

### Quick Start & Core Documentation
- [Quick Start Guide](INFORMATION/01_QUICK_START.md) — Get started with the system
- [Project Overview](INFORMATION/07_PROJECT_OVERVIEW.md) — High-level system architecture
- [Config Reference](INFORMATION/02_CONFIG_REFERENCE.md) — Configuration file structure
- [Data Pipeline Guide](INFORMATION/04_DATA_PIPELINE.md) — Data processing workflows
- [Model Training Guide](INFORMATION/05_MODEL_TRAINING.md) — Training workflows and best practices
- [Feature Selection Guide](INFORMATION/08_FEATURE_SELECTION.md) — Feature engineering and selection
- [Multi-Model Feature Selection](INFORMATION/MULTI_MODEL_FEATURE_SELECTION.md) — Ensemble approaches
- [Column Reference](INFORMATION/06_COLUMN_REFERENCE.md) — Data schema and column definitions
- [Migration Notes](INFORMATION/03_MIGRATION_NOTES.md) — Migration and upgrade guidance

### Component Documentation
- [Alpaca Trading](ALPACA_trading/README.md) — Paper trading service documentation
- [IBKR Trading](IBKR_trading/README.md) — Interactive Brokers integration
- [Data Processing](DATA_PROCESSING/README.md) — Data pipelines and feature engineering
- [Configuration Management](CONFIG/README.md) — Config system and overlays

### Training & Model Development
- [Feature Selection Guide](TRAINING/FEATURE_SELECTION_GUIDE.md) — Feature selection workflows
- [Training Optimization Guide](TRAINING/TRAINING_OPTIMIZATION_GUIDE.md) — Performance optimization
- [Training Experiments](TRAINING/EXPERIMENTS/README.md) — Experimental workflows
- [Quick Start Feature Ranking](NOTES/QUICK_START_FEATURE_RANKING.md) — Feature ranking quick start

### Technical Guides & Reference
- [GPU Setup Multi-Model](docs/GPU_SETUP_MULTI_MODEL.md) — GPU configuration for multi-model training
- [Comprehensive Feature Ranking](docs/COMPREHENSIVE_FEATURE_RANKING.md) — Feature ranking methodology
- [Target Discovery Update](docs/TARGET_DISCOVERY_UPDATE.md) — Target variable selection
- [Target to Feature Workflow](docs/TARGET_TO_FEATURE_WORKFLOW.md) — Workflow documentation
- [Validation Leak Audit](docs/VALIDATION_LEAK_AUDIT.md) — Leakage prevention and validation
- [Feature Importance Fix](docs/FEATURE_IMPORTANCE_FIX.md) — Feature importance corrections
- [Dataset Sizing Strategy](docs/DATASET_SIZING_STRATEGY.md) — Data sizing recommendations
- [Journald Logging](docs/JOURNALD_LOGGING.md) — System logging configuration
- [Code Review Bugs](docs/CODE_REVIEW_BUGS.md) — Known issues and fixes
- [Alpha Enhancement Roadmap](docs/ALPHA_ENHANCEMENT_ROADMAP.md) — Enhancement planning

### Fixes & Technical Notes
- [Leakage Fixes](docs/FIXES/) — Collection of leakage-related fixes and analyses
- [Target Leakage Clarification](docs/TARGET_LEAKAGE_CLARIFICATION.md) — Leakage prevention
- [Forward Return Leakage Analysis](docs/FWD_RET_20D_LEAKAGE_ANALYSIS.md) — Temporal leakage analysis

---

## About

I am an independent contractor specializing in **ML Systems + Quant Infrastructure + HPC Pipelines** for financial and research organizations. My work has been used in research environments, academic contexts, and internal experimentation by multiple developers and analysts.

**For organizations requiring custom development, enterprise integrations, or specialized consulting services, I provide contract-based solutions tailored to your needs.**

---

## Services for Organizations

**Primary Specialization:** ML Systems + Quant Infrastructure + HPC Pipelines

All other services extend from these core competencies.

### Core Expertise

**Machine Learning Pipeline Development**
- End-to-end ML infrastructure design and implementation
- Feature engineering and leakage-safe validation frameworks
- Multi-model ensemble systems and model zoo management
- GPU-accelerated training and inference workflows

**Quantitative Research Systems**
- Financial modeling and backtesting frameworks
- Walk-forward validation and research tooling
- Data processing pipelines with strict quality controls
- Performance optimization for research workloads

**High-Performance Computing**
- C++ inference engine development and optimization
- HPC workflow design for GPU clusters and cloud infrastructure
- Latency-critical systems for HFT and real-time applications
- Infrastructure-specific optimizations

**Systems Architecture & Engineering**
- Hybrid C++/Python system design
- Enterprise-grade deployment and integration
- Compliance-focused configurations and governance
- Code review, audit, and optimization services

### Engagement Model

- **Remote contract work** — All engagements conducted remotely
- **Defined deliverables** — Every project includes a formal Statement of Work (SOW)
- **Flexible engagement** — Hourly, project-based, or retainer arrangements
- **Rates** — Standard consulting starts at **$300/hr**, with higher rates for latency-critical, 24/7, or high-urgency work (see [`CONSULTING_POLICY.md`](CONSULTING_POLICY.md) and [`CONSULTING_PRICING.md`](CONSULTING_PRICING.md) for details)
- **Organizational focus** — Specialized support for institutions, research organizations, and enterprises

---

## Why Work With Me

- **Deep system-building background** — Production experience with large-scale ML and quant systems
- **Fast delivery speed** — Efficient workflows and clear communication reduce project timelines
- **Clear documentation** — Every deliverable includes comprehensive documentation and reproducible workflows
- **Strong communication** — Defined scope, milestones, and regular updates throughout engagement
- **Low onboarding cost** — Self-contained expertise reduces integration overhead
- **Specialized domain knowledge** — Focused expertise in ML pipelines, quant research, and HPC optimization

---

## Getting Started

### For Organizations Seeking Consulting Services

1. **Review consulting materials:**
   - [`CONSULTING.md`](CONSULTING.md) — Overview of services and engagement process
   - [`CONSULTING_POLICY.md`](CONSULTING_POLICY.md) — Professional terms, rates, and policies
   - [`CONSULTING_PRICING.md`](CONSULTING_PRICING.md) — Detailed pricing tiers and rate structure
   - [`MASTER_CONSULTING_AGREEMENT.md`](MASTER_CONSULTING_AGREEMENT.md) — Standard agreement framework
   - [`STATEMENT_OF_WORK.md`](STATEMENT_OF_WORK.md) — SOW template

2. **Contact:** jenn.lewis5789@gmail.com  
   **Subject:** Consulting Inquiry — [Your Organization Name]

3. **Initial discussion:** We'll discuss your requirements, scope, and timeline to prepare a formal SOW.

### For Developers & Researchers

See individual component READMEs for technical documentation:

- [`ALPACA_trading/README.md`](ALPACA_trading/README.md) — Paper trading service
- [`IBKR_trading/README.md`](IBKR_trading/README.md) — Interactive Brokers integration
- [`DATA_PROCESSING/README.md`](DATA_PROCESSING/README.md) — Data processing pipelines
- [`TRAINING/`](TRAINING/) — Model training and feature engineering workflows

---

## Repository Structure

```
trader/
├── ALPACA_trading/          # Alpaca paper trading service
├── IBKR_trading/            # Interactive Brokers live trading
├── DATA_PROCESSING/         # Data pipelines and feature engineering
├── TRAINING/                # Model training and research workflows
├── CONFIG/                  # Configuration management
├── docs/                    # Technical documentation
├── scripts/                 # Utility scripts and tools
└── [consulting docs]        # CONSULTING.md, CONSULTING_POLICY.md, etc.
```

---

## Professional Standards

- **Client neutrality** — I work with any organization that respects professional boundaries and contractual terms
- **Confidentiality** — All client materials and proprietary information are protected under standard NDAs
- **Quality deliverables** — Structured, documented, and production-ready code
- **Clear communication** — Defined scope, milestones, and expectations in every engagement

---

## Contact

**Email:** jenn.lewis5789@gmail.com  
**Subject:** Consulting Inquiry — [Your Organization Name]

For custom development, enterprise integrations, or specialized consulting services, please reach out to discuss your requirements.

---

## License

Open-source components are available under their respective licenses. Custom work and deliverables are governed by individual Statements of Work and the Master Consulting Agreement.

