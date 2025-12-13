# System Requirements

Hardware and software requirements for FoxML Core.

## Overview

FoxML Core is designed for **institutional-scale ML infrastructure**, not desktop or laptop environments. The system requires significant computational resources for production workloads.

## Verified Operating Ranges

### Verified Stable Range

**Tested and verified stable in continuous integration and benchmark testing:**

- **Memory (RAM)**: Up to **100 GB** - Verified stable under continuous workloads
- **CPU**: Multi-core processors (16+ cores recommended)
- **Storage**: SSD recommended for data processing pipelines
- **GPU**: Optional but recommended (7GB+ VRAM for GPU acceleration)

**Evidence**: Internal testing has verified stability and performance with workloads requiring up to 100 GB of physical memory in continuous operation.

### Targeted Capacity

**Targeted for enterprise and institutional deployment:**

- **Memory (RAM)**: **1 TB and beyond** - Targeted capacity for large-scale institutional use
- **Note**: Requires verification and tuning on final production hardware
- **Scaling**: Architecture designed to scale to institutional workloads

**Important**: The system has been verified stable up to 100 GB RAM. Scaling to 1 TB+ requires production hardware verification and may require configuration tuning.

## Minimum Requirements

### For Development and Testing

**Minimum viable configuration** (for small-scale testing and development):

- **OS**: Linux (Arch Linux, Ubuntu 22.04+, or similar distributions)
  - Environment tested on: Arch Linux (kernel 6.17+)
  - Build tools: GCC 11+ (gcc_linux-64, gxx_linux-64 from conda-forge)
- **Python**: 3.10 (as specified in `environment.yml`)
- **RAM**: 16 GB minimum (32 GB recommended for development)
- **CPU**: Multi-core processor (8+ cores)
- **Storage**: 50 GB+ free space for datasets and models
- **GPU**: Optional (CUDA 12.9 if using GPU acceleration, as specified in `environment.yml`)

**Limitations**: Minimum configuration is suitable for:
- Small-scale testing (< 10 symbols, < 50k samples)
- Development and debugging
- Learning the system architecture

**Not suitable for**:
- Production workloads
- Large-scale feature engineering
- Multi-target training pipelines
- Institutional research workflows

### For Production Workloads

**Recommended configuration** (for production and research use):

- **OS**: Linux (Arch Linux, Ubuntu 22.04+, or similar distributions)
  - Environment tested on: Arch Linux (kernel 6.17+)
  - Build tools: GCC 11+ (gcc_linux-64, gxx_linux-64 from conda-forge)
- **Python**: 3.10 (as specified in `environment.yml`)
- **RAM**: **64 GB minimum** (128 GB+ recommended for production)
- **CPU**: Multi-core processor (16+ cores, 32+ cores for large workloads)
- **Storage**: 500 GB+ SSD (1 TB+ for large datasets)
- **GPU**: Recommended (11GB+ VRAM for optimal performance)
  - NVIDIA GPU with CUDA support
  - CUDA toolkit 12.9 (as specified in `environment.yml`)
  - OpenCL drivers (for LightGBM GPU)
  - cuDNN 8.9+ (as specified in `environment.yml`)

## Memory Requirements by Workload

### Small-Scale Testing (< 10 symbols, < 50k samples)
- **RAM**: 16-32 GB
- **Use case**: Development, learning, small experiments

### Medium-Scale Research (10-50 symbols, 50k-500k samples)
- **RAM**: 64-128 GB
- **Use case**: Research workflows, feature engineering, model development

### Large-Scale Production (50+ symbols, 500k+ samples)
- **RAM**: 128 GB - 1 TB+
- **Use case**: Production pipelines, institutional research, multi-target training

### Verified Stable Range
- **RAM**: Up to **100 GB** - Confirmed stable in testing
- **Use case**: All production workloads within this range

### Targeted Capacity
- **RAM**: **1 TB and beyond** - Targeted for enterprise deployment
- **Use case**: Large-scale institutional workloads
- **Note**: Requires production hardware verification

## GPU Requirements

### GPU Acceleration (Optional but Recommended)

GPU acceleration provides 10-50x speedup for:
- Target ranking
- Feature selection
- Model training (LightGBM, XGBoost, CatBoost)

**Minimum GPU**:
- NVIDIA GPU with CUDA support
- 7 GB VRAM minimum
- CUDA toolkit 12.9 (as specified in `environment.yml`)
- cuDNN 8.9+ (as specified in `environment.yml`)

**Recommended GPU**:
- 11 GB+ VRAM for optimal performance
- Multiple GPUs supported for parallel workloads

**Performance**: GPU acceleration is most beneficial for large datasets (>100k samples). For smaller datasets (<50k samples), CPU may be faster due to GPU overhead.

## Storage Requirements

### Data Storage

- **Raw data**: Varies by dataset size (typically 10-100 GB for production datasets)
- **Processed features**: 50-500 GB depending on feature set size
- **Model artifacts**: 1-10 GB per training run
- **Results and metadata**: 1-5 GB per run

**Total recommended**: 500 GB - 1 TB+ free space for production workloads

### Storage Type

- **SSD recommended**: For data processing pipelines and model I/O
- **Network storage**: Supported for shared datasets (NFS, S3-compatible)

## Network Requirements

### For Data Acquisition

- Internet connection for data sources (yfinance, broker APIs)
- Network storage access if using shared datasets

### For Distributed Workloads

- Low-latency network for distributed training (planned/WIP)
- Shared filesystem access for multi-node workflows

## Software Requirements

### Operating System

- **Linux**: Arch Linux, Ubuntu 22.04+, or similar distributions
  - **Tested on**: Arch Linux (kernel 6.17+)
  - **Build requirements**: GCC 11+ (provided via conda-forge: gcc_linux-64, gxx_linux-64)
  - **Package management**: Conda (recommended) or pip
- **macOS**: Supported for development and testing (limited GPU support)
- **Windows**: Via WSL (Windows Subsystem for Linux) - not recommended for production

### Python Environment

- **Python**: 3.10 (as specified in `environment.yml`)
- **Package manager**: Conda (recommended, see `environment.yml`) or pip (see `requirements.txt`)
- **Virtual environment**: Required (conda env recommended)
  - Create with: `conda env create -f environment.yml`
  - Activate with: `conda activate foxml_env`

### Core Dependencies

See `requirements.txt` and `environment.yml` for complete dependency lists.

**Key dependencies**:
- pandas, numpy, scikit-learn
- LightGBM, XGBoost, CatBoost
- Polars (for high-throughput data processing)
- PyTorch (for neural network models)

## Cloud Deployment

### Recommended Cloud Instances

**For verified stable range (up to 100 GB RAM)**:
- AWS: r6i.4xlarge (128 GB RAM) or larger
- GCP: n2-highmem-32 (256 GB RAM) or larger
- Azure: Standard_E32s_v3 (256 GB RAM) or larger

**For targeted capacity (1 TB+)**:
- AWS: r6i.24xlarge (768 GB RAM) or x1e.32xlarge (3.9 TB RAM)
- GCP: n2-highmem-96 (768 GB RAM) or custom instances
- Azure: Standard_E96s_v3 (672 GB RAM) or larger

### Container Deployment

- Docker support: Yes (see deployment documentation)
- Kubernetes: Supported for orchestration
- Resource limits: Configure based on verified stable range

## Performance Expectations

### Verified Performance (Up to 100 GB RAM)

Based on internal testing and continuous integration:

- **Stability**: Verified stable under continuous workloads
- **Memory management**: Efficient memory usage with Polars-optimized pipelines
- **Throughput**: High-throughput data processing validated for production workloads
- **Scalability**: Architecture proven stable at scale

### Performance Characteristics

- **Data processing**: Optimized with Polars for high-throughput operations
- **Model training**: GPU acceleration provides 10-50x speedup on large datasets
- **Memory efficiency**: Memory management system handles large datasets efficiently

## Limitations and Considerations

### Not Suitable For

- **Laptop/desktop deployment**: System requires significant computational resources
- **Low-memory environments**: Minimum 16 GB RAM required, 64 GB+ recommended for production
- **Single-core systems**: Multi-core processors required for parallel workloads

### Development vs Production

- **Development**: Can run on smaller systems (16-32 GB RAM) for learning and small tests
- **Production**: Requires institutional-scale hardware (64 GB+ RAM, verified stable up to 100 GB)

## Getting Started

For installation and setup instructions, see:
- [Installation Guide](../01_tutorials/setup/INSTALLATION.md)
- [Environment Setup](../01_tutorials/setup/ENVIRONMENT_SETUP.md)
- [GPU Setup](../01_tutorials/setup/GPU_SETUP.md)

## Support

For questions about system requirements or deployment:
- **Email**: jenn.lewis5789@gmail.com
- **Documentation**: See [Documentation Index](../INDEX.md)

---

**Last Updated**: 2025-12-13  
**Verified Stable Range**: Up to 100 GB RAM (tested)  
**Targeted Capacity**: 1 TB+ RAM (enterprise deployment)
