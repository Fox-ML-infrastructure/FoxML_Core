# Distributed Training Architecture for Intelligence Layer

**Status**: Design Phase  
**Date**: 2025-12-08  
**Classification**: Internal Planning Document  
**Related**: [Intelligence Layer Overview](../../03_technical/research/INTELLIGENCE_LAYER.md), [Continuous Integrated Learning System](CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md)

## Executive Summary

This document outlines the architecture for distributing model training across multiple compute nodes, replacing the current sequential execution model. The distributed architecture enables:

- **Parallel target ranking** - Evaluate multiple targets simultaneously across nodes
- **Parallel feature selection** - Select features for multiple targets in parallel
- **Parallel model training** - Train multiple model families/targets concurrently
- **Resource optimization** - Better utilization of GPU/CPU resources across cluster
- **Fault tolerance** - Automatic retry and recovery from node failures
- **Scalability** - Linear scaling with number of compute nodes

**Current State**: All training runs sequentially (targets → features → models)  
**Target State**: Distributed task queue with worker nodes executing tasks in parallel

---

## 1. Current Sequential Architecture

### 1.1 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Single Node (Sequential)                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 1: Target Ranking                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Target 1 │→ │ Target 2 │→ │ Target 3 │→ ...            │
│  └──────────┘  └──────────┘  └──────────┘                  │
│                                                               │
│  Step 2: Feature Selection                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Target 1 │→ │ Target 2 │→ │ Target 3 │→ ...            │
│  └──────────┘  └──────────┘  └──────────┘                  │
│                                                               │
│  Step 3: Model Training                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Target 1 │→ │ Target 2 │→ │ Target 3 │→ ...            │
│  │ Family A │  │ Family A │  │ Family A │                  │
│  │ Family B │  │ Family B │  │ Family B │                  │
│  │ Family C │  │ Family C │  │ Family C │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Current Limitations

**Performance Bottlenecks:**
- **Sequential target evaluation**: 23 targets × 3 models × ~30s = ~20 minutes just for ranking
- **Sequential feature selection**: 23 targets × 14 model families × ~60s = ~5 hours
- **Sequential model training**: 23 targets × 14 families × ~120s = ~11 hours
- **Total wall-clock time**: ~16+ hours for full pipeline

**Resource Underutilization:**
- GPU idle during CPU-only model families
- CPU cores underutilized (only 1-2 models running at once)
- Memory not fully utilized
- Network bandwidth unused (no data distribution)

**Scalability Issues:**
- Cannot leverage multiple machines
- No horizontal scaling
- Single point of failure
- No load balancing

---

## 2. Distributed Architecture Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Coordinator Node                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Task       │  │   Result     │  │   State      │     │
│  │   Queue      │  │   Aggregator │  │   Manager    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         │ distributes      │ collects        │ tracks      │
│         ▼                  ▼                  ▼             │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │                  │                  │
    ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
    │           │      │           │      │           │
┌───▼───────────▼──┐ ┌─▼───────────▼──┐ ┌─▼───────────▼──┐
│  Worker Node 1   │ │  Worker Node 2 │ │  Worker Node N │
│  ┌────────────┐ │ │  ┌────────────┐ │ │  ┌────────────┐ │
│  │  Task      │ │ │  │  Task      │ │ │  │  Task      │ │
│  │  Executor  │ │ │  │  Executor  │ │ │  │  Executor  │ │
│  └────────────┘ │ │  └────────────┘ │ │  └────────────┘ │
│  ┌────────────┐ │ │  ┌────────────┐ │ │  ┌────────────┐ │
│  │  Resource  │ │ │  │  Resource  │ │ │  │  Resource  │ │
│  │  Manager   │ │ │  │  Manager   │ │ │  │  Manager   │ │
│  └────────────┘ │ │  └────────────┘ │ │  └────────────┘ │
│  GPU: Available │ │  GPU: Busy      │ │  CPU: Available │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.2 Core Components

#### 2.2.1 Coordinator Node

**Responsibilities:**
- Task queue management (distribute tasks to workers)
- Result aggregation (collect results from workers)
- State management (track task progress, failures, retries)
- Dependency resolution (ensure targets ranked before feature selection)
- Resource discovery (detect available workers, GPUs, CPUs)

**Components:**
- **Task Queue**: Redis/Celery/RabbitMQ for distributed task queue
- **Result Store**: Redis/PostgreSQL for storing results
- **State Manager**: Tracks task status (pending, running, completed, failed)
- **Scheduler**: Determines task priority and worker assignment

#### 2.2.2 Worker Nodes

**Responsibilities:**
- Execute tasks (target ranking, feature selection, model training)
- Report progress and results back to coordinator
- Manage local resources (GPU/CPU allocation)
- Handle failures gracefully (retry, checkpoint, recovery)

**Components:**
- **Task Executor**: Executes training tasks (calls existing training functions)
- **Resource Manager**: Manages GPU/CPU allocation per task
- **Progress Reporter**: Sends periodic updates to coordinator
- **Checkpoint Manager**: Saves intermediate results for recovery

#### 2.2.3 Shared Storage

**Requirements:**
- **Data Storage**: Shared filesystem (NFS/GlusterFS) or object storage (S3/MinIO) for training data
- **Model Storage**: Shared location for trained models
- **Cache Storage**: Shared cache for target rankings and feature selections
- **Config Storage**: Shared config files (read-only, versioned)

---

## 3. Task Distribution Strategy

### 3.1 Task Granularity

**Fine-Grained Tasks** (Recommended):
- **Target Ranking Task**: One target × one symbol
- **Feature Selection Task**: One target × one symbol
- **Model Training Task**: One target × one model family × one symbol

**Benefits:**
- Maximum parallelism (hundreds of tasks can run simultaneously)
- Better load balancing (small tasks distribute evenly)
- Faster failure recovery (only retry small failed task)

**Coarse-Grained Tasks** (Alternative):
- **Target Ranking Task**: All targets for one symbol
- **Feature Selection Task**: All targets for one symbol
- **Model Training Task**: All models for one target

**Benefits:**
- Fewer tasks to manage
- Less coordination overhead
- Better for small clusters

### 3.2 Task Types

#### 3.2.1 Target Ranking Task

```python
{
    "task_type": "rank_target",
    "task_id": "rank_target_AAPL_y_will_peak_60m_0.8",
    "target": "y_will_peak_60m_0.8",
    "symbol": "AAPL",
    "data_path": "s3://bucket/data/symbol=AAPL/AAPL.parquet",
    "config_hash": "abc123...",
    "dependencies": [],  # No dependencies
    "priority": 1,
    "resource_requirements": {
        "gpu": false,  # Can use GPU but not required
        "cpu_cores": 4,
        "memory_gb": 8
    }
}
```

#### 3.2.2 Feature Selection Task

```python
{
    "task_type": "select_features",
    "task_id": "select_features_AAPL_y_will_peak_60m_0.8",
    "target": "y_will_peak_60m_0.8",
    "symbol": "AAPL",
    "data_path": "s3://bucket/data/symbol=AAPL/AAPL.parquet",
    "config_hash": "abc123...",
    "dependencies": [
        "rank_target_AAPL_y_will_peak_60m_0.8"  # Must complete first
    ],
    "priority": 2,
    "resource_requirements": {
        "gpu": false,
        "cpu_cores": 8,
        "memory_gb": 16
    }
}
```

#### 3.2.3 Model Training Task

```python
{
    "task_type": "train_model",
    "task_id": "train_model_AAPL_y_will_peak_60m_0.8_lightgbm",
    "target": "y_will_peak_60m_0.8",
    "symbol": "AAPL",
    "model_family": "lightgbm",
    "features": ["feature_1", "feature_2", ...],  # From feature selection
    "data_path": "s3://bucket/data/symbol=AAPL/AAPL.parquet",
    "config_hash": "abc123...",
    "dependencies": [
        "select_features_AAPL_y_will_peak_60m_0.8"
    ],
    "priority": 3,
    "resource_requirements": {
        "gpu": true,  # LightGBM can use GPU
        "cpu_cores": 4,
        "memory_gb": 8
    }
}
```

### 3.3 Dependency Graph

```
Target Ranking Tasks (Level 1)
    │
    ├─→ Feature Selection Tasks (Level 2)
    │       │
    │       ├─→ Model Training Tasks (Level 3)
    │       │       ├─→ lightgbm
    │       │       ├─→ xgboost
    │       │       ├─→ random_forest
    │       │       └─→ ...
    │       │
    │       └─→ Model Training Tasks (Level 3)
    │               └─→ ...
    │
    └─→ Feature Selection Tasks (Level 2)
            └─→ ...
```

**Execution Strategy:**
1. **Level 1**: All target ranking tasks can run in parallel (no dependencies)
2. **Level 2**: Feature selection tasks start as their target ranking completes
3. **Level 3**: Model training tasks start as their feature selection completes

---

## 4. Communication Patterns

### 4.1 Task Distribution

**Push Model** (Recommended):
- Coordinator pushes tasks to workers based on resource availability
- Workers pull tasks from queue when ready
- Better for heterogeneous clusters (different GPU/CPU counts)

**Pull Model** (Alternative):
- Workers continuously poll coordinator for tasks
- Coordinator assigns tasks based on worker capabilities
- Simpler implementation, but more network traffic

### 4.2 Result Collection

**Immediate Return**:
- Workers return results immediately upon task completion
- Coordinator aggregates results as they arrive
- Best for real-time progress tracking

**Batch Return**:
- Workers batch results and send periodically
- Reduces network overhead
- Better for high-throughput scenarios

### 4.3 Progress Reporting

**Heartbeat Mechanism**:
- Workers send periodic heartbeats (every 30s)
- Coordinator tracks worker health
- Missing heartbeats trigger task reassignment

**Progress Updates**:
- Workers send progress updates (every 10% completion)
- Coordinator updates task status
- Enables real-time progress visualization

---

## 5. Resource Management

### 5.1 Resource Discovery

**Worker Registration:**
```python
{
    "worker_id": "worker-001",
    "hostname": "gpu-node-01",
    "resources": {
        "gpu_count": 2,
        "gpu_memory_gb": [24, 24],
        "cpu_cores": 32,
        "memory_gb": 128,
        "available": true
    },
    "capabilities": {
        "can_run_gpu_tasks": true,
        "can_run_cpu_tasks": true,
        "supported_model_families": ["lightgbm", "xgboost", "neural_network"]
    }
}
```

### 5.2 Resource Allocation

**GPU Allocation:**
- Assign GPU tasks to workers with available GPUs
- Track GPU memory usage per task
- Prevent oversubscription (don't assign more tasks than GPUs)

**CPU Allocation:**
- Assign CPU tasks based on available CPU cores
- Use CPU affinity to prevent thread contention
- Balance load across workers

**Memory Management:**
- Track memory usage per task
- Kill tasks that exceed memory limits
- Prioritize tasks with lower memory requirements

### 5.3 Load Balancing

**Strategies:**
1. **Round-Robin**: Distribute tasks evenly across workers
2. **Least-Loaded**: Assign to worker with fewest active tasks
3. **Resource-Aware**: Match task requirements to worker capabilities
4. **Affinity-Based**: Prefer workers that have cached data/configs

---

## 6. Fault Tolerance

### 6.1 Task Failure Handling

**Automatic Retry:**
- Retry failed tasks up to N times (default: 3)
- Exponential backoff between retries
- Mark as permanently failed after max retries

**Checkpointing:**
- Save intermediate results periodically
- Resume from checkpoint on failure
- Reduces wasted computation

**Task Reassignment:**
- Reassign tasks from failed workers
- Detect worker failures via heartbeat timeout
- Prevent duplicate execution (idempotent tasks)

### 6.2 Data Consistency

**Idempotent Tasks:**
- All tasks must be idempotent (safe to retry)
- Use deterministic task IDs
- Check for existing results before execution

**Result Deduplication:**
- Store results with task ID as key
- Ignore duplicate results
- Handle race conditions (multiple workers complete same task)

### 6.3 Recovery Procedures

**Coordinator Failure:**
- Persist task queue to disk
- Restore queue on coordinator restart
- Resume task distribution from checkpoint

**Worker Failure:**
- Detect via missing heartbeats
- Reassign in-progress tasks to other workers
- Mark worker as unavailable

---

## 7. Implementation Options

### 7.1 Option 1: Celery + Redis (Recommended)

**Architecture:**
- **Task Queue**: Redis
- **Message Broker**: Redis
- **Result Backend**: Redis
- **Worker Framework**: Celery

**Pros:**
- Mature, battle-tested framework
- Good Python integration
- Built-in retry, monitoring, scheduling
- Easy to scale workers

**Cons:**
- Requires Redis infrastructure
- Some overhead for small clusters
- Learning curve for advanced features

**Implementation:**
```python
from celery import Celery

app = Celery('intelligence_layer',
             broker='redis://coordinator:6379/0',
             backend='redis://coordinator:6379/0')

@app.task(bind=True, max_retries=3)
def rank_target_task(self, target, symbol, data_path, config_hash):
    # Execute target ranking
    result = rank_target(target, symbol, data_path)
    return result

@app.task(bind=True, max_retries=3)
def select_features_task(self, target, symbol, data_path, config_hash):
    # Execute feature selection
    result = select_features(target, symbol, data_path)
    return result
```

### 7.2 Option 2: Ray

**Architecture:**
- **Task Framework**: Ray
- **Object Store**: Ray's distributed object store
- **Scheduler**: Ray's built-in scheduler

**Pros:**
- Excellent for ML workloads
- Built-in distributed data handling
- Good GPU support
- Auto-scaling capabilities

**Cons:**
- Larger dependency footprint
- More complex setup
- Less mature ecosystem than Celery

**Implementation:**
```python
import ray

ray.init(address='coordinator:10001')

@ray.remote(num_gpus=1)
def rank_target_task(target, symbol, data_path, config_hash):
    # Execute target ranking
    result = rank_target(target, symbol, data_path)
    return result

# Distribute tasks
futures = [rank_target_task.remote(t, s, d, c) 
           for t, s, d, c in task_list]
results = ray.get(futures)
```

### 7.3 Option 3: Custom Task Queue (Simple)

**Architecture:**
- **Task Queue**: PostgreSQL/Redis
- **Worker Pool**: Multiprocessing/Threading
- **Coordination**: Custom Python code

**Pros:**
- Full control over behavior
- Minimal dependencies
- Easy to customize

**Cons:**
- More code to maintain
- Need to implement retry, monitoring, etc.
- Less battle-tested

---

## 8. Data Distribution

### 8.1 Shared Filesystem

**Option: NFS/GlusterFS**
- All nodes mount same filesystem
- Data accessible at same path on all nodes
- Simple, but single point of failure

**Option: Object Storage (S3/MinIO)**
- Data stored in object storage
- Workers download data as needed
- Better scalability, but network overhead

### 8.2 Data Locality

**Strategy: Pre-stage Data**
- Copy data to worker nodes before task execution
- Reduces network traffic during training
- Better for large datasets

**Strategy: On-Demand Download**
- Workers download data when task starts
- More flexible, but slower first access
- Good for small/medium datasets

### 8.3 Caching Strategy

**Shared Cache:**
- Store target rankings/feature selections in shared cache
- Workers check cache before computation
- Reduces redundant computation

**Local Cache:**
- Workers cache data locally
- Faster access for repeated tasks
- Requires cache invalidation strategy

---

## 9. Monitoring & Observability

### 9.1 Metrics to Track

**Task Metrics:**
- Tasks pending/running/completed/failed
- Average task duration per type
- Task failure rate
- Retry count distribution

**Resource Metrics:**
- GPU utilization per worker
- CPU utilization per worker
- Memory usage per worker
- Network I/O per worker

**Pipeline Metrics:**
- End-to-end pipeline duration
- Tasks completed per hour
- Throughput (tasks/second)
- Queue depth over time

### 9.2 Dashboards

**Coordinator Dashboard:**
- Task queue status
- Worker health
- Resource utilization
- Pipeline progress

**Worker Dashboard:**
- Active tasks
- Resource usage
- Task history
- Error logs

### 9.3 Alerting

**Critical Alerts:**
- Coordinator down
- High task failure rate (>10%)
- Worker failures
- Resource exhaustion

**Warning Alerts:**
- Slow task execution (>2x expected)
- High queue depth (>100 tasks)
- Low worker availability (<50%)

---

## 10. Migration Path

### 10.1 Phase 1: Proof of Concept (2-4 weeks)

**Goals:**
- Set up basic distributed infrastructure (Celery + Redis)
- Implement single task type (target ranking)
- Test with 2-3 worker nodes
- Validate correctness and performance

**Deliverables:**
- Working distributed target ranking
- Basic monitoring dashboard
- Documentation for setup

### 10.2 Phase 2: Feature Selection (2-3 weeks)

**Goals:**
- Add distributed feature selection
- Implement dependency resolution
- Test with full pipeline (ranking → selection)
- Optimize data distribution

**Deliverables:**
- Distributed feature selection
- Dependency-aware scheduler
- Performance benchmarks

### 10.3 Phase 3: Model Training (3-4 weeks)

**Goals:**
- Add distributed model training
- Implement GPU resource management
- Test with full pipeline (ranking → selection → training)
- Optimize resource allocation

**Deliverables:**
- Full distributed pipeline
- GPU-aware scheduling
- Production-ready system

### 10.4 Phase 4: Production Hardening (2-3 weeks)

**Goals:**
- Add comprehensive monitoring
- Implement advanced fault tolerance
- Performance tuning
- Documentation and runbooks

**Deliverables:**
- Production-ready system
- Complete documentation
- Operational runbooks

---

## 11. Expected Performance Improvements

### 11.1 Current Performance (Sequential)

**Example: 23 targets, 14 model families, 3 symbols**

- Target Ranking: 23 targets × 3 symbols × 30s = **20.7 minutes**
- Feature Selection: 23 targets × 3 symbols × 60s = **69 minutes**
- Model Training: 23 targets × 14 families × 3 symbols × 120s = **32.2 hours**
- **Total: ~34 hours**

### 11.2 Distributed Performance (10 workers)

**Assumptions:**
- 10 worker nodes (mix of GPU and CPU)
- Perfect load balancing
- No network overhead
- Tasks fully parallelizable

- Target Ranking: 23 × 3 / 10 = **2.1 minutes** (10x speedup)
- Feature Selection: 23 × 3 / 10 = **6.9 minutes** (10x speedup)
- Model Training: 23 × 14 × 3 / 10 = **9.7 hours** (3.3x speedup, GPU bottleneck)
- **Total: ~10 hours** (3.4x overall speedup)

### 11.3 Distributed Performance (50 workers)

- Target Ranking: **25 seconds** (50x speedup)
- Feature Selection: **1.4 minutes** (50x speedup)
- Model Training: **1.9 hours** (17x speedup, GPU bottleneck)
- **Total: ~2 hours** (17x overall speedup)

**Note**: Actual speedup depends on:
- Task granularity (finer = better parallelism)
- Data distribution overhead
- Network latency
- Resource contention
- Dependency bottlenecks

---

## 12. Configuration

### 12.1 Coordinator Config

```yaml
# CONFIG/distributed/coordinator.yaml
coordinator:
  task_queue:
    backend: "redis"  # redis, rabbitmq, postgresql
    host: "coordinator"
    port: 6379
    db: 0
  
  result_backend:
    backend: "redis"
    host: "coordinator"
    port: 6379
    db: 1
    ttl: 86400  # 24 hours
  
  scheduler:
    max_workers: 50
    task_timeout: 3600  # 1 hour
    retry_max: 3
    retry_backoff: 60  # 1 minute
  
  monitoring:
    enabled: true
    metrics_port: 9090
    dashboard_port: 8080
```

### 12.2 Worker Config

```yaml
# CONFIG/distributed/worker.yaml
worker:
  coordinator_url: "redis://coordinator:6379/0"
  worker_id: "worker-001"  # Auto-generated if not set
  hostname: "gpu-node-01"  # Auto-detected if not set
  
  resources:
    gpu_count: 2
    cpu_cores: 32
    memory_gb: 128
  
  task_execution:
    max_concurrent_tasks: 4
    task_timeout: 3600
    checkpoint_interval: 300  # 5 minutes
  
  data:
    cache_dir: "/local/cache"
    shared_storage: "s3://bucket/data"
    download_on_demand: true
```

---

## 13. Open Questions & Decisions Needed

### 13.1 Infrastructure

- [ ] **Task Queue Backend**: Celery + Redis vs Ray vs Custom?
- [ ] **Shared Storage**: NFS vs S3/MinIO vs GlusterFS?
- [ ] **Monitoring**: Prometheus + Grafana vs Custom dashboard?
- [ ] **Containerization**: Docker vs Kubernetes vs Bare metal?

### 13.2 Task Granularity

- [ ] **Fine-grained** (one target × one symbol) vs **Coarse-grained** (all targets per symbol)?
- [ ] **Model-level** (one model family) vs **Target-level** (all models per target)?

### 13.3 Resource Management

- [ ] **GPU Allocation**: Exclusive vs Shared?
- [ ] **CPU Allocation**: Per-task cores vs Shared pool?
- [ ] **Memory Limits**: Per-task limits vs Shared pool?

### 13.4 Data Distribution

- [ ] **Pre-stage** data on workers vs **On-demand** download?
- [ ] **Cache Strategy**: Shared cache vs Local cache?
- [ ] **Data Format**: Parquet vs HDF5 vs Arrow?

---

## 14. Related Documentation

- [Intelligence Layer Overview](../../03_technical/research/INTELLIGENCE_LAYER.md) - Current intelligence layer architecture
- [Continuous Integrated Learning System](CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md) - Adaptive learning system
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Current training workflow
- [Intelligent Trainer API](../../02_reference/api/INTELLIGENT_TRAINER_API.md) - API reference

---

## 15. Next Steps

1. **Review and Approve**: Get stakeholder approval on architecture
2. **Infrastructure Setup**: Set up Redis/Celery infrastructure
3. **Proof of Concept**: Implement Phase 1 (distributed target ranking)
4. **Benchmarking**: Measure performance improvements
5. **Iterate**: Refine based on results

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-08  
**Author**: AI Assistant (Auto)  
**Review Status**: Pending

