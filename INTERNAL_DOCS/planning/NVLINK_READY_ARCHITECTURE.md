# NVLink-Ready Architecture — Minimal Implementation Plan

**Status:** Design Phase  
**Owner:** Maintainer (Jennifer)  
**Scope:** FoxML Core GPU training infrastructure  
**Last Updated:** 2025-12-07

---

## Objective

Make FoxML Core **NVLink-ready** with **surgical, minimal changes**—not a 3-month yak shave. Think of it as "**NVLink-ready architecture**" rather than "full GB200 integration".

The goal is to enable future multi-GPU scaling without rewriting core logic when clients have NVLink-equipped systems.

---

## 1. What "NVLink-Ready" Actually Means

For FoxML, being NVLink-ready means:

1. **No hard-coded single-GPU assumptions**
   - No `cuda:0` baked anywhere
   - Everything goes through a *device allocation layer*

2. **A notion of "device groups"**
   - `cpu`
   - `single_gpu`
   - `nvlink_pod` (N GPUs in a shared high-bandwidth domain)

3. **Work is scheduled to device groups, not devices**
   - "Train these 120 models on the `nvlink_pod` group, max 4 per GPU"
   - "Run these 5 symbols on a single GPU group"

If we do just that, when a future client says "we have an 8x or 72x NVLink box", we're not rewriting core logic—only the allocator / scheduler.

---

## 2. Minimal Changes Required

### 2.1 GPU Topology & Group Abstraction

**Add to config system:**

```yaml
# CONFIG/training_config/gpu_config.yaml (or add to existing)

gpu:
  mode: "auto"        # auto | single | nvlink
  devices: [0, 1, 2, 3]   # logical GPU IDs
  max_models_per_gpu: 4
  allow_peer_access: true
  
  # NVLink-specific (optional, for future)
  nvlink:
    enabled: false  # Set to true when NVLink topology detected
    topology: "auto"  # auto | ring | mesh | custom
    bandwidth_test: false  # Run bandwidth test on startup
```

**Create device group abstraction:**

```python
# TRAINING/common/device_groups.py

from typing import List, Optional
from dataclasses import dataclass
import torch

@dataclass
class DeviceGroup:
    """Represents a group of devices (GPUs) that can work together."""
    ids: List[int]  # GPU IDs: [0], [0, 1, 2], etc.
    mode: str  # "single", "nvlink", "cpu"
    allow_peer_access: bool = True
    
    def primary(self) -> torch.device:
        """Get primary device (first GPU in group)."""
        if self.mode == "cpu":
            return torch.device("cpu")
        return torch.device(f"cuda:{self.ids[0]}")
    
    def all(self) -> List[torch.device]:
        """Get all devices in group."""
        if self.mode == "cpu":
            return [torch.device("cpu")]
        return [torch.device(f"cuda:{i}") for i in self.ids]
    
    def count(self) -> int:
        """Number of devices in group."""
        return len(self.ids)
    
    def is_multi_gpu(self) -> bool:
        """True if group has multiple GPUs."""
        return len(self.ids) > 1 and self.mode != "cpu"
    
    def __repr__(self) -> str:
        return f"DeviceGroup(ids={self.ids}, mode={self.mode})"


def create_device_group(
    mode: str = "auto",
    devices: Optional[List[int]] = None,
    allow_peer_access: bool = True
) -> DeviceGroup:
    """
    Create a device group from config.
    
    Args:
        mode: "auto" | "single" | "nvlink" | "cpu"
        devices: List of GPU IDs (None = auto-detect)
        allow_peer_access: Enable peer-to-peer access between GPUs
    
    Returns:
        DeviceGroup instance
    """
    if mode == "cpu":
        return DeviceGroup(ids=[], mode="cpu", allow_peer_access=False)
    
    # Auto-detect devices if not specified
    if devices is None:
        if not torch.cuda.is_available():
            return DeviceGroup(ids=[], mode="cpu", allow_peer_access=False)
        devices = list(range(torch.cuda.device_count()))
    
    if not devices:
        return DeviceGroup(ids=[], mode="cpu", allow_peer_access=False)
    
    # Determine mode if auto
    if mode == "auto":
        if len(devices) == 1:
            mode = "single"
        elif len(devices) > 1:
            # Check if NVLink is available (simplified check)
            # Full NVLink detection can be added later
            mode = "nvlink"  # Assume NVLink for multi-GPU for now
    
    # Enable peer access if multi-GPU
    if len(devices) > 1 and allow_peer_access:
        _enable_peer_access(devices)
    
    return DeviceGroup(ids=devices, mode=mode, allow_peer_access=allow_peer_access)


def _enable_peer_access(devices: List[int]) -> None:
    """Enable peer-to-peer access between GPUs."""
    import torch
    for i in devices:
        for j in devices:
            if i != j:
                try:
                    torch.cuda.set_device(i)
                    if torch.cuda.can_device_access_peer(i, j):
                        torch.cuda.device(i).__enter__()
                        # Peer access is enabled automatically in PyTorch
                except Exception:
                    pass  # Peer access not available
```

### 2.2 Centralize All Device Moves

**Create unified device transfer utility:**

```python
# TRAINING/common/device_utils.py

from typing import Any, Union, Dict, List
import torch
import numpy as np

def move_to_device(
    data: Any,
    device: Union[torch.device, str],
    non_blocking: bool = False
) -> Any:
    """
    Move data to device, handling dicts, lists, tensors, numpy arrays.
    
    This is the single point for all device transfers. Later, when we add
    NVLink optimizations, this is where we add:
    - non_blocking=True transfers
    - peer-to-peer copies (torch.cuda.comm.broadcast, etc.)
    - pinned-memory optimizations
    
    Args:
        data: Data to move (tensor, dict, list, numpy array, etc.)
        device: Target device
        non_blocking: Use non-blocking transfer (for async)
    
    Returns:
        Data on target device
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device, non_blocking=non_blocking)
    
    elif isinstance(data, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device, non_blocking) for item in data]
        return type(data)(moved)
    
    elif isinstance(data, (int, float, str, bool)):
        return data  # Scalars stay on CPU
    
    else:
        # Try to move if it has .to() method
        if hasattr(data, 'to'):
            return data.to(device, non_blocking=non_blocking)
        return data


def move_batch_to_device(
    batch: Union[Dict, List, torch.Tensor],
    device: Union[torch.device, str],
    non_blocking: bool = False
) -> Any:
    """
    Alias for move_to_device (for clarity in training code).
    
    Use this everywhere instead of scattered .cuda() / .to() calls.
    """
    return move_to_device(batch, device, non_blocking)
```

### 2.3 Multi-GPU Aware Scheduler

**Create job scheduler:**

```python
# TRAINING/common/job_scheduler.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .device_groups import DeviceGroup

@dataclass
class TrainingJob:
    """Represents a single training job."""
    job_id: str
    symbol: str
    target: str
    model_family: str
    priority: int = 0  # Higher = more important
    estimated_memory_mb: Optional[float] = None


class JobScheduler:
    """
    Schedules training jobs across device groups.
    
    For now, simple round-robin or load-balancing.
    Later, can add:
    - Memory-aware scheduling
    - Priority-based scheduling
    - Model family affinity (e.g., all LSTM on same GPU)
    """
    
    def __init__(self, device_group: DeviceGroup, max_models_per_gpu: int = 4):
        """
        Initialize scheduler.
        
        Args:
            device_group: Device group to schedule on
            max_models_per_gpu: Maximum concurrent models per GPU
        """
        self.device_group = device_group
        self.max_models_per_gpu = max_models_per_gpu
        self.job_counts = {gpu_id: 0 for gpu_id in device_group.ids}
    
    def schedule_jobs(
        self,
        jobs: List[TrainingJob],
        strategy: str = "round_robin"
    ) -> Dict[int, List[TrainingJob]]:
        """
        Schedule jobs across GPUs in device group.
        
        Args:
            jobs: List of jobs to schedule
            strategy: "round_robin" | "load_balance" | "affinity"
        
        Returns:
            Dict mapping GPU ID -> list of jobs assigned to that GPU
        """
        if not self.device_group.is_multi_gpu():
            # Single GPU: assign all jobs to GPU 0
            return {self.device_group.ids[0]: jobs}
        
        assignments: Dict[int, List[TrainingJob]] = {
            gpu_id: [] for gpu_id in self.device_group.ids
        }
        
        if strategy == "round_robin":
            for i, job in enumerate(jobs):
                gpu_id = self.device_group.ids[i % len(self.device_group.ids)]
                assignments[gpu_id].append(job)
        
        elif strategy == "load_balance":
            # Assign to GPU with fewest jobs
            for job in jobs:
                gpu_id = min(self.device_group.ids, key=lambda g: len(assignments[g]))
                assignments[gpu_id].append(job)
        
        elif strategy == "affinity":
            # Group by model family, assign families to GPUs
            # (Simplified for now)
            families = {}
            for job in jobs:
                if job.model_family not in families:
                    families[job.model_family] = []
                families[job.model_family].append(job)
            
            gpu_idx = 0
            for family_jobs in families.values():
                for job in family_jobs:
                    gpu_id = self.device_group.ids[gpu_idx % len(self.device_group.ids)]
                    assignments[gpu_id].append(job)
                    gpu_idx += 1
        
        # Enforce max_models_per_gpu limit
        for gpu_id, job_list in assignments.items():
            if len(job_list) > self.max_models_per_gpu:
                # Truncate (or could queue for later)
                assignments[gpu_id] = job_list[:self.max_models_per_gpu]
        
        return assignments
    
    def get_available_gpu(self) -> Optional[int]:
        """Get a GPU with available capacity."""
        for gpu_id in self.device_group.ids:
            if self.job_counts[gpu_id] < self.max_models_per_gpu:
                return gpu_id
        return None
```

### 2.4 GPU Topology Logging

**Add startup diagnostics:**

```python
# TRAINING/common/gpu_topology.py

import torch
import logging
from typing import Dict, Any
from .device_groups import DeviceGroup

logger = logging.getLogger(__name__)


def log_gpu_topology(device_group: DeviceGroup) -> Dict[str, Any]:
    """
    Log GPU topology information at startup.
    
    This helps see instantly how FoxML "sees" a client box.
    
    Returns:
        Dict with topology information
    """
    info = {
        "mode": device_group.mode,
        "device_count": device_group.count(),
        "devices": device_group.ids,
        "peer_access": device_group.allow_peer_access
    }
    
    if device_group.mode == "cpu":
        logger.info("🔧 Device Group: CPU only")
        return info
    
    logger.info(f"🔧 Device Group: {device_group.mode.upper()} mode")
    logger.info(f"   Devices: {device_group.ids}")
    
    for gpu_id in device_group.ids:
        torch.cuda.set_device(gpu_id)
        props = torch.cuda.get_device_properties(gpu_id)
        memory_gb = props.total_memory / (1024**3)
        logger.info(
            f"   GPU {gpu_id}: {props.name} "
            f"({memory_gb:.1f} GB, Compute {props.major}.{props.minor})"
        )
        info[f"gpu_{gpu_id}"] = {
            "name": props.name,
            "memory_gb": memory_gb,
            "compute_capability": f"{props.major}.{props.minor}"
        }
    
    # Check peer access if multi-GPU
    if device_group.is_multi_gpu():
        logger.info("   Checking peer-to-peer access...")
        for i in device_group.ids:
            for j in device_group.ids:
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    status = "✅" if can_access else "❌"
                    logger.info(f"   {status} GPU {i} → GPU {j}: {'Yes' if can_access else 'No'}")
                    info[f"peer_{i}_{j}"] = can_access
    
    return info
```

---

## 3. Integration Points

### 3.1 Update Model Trainers

**Replace hardcoded device usage:**

```python
# Before (BAD):
model = model.cuda()
data = data.cuda()

# After (GOOD):
from TRAINING.common.device_utils import move_to_device
from TRAINING.common.device_groups import DeviceGroup

device_group = create_device_group(mode="auto", devices=[0, 1, 2, 3])
device = device_group.primary()

model = move_to_device(model, device)
data = move_to_device(data, device)
```

### 3.2 Update Training Orchestrator

**Use scheduler for job distribution:**

```python
# In intelligent_trainer.py or train_with_strategies.py

from TRAINING.common.device_groups import create_device_group, log_gpu_topology
from TRAINING.common.job_scheduler import JobScheduler, TrainingJob

# Initialize device group from config
device_group = create_device_group(
    mode=config.get('gpu', {}).get('mode', 'auto'),
    devices=config.get('gpu', {}).get('devices'),
    allow_peer_access=config.get('gpu', {}).get('allow_peer_access', True)
)

# Log topology
log_gpu_topology(device_group)

# Create scheduler
scheduler = JobScheduler(
    device_group=device_group,
    max_models_per_gpu=config.get('gpu', {}).get('max_models_per_gpu', 4)
)

# Schedule jobs
jobs = [TrainingJob(...) for ... in targets]
assignments = scheduler.schedule_jobs(jobs, strategy="round_robin")

# Train on assigned GPUs
for gpu_id, gpu_jobs in assignments.items():
    torch.cuda.set_device(gpu_id)
    for job in gpu_jobs:
        train_model(job, device_group)
```

---

## 4. What NOT to Bother With Yet

**Don't overbuild:**

### ❌ No Full Model/Data Parallelism
- Just getting clean multi-GPU *scheduling* + device abstraction is enough for now
- Full parallelism can come later when there's actual demand

### ❌ No NCCL Plumbing or Custom Topology Files
- That's only worth doing when a real customer has a specific HW layout
- PyTorch/TensorFlow handle NCCL automatically for most cases

### ❌ No Premature MoE Engineering
- We can *treat* different model families/targets as "experts" for now
- Real MoE trickery can wait until there's a clear use case

### ❌ No Custom CUDA Kernels for Multi-GPU
- Use framework-provided multi-GPU support (PyTorch DDP, TensorFlow MirroredStrategy)
- Custom kernels are premature optimization

---

## 5. Concrete V1 NVLink-Ready Checklist

### Phase 1: Device Abstraction (Week 1)

- [ ] **Search repo for hardcoded device usage**
  - Find all `cuda:0`, `.cuda()`, `"cuda"` literals
  - Document locations in `docs/internal/planning/DEVICE_REFACTOR_AUDIT.md`

- [ ] **Implement GPU config block**
  - Add `gpu` section to `gpu_config.yaml` (or create new config)
  - Support: `mode`, `devices`, `max_models_per_gpu`, `allow_peer_access`

- [ ] **Create `DeviceGroup` abstraction**
  - `TRAINING/common/device_groups.py`
  - `create_device_group()` function
  - Support: `auto`, `single`, `nvlink`, `cpu` modes

- [ ] **Add GPU topology logging**
  - `TRAINING/common/gpu_topology.py`
  - `log_gpu_topology()` function
  - Log at startup in training orchestrator

### Phase 2: Device Transfer Centralization (Week 1-2)

- [ ] **Create `move_to_device()` utility**
  - `TRAINING/common/device_utils.py`
  - Handle: tensors, numpy arrays, dicts, lists, tuples
  - Support `non_blocking` parameter (for future async)

- [ ] **Replace all `.cuda()` / `.to("cuda")` calls**
  - Update model trainers
  - Update data loading/preprocessing
  - Update inference paths
  - Use `move_to_device()` everywhere

### Phase 3: Job Scheduler (Week 2)

- [ ] **Create `JobScheduler` class**
  - `TRAINING/common/job_scheduler.py`
  - `TrainingJob` dataclass
  - Support: `round_robin`, `load_balance`, `affinity` strategies

- [ ] **Integrate scheduler into training orchestrator**
  - Update `intelligent_trainer.py` or `train_with_strategies.py`
  - Schedule jobs across device group
  - Assign jobs to GPUs based on strategy

### Phase 4: Testing & Validation (Week 2-3)

- [ ] **Test on single GPU** (should work exactly as before)
- [ ] **Test on multi-GPU** (if available)
- [ ] **Verify no regressions** (all existing tests pass)
- [ ] **Document device group usage** in training tutorials

---

## 6. Success Criteria

**V1 is complete when:**

1. ✅ No hardcoded `cuda:0` or `.cuda()` calls in training code
2. ✅ All device usage goes through `DeviceGroup` and `move_to_device()`
3. ✅ Job scheduler can distribute work across multiple GPUs
4. ✅ GPU topology is logged at startup
5. ✅ Single-GPU behavior is unchanged (no regressions)
6. ✅ Multi-GPU works with simple round-robin scheduling

**Future (not in V1):**
- Full data parallelism
- Model parallelism
- NVLink bandwidth optimization
- Custom topology detection
- Advanced scheduling strategies

---

## 7. Implementation Notes

### Backward Compatibility

- **Default behavior:** If no config provided, auto-detect single GPU (current behavior)
- **Single GPU:** Works exactly as before, just goes through abstraction layer
- **No breaking changes:** All existing code continues to work

### Performance Impact

- **Overhead:** Minimal (just function call indirection)
- **Single GPU:** No performance change expected
- **Multi-GPU:** Should see speedup proportional to GPU count (for parallelizable workloads)

### Testing Strategy

1. **Unit tests:** DeviceGroup, move_to_device, scheduler
2. **Integration tests:** Full training run on single GPU (verify no regressions)
3. **Multi-GPU tests:** If hardware available, test job distribution
4. **Fallback tests:** CPU-only mode, no CUDA available

---

## 8. Future Enhancements (Post-V1)

Once V1 is complete and validated:

### Phase 2: NVLink Optimization
- Bandwidth testing and topology detection
- Peer-to-peer transfer optimization
- Non-blocking transfers with async execution

### Phase 3: Advanced Scheduling
- Memory-aware scheduling
- Model family affinity
- Dynamic load balancing
- Priority-based queuing

### Phase 4: Framework Integration
- PyTorch DDP integration
- TensorFlow MirroredStrategy support
- XGBoost/LightGBM multi-GPU support

---

## 9. References

- PyTorch Multi-GPU: https://pytorch.org/tutorials/beginner/dist_overview.html
- CUDA Peer-to-Peer: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access
- NVLink Architecture: https://www.nvidia.com/en-us/data-center/nvlink/

---

**Status:** Ready for implementation  
**Priority:** Medium (enables future scaling, no immediate requirement)  
**Estimated Effort:** 2-3 weeks for V1  
**Dependencies:** Current GPU training infrastructure

