# Avoid Long Runs

Best practices for avoiding long-running processes and performance issues.

## Overview

Long-running processes can cause:
- Memory leaks
- Performance degradation
- Resource exhaustion
- System instability

## Best Practices

### 1. Use Checkpoints

Save state regularly:

```python
# Save checkpoint every N iterations
if iteration % checkpoint_interval == 0:
    save_checkpoint(model, state)
```

### 2. Limit Iterations

Set maximum iterations:

```python
config = {
    "max_iterations": 1000,
    "early_stopping_rounds": 50
}
```

### 3. Monitor Resources

Track memory and CPU usage:

```python
import psutil

memory = psutil.virtual_memory()
if memory.percent > 80:
    logger.warning("High memory usage")
```

### 4. Use Streaming

For large datasets, use streaming:

```python
for batch in stream_data():
    process_batch(batch)
```

## Configuration

### Set Limits

```yaml
training:
  max_iterations: 1000
  early_stopping_rounds: 50
  checkpoint_interval: 100
```

### Memory Limits

```yaml
system:
  max_memory_gb: 8
  memory_warning_threshold: 0.8
```

## Monitoring

### Check Process Duration

```bash
ps aux | grep python
```

### Monitor Memory

```bash
htop
```

### Check Logs

```bash
tail -f logs/training.log | grep "memory\|duration"
```

## See Also

- [Performance Optimization](../implementation/PERFORMANCE_OPTIMIZATION.md) - Optimization tips
- [Training Optimization Guide](../../../TRAINING/TRAINING_OPTIMIZATION_GUIDE.md) - Training optimization

