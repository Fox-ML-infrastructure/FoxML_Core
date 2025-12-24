"""
Minimal integration test for pipeline hooks.

Tests:
1. Hook registration and deduplication
2. Deterministic ordering
3. Error handling modes
4. Execution trace
5. Plugin loading

Run: pytest TRAINING/common/test_pipeline_hooks.py
"""

import pytest
from TRAINING.common.pipeline_hooks import (
    PipelineHooks,
    register_hook,
    load_plugins
)


def test_hook_registration():
    """Test basic hook registration."""
    PipelineHooks.reset()
    
    def hook1(context):
        return context
    
    PipelineHooks.register('test_point', hook1, priority=50)
    
    hooks = PipelineHooks.list_hooks('test_point')
    assert 'test_point' in hooks
    assert len(hooks['test_point']) == 1


def test_deduplication():
    """Test that duplicate registrations are prevented."""
    PipelineHooks.reset()
    
    def hook1(context):
        return context
    
    # Register twice
    PipelineHooks.register('test_point', hook1, priority=50)
    PipelineHooks.register('test_point', hook1, priority=50)
    
    hooks = PipelineHooks.list_hooks('test_point')
    assert len(hooks['test_point']) == 1  # Should only register once


def test_deterministic_ordering():
    """Test that hooks execute in deterministic order (same priority = registration order)."""
    PipelineHooks.reset()
    
    execution_order = []
    
    def hook1(context):
        execution_order.append(1)
        return context
    
    def hook2(context):
        execution_order.append(2)
        return context
    
    def hook3(context):
        execution_order.append(3)
        return context
    
    # Register with same priority (should use registration_index + stable_tiebreak)
    PipelineHooks.register('test_point', hook1, priority=100)
    PipelineHooks.register('test_point', hook2, priority=100)
    PipelineHooks.register('test_point', hook3, priority=100)
    
    # Execute
    context = {'test': True}
    PipelineHooks.execute('test_point', context)
    
    # Should execute in registration order (deterministic)
    assert execution_order == [1, 2, 3]
    
    # Verify order is stable across multiple executions
    execution_order2 = []
    hook1_2 = lambda ctx: (execution_order2.append(1), ctx)[1]
    hook2_2 = lambda ctx: (execution_order2.append(2), ctx)[1]
    hook3_2 = lambda ctx: (execution_order2.append(3), ctx)[1]
    
    PipelineHooks.reset()
    PipelineHooks.register('test_point', hook1_2, priority=100)
    PipelineHooks.register('test_point', hook2_2, priority=100)
    PipelineHooks.register('test_point', hook3_2, priority=100)
    
    context2 = {'test': True}
    PipelineHooks.execute('test_point', context2)
    
    # Same order (deterministic)
    assert execution_order2 == [1, 2, 3]


def test_priority_ordering():
    """Test that priority determines execution order."""
    PipelineHooks.reset()
    
    execution_order = []
    
    def hook1(context):
        execution_order.append(1)
        return context
    
    def hook2(context):
        execution_order.append(2)
        return context
    
    def hook3(context):
        execution_order.append(3)
        return context
    
    # Register with different priorities
    PipelineHooks.register('test_point', hook3, priority=200)  # Last
    PipelineHooks.register('test_point', hook1, priority=50)    # First
    PipelineHooks.register('test_point', hook2, priority=100)   # Middle
    
    # Execute
    context = {'test': True}
    PipelineHooks.execute('test_point', context)
    
    # Should execute in priority order
    assert execution_order == [1, 2, 3]


def test_error_handling_log_and_continue():
    """Test default error handling (log and continue)."""
    PipelineHooks.reset()
    PipelineHooks.set_error_mode("log_and_continue")
    
    def failing_hook(context):
        raise ValueError("Test error")
    
    def succeeding_hook(context):
        context['succeeded'] = True
        return context
    
    PipelineHooks.register('test_point', failing_hook, priority=50)
    PipelineHooks.register('test_point', succeeding_hook, priority=100)
    
    context = {'test': True}
    result = PipelineHooks.execute('test_point', context)
    
    # Should continue and execute succeeding hook
    assert result['succeeded'] is True


def test_error_handling_raise():
    """Test raise error mode."""
    PipelineHooks.reset()
    PipelineHooks.set_error_mode("raise")
    
    def failing_hook(context):
        raise ValueError("Test error")
    
    PipelineHooks.register('test_point', failing_hook, priority=50)
    
    context = {'test': True}
    
    with pytest.raises(ValueError, match="Test error"):
        PipelineHooks.execute('test_point', context)


def test_error_handling_disable_hook():
    """Test disable_hook error mode."""
    PipelineHooks.reset()
    PipelineHooks.set_error_mode("disable_hook")
    
    def failing_hook(context):
        raise ValueError("Test error")
    
    def succeeding_hook(context):
        context['succeeded'] = True
        return context
    
    PipelineHooks.register('test_point', failing_hook, priority=50)
    PipelineHooks.register('test_point', succeeding_hook, priority=100)
    
    context = {'test': True}
    
    # First execution: failing hook runs and gets disabled
    result1 = PipelineHooks.execute('test_point', context)
    assert result1['succeeded'] is True
    
    # Second execution: failing hook is skipped
    context2 = {'test': True}
    result2 = PipelineHooks.execute('test_point', context2)
    assert result2['succeeded'] is True
    
    # Verify hook was disabled
    stats = PipelineHooks.get_execution_stats()
    assert len(stats['disabled_hooks']) > 0


def test_execution_trace():
    """Test that execution trace is stored in context."""
    PipelineHooks.reset()
    
    def hook1(context):
        return context
    
    PipelineHooks.register('test_point', hook1, priority=50)
    
    context = {'test': True}
    result = PipelineHooks.execute('test_point', context)
    
    # Should have hook_traces
    assert 'hook_traces' in result
    assert len(result['hook_traces']) == 1
    
    trace = result['hook_traces'][0]
    assert trace['hook_point'] == 'test_point'
    assert len(trace['hooks_executed']) == 1
    assert trace['hooks_executed'][0]['status'] == 'success'


def test_context_modification():
    """Test that hooks can modify context."""
    PipelineHooks.reset()
    
    def hook1(context):
        context['modified'] = True
        return context
    
    PipelineHooks.register('test_point', hook1, priority=50)
    
    context = {'test': True}
    result = PipelineHooks.execute('test_point', context)
    
    assert result['modified'] is True


def test_decorator_registration():
    """Test @register_hook decorator."""
    PipelineHooks.reset()
    
    @register_hook('test_point', priority=50)
    def hook1(context):
        return context
    
    hooks = PipelineHooks.list_hooks('test_point')
    assert len(hooks['test_point']) == 1


def test_plugin_loading():
    """Test plugin loading (with non-existent module - should not fail)."""
    PipelineHooks.reset()
    
    # Try to load non-existent module (should not fail)
    result = load_plugins(['TRAINING.common.nonexistent_module'], fail_on_error=False)
    
    assert len(result['loaded']) == 0
    assert len(result['failed']) == 1


def test_determinism_same_config_same_order():
    """Test that same plugin list = same hook order (determinism)."""
    PipelineHooks.reset()
    
    execution_order = []
    
    def hook_a(context):
        execution_order.append('a')
        return context
    
    def hook_b(context):
        execution_order.append('b')
        return context
    
    # Simulate loading plugins in sorted order
    # (In real usage, load_plugins_from_config sorts automatically)
    from TRAINING.common.pipeline_hooks import load_plugins
    
    # First run: load in order
    PipelineHooks.reset()
    PipelineHooks.register('test_point', hook_a, priority=100)
    PipelineHooks.register('test_point', hook_b, priority=100)
    
    context1 = {'test': True}
    PipelineHooks.execute('test_point', context1)
    order1 = execution_order.copy()
    
    # Second run: same order
    execution_order.clear()
    PipelineHooks.reset()
    PipelineHooks.register('test_point', hook_a, priority=100)
    PipelineHooks.register('test_point', hook_b, priority=100)
    
    context2 = {'test': True}
    PipelineHooks.execute('test_point', context2)
    order2 = execution_order.copy()
    
    # Should be identical (deterministic)
    assert order1 == order2


def test_stable_tiebreak():
    """Test that stable_tiebreak (module+qualname) provides deterministic ordering."""
    PipelineHooks.reset()
    
    execution_order = []
    
    # Create hooks with same priority but different modules/names
    def hook_z(context):
        execution_order.append('z')
        return context
    
    def hook_a(context):
        execution_order.append('a')
        return context
    
    # Register with same priority
    PipelineHooks.register('test_point', hook_z, priority=100)
    PipelineHooks.register('test_point', hook_a, priority=100)
    
    context = {'test': True}
    PipelineHooks.execute('test_point', context)
    
    # Order should be stable (based on registration_index + stable_tiebreak)
    # hook_z registered first, so it should execute first
    assert execution_order == ['z', 'a']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
