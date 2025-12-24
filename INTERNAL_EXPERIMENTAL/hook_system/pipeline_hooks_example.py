"""
Example: How to add a new feature using the hook system

This shows how to add leakage detection/fixing (or any new feature) 
without refactoring the orchestrator.

STEP 1: Add hook points in orchestrator (one line each)
--------------------------------------------------------
In intelligent_trainer.py, add hook calls at key points:

    # Before target ranking
    context = PipelineHooks.execute('before_target_ranking', {
        'stage': 'target_ranking',
        'output_dir': self.output_dir,
        'symbols': self.symbols,
        'data_dir': self.data_dir
    })
    
    rankings = rank_targets(...)
    
    # After target ranking
    context = PipelineHooks.execute('after_target_ranking', {
        'stage': 'target_ranking',
        'rankings': rankings,
        'output_dir': self.output_dir
    })
    
    # Before feature selection
    context = PipelineHooks.execute('before_feature_selection', {
        'stage': 'feature_selection',
        'target': target,
        'output_dir': feature_output_dir
    })
    
    selected_features = select_features_for_target(...)
    
    # After feature selection
    context = PipelineHooks.execute('after_feature_selection', {
        'stage': 'feature_selection',
        'target': target,
        'selected_features': selected_features,
        'output_dir': feature_output_dir
    })


STEP 2: Create your new module (no orchestrator changes needed)
-----------------------------------------------------------------
Create TRAINING/common/my_new_feature.py:

    from TRAINING.common.pipeline_hooks import PipelineHooks, register_hook
    from typing import Dict, Any
    
    @register_hook('after_feature_selection', priority=50)
    def check_and_fix_leakage(context: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"
        Hook that runs after feature selection to detect/fix leakage.
        
        This hook is automatically called - no need to modify orchestrator!
        \"\"\"
        target = context.get('target')
        selected_features = context.get('selected_features', [])
        output_dir = context.get('output_dir')
        
        if not target or not selected_features:
            return context  # Skip if no data
        
        try:
            from TRAINING.common.leakage_auto_fixer import LeakageAutoFixer
            
            # Do your leakage detection/fixing
            fixer = LeakageAutoFixer()
            fixed_features, fix_info = fixer.auto_fix_leakage(
                target_name=target,
                feature_list=selected_features,
                data_dir=context.get('data_dir'),
                output_dir=output_dir
            )
            
            # Update context with fixed features
            context['selected_features'] = fixed_features
            context['leakage_fix_info'] = fix_info
            
            logger.info(f"✅ Leakage check completed for {target}: {len(fixed_features)} features")
            
        except ImportError:
            logger.debug("LeakageAutoFixer not available, skipping")
        except Exception as e:
            logger.warning(f"Leakage check failed (non-critical): {e}")
        
        return context  # Return modified context
    
    # Hook automatically registers when module is imported
    # Just import this module somewhere (e.g., in __init__.py) and it works!


STEP 3: Enable your module (one import)
----------------------------------------
In TRAINING/common/__init__.py:

    # Import hooks module to register hooks
    try:
        from TRAINING.common import my_new_feature  # Registers hooks automatically
    except ImportError:
        pass  # Module not available, hooks just won't fire


That's it! No orchestrator refactoring needed.
"""

# This is just documentation - see above for usage pattern
