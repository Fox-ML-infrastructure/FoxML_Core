#!/usr/bin/env python3
"""
Fix feature_registry.yaml lag_bars values with correct minute/day-to-bar conversion.

Usage:
    python scripts/fix_feature_registry.py          # Dry run (preview changes)
    python scripts/fix_feature_registry.py --apply  # Apply changes (creates backup)
"""

import math
import re
import sys
from pathlib import Path

import yaml

INTERVAL_MINUTES = 5  # Canonical interval for this registry
AUTO_REJECT_MARKER = 'AUTO-REJECTED'


def bars_from_minutes(minutes: int) -> int:
    """Convert minutes to bars at the canonical interval."""
    return int(math.ceil(minutes / INTERVAL_MINUTES))


def bars_from_days(days: int) -> int:
    """Convert days to bars at the canonical interval (calendar days = 1440 min/day)."""
    return int(math.ceil((days * 1440) / INTERVAL_MINUTES))


# Priority 1: Calendar features (lag_bars=0 is correct, always unreject)
CALENDAR_FEATURES = {
    '_hour', '_day', '_month', '_weekday', '_quarter', '_year',
    'day_of_week', 'day_of_month', 'month_of_year', 'trading_day_of_month', 'hour_of_day'
}

# Priority 2: Complex indicators with manual lookbacks
COMPLEX_LOOKBACKS = {
    'macd_hist': 26, 'macd_signal': 26, 'macd_x_ret': 26, 'macd_x_vol': 26, 'macd_x_volume': 26,
    'ichimoku_chikou': 52, 'ichimoku_senkou_a': 52, 'ichimoku_senkou_b': 52,
    'mass_index': 25, 'awesome_oscillator': 34,
    'obv': 1, 'obv_ema': 20, 'cmf': 21, 'force_index': 13,
    'negative_volume_index': 1, 'price_volume_trend': 1,
    'vol_persistence': 20, 'liquidity_ratio': 20, 'turnover': 20,
    'trade_size_vol': 5, 'high_vol_frac': 1, 'market_impact': 1, 'price_impact': 1,
    'ease_of_movement': 14, 'fractal_low': 5, 'n_trades': 1,
    'day_x_volume': 1, 'stoch_x_vol': 14, 'stoch_x_ret': 14, 'bb_x_ret': 20,
    'adx_x_vol': 14, 'atr_x_vol': 14,
    'market_facilitation_index': 1,  # Single-bar derived indicator
}

# Regex patterns (in priority order)
VOL_RE = re.compile(r'^(realized_vol|rs_vol|yz_vol|gk_vol|parkinson_vol)_(\d+)$')
PERIOD_RE = re.compile(
    r'^(rsi|sma|ema|roc|cci|atr|adx|mfi|turnover|hull_ma|'
    r'stoch_[a-z]+|bb_[a-z]+|volume_[a-z_]+)_(\d+)$'
)
MIN_RE = re.compile(r'^(.+?)_(\d+)m$')
DAY_RE = re.compile(r'^(.+?)_(\d+)d$')


def is_auto_rejected(meta: dict) -> bool:
    """Check multiple fields for AUTO-REJECTED marker."""
    reason_text = " ".join(
        str(meta.get(k, "")) 
        for k in ("description", "rejected_reason", "notes", "reason")
    )
    return AUTO_REJECT_MARKER in reason_text


def compute_lag_bars(name: str) -> tuple:
    """Return (lag_bars, reason) or (None, '') if no rule matches."""
    
    # Priority 1: Calendar features handled separately
    if name in CALENDAR_FEATURES:
        return 0, 'calendar_metadata'
    
    # Priority 2: Complex overrides
    if name in COMPLEX_LOOKBACKS:
        return COMPLEX_LOOKBACKS[name], 'complex_override'
    
    # Priority 3: Volatility features
    m = VOL_RE.match(name)
    if m:
        return int(m.group(2)), 'volatility_window'
    
    # Priority 4: Period-in-name indicators
    m = PERIOD_RE.match(name)
    if m:
        return int(m.group(2)), 'indicator_period'
    
    # Priority 5: Duration suffix _Xm
    m = MIN_RE.match(name)
    if m:
        minutes = int(m.group(2))
        return bars_from_minutes(minutes), f'minutes_to_bars({minutes}m/{INTERVAL_MINUTES}m)'
    
    # Priority 6: Duration suffix _Xd
    m = DAY_RE.match(name)
    if m:
        days = int(m.group(2))
        return bars_from_days(days), f'days_to_bars({days}d*1440/{INTERVAL_MINUTES}m)'
    
    return None, ''


def fix_registry(registry_path: Path, dry_run: bool = True) -> dict:
    """Fix feature registry and return report."""
    
    # Create backup
    if not dry_run:
        backup_path = registry_path.with_suffix('.yaml.bak')
        backup_path.write_text(registry_path.read_text())
        print(f"Backup created: {backup_path}")
    
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    
    features = data.get('features', {})
    
    report = {
        'updated_lag': [],
        'unrejected': [],
        'still_zero': [],
        'unmatched': [],
        'calendar_fixed': [],
        'skipped_nonzero': [],
    }
    
    for name, meta in features.items():
        if not isinstance(meta, dict):
            continue
        
        current_lag = meta.get('lag_bars', 0)
        current_rejected = meta.get('rejected', False)
        auto_rejected = is_auto_rejected(meta)
        
        new_lag, reason = compute_lag_bars(name)
        
        # === CALENDAR FEATURES: Special handling ===
        if name in CALENDAR_FEATURES:
            meta['lag_bars'] = 0
            if current_rejected:  # Unconditionally unreject calendar features
                meta['rejected'] = False
                meta['description'] = 'Calendar/time metadata (lag_bars=0 is correct)'
                meta['source'] = 'metadata'
                report['calendar_fixed'].append(name)
            continue
        
        # === OTHER FEATURES ===
        if new_lag is not None:
            # Determine if we should update lag_bars
            # Only update if: current is 0 OR was auto-rejected
            should_update_lag = (current_lag == 0) or auto_rejected
            
            if should_update_lag and (current_lag != new_lag):
                meta['lag_bars'] = new_lag
                report['updated_lag'].append((name, current_lag, new_lag, reason))
                
                # Only unreject if: was auto-rejected AND we set a nonzero lag
                if auto_rejected and new_lag > 0:
                    meta['rejected'] = False
                    meta['description'] = f'Fixed: lag_bars={new_lag} ({reason})'
                    meta['source'] = 'derived'
                    report['unrejected'].append(name)
            elif current_lag != 0 and not auto_rejected:
                # Feature already has nonzero lag and wasn't auto-rejected - skip
                report['skipped_nonzero'].append((name, current_lag))
        else:
            # No rule matches
            if current_lag == 0 and auto_rejected:
                report['still_zero'].append(name)
            elif current_lag == 0:
                report['unmatched'].append(name)
    
    if not dry_run:
        with open(registry_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return report


def print_report(report: dict):
    print(f"\n{'='*60}")
    print("FEATURE REGISTRY FIX REPORT")
    print(f"{'='*60}\n")
    
    print(f"Updated lag_bars: {len(report['updated_lag'])}")
    for name, old, new, reason in report['updated_lag'][:30]:
        print(f"  {name}: {old} → {new} ({reason})")
    if len(report['updated_lag']) > 30:
        print(f"  ... and {len(report['updated_lag']) - 30} more")
    
    print(f"\nUnrejected (were AUTO-REJECTED, now fixed): {len(report['unrejected'])}")
    for name in report['unrejected'][:15]:
        print(f"  {name}")
    if len(report['unrejected']) > 15:
        print(f"  ... and {len(report['unrejected']) - 15} more")
    
    print(f"\nCalendar features fixed: {len(report['calendar_fixed'])}")
    for name in report['calendar_fixed']:
        print(f"  {name}")
    
    print(f"\nSkipped (already had nonzero lag): {len(report['skipped_nonzero'])}")
    for name, lag in report['skipped_nonzero'][:10]:
        print(f"  {name}: lag_bars={lag}")
    if len(report['skipped_nonzero']) > 10:
        print(f"  ... and {len(report['skipped_nonzero']) - 10} more")
    
    print(f"\nStill lag_bars=0 (no rule, still AUTO-REJECTED): {len(report['still_zero'])}")
    for name in report['still_zero']:
        print(f"  {name}")
    
    print(f"\nUnmatched (lag_bars=0, not rejected): {len(report['unmatched'])}")
    for name in report['unmatched']:
        print(f"  {name}")


def verify_fix(registry_path: Path) -> bool:
    """Verify no AUTO-REJECTED features remain with lag_bars > 0 and rejected=True."""
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    
    features = data.get('features', {})
    bad = []
    calendar_rejected = []
    
    for name, meta in features.items():
        if not isinstance(meta, dict):
            continue
        
        if is_auto_rejected(meta) and meta.get('rejected', False) and meta.get('lag_bars', 0) > 0:
            bad.append(name)
        
        if name in CALENDAR_FEATURES and meta.get('rejected', False):
            calendar_rejected.append(name)
    
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}\n")
    
    if bad:
        print(f"ERROR: {len(bad)} features are AUTO-REJECTED with lag > 0 but still rejected:")
        for name in bad:
            print(f"  {name}")
    else:
        print("✓ No AUTO-REJECTED features remain rejected with nonzero lag")
    
    if calendar_rejected:
        print(f"ERROR: {len(calendar_rejected)} calendar features are still rejected:")
        for name in calendar_rejected:
            print(f"  {name}")
    else:
        print("✓ All calendar features are unrejected")
    
    return len(bad) == 0 and len(calendar_rejected) == 0


if __name__ == '__main__':
    registry_path = Path('CONFIG/feature_registry.yaml')
    
    dry_run = '--apply' not in sys.argv
    
    if dry_run:
        print("DRY RUN MODE (use --apply to write changes)")
    else:
        print("APPLYING CHANGES to feature_registry.yaml")
    
    report = fix_registry(registry_path, dry_run=dry_run)
    print_report(report)
    
    if not dry_run:
        success = verify_fix(registry_path)
        if not success:
            print("\n⚠️  VERIFICATION FAILED - review output above")
            sys.exit(1)
        print("\n✅ Fix applied and verified successfully")
    else:
        print("\n[Dry run complete. Review above and run with --apply to commit changes.]")

