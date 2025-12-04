#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Model Switching Utility
Switch from daily models to intraday models when ready.
"""


import yaml
import os
from datetime import datetime

def switch_to_intraday(config_path: str = "IBKR_trading/config/ibkr_daily_test.yaml"):
    """
    Switch the IBKR configuration from daily to intraday models.
    
    Args:
        config_path: Path to the configuration file
    """
    print("🔄 Switching to intraday models...")
    
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model settings
    config['models']['daily']['enabled'] = False
    config['models']['intraday']['enabled'] = True
    config['model_switching']['current_mode'] = 'intraday'
    
    # Update rebalancing schedule for intraday
    config['rebalancing']['schedule'] = ["09:35", "10:30", "12:00", "14:30", "15:50"]
    
    # Update portfolio settings for intraday
    config['portfolio']['per_name_cap'] = 0.02  # 2% max per name for intraday
    config['rebalancing']['no_trade_threshold'] = 0.005  # 0.5% threshold for intraday
    
    # Update execution settings for intraday
    config['execution']['participation_cap'] = 0.07  # 7% of 1-min volume
    config['execution']['tif_seconds'] = 60  # 1 minute for intraday
    
    # Update rotation settings for intraday
    config['rebalancing']['rotation']['holding_period_min'] = 30  # 30 min minimum
    config['rebalancing']['rotation']['max_holding_days'] = 1  # 1 day maximum
    config['rebalancing']['rotation']['rotation_cost_threshold'] = 0.0005  # 5 bps
    
    # Update data settings for intraday
    config['data']['frequency'] = '5m'
    config['data']['lookback_days'] = 30  # 30 days of intraday data
    
    # Update features for intraday
    config['data']['features'] = [
        "returns", "volatility", "volume", "momentum", "mean_reversion", 
        "microstructure", "barrier_targets"
    ]
    
    # Update horizons
    config['models']['intraday']['horizons'] = [5, 10, 15, 30, 60]
    
    # Update performance settings for intraday
    config['performance']['batch_size'] = 5  # Smaller batches for intraday
    config['performance']['max_workers'] = 4  # More workers for intraday
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create backup of daily config
    backup_path = f"IBKR_trading/config/ibkr_daily_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration updated for intraday models!")
    print(f"📁 Daily config backed up to: {backup_path}")
    print("")
    print("🔧 Key changes made:")
    print("   • Model mode: daily → intraday")
    print("   • Horizons: [1] → [5, 10, 15, 30, 60]")
    print("   • Rebalancing: 2x daily → 5x intraday")
    print("   • Position caps: 5% → 2%")
    print("   • Data frequency: 1d → 5m")
    print("   • Features: added microstructure, barrier_targets")
    print("")
    print("🚀 Ready to use intraday models!")

def switch_back_to_daily(config_path: str = "IBKR_trading/config/ibkr_daily_test.yaml"):
    """
    Switch back to daily models if needed.
    
    Args:
        config_path: Path to the configuration file
    """
    print("🔄 Switching back to daily models...")
    
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Revert to daily settings
    config['models']['daily']['enabled'] = True
    config['models']['intraday']['enabled'] = False
    config['model_switching']['current_mode'] = 'daily'
    
    # Revert rebalancing schedule
    config['rebalancing']['schedule'] = ["09:35", "15:45"]
    
    # Revert portfolio settings
    config['portfolio']['per_name_cap'] = 0.05  # 5% max per name for daily
    config['rebalancing']['no_trade_threshold'] = 0.01  # 1% threshold for daily
    
    # Revert execution settings
    config['execution']['participation_cap'] = 0.10  # 10% of daily volume
    config['execution']['tif_seconds'] = 300  # 5 minutes for daily
    
    # Revert rotation settings
    config['rebalancing']['rotation']['holding_period_min'] = 60  # 1 hour minimum
    config['rebalancing']['rotation']['max_holding_days'] = 5  # 5 days maximum
    config['rebalancing']['rotation']['rotation_cost_threshold'] = 0.001  # 10 bps
    
    # Revert data settings
    config['data']['frequency'] = '1d'
    config['data']['lookback_days'] = 252  # 1 year of daily data
    
    # Revert features
    config['data']['features'] = [
        "returns", "volatility", "volume", "momentum", "mean_reversion"
    ]
    
    # Revert horizons
    config['models']['daily']['horizons'] = [1]
    
    # Revert performance settings
    config['performance']['batch_size'] = 10  # Larger batches for daily
    config['performance']['max_workers'] = 2  # Fewer workers for daily
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration reverted to daily models!")

def show_current_mode(config_path: str = "IBKR_trading/config/ibkr_daily_test.yaml"):
    """
    Show the current model mode.
    
    Args:
        config_path: Path to the configuration file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    current_mode = config['model_switching']['current_mode']
    daily_enabled = config['models']['daily']['enabled']
    intraday_enabled = config['models']['intraday']['enabled']
    
    print(f"📊 Current model mode: {current_mode}")
    print(f"   Daily models: {'✅ Enabled' if daily_enabled else '❌ Disabled'}")
    print(f"   Intraday models: {'✅ Enabled' if intraday_enabled else '❌ Disabled'}")
    
    if current_mode == 'daily':
        print("   Horizons: [1] (daily)")
        print("   Rebalancing: 2x daily (09:35, 15:45)")
    else:
        print("   Horizons: [5, 10, 15, 30, 60] (intraday)")
        print("   Rebalancing: 5x intraday (09:35, 10:30, 12:00, 14:30, 15:50)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "switch":
            switch_to_intraday()
        elif command == "revert":
            switch_back_to_daily()
        elif command == "status":
            show_current_mode()
        else:
            print("Usage: python switch_to_intraday.py [switch|revert|status]")
    else:
        print("🔄 IBKR Model Switching Utility")
        print("===============================")
        print("")
        print("Commands:")
        print("  python switch_to_intraday.py switch  - Switch to intraday models")
        print("  python switch_to_intraday.py revert  - Switch back to daily models")
        print("  python switch_to_intraday.py status  - Show current mode")
        print("")
        show_current_mode()
