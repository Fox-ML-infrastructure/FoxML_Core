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
Paper Trading CLI
Command-line interface for the paper trading system
"""


import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.engine.paper import PaperTradingEngine


def main():
    """Main function for running enhanced paper trading."""
    parser = argparse.ArgumentParser(description="Enhanced Paper Trading System")
    parser.add_argument("--daily", action="store_true", help="Run daily trading")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--setup-cron", action="store_true", help="Setup cron job")
    parser.add_argument("--cron", action="store_true", help="Running from cron job")
    parser.add_argument(
        "--config",
        type=str,
        default="config/enhanced_paper_trading_config.json",
        help="Configuration file path",
    )
    parser.add_argument("--profile", help="Profile configuration file path")
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode")

    args = parser.parse_args()

    try:
        # Initialize system
        system = PaperTradingEngine(args.config, args.profile)

        if args.daily or args.cron:
            # Run daily trading
            system.run_daily_trading()
            system.save_results()

            # Send cron notification if running from cron
            if args.cron and system.discord_notifier:
                system.discord_notifier.send_cron_notification(
                    "SUCCESS", "Daily trading completed successfully"
                )

        elif args.backtest:
            # Run backtest (implement if needed)
            print("Backtest functionality not yet implemented")

        elif args.setup_cron:
            # Setup cron job
            cron_command = f"0 9 * * 1-5 cd {os.getcwd()} && python {__file__} --cron"
            print("Add this to your crontab:")
            print(f"{cron_command}")
            print("\nTo edit crontab: crontab -e")

        else:
            # Run daily trading by default
            system.run_daily_trading()
            system.save_results()

    except Exception as e:
        # Send error notification
        if "system" in locals() and hasattr(system, "discord_notifier") and system.discord_notifier:
            system.discord_notifier.send_error_notification(str(e), "Daily trading execution")
        if "system" in locals() and hasattr(system, "trading_logger"):
            system.trading_logger.log_error(str(e), "Daily trading execution", e)
        raise


if __name__ == "__main__":
    main()
