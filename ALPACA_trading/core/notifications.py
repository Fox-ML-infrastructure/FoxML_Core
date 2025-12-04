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
Discord Notification System for Trading Alerts
"""


import logging
from dataclasses import dataclass
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig:
    """Discord webhook configuration."""

    webhook_url: str
    bot_name: str = "Trading Bot"
    bot_avatar: str = "https://cdn.discordapp.com/emojis/📈.png"
    enabled: bool = True


class DiscordNotifier:
    """Discord notification system for trading alerts."""

    def __init__(self, config: DiscordConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def send_message(self, content: str, embed: dict | None = None) -> bool:
        """Send a message to Discord."""
        if not self.config.enabled:
            return True

        try:
            payload = {
                "username": self.config.bot_name,
                "avatar_url": self.config.bot_avatar,
                "content": content,
            }

            if embed:
                payload["embeds"] = [embed]

            response = requests.post(self.config.webhook_url, json=payload, timeout=10)

            if response.status_code == 204:
                self.logger.info("Discord notification sent successfully")
                return True
            else:
                self.logger.error(f"Discord notification failed: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"Discord notification error: {e}")
            return False

    def send_startup_notification(self, system_info: dict) -> bool:
        """Send startup notification."""
        embed = {
            "title": "🚀 Trading System Started",
            "color": 0x00FF00,
            "fields": [
                {
                    "name": "💰 Initial Capital",
                    "value": f"${system_info.get('initial_capital', 0):,.0f}",
                    "inline": True,
                },
                {
                    "name": "📊 Strategies",
                    "value": f"{len(system_info.get('strategies', []))} active",
                    "inline": True,
                },
                {"name": "🎯 Target Return", "value": "65%+ annually", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Enhanced Trading System"},
        }

        return self.send_message("", embed)

    def send_trade_notification(self, trade_info: dict) -> bool:
        """Send trade execution notification."""
        action_emoji = "🟢" if trade_info.get("action") == "BUY" else "🔴"
        color = 0x00FF00 if trade_info.get("action") == "BUY" else 0xFF0000

        embed = {
            "title": f"{action_emoji} Trade Executed",
            "color": color,
            "fields": [
                {
                    "name": "Symbol",
                    "value": trade_info.get("symbol", "N/A"),
                    "inline": True,
                },
                {
                    "name": "Action",
                    "value": trade_info.get("action", "N/A"),
                    "inline": True,
                },
                {
                    "name": "Size",
                    "value": f"{trade_info.get('size', 0):.2f}",
                    "inline": True,
                },
                {
                    "name": "Price",
                    "value": f"${trade_info.get('price', 0):.2f}",
                    "inline": True,
                },
                {
                    "name": "Value",
                    "value": f"${trade_info.get('value', 0):,.0f}",
                    "inline": True,
                },
                {
                    "name": "Regime",
                    "value": f"{trade_info.get('regime', 'N/A')} ({trade_info.get('confidence', 0):.1%})",
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self.send_message("", embed)

    def send_daily_summary(self, summary: dict) -> bool:
        """Send daily performance summary."""
        total_return = summary.get("total_return", 0)
        color = 0x00FF00 if total_return >= 0 else 0xFF0000
        return_emoji = "📈" if total_return >= 0 else "📉"

        embed = {
            "title": f"{return_emoji} Daily Trading Summary",
            "color": color,
            "fields": [
                {
                    "name": "📊 Total Return",
                    "value": f"{total_return:+.2%}",
                    "inline": True,
                },
                {
                    "name": "💰 Current Capital",
                    "value": f"${summary.get('current_capital', 0):,.0f}",
                    "inline": True,
                },
                {
                    "name": "📈 Sharpe Ratio",
                    "value": f"{summary.get('sharpe_ratio', 0):.2f}",
                    "inline": True,
                },
                {
                    "name": "📉 Max Drawdown",
                    "value": f"{summary.get('max_drawdown', 0):.2%}",
                    "inline": True,
                },
                {
                    "name": "🔄 Total Trades",
                    "value": f"{summary.get('total_trades', 0)}",
                    "inline": True,
                },
                {
                    "name": "🎯 Regime",
                    "value": f"{summary.get('regime', 'N/A')} ({summary.get('regime_confidence', 0):.1%})",
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"},
        }

        return self.send_message("", embed)

    def send_cron_notification(self, status: str, details: str = "") -> bool:
        """Send cron job execution notification."""
        color = 0x00FF00 if status == "SUCCESS" else 0xFF0000
        status_emoji = "✅" if status == "SUCCESS" else "❌"

        embed = {
            "title": f"{status_emoji} Cron Job Execution",
            "color": color,
            "fields": [
                {"name": "Status", "value": status, "inline": True},
                {
                    "name": "Time",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "inline": True,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if details:
            embed["fields"].append({"name": "Details", "value": details, "inline": False})

        return self.send_message("", embed)

    def send_error_notification(self, error: str, context: str = "") -> bool:
        """Send error notification."""
        embed = {
            "title": "❌ Trading System Error",
            "color": 0xFF0000,
            "fields": [
                {
                    "name": "Error",
                    "value": error[:1000],  # Limit length
                    "inline": False,
                }
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if context:
            embed["fields"].append({"name": "Context", "value": context, "inline": False})

        return self.send_message("", embed)
