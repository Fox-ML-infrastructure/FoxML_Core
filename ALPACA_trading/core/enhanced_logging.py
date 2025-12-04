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
Enhanced Logging System for Trading Bot
"""


import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better formatting."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # Add emoji based on level
        emoji_map = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨",
        }
        emoji = emoji_map.get(levelname, "")

        # Format the message
        formatted = super().format(record)
        if emoji:
            formatted = f"{emoji} {formatted}"

        return formatted


class TradingLogger:
    """Enhanced logging system for trading operations."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.log_dir / "trades").mkdir(exist_ok=True)
        (self.log_dir / "performance").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)
        (self.log_dir / "system").mkdir(exist_ok=True)

        self.setup_loggers()

    def setup_loggers(self):
        """Setup all loggers with proper formatting."""

        # Main logger
        self.main_logger = self._setup_logger(
            "trading_bot", self.log_dir / "trading_bot.log", level=logging.INFO
        )

        # Trade logger
        self.trade_logger = self._setup_logger(
            "trades",
            self.log_dir / "trades" / f"trades_{datetime.now().strftime('%Y-%m')}.log",
            level=logging.INFO,
        )

        # Performance logger
        self.performance_logger = self._setup_logger(
            "performance",
            self.log_dir / "performance" / f"performance_{datetime.now().strftime('%Y-%m')}.log",
            level=logging.INFO,
        )

        # Error logger
        self.error_logger = self._setup_logger(
            "errors",
            self.log_dir / "errors" / f"errors_{datetime.now().strftime('%Y-%m')}.log",
            level=logging.ERROR,
        )

        # System logger
        self.system_logger = self._setup_logger(
            "system",
            self.log_dir / "system" / f"system_{datetime.now().strftime('%Y-%m')}.log",
            level=logging.INFO,
        )

    def _setup_logger(self, name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
        """Setup individual logger with file and console handlers."""

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,  # 10MB
        )
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Formatters
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_trade(self, trade_data: dict):
        """Log trade execution with detailed information."""
        trade_info = {
            "timestamp": datetime.now().isoformat(),
            "symbol": trade_data.get("symbol"),
            "action": trade_data.get("action"),
            "size": trade_data.get("size"),
            "price": trade_data.get("price"),
            "value": trade_data.get("value"),
            "regime": trade_data.get("regime"),
            "confidence": trade_data.get("confidence"),
            "signal_strength": trade_data.get("signal_strength"),
        }

        # Log to trade file
        self.trade_logger.info(f"TRADE: {json.dumps(trade_info, indent=2)}")

        # Log to main file
        self.main_logger.info(
            f"💰 Trade executed: {trade_data.get('action')} {trade_data.get('size'):.2f} "
            f"{trade_data.get('symbol')} @ ${trade_data.get('price'):.2f} "
            f"(Value: ${trade_data.get('value'):,.0f})"
        )

    def log_performance(self, performance_data: dict):
        """Log performance metrics."""
        perf_info = {
            "timestamp": datetime.now().isoformat(),
            "total_return": performance_data.get("total_return"),
            "current_capital": performance_data.get("current_capital"),
            "sharpe_ratio": performance_data.get("sharpe_ratio"),
            "max_drawdown": performance_data.get("max_drawdown"),
            "total_trades": performance_data.get("total_trades"),
            "regime": performance_data.get("regime"),
            "regime_confidence": performance_data.get("regime_confidence"),
        }

        # Log to performance file
        self.performance_logger.info(f"PERFORMANCE: {json.dumps(perf_info, indent=2)}")

        # Log to main file
        self.main_logger.info(
            f"📊 Performance: Return: {performance_data.get('total_return', 0):+.2%}, "
            f"Capital: ${performance_data.get('current_capital', 0):,.0f}, "
            f"Sharpe: {performance_data.get('sharpe_ratio', 0):.2f}"
        )

    def log_regime_detection(self, regime_data: dict):
        """Log regime detection results."""
        regime_info = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime_data.get("regime"),
            "confidence": regime_data.get("confidence"),
            "indicators": regime_data.get("indicators", {}),
        }

        # Log to system file
        self.system_logger.info(f"REGIME: {json.dumps(regime_info, indent=2)}")

        # Log to main file
        self.main_logger.info(
            f"🎯 Regime detected: {regime_data.get('regime')} "
            f"(confidence: {regime_data.get('confidence', 0):.1%})"
        )

    def log_signal_generation(self, signal_data: dict):
        """Log signal generation results."""
        signal_info = {
            "timestamp": datetime.now().isoformat(),
            "strategy": signal_data.get("strategy"),
            "symbol": signal_data.get("symbol"),
            "signal_strength": signal_data.get("signal_strength"),
            "confidence": signal_data.get("confidence"),
            "features_used": signal_data.get("features_used", []),
        }

        # Log to system file
        self.system_logger.info(f"SIGNAL: {json.dumps(signal_info, indent=2)}")

        # Log to main file
        self.main_logger.info(
            f"📡 Signal generated: {signal_data.get('strategy')} -> "
            f"{signal_data.get('symbol')} (strength: {signal_data.get('signal_strength', 0):.3f})"
        )

    def log_error(self, error: str, context: str = "", exception: Exception | None = None):
        """Log errors with context."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_details": str(exception) if exception else None,
        }

        # Log to error file
        self.error_logger.error(f"ERROR: {json.dumps(error_info, indent=2)}")

        # Log to main file
        self.main_logger.error(f"❌ Error: {error} (Context: {context})")

    def log_system_startup(self, system_info: dict):
        """Log system startup information."""
        startup_info = {
            "timestamp": datetime.now().isoformat(),
            "initial_capital": system_info.get("initial_capital"),
            "strategies": system_info.get("strategies", []),
            "symbols": system_info.get("symbols", []),
            "config_file": system_info.get("config_file"),
        }

        # Log to system file
        self.system_logger.info(f"STARTUP: {json.dumps(startup_info, indent=2)}")

        # Log to main file
        self.main_logger.info(
            f"🚀 System started: ${system_info.get('initial_capital', 0):,.0f} capital, "
            f"{len(system_info.get('strategies', []))} strategies active"
        )

    def log_cron_execution(self, status: str, details: str = ""):
        """Log cron job execution."""
        cron_info = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details,
        }

        # Log to system file
        self.system_logger.info(f"CRON: {json.dumps(cron_info, indent=2)}")

        # Log to main file
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        self.main_logger.info(f"{status_emoji} Cron execution: {status} - {details}")

    def create_daily_summary(self, date: str) -> str:
        """Create a daily summary of all activities."""
        summary_file = self.log_dir / "daily_summaries" / f"summary_{date}.md"
        summary_file.parent.mkdir(exist_ok=True)

        # Read today's logs and create summary
        summary_content = f"""# Daily Trading Summary - {date}

## 📊 Performance Overview
- **Date**: {date}
- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📈 Trading Activity
- Check trade logs: `logs/trades/trades_{date[:7]}.log`

## 📊 Performance Metrics
- Check performance logs: `logs/performance/performance_{date[:7]}.log`

## 🔧 System Activity
- Check system logs: `logs/system/system_{date[:7]}.log`

## ❌ Errors (if any)
- Check error logs: `logs/errors/errors_{date[:7]}.log`

---
*Generated by Enhanced Trading System*
"""

        with open(summary_file, "w") as f:
            f.write(summary_content)

        return str(summary_file)
