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
Logging Setup with Journald Support

Configures logging to send messages to:
1. Console (stdout/stderr)
2. Systemd journal (if available) - for monitoring over SSH
3. Optional file handler

Usage:
    from scripts.utils.logging_setup import setup_logging
    
    logger = setup_logging(script_name="rank_target_predictability")
    logger.info("This will appear in console and journald")
"""


import logging
import sys
from pathlib import Path
from typing import Optional

# Try to import systemd journal handler
try:
    from systemd import journal
    JOURNALD_AVAILABLE = True
except ImportError:
    JOURNALD_AVAILABLE = False
    try:
        # Alternative: cysystemd
        import cysystemd.journal as journal
        JOURNALD_AVAILABLE = True
    except ImportError:
        JOURNALD_AVAILABLE = False


class JournaldHandler(logging.Handler):
    """Logging handler that sends messages to systemd journal"""
    
    def __init__(self, level=logging.NOTSET, identifier=None):
        super().__init__(level)
        if not JOURNALD_AVAILABLE:
            raise ImportError("systemd journal not available")
        self.identifier = identifier
    
    def emit(self, record):
        """Send log record to journald"""
        try:
            msg = self.format(record)
            
            # Map Python log levels to journald priority
            priority_map = {
                logging.DEBUG: journal.LOG_DEBUG,
                logging.INFO: journal.LOG_INFO,
                logging.WARNING: journal.LOG_WARNING,
                logging.ERROR: journal.LOG_ERR,
                logging.CRITICAL: journal.LOG_CRIT,
            }
            priority = priority_map.get(record.levelno, journal.LOG_INFO)
            
            # Prepare extra fields
            extra_fields = {
                'CODE_FILE': record.pathname,
                'CODE_LINE': str(record.lineno),
                'CODE_FUNC': record.funcName,
            }
            if self.identifier:
                extra_fields['SYSLOG_IDENTIFIER'] = self.identifier
            
            # Send to journal
            journal.send(
                msg,
                priority=priority,
                **extra_fields
            )
        except Exception:
            # Don't let journald errors break logging
            self.handleError(record)


def setup_logging(
    script_name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    use_journald: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging with console, journald, and optional file handlers.
    
    Args:
        script_name: Name of the script (for logger name)
        level: Logging level (default: INFO)
        log_file: Optional file path for file logging
        use_journald: Whether to use journald (default: True)
        format_string: Custom format string (default: includes timestamp, level, message)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Journald handler (if available and requested)
    if use_journald and JOURNALD_AVAILABLE:
        try:
            journald_handler = JournaldHandler(level=level, identifier=script_name)
            # Journald handles formatting internally, but we can still format
            journald_handler.setFormatter(formatter)
            logger.addHandler(journald_handler)
            # Use a basic handler to log this message (avoid recursion)
            temp_handler = logging.StreamHandler(sys.stdout)
            temp_handler.setFormatter(formatter)
            temp_logger = logging.getLogger(f"{script_name}.setup")
            temp_logger.addHandler(temp_handler)
            temp_logger.setLevel(level)
            temp_logger.info(f"Journald logging enabled for {script_name}")
            temp_logger.removeHandler(temp_handler)
        except Exception as e:
            # Use basic logging for this warning
            logging.basicConfig(level=level, format=format_string)
            logging.warning(f"Failed to enable journald logging: {e}")
    elif use_journald and not JOURNALD_AVAILABLE:
        # Silent fallback - journald not available
        pass
    
    # File handler (if requested)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"File logging enabled: {log_file}")
    
    return logger

