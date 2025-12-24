# MIT License - see LICENSE file

"""
Logging Setup with Journald Support

Configures logging to send messages to:
1. Console (stdout/stderr)
2. Systemd journal (if available) - for monitoring over SSH
3. Optional file handler

Usage:
    from TRAINING.orchestration.utils.logging_setup import setup_logging
    
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
    
    # Prevent propagation to root logger to avoid duplicate log lines
    # Root logger should not have handlers - only module loggers should
    logger.propagate = False
    
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


def enable_run_logging(
    output_dir: Path,
    log_filename: str = "run.log",
    level: int = logging.INFO,
    also_capture_stdout: bool = True
) -> Optional[Path]:
    """
    Enable persistent file logging for a training run.
    
    Creates a log file in the output directory that captures all log messages.
    Optionally also redirects stdout/stderr to the file.
    
    Args:
        output_dir: Run output directory (e.g., RESULTS/runs/...)
        log_filename: Name of the log file (default: run.log)
        level: Logging level (default: INFO)
        also_capture_stdout: If True, also redirect print() and stderr to log file
    
    Returns:
        Path to the log file, or None if setup failed
    
    Example:
        from TRAINING.orchestration.utils.logging_setup import enable_run_logging
        
        log_path = enable_run_logging(output_dir)
        # Now all logging goes to both console and {output_dir}/logs/run.log
    """
    try:
        # Create logs directory
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / log_filename
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add to root logger so ALL modules are captured
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        # Ensure root logger level is at least as permissive
        if root_logger.level == 0 or root_logger.level > level:
            root_logger.setLevel(level)
        
        # Optionally redirect stdout/stderr to also go to file
        if also_capture_stdout:
            # Create a tee-like wrapper for stdout
            class TeeOutput:
                def __init__(self, original, log_file_handle):
                    self.original = original
                    self.log_file_handle = log_file_handle
                
                def write(self, message):
                    self.original.write(message)
                    if message.strip():  # Don't log empty lines
                        self.log_file_handle.write(message)
                        self.log_file_handle.flush()
                
                def flush(self):
                    self.original.flush()
                    self.log_file_handle.flush()
                
                def isatty(self):
                    return self.original.isatty()
            
            # Open log file for stdout/stderr capture (append mode since handler uses 'w')
            stdout_log = open(log_file, 'a')
            sys.stdout = TeeOutput(sys.__stdout__, stdout_log)
            sys.stderr = TeeOutput(sys.__stderr__, stdout_log)
        
        logging.getLogger(__name__).info(f"📋 Run logging enabled: {log_file}")
        
        return log_file
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to enable run logging: {e}")
        return None

