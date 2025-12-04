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
Centralized Logging Manager
Reads from config/logging_config.yaml and provides consistent logging across all scripts.
"""


import logging
import logging.handlers
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class CentralLoggingManager:
    """Central logging manager that reads from config/logging_config.yaml"""
    
    def __init__(self, config_path: str = "config/logging_config.yaml"):
        """Initialize logging manager with config."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load logging configuration from YAML file."""
        if not self.config_path.exists():
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Failed to load logging config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'default': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'date_format': '%Y-%m-%d %H:%M:%S'
            },
            'file_settings': {
                'log_dir': 'logs',
                'max_size': '10MB',
                'backup_count': 5
            },
            'console_settings': {
                'enabled': True,
                'colored': True,
                'show_timestamps': True
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create log directory
        log_dir = Path(self.config['file_settings']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup third-party library logging
        self._setup_third_party_logging()
        
    def _setup_third_party_logging(self):
        """Setup logging for third-party libraries."""
        third_party_config = self.config.get('third_party', {})
        
        # XGBoost
        if 'xgboost' in third_party_config:
            logging.getLogger('xgboost').setLevel(
                getattr(logging, third_party_config['xgboost']['level'])
            )
        
        # LightGBM
        if 'lightgbm' in third_party_config:
            logging.getLogger('lightgbm').setLevel(
                getattr(logging, third_party_config['lightgbm']['level'])
            )
        
        # PyTorch
        if 'torch' in third_party_config:
            logging.getLogger('torch').setLevel(
                getattr(logging, third_party_config['torch']['level'])
            )
        
        # Pandas
        if 'pandas' in third_party_config:
            logging.getLogger('pandas').setLevel(
                getattr(logging, third_party_config['pandas']['level'])
            )
        
        # NumPy
        if 'numpy' in third_party_config:
            logging.getLogger('numpy').setLevel(
                getattr(logging, third_party_config['numpy']['level'])
            )
    
    def get_logger(self, name: str, script_type: str = None, 
                   environment: str = None) -> logging.Logger:
        """Get configured logger for script or module."""
        logger = logging.getLogger(name)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Get configuration
        config = self._get_logger_config(script_type, environment)
        
        # Setup formatter
        formatter = logging.Formatter(
            config['format'],
            datefmt=config.get('date_format', '%Y-%m-%d %H:%M:%S')
        )
        
        # Setup console handler
        if config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Setup file handler
        if config.get('file', True):
            file_handler = self._create_file_handler(name, config)
            if file_handler:
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        # Set level
        logger.setLevel(getattr(logging, config['level']))
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def _get_logger_config(self, script_type: str = None, 
                          environment: str = None) -> Dict[str, Any]:
        """Get logger configuration based on type and environment."""
        config = self.config['default'].copy()
        
        # Override with script-specific config
        if script_type and 'scripts' in self.config:
            script_config = self.config['scripts'].get(script_type, {})
            config.update(script_config)
        
        # Override with environment-specific config
        if environment and 'environments' in self.config:
            env_config = self.config['environments'].get(environment, {})
            config.update(env_config)
        
        return config
    
    def _create_file_handler(self, name: str, config: Dict[str, Any]) -> Optional[logging.Handler]:
        """Create file handler with rotation."""
        try:
            log_dir = Path(self.config['file_settings']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename
            filename = f"{name}.log"
            if 'file' in config and isinstance(config['file'], str):
                filename = config['file']
            
            file_path = log_dir / filename
            
            # Create rotating file handler
            max_size = self._parse_size(self.config['file_settings']['max_size'])
            backup_count = self.config['file_settings']['backup_count']
            
            handler = logging.handlers.RotatingFileHandler(
                file_path, maxBytes=max_size, backupCount=backup_count
            )
            
            return handler
            
        except Exception as e:
            print(f"Failed to create file handler: {e}")
            return None
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '10MB') to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def setup_script_logging(self, script_name: str, script_type: str = None,
                           environment: str = None) -> logging.Logger:
        """Setup logging for a script."""
        return self.get_logger(script_name, script_type, environment)
    
    def setup_module_logging(self, module_name: str) -> logging.Logger:
        """Setup logging for a module."""
        return self.get_logger(module_name)
    
    def get_environment(self) -> str:
        """Get current environment from environment variable."""
        return os.environ.get('ENVIRONMENT', 'development')
    
    def setup_structured_logging(self, logger: logging.Logger) -> None:
        """Setup structured logging (JSON format)."""
        if not self.config.get('structured', {}).get('enabled', False):
            return
        
        # Create JSON formatter
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                 'pathname', 'filename', 'module', 'lineno', 'funcName',
                                 'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                                 'processName', 'process', 'getMessage']:
                        log_entry[key] = value
                
                return json.dumps(log_entry)
        
        # Apply JSON formatter to all handlers
        for handler in logger.handlers:
            handler.setFormatter(JSONFormatter())
    
    def add_performance_logging(self, logger: logging.Logger) -> None:
        """Add performance logging capabilities."""
        if not self.config.get('performance', {}).get('enabled', False):
            return
        
        # Add performance logging methods
        def log_performance(func):
            def wrapper(*args, **kwargs):
                start_time = datetime.now()
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                if duration > self.config['performance']['slow_threshold']:
                    logger.warning(f"Slow operation: {func.__name__} took {duration:.2f}s")
                else:
                    logger.debug(f"Operation: {func.__name__} took {duration:.2f}s")
                
                return result
            return wrapper
        
        # Add to logger
        logger.performance_wrapper = log_performance

# Global logging manager instance
_logging_manager: Optional[CentralLoggingManager] = None

def init_logging_manager(config_path: str = "config/logging_config.yaml") -> CentralLoggingManager:
    """Initialize global logging manager."""
    global _logging_manager
    _logging_manager = CentralLoggingManager(config_path)
    return _logging_manager

def get_logging_manager() -> Optional[CentralLoggingManager]:
    """Get global logging manager instance."""
    return _logging_manager

def get_logger(name: str, script_type: str = None, environment: str = None) -> logging.Logger:
    """Get logger using global manager."""
    if _logging_manager is None:
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    return _logging_manager.get_logger(name, script_type, environment)

def setup_script_logging(script_name: str, script_type: str = None, 
                        environment: str = None) -> logging.Logger:
    """Setup logging for a script using global manager."""
    if _logging_manager is None:
        init_logging_manager()
    return _logging_manager.setup_script_logging(script_name, script_type, environment)

def setup_module_logging(module_name: str) -> logging.Logger:
    """Setup logging for a module using global manager."""
    if _logging_manager is None:
        init_logging_manager()
    return _logging_manager.setup_module_logging(module_name)
