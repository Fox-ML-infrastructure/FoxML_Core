"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
Logging Configuration Utilities

Provides helper functions to access structured logging configuration
per module without scattering config lookups throughout the codebase.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# Resolve CONFIG directory
CONFIG_DIR = Path(__file__).resolve().parent


@dataclass
class ModuleLoggingConfig:
    """Per-module logging configuration"""
    level: str = "INFO"
    gpu_detail: bool = False
    cv_detail: bool = False
    edu_hints: bool = False
    detail: bool = False


@dataclass
class BackendLoggingConfig:
    """Backend library logging configuration"""
    native_verbosity: int = -1
    show_sparse_warnings: bool = True


class LoggingConfigManager:
    """Manages structured logging configuration"""
    
    _instance: Optional['LoggingConfigManager'] = None
    _config: Optional[Dict[str, Any]] = None
    _active_profile: Optional[str] = None
    
    def __init__(self, config_path: Optional[Path] = None, profile: Optional[str] = None):
        """Initialize logging config manager"""
        if config_path is None:
            config_path = CONFIG_DIR / "logging_config.yaml"
        
        self.config_path = config_path
        self._active_profile = profile or "default"
        self._load_config()
        self._apply_profile()
    
    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None, profile: Optional[str] = None) -> 'LoggingConfigManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls(config_path, profile)
        return cls._instance
    
    def _load_config(self):
        """Load logging configuration from YAML"""
        if not self.config_path.exists():
            self._config = self._get_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                self._config = data.get('logging', {})
        except Exception as e:
            logging.warning(f"Failed to load logging config from {self.config_path}: {e}, using defaults")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            'global_level': 'INFO',
            'modules': {},
            'backends': {
                'lightgbm': {'native_verbosity': -1, 'show_sparse_warnings': True},
                'xgboost': {'native_verbosity': 0, 'show_sparse_warnings': True},
                'tensorflow': {'native_verbosity': 1, 'show_sparse_warnings': True}
            },
            'profiles': {}
        }
    
    def _apply_profile(self):
        """Apply active profile to config"""
        if self._active_profile == "default" or not self._active_profile:
            return
        
        profiles = self._config.get('profiles', {})
        if self._active_profile not in profiles:
            logging.warning(f"Profile '{self._active_profile}' not found, using default")
            return
        
        profile = profiles[self._active_profile]
        
        # Merge profile into base config
        if 'global_level' in profile:
            self._config['global_level'] = profile['global_level']
        
        if 'modules' in profile:
            for module_name, module_overrides in profile['modules'].items():
                if module_name not in self._config.get('modules', {}):
                    self._config.setdefault('modules', {})[module_name] = {}
                self._config['modules'][module_name].update(module_overrides)
    
    def get_module_config(self, module_name: str) -> ModuleLoggingConfig:
        """Get module-specific logging configuration"""
        modules = self._config.get('modules', {})
        module_data = modules.get(module_name, {})
        
        return ModuleLoggingConfig(
            level=module_data.get('level', self._config.get('global_level', 'INFO')),
            gpu_detail=module_data.get('gpu_detail', False),
            cv_detail=module_data.get('cv_detail', False),
            edu_hints=module_data.get('edu_hints', False),
            detail=module_data.get('detail', False)
        )
    
    def get_backend_config(self, backend_name: str) -> BackendLoggingConfig:
        """Get backend-specific logging configuration"""
        backends = self._config.get('backends', {})
        backend_data = backends.get(backend_name, {})
        
        return BackendLoggingConfig(
            native_verbosity=backend_data.get('native_verbosity', -1),
            show_sparse_warnings=backend_data.get('show_sparse_warnings', True)
        )
    
    def get_global_level(self) -> str:
        """Get global logging level"""
        return self._config.get('global_level', 'INFO')


# Convenience functions for easy access
_logging_manager: Optional[LoggingConfigManager] = None


def init_logging_config(config_path: Optional[Path] = None, profile: Optional[str] = None):
    """Initialize logging configuration (call once at startup)"""
    global _logging_manager
    _logging_manager = LoggingConfigManager.get_instance(config_path, profile)
    
    # Set global logging level
    global_level = _logging_manager.get_global_level()
    logging.basicConfig(level=getattr(logging, global_level))
    
    return _logging_manager


def get_module_logging_config(module_name: str) -> ModuleLoggingConfig:
    """Get module logging config (lazy initialization if needed)"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingConfigManager.get_instance()
    return _logging_manager.get_module_config(module_name)


def get_backend_logging_config(backend_name: str) -> BackendLoggingConfig:
    """Get backend logging config (lazy initialization if needed)"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingConfigManager.get_instance()
    return _logging_manager.get_backend_config(backend_name)

