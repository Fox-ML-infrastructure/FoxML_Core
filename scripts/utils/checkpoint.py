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
Checkpoint Utility for Long-Running Ranking Scripts

Provides checkpoint/resume functionality for scripts that process items one-by-one.
Saves progress after each item to allow resuming from interruptions.

Usage:
    from scripts.utils.checkpoint import CheckpointManager
    
    checkpoint = CheckpointManager(
        checkpoint_file=output_dir / "checkpoint.json",
        item_key_fn=lambda item: item['name']  # Function to get unique key for item
    )
    
    # Load existing results
    completed = checkpoint.load_completed()
    
    # Process items, skipping completed ones
    for item in items:
        if item['name'] in completed:
            logger.info(f"Skipping {item['name']} (already completed)")
            continue
        
        # Process item
        result = process_item(item)
        
        # Save checkpoint after each item
        checkpoint.save_item(item['name'], result)
    
    # Get all results
    all_results = checkpoint.get_all_results()
"""


import json
import logging
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Set, List
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpointing for long-running item-by-item processing.
    
    Saves progress after each item is processed, allowing resumption from interruptions.
    """
    
    def __init__(
        self,
        checkpoint_file: Path,
        item_key_fn: Optional[Callable[[Any], str]] = None,
        auto_save: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_file: Path to checkpoint JSON file
            item_key_fn: Function to extract unique key from item (default: str(item))
            auto_save: Automatically save after each item (default: True)
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.item_key_fn = item_key_fn or (lambda x: str(x))
        self.auto_save = auto_save
        
        # Ensure checkpoint directory exists
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._completed_items: Dict[str, Any] = {}
        self._failed_items: Set[str] = set()
        self._metadata: Dict[str, Any] = {}
        
        # Load existing checkpoint
        self.load()
    
    def load(self) -> None:
        """Load checkpoint from file"""
        if not self.checkpoint_file.exists():
            logger.info(f"No existing checkpoint found at {self.checkpoint_file}")
            return
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            self._completed_items = data.get('completed_items', {})
            self._failed_items = set(data.get('failed_items', []))
            self._metadata = data.get('metadata', {})
            
            logger.info(
                f"Loaded checkpoint: {len(self._completed_items)} completed, "
                f"{len(self._failed_items)} failed"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            self._completed_items = {}
            self._failed_items = set()
            self._metadata = {}
    
    def save(self) -> None:
        """Save checkpoint to file"""
        try:
            data = {
                'completed_items': self._completed_items,
                'failed_items': list(self._failed_items),
                'metadata': {
                    **self._metadata,
                    'last_saved': time.time(),
                    'n_completed': len(self._completed_items),
                    'n_failed': len(self._failed_items)
                }
            }
            
            # Atomic write: write to temp file, then rename
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_serializer)
            
            temp_file.replace(self.checkpoint_file)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for dataclasses and numpy types"""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._json_serializer(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._json_serializer(item) for item in obj]
        else:
            return str(obj)
    
    def is_completed(self, item: Any) -> bool:
        """Check if item is already completed"""
        key = self.item_key_fn(item)
        return key in self._completed_items
    
    def is_failed(self, item: Any) -> bool:
        """Check if item previously failed"""
        key = self.item_key_fn(item)
        return key in self._failed_items
    
    def get_result(self, item: Any) -> Optional[Any]:
        """Get saved result for item"""
        key = self.item_key_fn(item)
        return self._completed_items.get(key)
    
    def save_item(self, item: Any, result: Any) -> None:
        """Save result for an item"""
        key = self.item_key_fn(item)
        self._completed_items[key] = result
        
        # Remove from failed if it succeeded
        self._failed_items.discard(key)
        
        if self.auto_save:
            self.save()
    
    def mark_failed(self, item: Any, error: Optional[str] = None) -> None:
        """Mark item as failed"""
        key = self.item_key_fn(item)
        self._failed_items.add(key)
        
        if error:
            logger.warning(f"Marked {key} as failed: {error}")
        
        if self.auto_save:
            self.save()
    
    def load_completed(self) -> Dict[str, Any]:
        """Get all completed items as dict"""
        return self._completed_items.copy()
    
    def get_all_results(self) -> List[Any]:
        """Get all results as a list (values only)"""
        return list(self._completed_items.values())
    
    def get_completed_keys(self) -> Set[str]:
        """Get set of completed item keys"""
        return set(self._completed_items.keys())
    
    def get_failed_keys(self) -> Set[str]:
        """Get set of failed item keys"""
        return self._failed_items.copy()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value"""
        self._metadata[key] = value
        if self.auto_save:
            self.save()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self._metadata.get(key, default)
    
    def clear_failed(self) -> None:
        """Clear failed items (useful for retry)"""
        self._failed_items.clear()
        if self.auto_save:
            self.save()
    
    def clear(self) -> None:
        """Clear all checkpoint data"""
        self._completed_items.clear()
        self._failed_items.clear()
        self._metadata.clear()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        logger.info("Cleared checkpoint")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get progress summary"""
        return {
            'n_completed': len(self._completed_items),
            'n_failed': len(self._failed_items),
            'checkpoint_file': str(self.checkpoint_file),
            'last_saved': self._metadata.get('last_saved'),
            **self._metadata
        }

