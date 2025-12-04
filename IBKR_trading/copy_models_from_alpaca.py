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
Copy Models from Alpaca to IBKR
Copy your existing Alpaca models to the IBKR test environment.
"""


import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelCopier:
    """
    Copy models from Alpaca to IBKR test environment.
    """
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.alpaca_model_path = "models/"  # Your Alpaca model path
        self.ibkr_model_path = "IBKR_trading/models/daily_models/"
        
    def setup_logging(self) -> logging.Logger:
        """Setup logging for the copier."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # File handler
        fh = logging.FileHandler("logs/model_copy.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def find_alpaca_models(self) -> list:
        """Find all Alpaca models."""
        self.logger.info("🔍 Searching for Alpaca models...")
        
        models = []
        
        # Look for common model file patterns
        model_patterns = [
            "*.pkl", "*.joblib", "*.h5", "*.hdf5", "*.pt", "*.pth", "*.onnx",
            "*.json", "*.yaml", "*.yml"
        ]
        
        for pattern in model_patterns:
            model_files = list(Path(self.alpaca_model_path).rglob(pattern))
            models.extend(model_files)
        
        # Remove duplicates and sort
        models = list(set(models))
        models.sort()
        
        self.logger.info(f"Found {len(models)} model files")
        for model in models:
            self.logger.info(f"  📁 {model}")
        
        return models
    
    def create_ibkr_model_directory(self) -> bool:
        """Create IBKR model directory structure."""
        self.logger.info("📁 Creating IBKR model directory structure...")
        
        try:
            # Create main directory
            os.makedirs(self.ibkr_model_path, exist_ok=True)
            
            # Create subdirectories for different model types
            subdirs = [
                "momentum",
                "mean_reversion", 
                "volatility",
                "volume",
                "ensemble",
                "features"
            ]
            
            for subdir in subdirs:
                os.makedirs(os.path.join(self.ibkr_model_path, subdir), exist_ok=True)
                self.logger.info(f"  📁 Created: {subdir}/")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create IBKR model directory: {e}")
            return False
    
    def copy_model_file(self, source_path: Path, dest_path: Path) -> bool:
        """Copy a single model file."""
        try:
            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, dest_path)
            
            # Verify the copy
            if dest_path.exists() and dest_path.stat().st_size > 0:
                self.logger.info(f"  ✅ Copied: {source_path.name}")
                return True
            else:
                self.logger.error(f"  ❌ Copy failed: {source_path.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"  ❌ Copy error for {source_path.name}: {e}")
            return False
    
    def copy_models(self) -> dict:
        """Copy all models from Alpaca to IBKR."""
        self.logger.info("📋 Starting model copy process...")
        
        # Find Alpaca models
        alpaca_models = self.find_alpaca_models()
        
        if not alpaca_models:
            self.logger.warning("⚠️ No Alpaca models found")
            return {"copied": 0, "failed": 0, "total": 0}
        
        # Create IBKR directory structure
        if not self.create_ibkr_model_directory():
            return {"copied": 0, "failed": 0, "total": 0}
        
        # Copy models
        copied = 0
        failed = 0
        
        for model_path in alpaca_models:
            # Determine destination path
            relative_path = model_path.relative_to(Path(self.alpaca_model_path))
            dest_path = Path(self.ibkr_model_path) / relative_path
            
            # Copy the model
            if self.copy_model_file(model_path, dest_path):
                copied += 1
            else:
                failed += 1
        
        # Summary
        total = copied + failed
        self.logger.info(f"📊 Copy summary: {copied}/{total} models copied successfully")
        
        if failed > 0:
            self.logger.warning(f"⚠️ {failed} models failed to copy")
        
        return {"copied": copied, "failed": failed, "total": total}
    
    def validate_copied_models(self) -> bool:
        """Validate that copied models are accessible."""
        self.logger.info("🔍 Validating copied models...")
        
        try:
            # Check if IBKR model directory exists
            if not os.path.exists(self.ibkr_model_path):
                self.logger.error(f"❌ IBKR model directory not found: {self.ibkr_model_path}")
                return False
            
            # Count model files
            model_files = list(Path(self.ibkr_model_path).rglob("*"))
            model_files = [f for f in model_files if f.is_file() and f.suffix in ['.pkl', '.joblib', '.h5', '.hdf5', '.pt', '.pth', '.onnx', '.json', '.yaml', '.yml']]
            
            self.logger.info(f"✅ Found {len(model_files)} model files in IBKR directory")
            
            # Test loading a few models
            test_models = model_files[:3]  # Test first 3 models
            
            for model_file in test_models:
                try:
                    # Try to load the model (basic validation)
                    if model_file.suffix in ['.pkl', '.joblib']:
                        import pickle
                        with open(model_file, 'rb') as f:
                            pickle.load(f)
                    elif model_file.suffix in ['.json']:
                        import json
                        with open(model_file, 'r') as f:
                            json.load(f)
                    elif model_file.suffix in ['.yaml', '.yml']:
                        import yaml
                        with open(model_file, 'r') as f:
                            yaml.safe_load(f)
                    
                    self.logger.info(f"  ✅ {model_file.name} loads successfully")
                    
                except Exception as e:
                    self.logger.warning(f"  ⚠️ {model_file.name} load test failed: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            return False
    
    def create_model_manifest(self) -> bool:
        """Create a manifest of copied models."""
        self.logger.info("📝 Creating model manifest...")
        
        try:
            manifest_path = os.path.join(self.ibkr_model_path, "model_manifest.json")
            
            # Find all model files
            model_files = list(Path(self.ibkr_model_path).rglob("*"))
            model_files = [f for f in model_files if f.is_file()]
            
            # Create manifest
            manifest = {
                "copy_date": datetime.now().isoformat(),
                "source_path": self.alpaca_model_path,
                "destination_path": self.ibkr_model_path,
                "total_files": len(model_files),
                "models": []
            }
            
            for model_file in model_files:
                manifest["models"].append({
                    "name": model_file.name,
                    "path": str(model_file.relative_to(Path(self.ibkr_model_path))),
                    "size_bytes": model_file.stat().st_size,
                    "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })
            
            # Save manifest
            import json
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"✅ Model manifest created: {manifest_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create model manifest: {e}")
            return False
    
    def run_copy_process(self) -> dict:
        """Run the complete model copy process."""
        self.logger.info("🚀 Starting model copy process...")
        
        # Copy models
        copy_results = self.copy_models()
        
        # Validate copied models
        validation_success = self.validate_copied_models()
        
        # Create manifest
        manifest_success = self.create_model_manifest()
        
        # Final summary
        self.logger.info("📊 Model copy process completed!")
        self.logger.info(f"  📁 Copied: {copy_results['copied']} models")
        self.logger.info(f"  ❌ Failed: {copy_results['failed']} models")
        self.logger.info(f"  🔍 Validation: {'✅ Passed' if validation_success else '❌ Failed'}")
        self.logger.info(f"  📝 Manifest: {'✅ Created' if manifest_success else '❌ Failed'}")
        
        return {
            "copy_results": copy_results,
            "validation_success": validation_success,
            "manifest_success": manifest_success
        }

def main():
    """Main function to run model copying."""
    print("📋 Model Copying from Alpaca to IBKR")
    print("====================================")
    
    # Initialize copier
    copier = ModelCopier()
    
    # Run copy process
    results = copier.run_copy_process()
    
    # Show results
    if results["copy_results"]["copied"] > 0:
        print(f"\n✅ Successfully copied {results['copy_results']['copied']} models")
        print(f"📁 Models copied to: {copier.ibkr_model_path}")
    else:
        print("\n❌ No models were copied")
    
    if results["copy_results"]["failed"] > 0:
        print(f"⚠️ {results['copy_results']['failed']} models failed to copy")
    
    return results

if __name__ == "__main__":
    results = main()
