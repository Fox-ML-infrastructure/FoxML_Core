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
Polars Optimization - Mega Script Integration
High-performance data processing using Polars for large-scale training.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional, List
import gc

logger = logging.getLogger(__name__)

# Try to import Polars
try:
    import polars as pl
    POLARS_AVAILABLE = True
    logger.info("✅ Polars available for high-performance processing")
except ImportError:
    POLARS_AVAILABLE = False
    logger.warning("⚠️ Polars not available, falling back to pandas")

class PolarsOptimizer:
    """High-performance data processing using Polars."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.use_polars = self.config.get('use_polars', POLARS_AVAILABLE)
        self.chunk_size = self.config.get('chunk_size', 1000000)  # 1M rows
        self.streaming = self.config.get('streaming', True)
        self.memory_mapping = self.config.get('memory_mapping', True)
        
    def convert_to_polars(self, df: pd.DataFrame) -> 'pl.DataFrame':
        """Convert pandas DataFrame to Polars DataFrame."""
        if not self.use_polars:
            return df
            
        try:
            # Convert to Polars with lazy evaluation
            pl_df = pl.from_pandas(df, rechunk=True)
            logger.info(f"✅ Converted to Polars: {pl_df.shape}")
            return pl_df
        except Exception as e:
            logger.warning(f"Polars conversion failed: {e}, using pandas")
            return df
    
    def optimize_dataframe(self, df: pd.DataFrame) -> 'pl.DataFrame':
        """Optimize DataFrame for processing (mega script approach)."""
        
        if not self.use_polars:
            return df
            
        logger.info(f"🔧 Optimizing DataFrame with Polars: {df.shape}")
        
        try:
            # Convert to Polars
            pl_df = self.convert_to_polars(df)
            
            # Optimize data types
            pl_df = self._optimize_dtypes(pl_df)
            
            # Optimize memory layout
            pl_df = pl_df.rechunk()
            
            logger.info(f"✅ DataFrame optimized: {pl_df.shape}")
            return pl_df
            
        except Exception as e:
            logger.warning(f"Polars optimization failed: {e}, using pandas")
            return df
    
    def _optimize_dtypes(self, df: 'pl.DataFrame') -> 'pl.DataFrame':
        """Optimize data types for memory efficiency."""
        
        # Convert float64 to float32 where possible
        for col in df.columns:
            if df[col].dtype == pl.Float64:
                # Check if values fit in float32
                if df[col].min() >= -3.4e38 and df[col].max() <= 3.4e38:
                    df = df.with_columns(pl.col(col).cast(pl.Float32))
        
        # Convert int64 to int32 where possible
        for col in df.columns:
            if df[col].dtype == pl.Int64:
                # Check if values fit in int32
                if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df = df.with_columns(pl.col(col).cast(pl.Int32))
        
        return df
    
    def process_large_dataset(self, df: pd.DataFrame, 
                            processing_func: callable,
                            chunk_size: int = None) -> pd.DataFrame:
        """Process large dataset in chunks (mega script approach)."""
        
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        if len(df) <= chunk_size:
            return processing_func(df)
        
        logger.info(f"🔧 Processing large dataset in chunks: {len(df)} rows, {chunk_size} per chunk")
        
        # Process in chunks
        results = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            processed_chunk = processing_func(chunk)
            results.append(processed_chunk)
            
            # Memory cleanup
            gc.collect()
            
            logger.info(f"Processed chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}")
        
        # Combine results
        result_df = pd.concat(results, ignore_index=True)
        logger.info(f"✅ Large dataset processing complete: {result_df.shape}")
        return result_df
    
    def streaming_processing(self, file_path: str, 
                           processing_func: callable,
                           chunk_size: int = None) -> pd.DataFrame:
        """Stream processing for very large files (mega script approach)."""
        
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        logger.info(f"🔧 Streaming processing: {file_path}")
        
        results = []
        chunk_count = 0
        
        try:
            # Stream processing with Polars
            if self.use_polars:
                for chunk in pl.scan_csv(file_path).iter_slices(chunk_size):
                    processed_chunk = processing_func(chunk.collect())
                    results.append(processed_chunk)
                    chunk_count += 1
                    
                    # Memory cleanup
                    gc.collect()
                    
                    logger.info(f"Processed streaming chunk {chunk_count}")
            else:
                # Fallback to pandas chunking
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    processed_chunk = processing_func(chunk)
                    results.append(processed_chunk)
                    chunk_count += 1
                    
                    # Memory cleanup
                    gc.collect()
                    
                    logger.info(f"Processed streaming chunk {chunk_count}")
        
        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            return pd.DataFrame()
        
        # Combine results
        if results:
            result_df = pd.concat(results, ignore_index=True)
            logger.info(f"✅ Streaming processing complete: {result_df.shape}")
            return result_df
        else:
            logger.warning("No results from streaming processing")
            return pd.DataFrame()
    
    def memory_efficient_join(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                            on: str, how: str = 'inner') -> pd.DataFrame:
        """Memory-efficient join operation."""
        
        if not self.use_polars:
            return df1.merge(df2, on=on, how=how)
        
        try:
            # Convert to Polars
            pl_df1 = self.convert_to_polars(df1)
            pl_df2 = self.convert_to_polars(df2)
            
            # Perform join
            result = pl_df1.join(pl_df2, on=on, how=how)
            
            # Convert back to pandas
            return result.to_pandas()
            
        except Exception as e:
            logger.warning(f"Polars join failed: {e}, using pandas")
            return df1.merge(df2, on=on, how=how)
    
    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get memory usage statistics."""
        
        memory_info = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'null_counts': df.isnull().sum().to_dict()
        }
        
        if self.use_polars:
            try:
                pl_df = self.convert_to_polars(df)
                memory_info['polars_memory_mb'] = pl_df.estimated_size() / 1024**2
            except Exception:
                pass
        
        return memory_info
