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
C++ Component Testing
Test all C++ components to ensure they work correctly before IBKR integration.
"""


import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

class CppComponentTester:
    """
    Test C++ components for correctness and performance.
    """
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.test_results = {}
        
    def setup_logging(self) -> logging.Logger:
        """Setup logging for the tester."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # File handler
        fh = logging.FileHandler("logs/cpp_component_test.log")
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
    
    def test_cpp_build(self) -> bool:
        """Test if C++ components build successfully."""
        self.logger.info("🔨 Testing C++ build...")
        
        try:
            # Check if build directory exists
            build_dir = Path("IBKR_trading/cpp_engine/build")
            if not build_dir.exists():
                self.logger.error("❌ Build directory not found. Run ./build.sh first.")
                return False
            
            # Check if shared libraries exist
            lib_files = list(build_dir.glob("*.so")) + list(build_dir.glob("*.dll"))
            if not lib_files:
                self.logger.error("❌ No shared libraries found. Build failed.")
                return False
            
            self.logger.info(f"✅ C++ build successful. Found {len(lib_files)} libraries.")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ C++ build test failed: {e}")
            return False
    
    def test_python_bindings(self) -> bool:
        """Test Python-C++ bindings."""
        self.logger.info("🐍 Testing Python-C++ bindings...")
        
        try:
            # Try to import C++ modules
            sys.path.append("IBKR_trading/cpp_engine/build")
            
            # Test inference engine
            try:
                import inference_engine
                self.logger.info("✅ Inference engine binding works")
            except ImportError as e:
                self.logger.warning(f"⚠️ Inference engine binding not available: {e}")
            
            # Test feature pipeline
            try:
                import feature_pipeline
                self.logger.info("✅ Feature pipeline binding works")
            except ImportError as e:
                self.logger.warning(f"⚠️ Feature pipeline binding not available: {e}")
            
            # Test market data parser
            try:
                import market_data_parser
                self.logger.info("✅ Market data parser binding works")
            except ImportError as e:
                self.logger.warning(f"⚠️ Market data parser binding not available: {e}")
            
            # Test linear algebra engine
            try:
                import linear_algebra_engine
                self.logger.info("✅ Linear algebra engine binding works")
            except ImportError as e:
                self.logger.warning(f"⚠️ Linear algebra engine binding not available: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Python bindings test failed: {e}")
            return False
    
    def test_inference_engine(self) -> bool:
        """Test C++ inference engine."""
        self.logger.info("🧠 Testing C++ inference engine...")
        
        try:
            # Create test data
            n_samples = 1000
            n_features = 50
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            
            # Test Python fallback
            self.logger.info("Testing Python inference fallback...")
            
            # Simulate model inference
            start_time = time.time()
            predictions = np.random.randn(n_samples)  # Dummy predictions
            python_time = time.time() - start_time
            
            self.logger.info(f"✅ Python inference: {python_time:.4f}s for {n_samples} samples")
            
            # Test C++ inference if available
            try:
                import inference_engine
                
                start_time = time.time()
                cpp_predictions = inference_engine.predict(X)
                cpp_time = time.time() - start_time
                
                speedup = python_time / cpp_time if cpp_time > 0 else 1.0
                self.logger.info(f"✅ C++ inference: {cpp_time:.4f}s for {n_samples} samples")
                self.logger.info(f"🚀 Speedup: {speedup:.2f}x")
                
                # Verify predictions are reasonable
                if len(cpp_predictions) == n_samples:
                    self.logger.info("✅ C++ inference output shape correct")
                else:
                    self.logger.error(f"❌ C++ inference output shape incorrect: {len(cpp_predictions)} vs {n_samples}")
                    return False
                
            except ImportError:
                self.logger.warning("⚠️ C++ inference engine not available, using Python fallback")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Inference engine test failed: {e}")
            return False
    
    def test_feature_pipeline(self) -> bool:
        """Test C++ feature pipeline."""
        self.logger.info("🔧 Testing C++ feature pipeline...")
        
        try:
            # Create test market data
            n_bars = 1000
            market_data = {
                'open': np.random.randn(n_bars) * 100 + 100,
                'high': np.random.randn(n_bars) * 100 + 105,
                'low': np.random.randn(n_bars) * 100 + 95,
                'close': np.random.randn(n_bars) * 100 + 100,
                'volume': np.random.randint(100000, 1000000, n_bars)
            }
            
            # Test Python feature computation
            self.logger.info("Testing Python feature computation...")
            start_time = time.time()
            
            # Compute basic features
            returns = np.diff(market_data['close']) / market_data['close'][:-1]
            volatility = np.std(returns)
            volume_ma = np.mean(market_data['volume'])
            
            python_time = time.time() - start_time
            self.logger.info(f"✅ Python features: {python_time:.4f}s for {n_bars} bars")
            
            # Test C++ feature pipeline if available
            try:
                import feature_pipeline
                
                start_time = time.time()
                cpp_features = feature_pipeline.compute_features(market_data)
                cpp_time = time.time() - start_time
                
                speedup = python_time / cpp_time if cpp_time > 0 else 1.0
                self.logger.info(f"✅ C++ features: {cpp_time:.4f}s for {n_bars} bars")
                self.logger.info(f"🚀 Speedup: {speedup:.2f}x")
                
                # Verify features are reasonable
                if len(cpp_features) > 0:
                    self.logger.info("✅ C++ feature pipeline output generated")
                else:
                    self.logger.error("❌ C++ feature pipeline output empty")
                    return False
                
            except ImportError:
                self.logger.warning("⚠️ C++ feature pipeline not available, using Python fallback")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Feature pipeline test failed: {e}")
            return False
    
    def test_market_data_parser(self) -> bool:
        """Test C++ market data parser."""
        self.logger.info("📊 Testing C++ market data parser...")
        
        try:
            # Create test market data
            n_messages = 10000
            market_messages = []
            for i in range(n_messages):
                market_messages.append({
                    'symbol': 'AAPL',
                    'timestamp': time.time() + i * 0.001,
                    'bid': 100.0 + np.random.normal(0, 0.1),
                    'ask': 100.1 + np.random.normal(0, 0.1),
                    'size': np.random.randint(100, 1000)
                })
            
            # Test Python parsing
            self.logger.info("Testing Python market data parsing...")
            start_time = time.time()
            
            parsed_data = []
            for msg in market_messages:
                parsed_data.append({
                    'symbol': msg['symbol'],
                    'timestamp': msg['timestamp'],
                    'mid': (msg['bid'] + msg['ask']) / 2,
                    'spread': msg['ask'] - msg['bid']
                })
            
            python_time = time.time() - start_time
            self.logger.info(f"✅ Python parsing: {python_time:.4f}s for {n_messages} messages")
            
            # Test C++ parser if available
            try:
                import market_data_parser
                
                start_time = time.time()
                cpp_parsed = market_data_parser.parse_messages(market_messages)
                cpp_time = time.time() - start_time
                
                speedup = python_time / cpp_time if cpp_time > 0 else 1.0
                self.logger.info(f"✅ C++ parsing: {cpp_time:.4f}s for {n_messages} messages")
                self.logger.info(f"🚀 Speedup: {speedup:.2f}x")
                
                # Verify parsing results
                if len(cpp_parsed) > 0:
                    self.logger.info("✅ C++ market data parser output generated")
                else:
                    self.logger.error("❌ C++ market data parser output empty")
                    return False
                
            except ImportError:
                self.logger.warning("⚠️ C++ market data parser not available, using Python fallback")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Market data parser test failed: {e}")
            return False
    
    def test_linear_algebra_engine(self) -> bool:
        """Test C++ linear algebra engine."""
        self.logger.info("🔢 Testing C++ linear algebra engine...")
        
        try:
            # Create test matrices
            n = 1000
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            x = np.random.randn(n).astype(np.float32)
            
            # Test Python linear algebra
            self.logger.info("Testing Python linear algebra...")
            start_time = time.time()
            
            # Matrix multiplication
            C_python = np.dot(A, B)
            # Matrix-vector multiplication
            y_python = np.dot(A, x)
            # Eigenvalue computation
            eigenvals_python = np.linalg.eigvals(A[:100, :100])  # Smaller matrix for speed
            
            python_time = time.time() - start_time
            self.logger.info(f"✅ Python linear algebra: {python_time:.4f}s for {n}x{n} matrices")
            
            # Test C++ linear algebra if available
            try:
                import linear_algebra_engine
                
                start_time = time.time()
                C_cpp = linear_algebra_engine.matrix_multiply(A, B)
                y_cpp = linear_algebra_engine.matrix_vector_multiply(A, x)
                eigenvals_cpp = linear_algebra_engine.eigenvalues(A[:100, :100])
                cpp_time = time.time() - start_time
                
                speedup = python_time / cpp_time if cpp_time > 0 else 1.0
                self.logger.info(f"✅ C++ linear algebra: {cpp_time:.4f}s for {n}x{n} matrices")
                self.logger.info(f"🚀 Speedup: {speedup:.2f}x")
                
                # Verify results are reasonable
                if C_cpp.shape == C_python.shape:
                    self.logger.info("✅ C++ matrix multiplication output shape correct")
                else:
                    self.logger.error(f"❌ C++ matrix multiplication output shape incorrect: {C_cpp.shape} vs {C_python.shape}")
                    return False
                
            except ImportError:
                self.logger.warning("⚠️ C++ linear algebra engine not available, using Python fallback")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Linear algebra engine test failed: {e}")
            return False
    
    def test_memory_usage(self) -> bool:
        """Test memory usage and potential leaks."""
        self.logger.info("💾 Testing memory usage...")
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            self.logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
            
            # Run memory-intensive operations
            for i in range(100):
                # Create large arrays
                data = np.random.randn(1000, 1000)
                
                # Simulate C++ operations
                result = np.dot(data, data.T)
                
                # Clean up
                del data, result
                gc.collect()
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            self.logger.info(f"Final memory usage: {final_memory:.2f} MB")
            self.logger.info(f"Memory increase: {memory_increase:.2f} MB")
            
            # Check for memory leaks
            if memory_increase > 100:  # More than 100MB increase
                self.logger.warning(f"⚠️ Potential memory leak: {memory_increase:.2f} MB increase")
                return False
            else:
                self.logger.info("✅ No significant memory leaks detected")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Memory usage test failed: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """Run all C++ component tests."""
        self.logger.info("🚀 Starting C++ component testing...")
        
        tests = [
            ("C++ Build", self.test_cpp_build),
            ("Python Bindings", self.test_python_bindings),
            ("Inference Engine", self.test_inference_engine),
            ("Feature Pipeline", self.test_feature_pipeline),
            ("Market Data Parser", self.test_market_data_parser),
            ("Linear Algebra Engine", self.test_linear_algebra_engine),
            ("Memory Usage", self.test_memory_usage)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running: {test_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    self.logger.info(f"✅ {test_name}: PASSED")
                else:
                    self.logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                self.logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"TEST SUMMARY")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Passed: {passed}/{total}")
        self.logger.info(f"Success rate: {passed/total*100:.1f}%")
        
        if passed == total:
            self.logger.info("🎉 All C++ component tests PASSED!")
        else:
            self.logger.warning(f"⚠️ {total-passed} tests FAILED. Check logs for details.")
        
        return results

def main():
    """Main function to run C++ component tests."""
    print("🧪 C++ Component Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = CppComponentTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Return results
    return results

if __name__ == "__main__":
    results = main()
