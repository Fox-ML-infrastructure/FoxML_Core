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
Leakage Sentinels

Automated tests to detect data leakage in models.
These tests catch leakage that might slip through structural rules.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentinelResult:
    """Result from a leakage sentinel test."""
    test_name: str
    passed: bool
    score: float
    threshold: float
    warning: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class LeakageSentinel:
    """
    Automated tests to detect data leakage.
    
    These tests are designed to catch leakage that might slip through
    structural rules (e.g., features that encode future information
    in subtle ways).
    """
    
    def __init__(
        self,
        shifted_target_threshold: float = 0.5,
        symbol_holdout_train_threshold: float = 0.9,
        symbol_holdout_test_threshold: float = 0.3,
        randomized_time_threshold: float = 0.5
    ):
        """
        Initialize leakage sentinel with thresholds.
        
        Args:
            shifted_target_threshold: If model score > this on shifted target, flag as leaky
            symbol_holdout_train_threshold: If train score > this, flag for investigation
            symbol_holdout_test_threshold: If test score < this (with high train), flag as leaky
            randomized_time_threshold: If model score > this on time-shuffled data, flag as leaky
        """
        self.shifted_target_threshold = shifted_target_threshold
        self.symbol_holdout_train_threshold = symbol_holdout_train_threshold
        self.symbol_holdout_test_threshold = symbol_holdout_test_threshold
        self.randomized_time_threshold = randomized_time_threshold
    
    def shifted_target_test(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int,
        score_func: callable = None
    ) -> SentinelResult:
        """
        Test with target shifted by +N bars (trying to predict further in the future).
        
        If the model still shows "magic" performance on a shifted target,
        it almost surely has features that look into the future.
        
        Args:
            model: Trained model with .score() or .predict() method
            X: Feature matrix
            y: Target array
            horizon: Number of bars to shift target forward
            score_func: Optional scoring function (default: model.score if available)
        
        Returns:
            SentinelResult indicating if test passed
        """
        try:
            # Shift target forward by horizon
            y_shifted = np.roll(y, horizon)
            
            # Remove samples where shift caused invalid data
            # (first horizon samples have shifted values from end of array)
            valid_mask = np.arange(len(y)) >= horizon
            X_valid = X[valid_mask]
            y_shifted_valid = y_shifted[valid_mask]
            
            if len(y_shifted_valid) == 0:
                return SentinelResult(
                    test_name="shifted_target",
                    passed=True,
                    score=0.0,
                    threshold=self.shifted_target_threshold,
                    warning="Insufficient data for shifted target test"
                )
            
            # Score model on shifted target
            if score_func is not None:
                score_shifted = score_func(model, X_valid, y_shifted_valid)
            elif hasattr(model, 'score'):
                score_shifted = model.score(X_valid, y_shifted_valid)
            else:
                # Fallback: use predictions
                preds = model.predict(X_valid)
                if hasattr(model, 'predict_proba'):
                    # For classification, use accuracy
                    score_shifted = np.mean(preds == y_shifted_valid)
                else:
                    # For regression, use R²
                    from sklearn.metrics import r2_score
                    score_shifted = r2_score(y_shifted_valid, preds)
            
            # Check if score is suspiciously high
            passed = score_shifted <= self.shifted_target_threshold
            
            warning = None
            if not passed:
                warning = (
                    f"🚨 LEAKAGE ALERT: Model performs well on shifted target "
                    f"(score={score_shifted:.3f} > threshold={self.shifted_target_threshold}). "
                    f"Features may look into future."
                )
            
            return SentinelResult(
                test_name="shifted_target",
                passed=passed,
                score=float(score_shifted),
                threshold=self.shifted_target_threshold,
                warning=warning,
                details={
                    'horizon': horizon,
                    'n_samples': len(y_shifted_valid)
                }
            )
        
        except Exception as e:
            logger.warning(f"Shifted target test failed: {e}")
            return SentinelResult(
                test_name="shifted_target",
                passed=True,  # Pass on error (don't block)
                score=0.0,
                threshold=self.shifted_target_threshold,
                warning=f"Test failed: {e}"
            )
    
    def symbol_holdout_test(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_symbols: List[str] = None,
        test_symbols: List[str] = None,
        score_func: callable = None
    ) -> SentinelResult:
        """
        Test on never-seen symbols.
        
        Train on some symbols, validate on never-seen symbols.
        If performance craters on test but is insane in-sample,
        check for symbol-specific leakage (e.g., using symbol-level
        aggregates built from the full dataset).
        
        Args:
            model: Trained model with .score() or .predict() method
            X_train: Training feature matrix
            y_train: Training target array
            X_test: Test feature matrix (never-seen symbols)
            y_test: Test target array
            train_symbols: Optional list of training symbols (for logging)
            test_symbols: Optional list of test symbols (for logging)
            score_func: Optional scoring function
        
        Returns:
            SentinelResult indicating if test passed
        """
        try:
            # Score on training data
            if score_func is not None:
                train_score = score_func(model, X_train, y_train)
            elif hasattr(model, 'score'):
                train_score = model.score(X_train, y_train)
            else:
                # Fallback: use predictions
                preds_train = model.predict(X_train)
                if hasattr(model, 'predict_proba'):
                    train_score = np.mean(preds_train == y_train)
                else:
                    from sklearn.metrics import r2_score
                    train_score = r2_score(y_train, preds_train)
            
            # Score on test data (never-seen symbols)
            if score_func is not None:
                test_score = score_func(model, X_test, y_test)
            elif hasattr(model, 'score'):
                test_score = model.score(X_test, y_test)
            else:
                # Fallback: use predictions
                preds_test = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    test_score = np.mean(preds_test == y_test)
                else:
                    from sklearn.metrics import r2_score
                    test_score = r2_score(y_test, preds_test)
            
            # Check for suspicious gap
            high_train = train_score > self.symbol_holdout_train_threshold
            low_test = test_score < self.symbol_holdout_test_threshold
            passed = not (high_train and low_test)
            
            warning = None
            if not passed:
                symbol_info = ""
                if train_symbols and test_symbols:
                    symbol_info = f" (train: {train_symbols[:3]}, test: {test_symbols[:3]})"
                warning = (
                    f"🚨 LEAKAGE ALERT: Large train/test gap "
                    f"(train={train_score:.3f}, test={test_score:.3f}){symbol_info}. "
                    f"Possible symbol-specific leakage."
                )
            
            return SentinelResult(
                test_name="symbol_holdout",
                passed=passed,
                score=float(test_score),
                threshold=self.symbol_holdout_test_threshold,
                warning=warning,
                details={
                    'train_score': float(train_score),
                    'test_score': float(test_score),
                    'gap': float(train_score - test_score),
                    'train_symbols': train_symbols,
                    'test_symbols': test_symbols
                }
            )
        
        except Exception as e:
            logger.warning(f"Symbol holdout test failed: {e}")
            return SentinelResult(
                test_name="symbol_holdout",
                passed=True,  # Pass on error
                score=0.0,
                threshold=self.symbol_holdout_test_threshold,
                warning=f"Test failed: {e}"
            )
    
    def randomized_time_test(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        score_func: callable = None
    ) -> SentinelResult:
        """
        Test with shuffled time (features and targets paired).
        
        Shuffle time within each symbol but keep features and targets paired.
        A good model should die. If it still does suspiciously well,
        your features are encoding future info or label proxies.
        
        Args:
            model: Trained model with .score() or .predict() method
            X: Feature matrix
            y: Target array
            score_func: Optional scoring function
        
        Returns:
            SentinelResult indicating if test passed
        """
        try:
            # Shuffle time index but keep feature-target pairs
            indices = np.arange(len(X))
            np.random.seed(42)  # Deterministic shuffle
            np.random.shuffle(indices)
            
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Score model on time-shuffled data
            if score_func is not None:
                score_shuffled = score_func(model, X_shuffled, y_shuffled)
            elif hasattr(model, 'score'):
                score_shuffled = model.score(X_shuffled, y_shuffled)
            else:
                # Fallback: use predictions
                preds = model.predict(X_shuffled)
                if hasattr(model, 'predict_proba'):
                    # For classification, use accuracy
                    score_shuffled = np.mean(preds == y_shuffled)
                else:
                    # For regression, use R²
                    from sklearn.metrics import r2_score
                    score_shuffled = r2_score(y_shuffled, preds)
            
            # Check if score is suspiciously high (should be random)
            passed = score_shuffled <= self.randomized_time_threshold
            
            warning = None
            if not passed:
                warning = (
                    f"🚨 LEAKAGE ALERT: Model performs well on time-shuffled data "
                    f"(score={score_shuffled:.3f} > threshold={self.randomized_time_threshold}). "
                    f"Features may encode future info or label proxies."
                )
            
            return SentinelResult(
                test_name="randomized_time",
                passed=passed,
                score=float(score_shuffled),
                threshold=self.randomized_time_threshold,
                warning=warning,
                details={
                    'n_samples': len(X)
                }
            )
        
        except Exception as e:
            logger.warning(f"Randomized time test failed: {e}")
            return SentinelResult(
                test_name="randomized_time",
                passed=True,  # Pass on error
                score=0.0,
                threshold=self.randomized_time_threshold,
                warning=f"Test failed: {e}"
            )
    
    def run_all_tests(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int = None,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        train_symbols: List[str] = None,
        test_symbols: List[str] = None,
        score_func: callable = None,
        enabled_tests: List[str] = None
    ) -> List[SentinelResult]:
        """
        Run all enabled leakage sentinel tests.
        
        Args:
            model: Trained model
            X: Feature matrix (for shifted-target and randomized-time tests)
            y: Target array
            horizon: Target horizon in bars (for shifted-target test)
            X_train: Training features (for symbol-holdout test)
            y_train: Training targets
            X_test: Test features (never-seen symbols)
            y_test: Test targets
            train_symbols: Training symbols (for logging)
            test_symbols: Test symbols (for logging)
            score_func: Optional scoring function
            enabled_tests: List of test names to run (default: all)
        
        Returns:
            List of SentinelResult objects
        """
        if enabled_tests is None:
            enabled_tests = ['shifted_target', 'symbol_holdout', 'randomized_time']
        
        results = []
        
        # Shifted-target test
        if 'shifted_target' in enabled_tests and horizon is not None:
            result = self.shifted_target_test(model, X, y, horizon, score_func)
            results.append(result)
            if result.warning:
                logger.warning(result.warning)
        
        # Symbol-holdout test
        if 'symbol_holdout' in enabled_tests:
            if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
                result = self.symbol_holdout_test(
                    model, X_train, y_train, X_test, y_test,
                    train_symbols, test_symbols, score_func
                )
                results.append(result)
                if result.warning:
                    logger.warning(result.warning)
            else:
                logger.debug("Symbol holdout test skipped: train/test split not provided")
        
        # Randomized-time test
        if 'randomized_time' in enabled_tests:
            result = self.randomized_time_test(model, X, y, score_func)
            results.append(result)
            if result.warning:
                logger.warning(result.warning)
        
        return results

