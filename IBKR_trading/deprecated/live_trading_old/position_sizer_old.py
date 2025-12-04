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
Position Sizer - Convert Alpha to Target Weights
===============================================

Convert alpha to target weights with risk management.
"""


import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PositionSizer:
    """Convert alpha to target weights."""
    
    def __init__(self, config):
        self.config = config
        self.z_max = config.get('z_max', 3.0)
        self.max_weight = config.get('max_weight', 0.05)
        self.gross_target = config.get('gross_target', 0.5)
        self.no_trade_band = config.get('no_trade_band', 0.008)
        self.use_risk_parity = config.get('use_risk_parity', False)
        self.ridge_lambda = config.get('ridge_lambda', 0.01)
        
        logger.info(f"PositionSizer initialized: z_max={self.z_max}, max_weight={self.max_weight}, "
                   f"gross_target={self.gross_target}, no_trade_band={self.no_trade_band}")
    
    def size_positions(self, alpha: Dict[str, float], 
                      market_data: Dict,
                      current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Convert alpha to target weights.
        
        Args:
            alpha: Alpha values by symbol
            market_data: Market data for risk calculation
            current_weights: Current portfolio weights
        
        Returns:
            Target weights by symbol
        """
        try:
            if not alpha:
                logger.warning("No alpha provided")
                return current_weights
            
            # 1. Vol scaling
            z = self._vol_scaling(alpha, market_data)
            
            # 2. Cross-sectional standardization
            z_std = self._cross_sectional_standardize(z)
            
            # 3. Risk parity (optional)
            if self.use_risk_parity:
                w_raw = self._risk_parity_ridge(z_std, market_data)
            else:
                w_raw = z_std
            
            # 4. Apply caps
            w_capped = self._apply_caps(w_raw)
            
            # 5. Renormalize to target gross
            w_gross = self._renormalize_to_gross(w_capped)
            
            # 6. No-trade band
            w_final = self._apply_no_trade_band(w_gross, current_weights)
            
            logger.info(f"Position sizing complete: {len(w_final)} symbols")
            return w_final
            
        except Exception as e:
            logger.error(f"Error in position sizing: {e}")
            return current_weights
    
    def _vol_scaling(self, alpha: Dict[str, float], market_data: Dict) -> Dict[str, float]:
        """Apply volatility scaling."""
        try:
            vol_short = market_data.get('vol_short', {})
            z = {}
            
            for symbol, a in alpha.items():
                vol = vol_short.get(symbol, 0.15)
                vol_clipped = max(vol, 1e-8)  # Avoid division by zero
                z[symbol] = np.clip(a / vol_clipped, -self.z_max, self.z_max)
                
                logger.debug(f"Symbol {symbol}: alpha={a:.4f}, vol={vol:.4f}, z={z[symbol]:.4f}")
            
            return z
            
        except Exception as e:
            logger.error(f"Error in vol scaling: {e}")
            return alpha
    
    def _cross_sectional_standardize(self, z: Dict[str, float]) -> Dict[str, float]:
        """Cross-sectional z-score standardization."""
        try:
            if len(z) <= 1:
                return z
            
            values = np.array(list(z.values()))
            symbols = list(z.keys())
            
            # Calculate z-score
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val > 1e-8:
                z_std = (values - mean_val) / std_val
            else:
                z_std = values
            
            return dict(zip(symbols, z_std))
            
        except Exception as e:
            logger.error(f"Error in cross-sectional standardization: {e}")
            return z
    
    def _risk_parity_ridge(self, z_std: Dict[str, float], market_data: Dict) -> Dict[str, float]:
        """Apply risk parity with ridge regularization."""
        try:
            symbols = list(z_std.keys())
            z_vec = np.array([z_std[s] for s in symbols])
            
            # Get covariance matrix
            cov = market_data.get('covariance', np.eye(len(symbols)))
            
            # Ensure covariance is positive definite
            cov = cov + 1e-6 * np.eye(len(symbols))
            
            # Ridge solution: w = lambda * (C + lambda*I)^(-1) * z
            try:
                cov_inv = np.linalg.inv(cov + self.ridge_lambda * np.eye(len(symbols)))
                w_raw = self.ridge_lambda * cov_inv @ z_vec
            except np.linalg.LinAlgError:
                logger.warning("Covariance matrix inversion failed, using identity")
                w_raw = z_vec
            
            return dict(zip(symbols, w_raw))
            
        except Exception as e:
            logger.error(f"Error in risk parity: {e}")
            return z_std
    
    def _apply_caps(self, w_raw: Dict[str, float]) -> Dict[str, float]:
        """Apply position caps."""
        try:
            w_capped = {}
            for symbol, w in w_raw.items():
                w_capped[symbol] = np.clip(w, -self.max_weight, self.max_weight)
            
            logger.debug(f"Applied caps: max_weight={self.max_weight}")
            return w_capped
            
        except Exception as e:
            logger.error(f"Error applying caps: {e}")
            return w_raw
    
    def _renormalize_to_gross(self, w_capped: Dict[str, float]) -> Dict[str, float]:
        """Renormalize to target gross exposure."""
        try:
            if not w_capped:
                return w_capped
            
            # Calculate total absolute weight
            total_abs = sum(abs(w) for w in w_capped.values())
            
            if total_abs > 0:
                scale = self.gross_target / total_abs
                w_gross = {s: w * scale for s, w in w_capped.items()}
                
                # Verify gross exposure
                actual_gross = sum(abs(w) for w in w_gross.values())
                logger.debug(f"Renormalized to gross: target={self.gross_target:.3f}, "
                           f"actual={actual_gross:.3f}")
            else:
                w_gross = w_capped
            
            return w_gross
            
        except Exception as e:
            logger.error(f"Error in gross renormalization: {e}")
            return w_capped
    
    def _apply_no_trade_band(self, w_target: Dict[str, float], 
                            w_current: Dict[str, float]) -> Dict[str, float]:
        """Apply no-trade band to reduce turnover."""
        try:
            w_final = {}
            
            for symbol in w_target.keys():
                current = w_current.get(symbol, 0.0)
                target = w_target[symbol]
                drift = abs(target - current)
                
                if drift > self.no_trade_band:
                    w_final[symbol] = target
                    logger.debug(f"Symbol {symbol}: trading (drift={drift:.4f} > {self.no_trade_band:.4f})")
                else:
                    w_final[symbol] = current
                    logger.debug(f"Symbol {symbol}: no trade (drift={drift:.4f} <= {self.no_trade_band:.4f})")
            
            # Calculate turnover
            turnover = sum(abs(w_final[s] - w_current.get(s, 0)) for s in w_final.keys())
            logger.info(f"No-trade band applied: turnover={turnover:.3f}")
            
            return w_final
            
        except Exception as e:
            logger.error(f"Error in no-trade band: {e}")
            return w_target

class AdvancedPositionSizer(PositionSizer):
    """Advanced position sizer with additional features."""
    
    def __init__(self, config):
        super().__init__(config)
        self.use_sector_neutral = config.get('use_sector_neutral', False)
        self.use_momentum_tilt = config.get('use_momentum_tilt', False)
        self.sector_lambda = config.get('sector_lambda', 0.1)
        self.momentum_lambda = config.get('momentum_lambda', 0.05)
    
    def size_positions_advanced(self, alpha: Dict[str, float], 
                              market_data: Dict,
                              current_weights: Dict[str, float]) -> Dict[str, float]:
        """Advanced position sizing with additional constraints."""
        try:
            # Get base weights
            w_base = self.size_positions(alpha, market_data, current_weights)
            
            # Apply sector neutralization
            if self.use_sector_neutral:
                w_base = self._apply_sector_neutral(w_base, market_data)
            
            # Apply momentum tilt
            if self.use_momentum_tilt:
                w_base = self._apply_momentum_tilt(w_base, market_data)
            
            return w_base
            
        except Exception as e:
            logger.error(f"Error in advanced position sizing: {e}")
            return w_base
    
    def _apply_sector_neutral(self, w: Dict[str, float], market_data: Dict) -> Dict[str, float]:
        """Apply sector neutralization."""
        try:
            # Get sector data
            sectors = market_data.get('sectors', {})
            if not sectors:
                return w
            
            # Calculate sector weights
            sector_weights = {}
            for symbol, weight in w.items():
                sector = sectors.get(symbol, 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            # Calculate sector-neutral adjustment
            sector_adj = {}
            for sector, sector_weight in sector_weights.items():
                sector_adj[sector] = -sector_weight / len([s for s in sectors.values() if s == sector])
            
            # Apply adjustment
            w_neutral = {}
            for symbol, weight in w.items():
                sector = sectors.get(symbol, 'Unknown')
                w_neutral[symbol] = weight + sector_adj.get(sector, 0)
            
            logger.info(f"Applied sector neutralization")
            return w_neutral
            
        except Exception as e:
            logger.error(f"Error in sector neutralization: {e}")
            return w
    
    def _apply_momentum_tilt(self, w: Dict[str, float], market_data: Dict) -> Dict[str, float]:
        """Apply momentum tilt."""
        try:
            # Get momentum data
            momentum = market_data.get('momentum', {})
            if not momentum:
                return w
            
            # Apply momentum tilt
            w_tilted = {}
            for symbol, weight in w.items():
                mom = momentum.get(symbol, 0.0)
                w_tilted[symbol] = weight + self.momentum_lambda * mom
            
            logger.info(f"Applied momentum tilt")
            return w_tilted
            
        except Exception as e:
            logger.error(f"Error in momentum tilt: {e}")
            return w

class PositionValidator:
    """Validate position sizing results."""
    
    def __init__(self, config):
        self.config = config
        self.max_gross = config.get('max_gross', 1.0)
        self.max_net = config.get('max_net', 0.1)
        self.max_single_weight = config.get('max_single_weight', 0.1)
    
    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Validate position weights."""
        try:
            if not weights:
                return {'valid': False, 'errors': ['No weights provided']}
            
            errors = []
            warnings = []
            
            # Calculate metrics
            total_abs = sum(abs(w) for w in weights.values())
            total_net = sum(weights.values())
            max_weight = max(abs(w) for w in weights.values())
            
            # Check gross exposure
            if total_abs > self.max_gross:
                errors.append(f"Gross exposure {total_abs:.3f} exceeds limit {self.max_gross:.3f}")
            
            # Check net exposure
            if abs(total_net) > self.max_net:
                warnings.append(f"Net exposure {total_net:.3f} exceeds limit {self.max_net:.3f}")
            
            # Check single position
            if max_weight > self.max_single_weight:
                errors.append(f"Max single weight {max_weight:.3f} exceeds limit {self.max_single_weight:.3f}")
            
            # Check for NaN or infinite values
            for symbol, weight in weights.items():
                if not np.isfinite(weight):
                    errors.append(f"Invalid weight for {symbol}: {weight}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'metrics': {
                    'gross_exposure': total_abs,
                    'net_exposure': total_net,
                    'max_weight': max_weight,
                    'num_positions': len(weights)
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating weights: {e}")
            return {'valid': False, 'errors': [str(e)]}
