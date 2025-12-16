# IBKR Intraday Trading System Analysis

## Overview

The IBKR system is specifically designed for **intraday trading** with sophisticated multi-timeframe horizon management and barrier target integration. This analysis examines how it handles 5m, 10m, 15m, and 30m horizons with barrier targets.

## ️ **Multi-Timeframe Architecture**

### **Horizon Configuration**
```yaml
# From ibkr_enhanced.yaml
horizons: ["5m", "10m", "15m", "30m", "60m"]
families: [
  "LightGBM", "XGBoost", "MLP", "TabCNN", "TabLSTM", "TabTransformer",
  "RewardBased", "QuantileLightGBM", "NGBoost", "GMMRegime",
  "ChangePoint", "FTRLProximal", "VAE", "GAN", "Ensemble", "MetaLearning"
]
```

### **Intraday Rebalancing Schedule**
```python
# From intraday_rebalancer.py
rebalance_times = ["09:35", "10:30", "12:00", "14:30", "15:50"]
```

**Key Features:**
- **5-minute cadence** for model inference
- **Scheduled rebalancing** at optimal times
- **Window-aware execution** (pre-open, open, core, close)
- **Volatility-aware sizing** with EWMA tracking

## **Horizon Arbitration System**

### **Cost-Aware Horizon Selection**
```python
# From horizon_arbiter.py
def net_score(self, alpha_h: float, spread_bps: float, vol_bps: float,
              impact_bps: float, h_minutes: int) -> float:
    """
    Formula: net_h = α_h - k₁×spread_bps - k₂×σ×√(h/5) - k₃×impact_bps
    """
    net = alpha_h
    net -= self.k_spread * spread_bps
    vol_penalty = self.k_vol * vol_bps * np.sqrt(h_minutes / 5.0)
    net -= vol_penalty
    net -= self.k_imp * impact_bps
    return net
```

**Mathematical Foundation:**
- **Cost Penalties**: Spread, volatility timing, market impact
- **Horizon Scaling**: Longer horizons penalized in 5m loop
- **Adjacent Blending**: 70% primary + 30% adjacent horizon
- **Threshold Gating**: Entry/exit based on net scores

### **Horizon Selection Process**
```python
# From ibkr_live_exec.py
# 4) Build horizon alphas & arbitrate
for sym in trade_syms:
    sym_preds = raw_preds.get(sym, {})
    alpha_by_h = self._within_horizon_alpha(sym_preds, sym)

    # Across-horizon arbitration (cost-aware)
    h_star, alpha_star, scores = self.arbiter.choose(alpha_by_h, md[sym], self.thresholds)
```

## **Barrier Target Integration**

### **Barrier Target Processing**
```python
# From barrier_gates.py
class BarrierGates:
    def allow_long_entry(self, predictions: BarrierPredictions, alpha_5m: float):
        """
        Formula: Block if P(will_peak_5m) > 0.6 OR P(y_will_peak_5m) > 0.6
                 Prefer if P(will_valley_5m) > 0.55 AND ΔP > 0
        """
        # Check blocking conditions
        if predictions.will_peak_5m > self.thresholds.block_peak:
            return False, "peak_risk", 0.0

        if predictions.y_will_peak_5m > self.thresholds.block_y_peak:
            return False, "y_peak_risk", 0.0

        # Check for valley preference
        if (predictions.will_valley_5m > self.thresholds.prefer_valley and
            predictions.price_change > 0):
            return True, "valley_bounce_pref", 1.0
```

### **Barrier Target Types**
```python
@dataclass
class BarrierPredictions:
    will_peak_5m: float      # P(price will peak in 5m)
    will_valley_5m: float   # P(price will valley in 5m)
    y_will_peak_5m: float   # P(yesterday's pattern will peak)
    y_will_valley_5m: float # P(yesterday's pattern will valley)
    price_change: float     # ΔP for valley bounce detection
```

**Key Features:**
- **Peak Risk Blocking**: Prevents entries near local tops
- **Valley Preference**: Favors entries near local bottoms
- **Historical Pattern Matching**: Uses yesterday's patterns
- **Position Size Adjustment**: Reduces size based on peak risk

## **Intraday Execution Strategy**

### **Time-in-Force by Horizon**
```yaml
# From ibkr_enhanced.yaml
tif_s:
  "5m": 15    # 15 seconds for 5m horizon
  "10m": 25   # 25 seconds for 10m horizon
  "15m": 30   # 30 seconds for 15m horizon
  "30m": 45   # 45 seconds for 30m horizon
  "60m": 60   # 60 seconds for 60m horizon
```

### **Execution Windows**
```python
# From intraday_rebalancer.py
class RebalanceWindow(Enum):
    PRE_OPEN = "pre_open"      # 09:20: compute targets, stage positions
    OPEN = "open"              # 09:30–09:45: widen bands ×2, halve sizes
    CORE = "core"              # Normal session: event-triggered rebalances
    CLOSE = "close"            # 15:45: shrink bands, converge to EOD
```

### **Open Window Protection**
```python
def apply_open_protection(self, target_weights: Dict[str, float]) -> Dict[str, float]:
    """Apply open window protection by reducing position sizes."""
    if self.is_open_window():
        protected_weights = {}
        for symbol, weight in target_weights.items():
            protected_weights[symbol] = weight * self.config.open_size_multiplier  # 0.5x
        return protected_weights
```

## **Risk Management Framework**

### **Volatility-Aware Sizing**
```python
def scale_signals(self, scores: Dict[str, float]) -> Dict[str, float]:
    """Convert signals to target weights with vol scaling."""
    for symbol, score in scores.items():
        if symbol in self.vol_estimates and self.vol_estimates[symbol] > 0:
            # Vol scaling: z_i = score_i / σ_i
            z = score / self.vol_estimates[symbol]
            z = np.clip(z, -self.config.z_max, self.config.z_max)
            z_scores[symbol] = z
```

### **No-Trade Bands**
```python
def apply_no_trade_band(self, target_weights: Dict[str, float]) -> Dict[str, float]:
    """Apply no-trade band to avoid over-trading."""
    threshold = self.config.no_trade_threshold  # 0.8% NAV

    # Double threshold during open window
    if self.is_open_window():
        threshold *= self.config.open_no_trade_multiplier  # 2.0x

    # Only trade symbols that exceed threshold
    for symbol in target_weights:
        if drift[symbol] > threshold:
            final_weights[symbol] = target_weights[symbol]
```

### **Volatility Clamps**
```python
def apply_vol_clamps(self, target_weights: Dict[str, float]) -> Dict[str, float]:
    """Apply volatility clamps to reduce risk during high-vol periods."""
    if self.realized_vol > self.config.daily_vol_target * self.config.vol_clamp_threshold:
        # Scale down all positions
        for symbol, weight in target_weights.items():
            clamped_weights[symbol] = weight * self.config.vol_clamp_multiplier  # 0.7x
```

## **Trading Decision Flow**

### **Complete Decision Pipeline**
```python
# From ibkr_live_exec.py
def run(self):
    while not self.stop_event.is_set():
        # 1. Get market data
        md = self._get_market_data()

        # 2. Get model predictions
        raw_preds = self.inference_engine.predict(md)

        # 3. Prefilter universe
        trade_syms = self.prefilter.filter(md, raw_preds)

        # 4. Build horizon alphas & arbitrate
        for sym in trade_syms:
            alpha_by_h = self._within_horizon_alpha(sym_preds, sym)
            h_star, alpha_star, scores = self.arbiter.choose(alpha_by_h, md[sym], self.thresholds)

            # 5. Barrier gating (5m focused)
            bar = self._barrier_preds(sym_preds)
            ok, why = self.barriers.allow_long_entry(
                p_peak5=bar.get("will_peak_5m", 0.0),
                p_valley5=bar.get("will_valley_5m", 0.0),
                p_y_peak5=bar.get("y_will_peak_5m", 0.0)
            )

            # 6. Cost model & risk sizing
            costs = self.cost_model.calculate_costs(sym, vars(md[sym]))
            net_alpha_bps = alpha_star - costs.total

            # 7. Position size (risk-aware)
            size_dollars = self.risk_manager.size_from_signal(sym, net_alpha_bps, md[sym], portfolio)

            # 8. Execution plan (TIF+step-ups)
            side = "BUY" if net_alpha_bps > 0 else "SELL"
            steps = self.exec_policy.plan(side, size_dollars, h_star)
```

## **Configuration Parameters**

### **Barrier Target Thresholds**
```yaml
# From ibkr_enhanced.yaml
barrier:
  block_peak: 0.60      # Block long entry if P(peak) > 0.60
  block_y_peak: 0.60    # Block long entry if P(y_peak) > 0.60
  prefer_valley: 0.55    # Prefer long entry if P(valley) > 0.55
  exit_peak: 0.65       # Exit long if P(peak) > 0.65
  exit_valley: 0.65      # Exit short if P(valley) > 0.65
  min_alpha: 0.0         # Minimum alpha for entry
```

### **Execution Parameters**
```yaml
execution:
  default: "marketable_limit"
  step_up_ticks: [0, 1, 2]
  tif_s:
    "5m": 15
    "10m": 25
    "15m": 30
    "30m": 45
  bracket:
    enabled: true
    tp_bps: 8      # Take profit at 8 bps
    sl_bps: 10     # Stop loss at 10 bps
```

### **Risk Parameters**
```yaml
risk:
  max_daily_loss: 0.02           # 2% max daily loss
  max_drawdown: 0.15             # 15% max drawdown
  max_weight_per_symbol: 0.25    # 25% max per symbol
  max_gross_exposure: 0.5        # 50% max gross exposure
  order_rate_limit: 10           # 10 orders per minute max
```

## **Performance Optimization**

### **Cost-Aware Decision Making**
- **Spread Penalties**: Avoid wide spreads
- **Volatility Timing**: Penalize longer horizons in 5m loop
- **Market Impact**: Account for execution costs
- **Adjacent Blending**: Smooth horizon transitions

### **Execution Efficiency**
- **Time-in-Force Optimization**: Shorter TIF for shorter horizons
- **Step-up Orders**: Gradual execution to minimize impact
- **Bracket Orders**: Automatic profit/loss management
- **Microstructure Awareness**: Prefer mid-peg orders

### **Risk Management**
- **Volatility Clamps**: Reduce size during high-vol periods
- **No-Trade Bands**: Avoid over-trading
- **Open Protection**: Halve sizes during market open
- **Barrier Gating**: Prevent entries near peaks/valleys

## **Key Advantages**

### **1. Multi-Timeframe Intelligence**
- **Horizon Arbitration**: Cost-aware selection across 5m, 10m, 15m, 30m, 60m
- **Adjacent Blending**: Smooth transitions between horizons
- **Timing Optimization**: Shorter horizons preferred in 5m loop

### **2. Barrier Target Protection**
- **Peak Risk Blocking**: Prevents entries near local tops
- **Valley Preference**: Favors entries near local bottoms
- **Historical Patterns**: Uses yesterday's patterns for prediction
- **Position Sizing**: Adjusts size based on peak/valley risk

### **3. Intraday Execution Excellence**
- **Window-Aware**: Different strategies for pre-open, open, core, close
- **Volatility-Aware**: Dynamic sizing based on realized volatility
- **Cost-Optimized**: Comprehensive cost modeling and optimization
- **Risk-Controlled**: Multiple layers of risk management

### **4. Production-Ready Architecture**
- **Safety Guards**: Pre-trade, margin, short-sale, rate limiting
- **Fault Tolerance**: Automatic reconnection, error handling
- **Monitoring**: Comprehensive metrics and health checks
- **Recovery**: Disaster recovery and state reconciliation

## **Conclusion**

The IBKR system provides a **sophisticated, production-ready framework** for intraday trading with:

- **Multi-timeframe horizon management** (5m, 10m, 15m, 30m, 60m)
- **Barrier target integration** for microstructure protection
- **Cost-aware decision making** with comprehensive optimization
- **Risk management** with multiple safety layers
- **Execution excellence** with window-aware strategies

The system is specifically designed for **intraday trading** with **barrier targets** as a core component, making it ideal for high-frequency, short-horizon trading strategies.

---

*Analysis Date: 2025-01-01*
*System Status: Production Ready*
*Key Focus: Intraday Trading with Barrier Targets*
