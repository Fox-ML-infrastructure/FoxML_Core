# IBKR Intraday Plan - Pressure Test Upgrades

## Overview
12 surgical improvements for alpha retention under friction, robustness to staleness, execution realism, and cleaner control loops.

## 1) Conformal Gates (Alpha Retention)

```python
# conformal_gate.py
class ConformalGate:
    def __init__(self, q_lo=0.2, q_hi=0.8):
        self.q_lo, self.q_hi = q_lo, q_hi
        self.calibration_store = {}

    def allow(self, alpha_point_bps, q_lo_bps, q_hi_bps, est_cost_bps):
        """Only act when distribution clears costs by margin."""
        if q_lo_bps > est_cost_bps:   # long
            return True, "conf_long"
        if q_hi_bps < -est_cost_bps:  # short
            return True, "conf_short"
        return False, "conf_block"

    def update_calibration(self, symbol, horizon, realized_returns):
        """Update conformal intervals from realized returns."""
        # Rolling window calibration
        pass
```

## 2) Triple-Barrier Labels (Vol-Scaled)

```python
# labeling.py
def triple_barrier(price, vol_5m, pt_mult=1.5, sl_mult=1.0, max_h=6):
    """Vol-scaled triple barrier labels."""
    pt = price * (1 + pt_mult * vol_5m)
    sl = price * (1 - sl_mult * vol_5m)
    # Walk forward up to H, emit {+1, -1, 0} outcome + time_to_event
    return outcome, time_to_event
```

## 3) Horizon Arbitration 2.0 (Meta-Selector)

```python
# horizon_meta.py
class HorizonArbiter2:
    def __init__(self, meta_model, cost_model, bandit=None):
        self.meta = meta_model  # softmax weights
        self.cost = cost_model
        self.bandit = bandit

    def choose(self, alpha_by_h, md_row, thresholds):
        feats = self.make_state_features(md_row)
        w = self.meta.predict_proba(feats)   # dict {h: weight}
        alpha_net = {}
        for h, a in alpha_by_h.items():
            c = self.cost.estimate(md_row, h)
            alpha_net[h] = a - c
        alpha_star = sum(w[h]*alpha_net[h] for h in alpha_net)
        h_star = max(alpha_net, key=alpha_net.get)
        if self.bandit:
            w = self.bandit.update(alpha_net)
        return h_star, alpha_star, {"w": w, "alpha_net": alpha_net}
```

## 4) Event-Driven Rebalancing

```python
# rebalance_trigger.py
class EventDrivenRebalancer:
    def should_rebalance(self, symbol, current_weight, target_weight,
                        spread_bps, alpha_conf, regime_switched):
        """Event-driven rebalancing triggers."""
        drift = abs(target_weight - current_weight)
        no_trade_threshold = 0.008

        return (drift > no_trade_threshold or
                (spread_bps < 5.0 and alpha_conf > 2.0) or
                regime_switched)
```

## 5) Execution Micro-Planner

```python
# exec_microplanner.py
class ExecutionMicroPlanner:
    def plan(self, side, qty, tif_s, px_ref, spread, lot=1):
        """Queue-aware execution slices."""
        steps = []
        # 1) Start mid-peg with small slice
        # 2) Schedule step-ups at t = [0, tif/3, 2tif/3]
        # 3) If remaining at TIF-ε, cross with capped slippage
        # 4) Enforce per-minute order budget
        return steps
```

## 6) Live Cost Model

```python
# cost_model.py
class LiveCostModel:
    def __init__(self):
        self.ewma_slippage = {}
        self.impact_curve = {}

    def estimate(self, symbol, horizon, spread_bps, vol_bps, participation_rate):
        """Intraday cost estimation."""
        base_cost = (1.0 * spread_bps +
                    0.15 * vol_bps * np.sqrt(horizon/5) +
                    1.0 * participation_rate**0.6)
        base_cost += self.ewma_slippage.get(symbol, 0)
        return base_cost
```

## 7) Risk: Correlation-Aware

```python
# risk_correlation.py
class CorrelationAwareRisk:
    def __init__(self, corr_window=15):  # 15min window
        self.corr_matrix = None
        self.corr_window = corr_window

    def risk_parity_tilt(self, target_weights):
        """Convert to risk parity before guardrails."""
        if self.corr_matrix is not None:
            # Apply correlation penalty
            risk_adj = np.dot(target_weights, np.dot(self.corr_matrix, target_weights))
            return target_weights / (1 + risk_adj)
        return target_weights
```

## 8) Barrier Gating: Exit Symmetry

```python
# barrier_gates_enhanced.py
class EnhancedBarrierGates:
    def partial_exit_signal(self, predictions, current_position, alpha_5m):
        """Scale out on peak risk, tighten SL on valley collapse."""
        if predictions.will_peak_5m > 0.65 and current_position > 0:
            return True, "scale_out_50pct"
        if predictions.will_valley_5m < 0.3 and current_position > 0:
            return True, "tighten_sl"
        return False, "hold"
```

## 9) Staleness & TTLs

```python
# staleness_guard.py
class StalenessGuard:
    def __init__(self):
        self.quote_ttl = 1.0      # 1 second
        self.bar_ttl = 90.0       # 90 seconds
        self.pred_ttl_mult = 2.0  # 2x horizon

    def check_data_freshness(self, symbol, quote_time, bar_time, pred_time, horizon_min):
        """Block stale data."""
        now = time.time()
        if (now - quote_time) > self.quote_ttl:
            return False, "stale_quote"
        if (now - bar_time) > self.bar_ttl:
            return False, "stale_bar"
        if (now - pred_time) > (horizon_min * 60 * self.pred_ttl_mult):
            return False, "stale_pred"
        return True, "fresh"
```

## 10) Deterministic Renormalization

```python
# renormalizer.py
def renorm_weights(weights):
    """Always renormalize surviving weights."""
    weights = {k: v for k, v in weights.items() if v > 0}
    s = sum(weights.values())
    return {k: v/s for k, v in weights.items()} if s > 0 else {}
```

## 11) Monitoring Dashboard

```python
# monitoring.py
class TradingMonitor:
    def log_cycle_metrics(self, cycle_data):
        """Log every cycle metrics."""
        metrics = {
            "alpha_budget": cycle_data["pred_net_bps"] - cycle_data["realized_slippage_bps"],
            "conf_coverage": cycle_data["trades_in_band"] / cycle_data["total_trades"],
            "barrier_calibration": cycle_data["pred_vs_empirical_peak_prob"],
            "execution_sla": cycle_data["fill_rate_before_tif"],
            "churn": cycle_data["turnover"] / cycle_data["no_trade_band"]
        }
        return metrics
```

## 12) Offline Tests

```python
# test_contracts.py
def test_ttl_contracts():
    """TTL contract tests."""
    assert quote_age < 1.0, "Quote TTL violation"
    assert bar_age < 90.0, "Bar TTL violation"
    assert pred_age < horizon_min * 120, "Prediction TTL violation"

def test_shape_contracts():
    """Shape/NaN contract tests."""
    assert len(predictions) == N, "Prediction length mismatch"
    assert all(np.isfinite(predictions)), "Non-finite predictions"
    assert predictions.ndim == 1, "Prediction dimension mismatch"
```

## Integration Points

### A) Enhanced Decision Pipeline
```python
# ibkr_live_exec_enhanced.py
def enhanced_decision_cycle(self):
    # 1. Staleness check
    fresh, reason = self.staleness_guard.check_data_freshness(...)
    if not fresh:
        return

    # 2. Conformal gating
    alpha_pt, q_lo, q_hi = self.get_calibrated_alpha(symbol, horizon)
    ok, why = self.conformal_gate.allow(alpha_pt, q_lo, q_hi, est_cost)
    if not ok:
        return

    # 3. Horizon arbitration 2.0
    h_star, alpha_star, scores = self.arbiter2.choose(alpha_by_h, md, thresholds)

    # 4. Enhanced barrier gating
    ok, why, size_mult = self.enhanced_barriers.allow_entry(predictions, alpha_star)
    if not ok:
        return

    # 5. Execution micro-planner
    steps = self.microplanner.plan(side, qty, tif_s[h_star], px_ref, spread)
    for step in steps:
        self.broker.submit(step)
```

### B) Event-Driven Rebalancing
```python
# Replace fixed schedule with event triggers
def should_rebalance_now(self, symbol):
    return (self.event_rebalancer.should_rebalance(symbol, ...) or
            self.is_safety_sweep_time())  # 14:30 safety net
```

## Verification Checklist

- [ ] **Calibrated gates live**: Conformal coverage ~80%, barrier reliability slope ~1
- [ ] **Selector adapts**: Horizon weights vary by regime, net bps improves
- [ ] **Message budget respected**: No IBKR throttling, TIF SLA ≥95%
- [ ] **Churn under control**: Turnover reduction at equal/better net bps
- [ ] **No silent failures**: Stale TTL/NaN causes per-symbol skip, not global abort
- [ ] **Renormalize always**: Post-guardrail weights sum to 1.0 when positions >0

## What to Cut (Risk Reduction)

- Drop GAN/VAE/MetaLearning for live until they pass output-shape/calibration tests
- Replace fixed rebalancing times with event-driven + single safety sweep (14:30)
- Remove "yesterday pattern" unless it shows net value after costs in OOS

---

*Implementation Priority: Conformal Gates → Horizon Arbiter 2.0 → Execution Micro-Planner → Risk Correlation*
