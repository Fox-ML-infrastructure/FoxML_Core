#!/usr/bin/env python3

# -*- coding: utf-8 -*-
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
IBKR Live Intraday Execution Orchestrator (single-file wiring)
- Focus: 5–10m horizons with barrier gates, cost-aware ensemble, and strict safety.
- Run: python live_trading/ibkr_live_exec.py --config config/ibkr_live.yaml --symbols AAPL,MSFT
"""


import os, sys, time, json, signal, argparse, logging, pathlib, math, threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

# --- Third-party (install): ib-insync, pyyaml, numpy, pandas
try:
    import yaml
    from ib_insync import IB
except Exception as e:
    print("Missing deps: pip install ib-insync pyyaml", file=sys.stderr); raise

# === Imports from your codebase (create these modules per your plan) ===
# If not yet implemented, keep the TODOs to wire later.
try:
    # Brokers & infra
    from brokers.ibkr_broker import IBKRBrokerAdapter
    from brokers.ibkr_orders import OrderResult
    # Live trading components
    from live_trading.ibkr_model_loader import IBKRModelLoader
    from live_trading.ibkr_feature_pipeline import IBKRFeaturePipeline
    from live_trading.ibkr_inference_engine import IBKRModelInference
    from live_trading.cost_model import CostModel
    from live_trading.decision_optimizer import DecisionOptimizer
    from live_trading.ensemble_weight_optimizer import EnsembleWeightOptimizer
    # Risk & safety
    from live_trading.risk_manager import IBKRRiskManager
    from live_trading.kill_switch import IBKRKillSwitch
    from live_trading.guards import PreTradeGuards, MarginGate, ShortSaleGuard, nbbo_sane
    from live_trading.reconcile import OrderReconciler
    from live_trading.rate_limiter import RateLimiter
    # Balancing & execution
    from live_trading.zoo_balancer import ZooBalancer
    from live_trading.horizon_arbiter import HorizonArbiter
    from live_trading.barrier_gates import BarrierGates
    from live_trading.exec_policy import ShortHorizonExecutionPolicy
    # Health & modes & prefilter & tca
    from live_trading.health import DriftHealth, ModeController
    from live_trading.prefilter import UniversePrefilter
    from live_trading.tca import TradeCostAnalytics
except Exception:
    # Minimal fallbacks so the file imports; you should implement proper modules.
    class IBKRRiskManager: 
        def __init__(self, cfg): self.cfg = cfg
        def apply_risk_constraints(self, decisions, positions, pv): return decisions
        def size_from_signal(self, sym, alpha_bps, md, portfolio): 
            cap = self.cfg.get("max_weight_per_symbol", 0.25)
            return max(0, min(cap * pv, pv * 0.01))  # placeholder
    class IBKRKillSwitch:
        def __init__(self, cfg): self.cfg = cfg; self.flag = pathlib.Path("kill.flag")
        def check_kill_conditions(self) -> bool: return self.flag.exists()
        def handle_exception(self, e): logging.exception(e)
    class ZooBalancer:
        def __init__(self, **kw): pass
        def standardize(self, preds, stats): return preds
        def calibrate(self, z, calibrators): return z
        def weights(self, cal, ic, cost_share, corr_mat, horizon): 
            n = len(cal) or 1; return {k: 1.0/n for k in cal}
        def blend(self, cal, w): 
            return sum(cal.get(f,0)*w.get(f,0) for f in w)
    class HorizonArbiter:
        def __init__(self, **kw): pass
        def choose(self, horizon_alphas, md, thresholds):
            if not horizon_alphas: return None, 0.0, {}
            h_star = max(horizon_alphas, key=horizon_alphas.get)
            return h_star, horizon_alphas[h_star], {}
    class BarrierGates:
        def allow_long_entry(self, p_peak5, p_valley5, p_y_peak5, thresholds): return True, "ok"
        def long_exit_signal(self, p_peak5, alpha5): return False
    class ShortHorizonExecutionPolicy:
        def __init__(self, **kw): pass
        def plan(self, side, mid_px, tick, score): return [{"limit": mid_px, "tif_s": 15}]
    class MarginGate:
        def __init__(self, ib): pass
        def what_if_ok(self, order, contract): return True, "ok"
    class ShortSaleGuard:
        def __init__(self, ib, tick=0.01): pass
        def check(self, sym, md, side): return True, "ok"
    def nbbo_sane(md): 
        try:
            return (md.quote_age_ms <= 200 and md.ask > md.bid), "ok"
        except: return False, "nbbo_missing"
    class RateLimiter:
        def __init__(self, rate=45, burst=50): self.rate=rate; self.last=time.time(); self.tokens=burst; self.burst=burst
        def acquire(self, n=1):
            now=time.time(); self.tokens = min(self.burst, self.tokens+(now-self.last)*self.rate); self.last=now
            if self.tokens >= n: self.tokens -= n; return True
            time.sleep(max(0, (n-self.tokens)/self.rate)); self.tokens=0; return True
    class UniversePrefilter:
        def __init__(self, k=20, min_score_bps=2.0): self.k=k; self.min=min_score_bps
        def pick(self, candidates, cost_model): 
            ranked=sorted(((s, v["alpha_hint_bps"]-v.get("spread_bps",0)) for s,v in candidates.items()),
                          key=lambda x:x[1], reverse=True)
            return [s for s,_ in ranked[:self.k]]
    class DriftHealth:
        def __init__(self): self.bad=0
        def update(self, *a, **k): pass
        def unhealthy(self): return False
        def critical(self): return False
    class ModeController:
        def __init__(self, cfg): self.mode="LIVE"; self.cfg=cfg; self.kill_file=pathlib.Path(cfg.get("kill_file","panic.flag"))
        def evaluate(self, health, broker_lat_ms, exc_count): 
            if self.kill_file.exists(): self.mode="OBSERVE"
            return self.mode
    class OrderReconciler:
        def __init__(self, broker, wal_path="state/wal_orders.jsonl"): self.broker=broker; pathlib.Path("state").mkdir(exist_ok=True); self.wal=pathlib.Path(wal_path)
        def submit(self, intent): 
            with self.wal.open("a") as f: f.write(json.dumps({"ts":time.time(),"intent":intent})+"\n")
            return self.broker.place_order(intent)
        def reconcile(self): pass
    class TradeCostAnalytics:
        def __init__(self, cfg): pass
        def record_fill(self, *a, **k): pass
        def nightly_autotune(self): pass
    class CostModel:
        def __init__(self, cfg): pass
        def calculate_costs(self, symbol, md): 
            class C: total=md.get("spread_bps",1.0)+md.get("impact_bps",0.5)
            return C()
        def quick(self, sym, spread_bps, vol_bps): return spread_bps + 0.5*vol_bps
    class DecisionOptimizer:
        def __init__(self, cfg): self.cfg=cfg

# --- Simple market data shape for guards/exec policy glue ---
@dataclass
class MarketData:
    bid: float
    ask: float
    mid: float
    last: float
    bid_size: float
    ask_size: float
    volume: float
    spread_bps: float
    vol_5m_bps: float = 0.0
    impact_bps: float = 0.0
    quote_age_ms: int = 0
    ssr_active: bool = False
    is_halted: bool = False
    luld_active: bool = False
    is_opening_auction: bool = False
    since_open_s: int = 0
    until_close_s: int = 0
    tick: float = 0.01
    nba: float = field(init=False)
    nbb: float = field(init=False)
    def __post_init__(self):
        self.nbb = self.bid; self.nba = self.ask

# === Orchestrator app ===

class IBKRIntradayApp:
    def __init__(self, cfg_path: str, symbols: List[str]):
        self.cfg_path = cfg_path
        self.cfg = self._load_cfg(cfg_path)
        self.symbols = symbols or self.cfg.get("symbols", [])
        self.log = logging.getLogger("IBKRIntradayApp")
        self.stop_event = threading.Event()
        self.exc_count = 0

        # IBKR
        self.ib = IB()
        self.broker = IBKRBrokerAdapter(self.cfg["ibkr"]) if "ibkr" in self.cfg else None

        # Core components
        self.model_loader = IBKRModelLoader(self.cfg.get("model_dir", "models"))
        self.feature_pipeline = IBKRFeaturePipeline(self.model_loader)
        self.cost_model = CostModel(self.cfg.get("costs", {}))
        self.inference_engine = IBKRModelInference(self.model_loader, self.cost_model)
        self.decision_optimizer = DecisionOptimizer(self.cfg.get("decision_optimization", {}))
        self.risk_manager = IBKRRiskManager(self.cfg.get("risk", {}))
        self.kill_switch = IBKRKillSwitch(self.cfg.get("kill_switches", {}))

        # Ensemble & decisions
        self.balancer = ZooBalancer(**self.cfg.get("decision_optimization", {}).get("ensemble", {}))
        self.arbiter = HorizonArbiter()
        self.barriers = BarrierGates()
        self.exec_policy = ShortHorizonExecutionPolicy(**self.cfg.get("decision_optimization", {}).get("execution", {}))

        # Safety & ops
        self.pretrade = PreTradeGuards(self.cfg.get("safety", {}), md_source=None, broker=self.broker) \
                        if "safety" in self.cfg else None
        self.margin_gate = MarginGate(self.ib)
        self.short_guard = ShortSaleGuard(self.ib)
        self.rate_limiter = RateLimiter(**self.cfg.get("ibkr", {}).get("pacing", {}))
        self.reconciler = OrderReconciler(self.broker, wal_path="state/wal_orders.jsonl")
        self.health = DriftHealth()
        self.mode = ModeController(self.cfg.get("health", {"kill_file":"panic.flag"}))
        self.tca = TradeCostAnalytics(self.cfg.get("tca", {}))
        self.horizons = self.cfg.get("horizons", ["5m","10m","15m","30m","60m"])
        self.families = self.cfg.get("families", [])
        self.thresholds = self.cfg.get("decision_optimization", {}).get("thresholds", {"enter_bps":2.0})
        self.barrier_th = self.cfg.get("decision_optimization", {}).get("barrier", {"block_peak":0.60, "block_y_peak":0.60, "prefer_valley":0.55})
        self.prefilter = UniversePrefilter(**self.cfg.get("prefilter", {"top_k":20, "min_score_bps":2.0}))

    # --- Setup & lifecycle ---

    def _load_cfg(self, p: str) -> Dict:
        with open(p, "r") as f:
            base = yaml.safe_load(f) or {}
        # Optionally merge cost_model.yaml & decision_optimization.yaml if provided
        for extra in ("config/cost_model.yaml", "config/decision_optimization.yaml"):
            if pathlib.Path(extra).exists():
                with open(extra, "r") as ef:
                    sub = yaml.safe_load(ef) or {}
                    base.update(sub)
        return base

    def connect(self):
        log = self.log
        ibcfg = self.cfg.get("ibkr", {})
        host, port, client_id = ibcfg.get("host","127.0.0.1"), ibcfg.get("port",7497), ibcfg.get("client_id",1)
        log.info(f"Connecting IBKR TWS/Gateway {host}:{port} (client_id={client_id})...")
        self.ib.connect(host, port, clientId=client_id, readonly=False, timeout=5.0)
        assert self.ib.isConnected(), "IBKR connect failed"
        self.broker.connect()  # your adapter should connect to same session
        log.info("Connected to IBKR.")

    def warmup(self):
        self.log.info("Loading models & calibrators...")
        self.model_loader.load_intraday_models(self.horizons, self.families)
        self.model_loader.load_ensemble_weights(self.cfg.get("ensemble_weights", "ensemble_weights.json"))
        self.model_loader.load_decision_weights(self.cfg.get("decision_weights", "decision_weights.json"))
        self.log.info("Warming subscriptions & caches...")
        # TODO: subscribe market data here using your broker adapter if needed

    def shutdown(self):
        self.log.info("Shutting down: reconcile and disconnect...")
        try:
            self.reconciler.reconcile()
        finally:
            if self.ib.isConnected(): self.ib.disconnect()

    # --- Signal handling & panic ---

    def _install_signals(self):
        def handler(sig, frame):
            self.log.warning(f"Signal {sig} received; stopping...")
            self.stop_event.set()
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, handler)

    def _check_panic(self):
        panic_file = pathlib.Path(self.cfg.get("dr", {}).get("panic_file", "panic.flag"))
        if panic_file.exists():
            self.log.error("PANIC flag present; flattening all positions and exiting.")
            try:
                self.broker.flatten_all()
            finally:
                self.stop_event.set()
            return True
        return False

    # --- Core helpers ---

    def _get_market_data_snap(self, syms: List[str]) -> Dict[str, MarketData]:
        # Use your broker adapter to build MarketData dataclasses
        snaps = {}
        raw = self.broker.get_market_data(syms)  # implement to be low-latency
        for s, md in raw.items():
            # Normalize into MarketData. Ensure these fields are present.
            snaps[s] = MarketData(
                bid=md.bid, ask=md.ask, mid=(md.bid+md.ask)/2 if md.bid and md.ask else md.last,
                last=md.last, bid_size=md.bid_size, ask_size=md.ask_size, volume=md.volume,
                spread_bps=0.0 if not md.bid or not md.ask else 10000.0*(md.ask-md.bid)/((md.ask+md.bid)/2),
                vol_5m_bps=getattr(md,"vol_5m_bps",0.0), impact_bps=getattr(md,"impact_bps",0.0),
                quote_age_ms=getattr(md,"quote_age_ms",0), ssr_active=getattr(md,"ssr_active",False),
                is_halted=getattr(md,"is_halted",False), luld_active=getattr(md,"luld_active",False),
                is_opening_auction=getattr(md,"is_opening_auction",False),
                since_open_s=getattr(md,"since_open_s",0), until_close_s=getattr(md,"until_close_s",0),
                tick=getattr(md,"tick",0.01)
            )
        return snaps

    def _prefilter(self, md: Dict[str, MarketData]) -> List[str]:
        # Cheap hints; replace with your own proxies
        candidates = {}
        for s, m in md.items():
            if m.bid and m.ask:
                microtrend = (m.last - m.mid) / (m.mid or 1)
                candidates[s] = {"alpha_hint_bps": 10000*microtrend, "spread_bps": m.spread_bps, "vol_bps": m.vol_5m_bps}
        return self.prefilter.pick(candidates, self.cost_model)

    def _compute_predictions(self, features_by_sym: Dict[str, 'pd.DataFrame']) -> Dict[str, Dict[str, float]]:
        return self.inference_engine.predict_all_models(features_by_sym)

    def _within_horizon_alpha(self, sym_preds: Dict[str, float], sym: str) -> Dict[str, float]:
        # sym_preds keys like "Family_5m"
        alpha_by_h = {}
        for h in self.horizons:
            fam_preds = {k.split("_")[0]: v for k,v in sym_preds.items() if k.endswith(f"_{h}")}
            if not fam_preds: 
                continue
            # TODO: plug real rolling stats/calibrators/ic/corr/cost_share from your store
            z = self.balancer.standardize(fam_preds, stats={f:(0.0,1.0) for f in fam_preds})
            cal = self.balancer.calibrate(z, calibrators={f:(lambda x:x) for f in fam_preds})
            w = self.balancer.weights(cal, ic={f:0.05 for f in fam_preds}, cost_share={f:0.2 for f in fam_preds},
                                      corr_mat=__import__("numpy").eye(len(fam_preds)), horizon=h)
            alpha_by_h[h] = self.balancer.blend(cal, w)
        return alpha_by_h

    def _barrier_preds(self, sym_preds: Dict[str, float]) -> Dict[str, float]:
        # Expect keys like "Barrier_peak_5m"; adapt to your loader's naming
        out = {
            "will_peak_5m": sym_preds.get("will_peak_5m", 0.0),
            "will_valley_5m": sym_preds.get("will_valley_5m", 0.0),
            "y_will_peak_5m": sym_preds.get("y_will_peak_5m", 0.0),
        }
        return out

    def _net_and_execute(self, intents: Dict[str, Dict]):
        # Net across horizons per symbol (simple), guard pacing, margin, short-sale, then submit
        for sym, intent in intents.items():
            if not intent: 
                continue
            side, size = intent["side"], intent["size"]
            md = intent["md"]; steps = intent["steps"]; h = intent["horizon"]

            # NBBO sanity
            ok, why = nbbo_sane(md)
            if not ok: 
                logging.info(f"{sym}: NBBO sanity fail: {why}"); 
                continue

            # Short-sale guard if SELL
            ok, why = self.short_guard.check(sym, md, side)
            if not ok:
                logging.info(f"{sym}: short guard fail: {why}")
                continue

            # Build broker intent payload expected by your adapter
            broker_intent = {
                "symbol": sym, "side": side, "quantity": size,
                "timeInForce": "DAY", "steps": steps, "idemp": f"{sym}-{int(time.time()*1000)}",
                "horizon": h, "limit_plan": steps
            }

            # Rate limit (API pacing)
            self.rate_limiter.acquire(1)

            # Margin what-if (optional: move into broker.place_intent)
            # Here we assume broker.place_order handles what-if; else call MarginGate

            # Submit via reconciler (idempotent + WAL)
            try:
                res = self.reconciler.submit(broker_intent)
                logging.info(f"{sym}: order submitted -> {res}")
            except Exception as e:
                self.exc_count += 1
                logging.exception(f"{sym}: submit error: {e}")

    # --- Main loop ---

    def run(self):
        self._install_signals()
        self.connect()
        self.warmup()

        log = self.log
        cadence_s = int(self.cfg.get("cadence_seconds", 300))  # default 5m
        reconcile_every_s = int(self.cfg.get("reconcile", {}).get("run_every_s", 60))
        last_reconcile = 0

        log.info("🚀 Starting IBKR intraday loop")
        while not self.stop_event.is_set():
            try:
                if self._check_panic(): break
                if self.kill_switch.check_kill_conditions():
                    log.warning("🛑 Kill switch triggered; stopping.")
                    break

                t0 = time.time()

                # 1) Market data snapshot
                md = self._get_market_data_snap(self.symbols)
                if self.pretrade is not None:
                    # optional global checks can live here
                    pass

                # 2) Prefilter cheap top-K to save compute
                trade_syms = self._prefilter(md)
                if not trade_syms:
                    time.sleep(min(1.0, cadence_s/10)); 
                    continue

                # 3) Features (incremental) & predictions
                features = self.feature_pipeline.compute_features({s: md[s] for s in trade_syms if s in md})
                raw_preds = self._compute_predictions(features)

                # 4) Build horizon alphas & arbitrate
                decisions = {}
                for sym in trade_syms:
                    sym_preds = raw_preds.get(sym, {})
                    if not sym_preds:
                        continue

                    # Horizon blend
                    alpha_by_h = self._within_horizon_alpha(sym_preds, sym)
                    if not alpha_by_h:
                        continue

                    # Across-horizon arbitration (cost-aware inside arbiter)
                    h_star, alpha_star, scores = self.arbiter.choose(alpha_by_h, md[sym], self.thresholds)
                    if not h_star:
                        continue

                    # Barrier gating (5m focused)
                    bar = self._barrier_preds(sym_preds)
                    ok, why = self.barriers.allow_long_entry(
                        p_peak5=bar.get("will_peak_5m", 0.0),
                        p_valley5=bar.get("will_valley_5m", 0.0),
                        p_y_peak5=bar.get("y_will_peak_5m", 0.0),
                        thresholds=type("TH", (), self.barrier_th)()
                    )
                    if not ok:
                        logging.info(f"{sym}: barrier gate blocked: {why}")
                        continue

                    # Cost model & risk sizing
                    costs = self.cost_model.calculate_costs(sym, vars(md[sym]))
                    net_alpha_bps = alpha_star - costs.total
                    if net_alpha_bps < self.thresholds.get("enter_bps", 2.0):
                        continue

                    # Position size (risk-aware)
                    portfolio = self.broker.portfolio_view()
                    size_dollars = self.risk_manager.size_from_signal(sym, net_alpha_bps, md[sym], portfolio)
                    if size_dollars <= 0:
                        continue

                    # Execution plan (TIF+step-ups)
                    side = "BUY" if net_alpha_bps > 0 else "SELL"
                    steps = self.exec_policy.plan(side, md[sym].mid, md[sym].tick, score=scores.get(h_star, 0.0))

                    # Pre-trade guards (fail-closed)
                    if self.pretrade is not None:
                        ok, reason = self.pretrade.check_symbol(sym, datetime.now(timezone.utc))
                        if not ok:
                            logging.info(f"{sym}: pretrade guard blocked: {reason}")
                            continue

                    # Save decision
                    decisions[sym] = {"side": side, "size": size_dollars / max(md[sym].mid, 1e-6),
                                      "steps": steps, "horizon": h_star, "md": md[sym], "alpha_bps": net_alpha_bps}

                # 5) Netting across horizons (already per sym); could add multi-sym churn suppression here

                # 6) Risk constraints portfolio-wide
                risk_adj = self.risk_manager.apply_risk_constraints(decisions, self.broker.positions, self.broker.portfolio_value)

                # 7) Execute through reconciler (idempotent)
                self._net_and_execute(risk_adj)

                # 8) Periodics: reconcile, health/mode, TCA/autotune (off-cadence)
                now = time.time()
                if now - last_reconcile >= reconcile_every_s:
                    self.reconciler.reconcile()
                    last_reconcile = now

                mode = self.mode.evaluate(self.health, broker_latency_ms=0, exception_count=self.exc_count)
                if mode != "LIVE":
                    logging.warning(f"Mode switched to {mode}")

                # 9) Sleep until next slice within cadence to keep loop responsive
                elapsed = time.time() - t0
                sleep_s = max(0.25, min(1.0, (cadence_s/10) - elapsed))  # fine-grained loop; your bar builder drives 5m rolls
                time.sleep(sleep_s)

            except Exception as e:
                self.exc_count += 1
                logging.exception(f"Loop error: {e}")
                self.kill_switch.handle_exception(e)
                time.sleep(1.0)

        self.shutdown()


# --- CLI & bootstrap ---

def setup_logging(verbosity: int = 1, logfile: Optional[str] = None):
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        pathlib.Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

def parse_args():
    ap = argparse.ArgumentParser(description="IBKR Live Intraday Execution Orchestrator")
    ap.add_argument("--config", required=True, help="Path to ibkr_live.yaml (can merge other configs)")
    ap.add_argument("--symbols", type=str, default="", help="Comma-separated symbols (override config)")
    ap.add_argument("--log", type=str, default="logs/ibkr_live_exec.log")
    ap.add_argument("-v", "--verbose", action="count", default=1)
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose, args.log)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else []
    app = IBKRIntradayApp(cfg_path=args.config, symbols=symbols)
    app.run()

if __name__ == "__main__":
    main()
