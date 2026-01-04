# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
PyTorch Base Trainer for Sequential Models
==========================================

Minimal, reusable PyTorch training loop for (B, T, F) → scalar.
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.debug("Config loader not available; using hardcoded defaults")

# CRITICAL: Guard torch import to prevent libiomp5 in CPU-only children
_TORCH_DISABLED = os.getenv("TRAINER_CHILD_NO_TORCH", "0") == "1"
if not _TORCH_DISABLED:
    import torch
    from torch.utils.data import Dataset, DataLoader
else:
    # Sentinel values so module loads but torch is None
    torch = None
    Dataset = object
    DataLoader = None

logger = logging.getLogger(__name__)

class _SeqDataset(Dataset):
    """Simple dataset for sequential data."""
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self): 
        return self.X.shape[0]
    
    def __getitem__(self, i): 
        return self.X[i], self.y[i]

class SeqTorchTrainerBase:
    """Minimal, reusable PyTorch training loop for (B, T, F) → scalar."""
    
    def _get_max_norm(self) -> float:
        """Get gradient clipping max_norm from config, with fallback to default."""
        if _CONFIG_AVAILABLE:
            try:
                max_norm = get_cfg("safety.gradient_clipping.max_norm", default=1.0, config_name="safety_config")
                return float(max_norm)
            except Exception as e:
                logger.debug(f"Failed to load max_norm from config: {e}")
        return 1.0  # Default fallback
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = {**{
            "epochs": 30, 
            "batch_size": 512,
            "lr": 1e-3, 
            "weight_decay": 0.0,
            "num_workers": 2, 
            "pin_memory": True,
            "amp": True, 
            "early_stop": 6,
        }, **(config or {})}
        
        # STRICT MODE: Force CPU for determinism
        force_cpu = False
        try:
            from TRAINING.common.determinism import is_strict_mode
            if is_strict_mode():
                force_cpu = True
                import logging
                logging.getLogger(__name__).info("[PyTorch] Strict mode: forcing CPU (GPU disabled for determinism)")
        except ImportError:
            pass  # determinism module not available, continue normally
        
        self.device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(int(self.config.get("num_threads", 1)))
        self.model.to(self.device)
        
        if self.config["amp"]:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        else:
            self.scaler = None

    def _make_loaders(self, X, y):
        """Create train/validation data loaders."""
        n = len(X)
        v = max(1024, int(0.1 * n))
        ds_tr = _SeqDataset(X[:-v], y[:-v])
        ds_va = _SeqDataset(X[-v:], y[-v:])
        
        kw = dict(
            batch_size=self.config["batch_size"], 
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"], 
            drop_last=False
        )
        return DataLoader(ds_tr, shuffle=True, **kw), DataLoader(ds_va, shuffle=False, **kw)

    def train(self, X, y):
        """Train the model with early stopping."""
        # Configure PyTorch threading
        if torch.cuda.is_available():
            torch.set_num_threads(1)  # Keep CPU light for GPU
        else:
            torch.set_num_threads(self.config.get("num_threads", 4))
        
        dl_tr, dl_va = self._make_loaders(X, y)
        
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"], 
            weight_decay=self.config["weight_decay"]
        )
        crit = torch.nn.MSELoss()
        
        best = {"loss": float("inf"), "state": None, "epoch": -1}
        patience = 0
        
        for ep in range(self.config["epochs"]):
            # Training
            self.model.train()
            t0 = time.time()
            loss_sum = 0.0
            nobs = 0
            
            for xb, yb in dl_tr:
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        pred = self.model(xb)
                        loss = crit(pred, yb)
                    self.scaler.scale(loss).backward()
                    # Load max_norm from config if available
                    max_norm = self._get_max_norm()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    pred = self.model(xb)
                    loss = crit(pred, yb)
                    loss.backward()
                    # Load max_norm from config if available
                    max_norm = self._get_max_norm()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    opt.step()
                
                bs = xb.shape[0]
                loss_sum += loss.item() * bs
                nobs += bs

            # Validation
            self.model.eval()
            va_sum = 0.0
            vn = 0
            
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    l = crit(pred, yb)
                    bs = xb.shape[0]
                    va_sum += l.item() * bs
                    vn += bs
            
            va = va_sum / max(1, vn)
            
            logger.info(f"[TorchSeq] ep {ep+1}/{self.config['epochs']} train_n={nobs} val_loss={va:.6f} ({time.time()-t0:.1f}s)")
            
            if va < best["loss"] - 1e-6:
                best.update({"loss": va, "state": self.model.state_dict(), "epoch": ep})
                patience = 0
            else:
                patience += 1
                if patience >= self.config["early_stop"]:
                    logger.info(f"[TorchSeq] Early stop at ep {ep+1}, best={best['loss']:.6f} @ {best['epoch']+1}")
                    break

        if best["state"] is not None:
            self.model.load_state_dict(best["state"])
        return self.model

    @torch.no_grad()
    def predict(self, X):
        """Make predictions on new data."""
        self.model.eval()
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        preds = []
        
        for i in range(0, len(X), 4096):
            out = self.model(X[i:i+4096])
            preds.append(out.float().detach().cpu().numpy().reshape(-1))
        
        return np.nan_to_num(np.concatenate(preds), nan=0.0).astype(np.float32)
