"""Training loop: AdamW, cosine schedule from 1e-3, gradient clipping at 1.0, early stopping on macro-F1."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score

from .losses import class_balanced_weights, xplainonco_loss


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip: float = 1.0
    lam1: float = 1e-3  # attention-entropy weight (paper grid: 1e-4 .. 1e-1)
    lam2: float = 1e-4  # gate L1 weight
    cb_beta: float = 0.999
    patience: int = 12
    cond_lr_mult: float = 0.1  # conditioning nets learn slower, so they grow only as far as the loss rewards
    seed: int = 0
    verbose: bool = False


@torch.no_grad()
def predict_proba(model, X: torch.Tensor, z: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        out.append(torch.softmax(model(X[i : i + batch_size], z[i : i + batch_size]), -1))
    return torch.cat(out).cpu().numpy()


def fit(model, X, z, y, train_idx, val_idx, cfg: TrainConfig):
    """Train on ``train_idx``; early-stop on macro-F1 over ``val_idx``. Returns the best-state model and history."""
    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    n_classes = model.cfg.n_classes
    cw = class_balanced_weights(y[train_idx].numpy(), n_classes, cfg.cb_beta)
    cond = [p for n, p in model.named_parameters() if any(k in n for k in (".gamma.", ".kappa.", ".gamma0."))]
    cond_ids = {id(p) for p in cond}
    rest = [p for p in model.parameters() if id(p) not in cond_ids]
    groups = [{"params": rest, "lr": cfg.lr}]
    if cond:
        groups.append({"params": cond, "lr": cfg.lr * cfg.cond_lr_mult})
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    best, best_state, bad = -1.0, None, 0
    hist = []
    tr = torch.as_tensor(train_idx)
    for ep in range(cfg.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        tot = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            bi = perm[i : i + cfg.batch_size]
            if len(bi) < 2:
                continue
            logits = model(X[bi], z[bi])
            ent, l1 = model.regularisers()
            loss, ce = xplainonco_loss(logits, y[bi], cw, ent, l1, cfg.lam1, cfg.lam2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        p = predict_proba(model, X[val_idx], z[val_idx])
        f1 = f1_score(y[val_idx].numpy(), p.argmax(1), average="macro")
        hist.append({"epoch": ep, "loss": tot / len(tr), "val_macro_f1": float(f1)})
        if cfg.verbose:
            print(f"  epoch {ep:3d}  loss {tot / len(tr):.4f}  val macro-F1 {f1:.3f}")
        if f1 > best + 1e-4:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist
