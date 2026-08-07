"""Masked transformer for MC_Maze val (custom, GRAFT-simplified).

Architecture (see masked_transformer.md):
  spikes[B, T, Hi]
    -> Linear(Hi -> d)  (per-neuron read-in via row of E_in)
    + learned pos[T, d]
    -> 4x TransformerEncoderLayer (d=128, heads=4, ffn=512, gelu, pre-norm)
    -> Linear(d -> Hi+Ho)
    -> softplus (positive rates)

Training loss:
  - random 25% of (t, n) heldin entries zeroed in input.
  - Poisson NLL on heldin, only at masked entries.
  - Poisson NLL on heldout, at all entries (co-smoothing target).
  - L = L_heldin_masked + L_heldout.

Eval:
  - forward pass unmasked heldin -> rates for all Hi+Ho at all T.
  - co-bps + vel R² via nlb_tools.evaluation.evaluate.

Config chosen by env var MT_CONFIG (default v1).
"""
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

DATASET_NAME = "mc_maze"
BIN_SIZE_MS = 5
PHASE = "val"
BINSUF = "" if BIN_SIZE_MS == 5 else f"_{BIN_SIZE_MS}"
OUTPUT_KEY = DATASET_NAME + BINSUF

_BASE = dict(d_model=128, n_layers=4, n_heads=4, ffn=512, dropout=0.1,
             mask_rate=0.25, lr=3e-4, wd=1e-4, batch=64, epochs=500,
             warmup=500, min_lr=3e-5)
CONFIGS = {
    "v1": {**_BASE},
    "v2": {**_BASE, "mask_rate": 0.40},
    "v3": {**_BASE, "d_model": 192, "ffn": 768, "n_heads": 6},
    "v4": {**_BASE, "n_layers": 6},
    "v4_long": {**_BASE, "n_layers": 6, "epochs": 1500, "warmup": 1500},
    "v5": {**_BASE, "d_model": 192, "ffn": 768, "n_heads": 6, "n_layers": 6, "mask_rate": 0.40},
    "smoke": {**_BASE, "epochs": 10, "warmup": 50},
}
CONFIG = os.getenv("MT_CONFIG", "v1")
if CONFIG not in CONFIGS:
    raise ValueError(f"unknown MT_CONFIG {CONFIG}; choose from {list(CONFIGS)}")
C = CONFIGS[CONFIG]
D_MODEL, N_LAYERS, N_HEADS, FFN = C["d_model"], C["n_layers"], C["n_heads"], C["ffn"]
DROPOUT, MASK_RATE = C["dropout"], C["mask_rate"]
LR, WD, BATCH_SIZE, EPOCHS = C["lr"], C["wd"], C["batch"], C["epochs"]
WARMUP, MIN_LR = C["warmup"], C["min_lr"]
VAL_SPLIT = 0.1
SEED = 0

NPZPATH = Path(__file__).resolve().parents[1] / "data" / f"{DATASET_NAME}_{BIN_SIZE_MS}ms.npz"
SAVEPATH = Path(__file__).resolve().parents[1] / "outputs" / f"{OUTPUT_KEY}_mt_{CONFIG}_output_{PHASE}.h5"


class MaskedTransformer(nn.Module):
    def __init__(self, n_heldin, n_out, T, d_model, n_layers, n_heads, ffn, dropout):
        super().__init__()
        self.read_in = nn.Linear(n_heldin, d_model, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, T, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.read_out = nn.Linear(d_model, n_out)

    def forward(self, x):
        h = self.read_in(x) + self.pos
        h = self.encoder(h)
        return nn.functional.softplus(self.read_out(h))


def poisson_nll_elementwise(rates, spikes, eps=1e-8):
    return rates - spikes * torch.log(rates + eps)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"Loading pre-baked tensors from {NPZPATH} ...")
    npz = np.load(NPZPATH)
    x_train = npz["train_heldin"].astype(np.float32)      # (N, T, Hi)
    y_train_ho = npz["train_heldout"].astype(np.float32)  # (N, T, Ho)
    x_eval = npz["eval_heldin"].astype(np.float32)        # (M, T, Hi)

    n_train, T, n_heldin = x_train.shape
    n_heldout = y_train_ho.shape[2]
    n_out = n_heldin + n_heldout
    print(f"train {x_train.shape} heldout {y_train_ho.shape} eval {x_eval.shape}")

    y_train = np.concatenate([x_train, y_train_ho], axis=2)  # (N, T, Hi+Ho)

    perm = np.random.permutation(n_train)
    n_val = int(round(n_train * VAL_SPLIT))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    x_tr = torch.from_numpy(x_train[tr_idx]).to(device)
    y_tr = torch.from_numpy(y_train[tr_idx]).to(device)
    x_val = torch.from_numpy(x_train[val_idx]).to(device)
    y_val = torch.from_numpy(y_train[val_idx]).to(device)

    print(f"config={CONFIG} d={D_MODEL} L={N_LAYERS} H={N_HEADS} ffn={FFN} drop={DROPOUT} "
          f"mask={MASK_RATE} lr={LR} wd={WD} bs={BATCH_SIZE} ep={EPOCHS}")
    model = MaskedTransformer(n_heldin, n_out, T, D_MODEL, N_LAYERS, N_HEADS, FFN, DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n_batches = int(np.ceil(len(tr_idx) / BATCH_SIZE))
    total_steps = n_batches * EPOCHS

    def lr_mul(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        progress = min(1.0, (step - WARMUP) / max(1, total_steps - WARMUP))
        floor = MIN_LR / LR
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mul)

    best_val = float("inf")
    best_state = None
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        perm_e = torch.randperm(len(tr_idx))
        train_loss = 0.0
        for b in range(n_batches):
            idx = perm_e[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            x = x_tr[idx]
            y = y_tr[idx]
            keep = (torch.rand_like(x) >= MASK_RATE).float()
            rates = model(x * keep)
            hi_err = poisson_nll_elementwise(rates[:, :, :n_heldin], x)
            miss = 1.0 - keep
            denom = miss.sum().clamp(min=1.0)
            l_hi = (hi_err * miss).sum() / denom
            l_ho = poisson_nll_elementwise(rates[:, :, n_heldin:], y[:, :, n_heldin:]).mean()
            loss = l_hi + l_ho
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            train_loss += loss.item() * len(idx)
            step += 1
        train_loss /= len(tr_idx)

        model.eval()
        with torch.no_grad():
            val_rates = model(x_val)
            val_hi = poisson_nll_elementwise(val_rates[:, :, :n_heldin], x_val).mean()
            val_ho = poisson_nll_elementwise(val_rates[:, :, n_heldin:], y_val[:, :, n_heldin:]).mean()
            val_loss = (val_hi + val_ho).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"best {best_val:.4f}  lr {opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_rates = model(torch.from_numpy(x_train).to(device)).cpu().numpy()
        eval_rates = model(torch.from_numpy(x_eval).to(device)).cpu().numpy()

    output_dict = {
        OUTPUT_KEY: {
            "train_rates_heldin": train_rates[:, :, :n_heldin],
            "train_rates_heldout": train_rates[:, :, n_heldin:],
            "eval_rates_heldin": eval_rates[:, :, :n_heldin],
            "eval_rates_heldout": eval_rates[:, :, n_heldin:],
        }
    }
    # Save h5 in the format nlb_tools.evaluate expects (matches GRU/GPFA h5).
    SAVEPATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(SAVEPATH, "w") as f:
        g = f.create_group(OUTPUT_KEY)
        for name, arr in output_dict[OUTPUT_KEY].items():
            g.create_dataset(name, data=arr)
    print(f"Wrote {SAVEPATH}")
    print("Next: score all transformer outputs with: python eval_all_mt.py")


if __name__ == "__main__":
    main()
