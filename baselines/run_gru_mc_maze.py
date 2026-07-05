"""Simple bidirectional GRU decoder for MC_Maze val.

Predicts firing rates for held-in + held-out neurons directly from held-in spikes.
Trained with Poisson NLL over concatenated (heldin, heldout) targets on train trials.

Architecture:
  spikes (B, T, n_heldin)
    -> BiGRU (hidden=hidden_size, layers=n_layers, dropout)
    -> Linear(2*hidden_size -> n_heldin + n_heldout)
    -> softplus (positive rates)

Config chosen by env var GRU_CONFIG (default v1 = as-shipped in W27 commit 4810441).
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import (
    make_eval_input_tensors,
    make_eval_target_tensors,
    make_train_input_tensors,
    save_to_h5,
)
from nlb_tools.nwb_interface import NWBDataset

DATASET_NAME = "mc_maze"
BIN_SIZE_MS = 5
PHASE = "val"
BINSUF = "" if BIN_SIZE_MS == 5 else f"_{BIN_SIZE_MS}"
OUTPUT_KEY = DATASET_NAME + BINSUF

CONFIGS = {
    "v1": dict(hidden=128, layers=2, dropout=0.30, lr=3e-3, batch=64, epochs=200, cosine=False, weight_decay=0.0),
    "v2": dict(hidden=192, layers=2, dropout=0.35, lr=3e-3, batch=64, epochs=500, cosine=True,  weight_decay=1e-5),
}
CONFIG = os.getenv("GRU_CONFIG", "v1")
if CONFIG not in CONFIGS:
    raise ValueError(f"unknown GRU_CONFIG {CONFIG}; choose from {list(CONFIGS)}")
C = CONFIGS[CONFIG]
HIDDEN_SIZE, N_LAYERS, DROPOUT = C["hidden"], C["layers"], C["dropout"]
LR, BATCH_SIZE, EPOCHS = C["lr"], C["batch"], C["epochs"]
COSINE, WEIGHT_DECAY = C["cosine"], C["weight_decay"]
VAL_SPLIT = 0.1
SEED = 0

DATAPATH = Path(__file__).resolve().parents[1] / "data" / "000128" / "sub-Jenkins"
SAVEPATH = Path(__file__).resolve().parents[1] / "outputs" / f"{OUTPUT_KEY}_gru_{CONFIG}_output_{PHASE}.h5"


class BiGRURates(nn.Module):
    def __init__(self, n_heldin, n_out, hidden_size, n_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_heldin, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * hidden_size, n_out)

    def forward(self, x):
        h, _ = self.gru(x)
        return nn.functional.softplus(self.head(h))


def poisson_nll(rates, spikes, eps=1e-8):
    return (rates - spikes * torch.log(rates + eps)).mean()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cpu"

    print(f"Loading NWB from {DATAPATH} ...")
    dataset = NWBDataset(
        str(DATAPATH),
        "*full",
        skip_fields=[
            "hand_pos", "cursor_pos", "eye_pos",
            "muscle_vel", "muscle_len", "joint_vel", "joint_ang", "force",
        ],
    )
    dataset.resample(BIN_SIZE_MS)

    train_split, eval_split = "train", "val"
    train_dict = make_train_input_tensors(dataset, DATASET_NAME, train_split, save_file=False)
    x_train = train_dict["train_spikes_heldin"].astype(np.float32)      # (N, T, Hi)
    y_train_ho = train_dict["train_spikes_heldout"].astype(np.float32)  # (N, T, Ho)
    eval_dict = make_eval_input_tensors(dataset, DATASET_NAME, eval_split, save_file=False)
    x_eval = eval_dict["eval_spikes_heldin"].astype(np.float32)         # (M, T, Hi)

    n_train, T, n_heldin = x_train.shape
    n_heldout = y_train_ho.shape[2]
    n_out = n_heldin + n_heldout
    n_eval = x_eval.shape[0]
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

    print(f"config={CONFIG} hidden={HIDDEN_SIZE} layers={N_LAYERS} dropout={DROPOUT} lr={LR} bs={BATCH_SIZE} epochs={EPOCHS} cosine={COSINE} wd={WEIGHT_DECAY}")
    model = BiGRURates(n_heldin, n_out, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1) if COSINE else None

    best_val = float("inf")
    best_state = None
    n_batches = int(np.ceil(len(tr_idx) / BATCH_SIZE))
    for epoch in range(EPOCHS):
        model.train()
        perm_e = torch.randperm(len(tr_idx))
        train_loss = 0.0
        for b in range(n_batches):
            idx = perm_e[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            rates = model(x_tr[idx])
            loss = poisson_nll(rates, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(idx)
        train_loss /= len(tr_idx)

        if sched is not None:
            sched.step()
        model.eval()
        with torch.no_grad():
            val_rates = model(x_val)
            val_loss = poisson_nll(val_rates, y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  best {best_val:.4f}")

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
    SAVEPATH.parent.mkdir(parents=True, exist_ok=True)
    save_to_h5(output_dict, str(SAVEPATH), overwrite=True)
    print(f"Wrote {SAVEPATH}")

    target_dict = make_eval_target_tensors(
        dataset, DATASET_NAME, train_split, eval_split, save_file=False, include_psth=False
    )
    print("Evaluation:")
    print(evaluate(target_dict, output_dict))


if __name__ == "__main__":
    main()
