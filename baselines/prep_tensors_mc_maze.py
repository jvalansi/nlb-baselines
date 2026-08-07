"""Bake MC_Maze train + eval tensors to npz.

Run on a CPU box with nlb_tools' pinned pandas 1.3.4 — pandas 2.x drops trials
differently (NaN handling), which causes eval-time mismatches between input
and target trial counts.

Output: data/mc_maze_5ms.npz with keys
    train_heldin  (N, T, Hi)
    train_heldout (N, T, Ho)
    eval_heldin   (M, T, Hi)
"""
from pathlib import Path

import numpy as np

from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors
from nlb_tools.nwb_interface import NWBDataset

DATASET_NAME = "mc_maze"
BIN_SIZE_MS = 5
DATAPATH = Path(__file__).resolve().parents[1] / "data" / "000128" / "sub-Jenkins"
OUTPATH = Path(__file__).resolve().parents[1] / "data" / f"{DATASET_NAME}_{BIN_SIZE_MS}ms.npz"


def main():
    print(f"Loading NWB from {DATAPATH} ...")
    ds = NWBDataset(str(DATAPATH), "*full", skip_fields=[
        "hand_pos", "cursor_pos", "eye_pos",
        "muscle_vel", "muscle_len", "joint_vel", "joint_ang", "force",
    ])
    ds.resample(BIN_SIZE_MS)

    train_dict = make_train_input_tensors(ds, DATASET_NAME, "train", save_file=False)
    eval_dict = make_eval_input_tensors(ds, DATASET_NAME, "val", save_file=False)

    train_heldin = train_dict["train_spikes_heldin"].astype(np.float32)
    train_heldout = train_dict["train_spikes_heldout"].astype(np.float32)
    eval_heldin = eval_dict["eval_spikes_heldin"].astype(np.float32)

    print(f"train_heldin  {train_heldin.shape}")
    print(f"train_heldout {train_heldout.shape}")
    print(f"eval_heldin   {eval_heldin.shape}")

    OUTPATH.parent.mkdir(exist_ok=True)
    np.savez(OUTPATH, train_heldin=train_heldin, train_heldout=train_heldout, eval_heldin=eval_heldin)
    print(f"Wrote {OUTPATH} ({OUTPATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
