"""Evaluate all transformer h5 outputs and print a table.

Loads MC_Maze targets once, then scores every mc_maze_{mt,ar,vq,hmm}_*_output_val.h5
file it finds in ./outputs/.
"""
from pathlib import Path

import h5py

from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import make_eval_target_tensors
from nlb_tools.nwb_interface import NWBDataset

DATASET_NAME = "mc_maze"
BIN_SIZE_MS = 5
DATAPATH = Path(__file__).resolve().parent / "data" / "000128" / "sub-Jenkins"
DIR = Path(__file__).resolve().parent / "outputs"


def load_h5(path):
    with h5py.File(path, "r") as f:
        k = list(f.keys())[0]
        return {k: {name: f[k][name][...] for name in f[k]}}


def main():
    print(f"Loading NWB from {DATAPATH} ...")
    ds = NWBDataset(str(DATAPATH), "*full", skip_fields=[
        "hand_pos", "cursor_pos", "eye_pos",
        "muscle_vel", "muscle_len", "joint_vel", "joint_ang", "force",
    ])
    ds.resample(BIN_SIZE_MS)
    target_dict = make_eval_target_tensors(
        ds, DATASET_NAME, "train", "val", save_file=False, include_psth=False
    )

    patterns = ["mc_maze_mt_*_output_val.h5", "mc_maze_ar_*_output_val.h5", "mc_maze_vq_*_output_val.h5", "mc_maze_hmm_*_output_val.h5"]
    paths = sorted(p for pat in patterns for p in DIR.glob(pat))
    print(f"Found {len(paths)} h5 files")
    print()
    print(f"{'config':<15}  {'co-bps':>8}  {'vel R2':>8}")
    print("-" * 37)
    for p in paths:
        name = p.stem.replace("mc_maze_", "").replace("_output_val", "")
        try:
            res = evaluate(target_dict, load_h5(p))[0][f"{DATASET_NAME}_split"]
            print(f"{name:<32}  {res['co-bps']:>8.4f}  {res['vel R2']:>8.4f}")
        except Exception as e:
            print(f"{name:<32}  ERROR  {e!r}")


if __name__ == "__main__":
    main()
