"""Evaluate all transformer h5 outputs and print a table.

Loads the NLB_DATASET targets once, scores every
<dataset>_{mt,ar,vq,hmm}_*_output_val.h5 file it finds in ./outputs/.
"""
import sys
from pathlib import Path

import h5py

from nlb_tools.evaluation import evaluate
from nlb_tools.make_tensors import make_eval_target_tensors
from nlb_tools.nwb_interface import NWBDataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "baselines"))
from nlb_dataset import DATASET_NAME, DATAPATH, NWB_PREFIX  # noqa: E402
BIN_SIZE_MS = 5
DIR = Path(__file__).resolve().parent / "outputs"


def load_h5(path):
    with h5py.File(path, "r") as f:
        k = list(f.keys())[0]
        return {k: {name: f[k][name][...] for name in f[k]}}


def main():
    print(f"Loading NWB from {DATAPATH} ...")
    ds = NWBDataset(str(DATAPATH), NWB_PREFIX, skip_fields=[
        "hand_pos", "cursor_pos", "eye_pos",
        "muscle_vel", "muscle_len", "joint_vel", "joint_ang", "force",
    ])
    ds.resample(BIN_SIZE_MS)
    target_dict = make_eval_target_tensors(
        ds, DATASET_NAME, "train", "val", save_file=False, include_psth=False
    )

    patterns = [f"{DATASET_NAME}_{m}_*_output_val.h5" for m in ("mt", "ar", "vq", "hmm")]
    paths = sorted(p for pat in patterns for p in DIR.glob(pat))
    print(f"Found {len(paths)} h5 files")
    print()
    print(f"{'config':<15}  {'co-bps':>8}  {'vel R2':>8}")
    print("-" * 37)
    for p in paths:
        name = p.stem.replace(f"{DATASET_NAME}_", "").replace("_output_val", "")
        try:
            res = evaluate(target_dict, load_h5(p))[0][f"{DATASET_NAME}_split"]
            print(f"{name:<32}  {res['co-bps']:>8.4f}  {res['vel R2']:>8.4f}")
        except Exception as e:
            print(f"{name:<32}  ERROR  {e!r}")


if __name__ == "__main__":
    main()
