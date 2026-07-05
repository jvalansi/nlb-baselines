"""MC_Maze spike-smoothing baseline (NLB'21).

Adapted from nlb_tools/examples/baselines/smoothing/run_smoothing.py:
- dataset fixed to mc_maze
- phase='val' so we can evaluate locally (test submission closed Jan 2026)
- datapath points at bci/nlb_tools/data/
"""
from pathlib import Path

import numpy as np
import scipy.signal as signal
from sklearn.linear_model import PoissonRegressor

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
KERN_SD = 50
ALPHA = 0.01
PHASE = "val"
LOG_OFFSET = 1e-4
BINSUF = "" if BIN_SIZE_MS == 5 else f"_{BIN_SIZE_MS}"
OUTPUT_KEY = DATASET_NAME + BINSUF

DATAPATH = Path(__file__).resolve().parents[1] / "data" / "000128" / "sub-Jenkins"
SAVEPATH = Path(__file__).resolve().parents[1] / "outputs" / f"{OUTPUT_KEY}_smoothing_output_{PHASE}.h5"


def main() -> None:
    print(f"Loading NWB from {DATAPATH} ...")
    dataset = NWBDataset(
        str(DATAPATH),
        "*full",
        skip_fields=[
            "hand_pos",
            "cursor_pos",
            "eye_pos",
            "muscle_vel",
            "muscle_len",
            "joint_vel",
            "joint_ang",
            "force",
        ],
    )
    dataset.resample(BIN_SIZE_MS)

    if PHASE == "val":
        train_split = "train"
        eval_split = "val"
    else:
        train_split = ["train", "val"]
        eval_split = "test"

    train_dict = make_train_input_tensors(dataset, DATASET_NAME, train_split, save_file=False)
    train_spikes_heldin = train_dict["train_spikes_heldin"]
    train_spikes_heldout = train_dict["train_spikes_heldout"]

    eval_dict = make_eval_input_tensors(dataset, DATASET_NAME, eval_split, save_file=False)
    eval_spikes_heldin = eval_dict["eval_spikes_heldin"]

    tlen = train_spikes_heldin.shape[1]
    num_heldout = train_spikes_heldout.shape[2]

    window = signal.windows.gaussian(int(6 * KERN_SD / BIN_SIZE_MS), int(KERN_SD / BIN_SIZE_MS), sym=True)
    window /= np.sum(window)

    def filt(x):
        return np.convolve(x, window, "same")

    train_spksmth_heldin = np.apply_along_axis(filt, 1, train_spikes_heldin)
    eval_spksmth_heldin = np.apply_along_axis(filt, 1, eval_spikes_heldin)

    flatten2d = lambda x: x.reshape(-1, x.shape[2])
    train_spksmth_heldin_s = flatten2d(train_spksmth_heldin)
    train_spikes_heldout_s = flatten2d(train_spikes_heldout)
    eval_spksmth_heldin_s = flatten2d(eval_spksmth_heldin)

    train_lograte_heldin_s = np.log(train_spksmth_heldin_s + LOG_OFFSET)
    eval_lograte_heldin_s = np.log(eval_spksmth_heldin_s + LOG_OFFSET)

    print(f"Fitting {num_heldout} Poisson GLMs (heldin log-rates -> heldout spikes)...")
    train_pred, eval_pred = [], []
    for chan in range(num_heldout):
        pr = PoissonRegressor(alpha=ALPHA, max_iter=500)
        pr.fit(train_lograte_heldin_s, train_spikes_heldout_s[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            oldmax = pr.max_iter
            pr = PoissonRegressor(alpha=ALPHA, max_iter=oldmax * 5)
            pr.fit(train_lograte_heldin_s, train_spikes_heldout_s[:, chan])
            print(f"  chan {chan}: retrained with max_iter={pr.max_iter}")
        train_pred.append(pr.predict(train_lograte_heldin_s))
        eval_pred.append(pr.predict(eval_lograte_heldin_s))

    train_spksmth_heldout = np.clip(np.vstack(train_pred).T, 1e-9, 1e20).reshape((-1, tlen, num_heldout))
    eval_spksmth_heldout = np.clip(np.vstack(eval_pred).T, 1e-9, 1e20).reshape((-1, tlen, num_heldout))

    output_dict = {
        OUTPUT_KEY: {
            "train_rates_heldin": train_spksmth_heldin,
            "train_rates_heldout": train_spksmth_heldout,
            "eval_rates_heldin": eval_spksmth_heldin,
            "eval_rates_heldout": eval_spksmth_heldout,
        }
    }
    SAVEPATH.parent.mkdir(parents=True, exist_ok=True)
    save_to_h5(output_dict, str(SAVEPATH), overwrite=True)
    print(f"Wrote {SAVEPATH}")

    if PHASE == "val":
        target_dict = make_eval_target_tensors(
            dataset, DATASET_NAME, train_split, eval_split, save_file=False, include_psth=False
        )
        print("Evaluation:")
        print(evaluate(target_dict, output_dict))


if __name__ == "__main__":
    main()
