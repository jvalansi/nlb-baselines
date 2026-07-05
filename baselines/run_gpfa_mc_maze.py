"""GPFA baseline for MC_Maze val phase.

Adapted from nlb_tools/examples/baselines/gpfa/run_gpfa.py.
- dataset_name = mc_maze; phase = val
- datapath points at bci/nlb_tools/data/
- include_psth=False to avoid the multiprocessing Pool cleanup issue
- scipy shim: elephant 1.0.0 imports scipy.integrate.simps (removed in scipy 1.11);
  alias to scipy.integrate.simpson before importing elephant.
"""
import scipy.integrate as _si
_si.simps = _si.simpson  # noqa: E402

from pathlib import Path

import numpy as np
import neo
import quantities as pq
from elephant.gpfa import GPFA
from sklearn.linear_model import PoissonRegressor, Ridge

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
LATENT_DIM = 52
ALPHA1 = 0.01  # Ridge alpha for held-in rates
ALPHA2 = 0.0   # Poisson alpha for held-out rates
PHASE = "val"
BINSUF = "" if BIN_SIZE_MS == 5 else f"_{BIN_SIZE_MS}"
OUTPUT_KEY = DATASET_NAME + BINSUF

DATAPATH = Path(__file__).resolve().parents[1] / "data" / "000128" / "sub-Jenkins"
SAVEPATH = Path(__file__).resolve().parents[1] / "outputs" / f"{OUTPUT_KEY}_gpfa_output_{PHASE}.h5"


def array_to_spiketrains(array, bin_size_ms):
    result = []
    for trial_idx in range(len(array)):
        trial = []
        for chan in range(array.shape[2]):
            times = np.nonzero(array[trial_idx, :, chan])[0]
            counts = array[trial_idx, times, chan].astype(int)
            times = np.repeat(times, counts)
            trial.append(neo.SpikeTrain(times * bin_size_ms * pq.ms, t_stop=array.shape[1] * bin_size_ms * pq.ms))
        result.append(trial)
    return result


def main():
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

    train_split = "train"
    eval_split = "val"

    train_dict = make_train_input_tensors(dataset, DATASET_NAME, train_split, save_file=False)
    train_spikes_heldin = train_dict["train_spikes_heldin"]
    train_spikes_heldout = train_dict["train_spikes_heldout"]
    eval_dict = make_eval_input_tensors(dataset, DATASET_NAME, eval_split, save_file=False)
    eval_spikes_heldin = eval_dict["eval_spikes_heldin"]

    print(f"train {train_spikes_heldin.shape} heldout {train_spikes_heldout.shape} eval {eval_spikes_heldin.shape}")

    print("Converting to neo.SpikeTrains ...")
    train_st = array_to_spiketrains(train_spikes_heldin, BIN_SIZE_MS)
    eval_st = array_to_spiketrains(eval_spikes_heldin, BIN_SIZE_MS)

    print(f"Fitting GPFA (latent_dim={LATENT_DIM}, bin={BIN_SIZE_MS} ms) ...")
    gpfa = GPFA(bin_size=(BIN_SIZE_MS * pq.ms), x_dim=LATENT_DIM)
    train_factors = gpfa.fit_transform(train_st)
    eval_factors = gpfa.transform(eval_st)

    train_factors_s = np.vstack([train_factors[i].T for i in range(len(train_factors))])
    eval_factors_s = np.vstack([eval_factors[i].T for i in range(len(eval_factors))])

    hi_chan = train_spikes_heldin.shape[2]
    ho_chan = train_spikes_heldout.shape[2]
    tlen = train_spikes_heldin.shape[1]
    num_train = len(train_st)
    num_eval = len(eval_st)

    train_spikes_heldin_s = train_spikes_heldin.reshape(-1, hi_chan)
    train_spikes_heldout_s = train_spikes_heldout.reshape(-1, ho_chan)
    eval_spikes_heldin_s = eval_spikes_heldin.reshape(-1, hi_chan)

    print("Fitting rectified linear regression (factors -> heldin) ...")
    all_factors = np.vstack([train_factors_s, eval_factors_s])
    all_heldin = np.vstack([train_spikes_heldin_s, eval_spikes_heldin_s])
    ridge = Ridge(alpha=ALPHA1)
    ridge.fit(all_factors, all_heldin)
    train_rates_heldin_s = np.clip(ridge.predict(train_factors_s), 1e-10, None)
    eval_rates_heldin_s = np.clip(ridge.predict(eval_factors_s), 1e-10, None)

    print(f"Fitting {ho_chan} Poisson GLMs (heldin rates -> heldout spikes) ...")
    train_pred, eval_pred = [], []
    for chan in range(ho_chan):
        pr = PoissonRegressor(alpha=ALPHA2, max_iter=500)
        pr.fit(train_rates_heldin_s, train_spikes_heldout_s[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            pr = PoissonRegressor(alpha=ALPHA2, max_iter=pr.max_iter * 5)
            pr.fit(train_rates_heldin_s, train_spikes_heldout_s[:, chan])
        train_pred.append(pr.predict(train_rates_heldin_s))
        eval_pred.append(pr.predict(eval_rates_heldin_s))
    train_rates_heldout_s = np.clip(np.vstack(train_pred).T, 1e-10, None)
    eval_rates_heldout_s = np.clip(np.vstack(eval_pred).T, 1e-10, None)

    output_dict = {
        OUTPUT_KEY: {
            "train_rates_heldin": train_rates_heldin_s.reshape(num_train, tlen, hi_chan),
            "train_rates_heldout": train_rates_heldout_s.reshape(num_train, tlen, ho_chan),
            "eval_rates_heldin": eval_rates_heldin_s.reshape(num_eval, tlen, hi_chan),
            "eval_rates_heldout": eval_rates_heldout_s.reshape(num_eval, tlen, ho_chan),
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
