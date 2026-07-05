"""Trial-averaged predicted rates for a sample heldout neuron across all four models.

Loads the four MC_Maze val-phase h5 outputs, picks the heldout neuron with the
highest across-trials variance under the smoothing model (as a proxy for
"informative"), trial-averages, and plots one line per model.

Reads h5 files from ./outputs/ (or a directory passed as arg).
Writes ./figs/rates_sample_neuron.png.
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "smoothing": "mc_maze_smoothing_output_val.h5",
    "GPFA": "mc_maze_gpfa_output_val.h5",
    "GRU-v1": "mc_maze_gru_v1_output_val.h5",
    "GRU-v2": "mc_maze_gru_v2_output_val.h5",
}
DATASET_KEY = "mc_maze"
BIN_MS = 5


def load_rates(h5_path):
    with h5py.File(h5_path, "r") as f:
        return f[DATASET_KEY]["eval_rates_heldout"][:]


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "outputs"
    out_path = Path(__file__).parent / "figs" / "rates_sample_neuron.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rates = {name: load_rates(data_dir / fname) for name, fname in MODELS.items()}
    for name, arr in rates.items():
        print(f"{name}: shape {arr.shape}")

    smoothing = rates["smoothing"]
    per_neuron_var = smoothing.mean(axis=0).var(axis=0)
    top_neurons = np.argsort(per_neuron_var)[::-1][:1]
    n = int(top_neurons[0])
    print(f"picked heldout neuron index {n} (across-trials variance rank 1 under smoothing)")

    T = smoothing.shape[1]
    t_ms = np.arange(T) * BIN_MS

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    for name, arr in rates.items():
        mean_rate = arr[:, :, n].mean(axis=0) * (1000.0 / BIN_MS)
        ax.plot(t_ms, mean_rate, label=name, linewidth=1.8)
    ax.set_xlabel("time in trial (ms)")
    ax.set_ylabel("firing rate (spikes / s)")
    ax.set_title(f"MC_Maze val: trial-averaged predicted rate, heldout neuron #{n}")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
