"""Which NLB dataset the scripts run on: env NLB_DATASET (default mc_maze).

The run_*_mc_maze.py scripts predate this switch and keep their names; outputs are
named <dataset>_*_output_val.h5 and data/<dataset>_5ms.npz. Only prep_tensors,
run_masked_transformer, run_vq_ar_transformer and eval_all_mt read it; the other
runners are MC_Maze only.
"""
import os
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data"
# name -> (NWB dir, NWBDataset prefix)
DATASETS = {
    "mc_maze": (_DATA / "000128" / "sub-Jenkins", "*full"),
    "mc_rtt": (_DATA / "000129" / "sub-Indy", ""),
}
DATASET_NAME = os.getenv("NLB_DATASET", "mc_maze")
if DATASET_NAME not in DATASETS:
    raise ValueError(f"unknown NLB_DATASET {DATASET_NAME}; choose from {list(DATASETS)}")
DATAPATH, NWB_PREFIX = DATASETS[DATASET_NAME]
