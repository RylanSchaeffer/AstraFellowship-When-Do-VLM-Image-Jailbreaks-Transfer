import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot
import src.utils


# refresh = True
refresh = False

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "swhukev7",  # Prism with N_Choose_1 Jailbreaks
]

runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=True,
)


runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    keys=[
        "epoch",
        "linear/val_top1",
        "linear/val_probs_mean",
        "pretrain/percent_error_step",
        "pretrain/total_loss_step",
        "pretrain/1_minus_average_centroid_norm_step",
    ],
    max_num_samples_per_run=100000,
)


print("Finished notebooks/00_prismatic_results.py!")
