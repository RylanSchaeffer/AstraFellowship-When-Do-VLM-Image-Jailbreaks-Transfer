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
    "8maxlygp",  # MMCR Sweep Part 1
    "zaqw7l33",  # MMCR Sweep Part 2
]


runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="maximum-manifold-capacity-representations",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=True,
)


# Extract the embedding dimensionality.
def extract_embedding_dimensionality(row: pd.Series):
    # TODO: Why are some stored as dicts and others as strings?
    if isinstance(row["projection_kwargs"], dict):
        projection_kwargs = row["projection_kwargs"]
    else:
        projection_kwargs = ast.literal_eval(row["projection_kwargs"])
    return ast.literal_eval(projection_kwargs["layer_widths"])[-1]


runs_configs_df["embedding_dimensionality"] = runs_configs_df.apply(
    extract_embedding_dimensionality,
    axis=1,
)

runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="maximum-manifold-capacity-representations",
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
