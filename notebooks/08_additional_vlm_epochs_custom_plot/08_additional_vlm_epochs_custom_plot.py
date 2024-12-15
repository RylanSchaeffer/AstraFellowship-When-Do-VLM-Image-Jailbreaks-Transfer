import ast
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot


refresh = False
finished_only = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,  # delete existing results if they exist (data is not deleted)
)

# Eyeballed because too lazy to set up the repo again.
df = pd.DataFrame.from_dict(
    {
        "Eval VLM Training Epoch": [1.0, 1.25, 1.5, 2.0, 3.0]
        + [1.0, 1.25, 1.5, 2.0, 3.0],
        "Score": [0.925, 1.225, 1.3, 1.375, 1.75] + [0.875, 0.85, 0.8, 0.675, 0.375],
        "Metric": ["Cross Entropy Loss"] * 5 + ["Claude 3 Opus"] * 5,
        "Attacked": [True, False, False, False, False] * 2,
    }
)
df["Eval VLM\nTraining Epoch"] = df["Eval VLM Training Epoch"]


plt.close()
g = sns.relplot(
    data=df,
    kind="scatter",
    x="Eval VLM Training Epoch",
    y="Score",
    hue="Eval VLM\nTraining Epoch",
    style="Attacked",
    style_order=[True, False],
    col="Metric",
    col_order=["Cross Entropy Loss", "Claude 3 Opus"],
    palette="cool",
    facet_kws=dict(sharey=False),
    s=250,
    markers={True: "o", False: "X"},
)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
unique_metrics_order = [
    "loss/avg_epoch",
    "loss/score_model=claude3opus",
]
for ax, key in zip(g.axes.flat, unique_metrics_order):
    ax.set_ylabel(src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[key])
    # ax.set_ylim(src.globals.METRICS_TO_BOUNDS_DICT[key])
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="y=score_x=vlm_training_epoch_hue=vlm_training_epoch",
)
# plt.show()
