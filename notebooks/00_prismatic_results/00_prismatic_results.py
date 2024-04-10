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
    "xk0nv314",  # Prismatic with N_Choose_1 Jailbreaks
]

runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=False,
)


runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=10000,
)

plt.close()
g = sns.relplot(
    data=runs_histories_df,
    x="n_gradient_step",
    y="eval/loss_model=avg",
    row="attack_models_str",
    col="eval_model_str",
)

g.set_axis_labels("Gradient Step", "Cross Entropy of Target | Prompt, Image")
plt.show()


plt.close()
g = sns.relplot(
    data=runs_histories_df,
    x="n_gradient_step",
    y="eval/generation_begins_with_target",
    row="attack_models_str",
    col="eval_model_str",
)

g.set_axis_labels("Gradient Step", "Cross Entropy of Target | Prompt, Image")
plt.show()


print("Finished notebooks/00_prismatic_results.py!")
