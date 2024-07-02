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
import src.utils


refresh = False
finished_only = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,  # delete existing results if they exist (data is not deleted)
)

sweep_ids = [
    "wkxrq2t2",
]


# Download wandb results for all runs in the given sweep.
wandb_username = "danvalentine256"  # Hard coded because Dan ran these experiments.
# wandb_username=src.utils.retrieve_wandb_username()
eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,  # will use local cache if this is false
    finished_only=finished_only,
    wandb_username=wandb_username,
    filetype="csv",
)


eval_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=eval_runs_configs_df,
    col_name="data",
    key_in_dict="dataset",
    new_col_name="eval_dataset",
)
eval_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=eval_runs_configs_df,
    col_name="data",
    key_in_dict="split",
    new_col_name="eval_dataset_split",
)

eval_runs_configs_df.rename(
    columns={"run_id": "eval_run_id", "wandb_attack_run_id": "attack_run_id"},
    inplace=True,
)

# Download attack runs.
attack_run_ids = eval_runs_configs_df["attack_run_id"].unique()
print("Attack Run IDs: ", attack_run_ids.tolist())
attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs_by_run_ids(
    wandb_project_path="universal-vlm-jailbreak",
    wandb_username=wandb_username,
    data_dir=data_dir,
    run_ids=attack_run_ids,
    refresh=refresh,
    finished_only=finished_only,
    filetype="csv",
)
attack_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=attack_runs_configs_df,
    col_name="data",
    key_in_dict="dataset",
    new_col_name="attack_dataset",
)
attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)


# Join attack run data into to evals df.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[["attack_run_id", "attack_dataset"]],
    how="left",
    left_on="attack_run_id",
    right_on="attack_run_id",
)

# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_username,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    # finished_only=finished_only,
    wandb_run_history_samples=1000000,
    filetype="csv",
)
# This col is not populated on this df
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)
eval_runs_histories_df.rename(columns={"run_id": "eval_run_id"}, inplace=True)

eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "eval_run_id",
            "attack_run_id",
            "model_to_eval",
            "models_to_attack",
            "attack_dataset",
            "eval_dataset",
        ]
    ],
    how="inner",
    on="eval_run_id",
)

eval_runs_histories_df["Same Data Distribution"] = (
    eval_runs_histories_df["attack_dataset"] == eval_runs_histories_df["eval_dataset"]
)
eval_runs_histories_df["Same Model"] = (
    eval_runs_histories_df["models_to_attack"]
    == eval_runs_histories_df["model_to_eval"]
)

eval_runs_histories_df["one_minus_score_model=claude3opus"] = (
    1.0 - eval_runs_histories_df["loss/score_model=claude3opus"]
)
eval_runs_histories_df["one_minus_score_model=harmbench"] = (
    1.0 - eval_runs_histories_df["loss/score_model=harmbench"]
)
eval_runs_histories_df["one_minus_score_model=llamaguard2"] = (
    1.0 - eval_runs_histories_df["loss/score_model=llamaguard2"]
)
eval_runs_histories_df.rename(
    columns={
        "attack_dataset": "Attack Dataset (Train Split)",
        "eval_dataset": "Eval Dataset (Val Split)",
        # "attack_subset": "Attack Topic (Train Split)",
        # "eval_subset": "Eval Topic (Val Split)",
        "model_to_eval": "Evaluated Model",
        "models_to_attack": "Attacked Model",
    },
    inplace=True,
)

eval_model_order = [
    "{'one-stage+7b'}",
    "{'train-1.25-epochs+7b'}",
    "{'train-1.5-epochs+7b'}",
    "{'train-2-epochs+7b'}",
    "{'train-3-epochs+7b'}",
]
eval_model_num_epochs = {
    "{'one-stage+7b'}": 1.0,
    "{'train-1.25-epochs+7b'}": 1.25,
    "{'train-1.5-epochs+7b'}": 1.5,
    "{'train-2-epochs+7b'}": 2.0,
    "{'train-3-epochs+7b'}": 3.0,
}

additional_epochs_eval_runs_histories_df = eval_runs_histories_df[
    (eval_runs_histories_df["Attacked Model"] == "{'one-stage+7b'}")
    & (eval_runs_histories_df["Evaluated Model"].isin(eval_model_order))
].copy()
additional_epochs_eval_runs_histories_df[
    "Eval VLM\nTraining Epochs"
] = additional_epochs_eval_runs_histories_df["Evaluated Model"].map(
    eval_model_num_epochs
)
unique_metrics_order = [
    "loss/avg_epoch",
    "loss/score_model=claude3opus",
]
unique_metrics_nice_strings_order = [
    src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric] for metric in unique_metrics_order
]


additional_epochs_eval_runs_histories_tall_df = (
    additional_epochs_eval_runs_histories_df.melt(
        id_vars=[
            "Attack Dataset (Train Split)",
            "Eval Dataset (Val Split)",
            "Eval VLM\nTraining Epochs",
            "optimizer_step_counter_epoch",
        ],
        value_vars=unique_metrics_order,
        var_name="Metric",
        value_name="Score",
    )
)

# Convert metrics to nice strings.
additional_epochs_eval_runs_histories_tall_df[
    "Original Metric"
] = additional_epochs_eval_runs_histories_tall_df["Metric"]
additional_epochs_eval_runs_histories_tall_df[
    "Metric"
] = additional_epochs_eval_runs_histories_tall_df["Metric"].map(
    lambda k: src.globals.METRICS_TO_TITLE_STRINGS_DICT.get(k, k)
)

additional_epochs_eval_runs_histories_tall_df["Attacked VLM\nTraining Epochs"] = 1.0

plt.close()
g = sns.relplot(
    data=additional_epochs_eval_runs_histories_tall_df,
    kind="line",
    x="optimizer_step_counter_epoch",
    y="Score",
    col="Metric",
    col_order=unique_metrics_nice_strings_order,
    hue="Eval VLM\nTraining Epochs",
    style="Attacked VLM\nTraining Epochs",
    # palette=custom_palette,
    palette="cool",
    aspect=0.75,
    linewidth=3,
    facet_kws={"margin_titles": True, "sharey": False},
)
g.set(xlim=(0, 50000))
g.set_axis_labels("Gradient Step")
g.fig.suptitle("Transfer Between VLM Training Checkpoints", y=1.0, fontsize=35)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
for ax, key in zip(g.axes.flat, unique_metrics_order):
    ax.set_ylabel(src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[key])
    ax.set_ylim(src.globals.METRICS_TO_BOUNDS_DICT[key])
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
g.legend._legend_box.align = "center"
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"score_vs_optimizer_step_by_num_vlm_epoch_split_metric_lineplot",
)
# plt.show()


print("Finished notebooks/04_additional_vlm_epochs!")
