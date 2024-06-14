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
    "x714akbx",  # train-1.25-epochs+7b
    "b5oob18s",  # train-2-epochs+7b
    "wuxm0jp4",  # llava-lrv+7b, llava-lvis4v+7b
    "x38im6cm",  # train-1.5-epochs+7b, train-2-epochs+7b, train-3-epochs+7b
    "kkypbhgu",  # one-stage+7b, train-1.25-epochs+7b, train-3-epochs+7b
    "u3bmmese",
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
# Not entirely sure on the specifics here - high level it is metric samples from the history of each eval run
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

# TODO: We want to only attack the base model, but we didn't run the correct evals.
additional_epochs_eval_runs_histories_df = eval_runs_histories_df[
    # (
    #     eval_runs_histories_df["Attacked Model"]
    #     == "{'one-stage+7b', 'llava-lvis4v-lrv+7b'}"
    # )
    # & (
    eval_runs_histories_df["Evaluated Model"].isin(eval_model_order)
    # )
]

# TODO: This is currently untested.
# Create a colormap from the "coolwarm" palette
cmap = sns.color_palette("coolwarm", as_cmap=True)

# Define the values you want to map to colors
values = [1, 1.25, 1.5, 2, 3]

# Map the values to colors using the colormap
colors = [cmap(val / max(values)) for val in values]

# Create the custom palette
custom_palette = sns.color_palette(colors)

# Use the custom palette in your plot
sns.set_palette(custom_palette)


for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT:
    plt.close()
    g = sns.relplot(
        data=additional_epochs_eval_runs_histories_df,
        kind="line",
        x="optimizer_step_counter_epoch",
        y=metric,
        hue="Evaluated Model",
        hue_order=eval_model_order,
        style="Same Model",
        style_order=[False, True],
        col="Attacked Model",
        # row="Eval Dataset (Val Split)",
        facet_kws={"margin_titles": True},
    )
    g.set_axis_labels(
        "Gradient Step", src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]
    )
    g.fig.suptitle("Transfer Across VLM Training Checkpoints", y=1.0)
    # g.set_titles(col_template="{col_name}", row_template="{row_name}")
    # g._legend.set_title("Evaluated Model")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
    g.set(ylim=src.globals.METRICS_TO_BOUNDS_DICT[metric])
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={idx}_model_type={attack_models[:20]}",
    )
    g.set(
        xscale="log",
        yscale="log",
        ylim=(0.95 * eval_runs_histories_df[metric].min(), None),
    )
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={idx}_model_type={attack_models[:20]}",
    )
    # plt.show()

eval_runs_histories_tall_df = eval_runs_histories_df.melt(
    id_vars=[
        "Attack Dataset (Train Split)",
        "Eval Dataset (Val Split)",
        "Same Data Distribution",
        # "Attack Topic (Train Split)",
        # "Eval Topic (Val Split)",
        # "Same Topic",
        "optimizer_step_counter_epoch",
    ],
    value_vars=["loss/avg_epoch"],
    var_name="Metric",
    value_name="Score",
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].replace(
    src.globals.METRICS_TO_TITLE_STRINGS_DICT
)
models_to_types = {
    "{'one-stage+7b'}": "Base",
    "{'reproduction-llava-v15+7b'}": "2-stage training",
    "{'train-1.25-epochs+7b'}": "Additional Epochs",
    "{'train-1.5-epochs+7b'}": "Additional Epochs",
    "{'train-2-epochs+7b'}": "Additional Epochs",
    "{'train-3-epochs+7b'}": "Additional Epochs",
    "{'llava-lvis4v+7b'}": "Additional Training Data",
    "{'llava-lrv+7b'}": "Additional Training Data",
    "{'llava-lvis4v-lrv+7b'}": "Additional Training Data",
}
eval_runs_histories_df["Eval Model Type"] = eval_runs_histories_df[
    "Evaluated Model"
].map(models_to_types)
eval_runs_histories_df["Attack Model Type"] = eval_runs_histories_df[
    "Attacked Model"
].map(models_to_types)
