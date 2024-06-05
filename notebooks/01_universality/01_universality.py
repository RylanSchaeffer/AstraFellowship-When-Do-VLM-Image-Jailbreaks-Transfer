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


# refresh = True
refresh = False

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "zyf0lb9y",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 1)
    "s754hflc",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 2)
    "jl9as45o",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 3)
    "1yoxmmrk",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 4)
    "bjg1o5ko",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 5)
    "8nrhoa2q",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 6)
]

eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=True,
)
eval_runs_configs_df["eval_dataset"] = eval_runs_configs_df["data"].apply(
    lambda x: x["dataset"] if isinstance(x, dict) else ast.literal_eval(x)["dataset"]
)
eval_runs_configs_df.rename(
    columns={"run_id": "eval_run_id", "wandb_attack_run_id": "attack_run_id"},
    inplace=True,
)

unique_attack_run_ids = eval_runs_configs_df["attack_run_id"].unique()
api = wandb.Api(timeout=600)
attack_wandb_sweep_ids = np.unique(
    [
        api.run(f"rylan/universal-vlm-jailbreak/{run_id}").sweep.id
        for run_id in unique_attack_run_ids
    ]
).tolist()
attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak",
    data_dir=data_dir,
    sweep_ids=attack_wandb_sweep_ids,
    refresh=refresh,
    finished_only=False,
)
attack_runs_configs_df["attack_dataset"] = attack_runs_configs_df["data"].apply(
    lambda x: x["dataset"] if isinstance(x, dict) else ast.literal_eval(x)["dataset"]
)
attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)

# Add the attacked dataset.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[
        [
            "attack_run_id",
            "attack_dataset",
        ]
    ],
    how="left",
    on="attack_run_id",
)

universality_runs_configs_df = eval_runs_configs_df[
    eval_runs_configs_df["models_to_attack"] == eval_runs_configs_df["model_to_eval"]
]

unique_metrics_order = [
    "loss/avg_epoch",
    "one_minus_score_model=claude3opus",
    "one_minus_score_model=harmbench",
    "one_minus_score_model=llamaguard2",
]
unique_datasets_order = ["advbench", "rylan_anthropic_hhh"]


unique_metrics_nice_strings_order = [
    src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric] for metric in unique_metrics_order
]

unique_datasets_nice_strings_order = [
    src.globals.DATASETS_TO_NICE_STRINGS_DICT[dataset]
    for dataset in unique_datasets_order
]


# universality_runs_configs_tall_df = universality_runs_configs_df.melt(
#     id_vars=["attack_dataset", "eval_dataset"],
#     value_vars=unique_metrics_order,
#     var_name="Metric",
#     value_name="Score",
# )
# universality_runs_configs_tall_df["Same Data Distribution"] = (
#     universality_runs_configs_tall_df["attack_dataset"]
#     == universality_runs_configs_tall_df["eval_dataset"]
# )
# universality_runs_configs_tall_df.rename(
#     columns={
#         "attack_dataset": "Attack Dataset (Train Split)",
#         "eval_dataset": "Eval Dataset (Val Split)",
#     },
#     inplace=True,
# )
# # Convert metrics to nice strings.
# universality_runs_configs_tall_df["Metric"] = universality_runs_configs_tall_df[
#     "Metric"
# ].replace(src.globals.METRICS_TO_NICE_STRINGS_DICT)


# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=50000,
    filetype="csv",
)
# Drop the vestigal "models_to_attack" column.
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)
eval_runs_histories_df.rename(
    columns={"run_id": "eval_run_id", "wandb_attack_run_id": "attack_run_id"},
    inplace=True,
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


eval_runs_histories_extended_df = eval_runs_histories_df.merge(
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
    how="left",
    on="eval_run_id",
)

# Keep only the runs that were attacked and evaluated on the same dataset.
universality_runs_histories_extended_df = eval_runs_histories_extended_df[
    eval_runs_histories_extended_df["models_to_attack"]
    == eval_runs_histories_extended_df["model_to_eval"]
]


universality_runs_histories_tall_df = universality_runs_histories_extended_df.melt(
    id_vars=["attack_dataset", "eval_dataset", "optimizer_step_counter_epoch"],
    value_vars=unique_metrics_order,
    var_name="Metric",
    value_name="Score",
)
universality_runs_histories_tall_df["Same Data Distribution"] = (
    universality_runs_histories_tall_df["attack_dataset"]
    == universality_runs_histories_tall_df["eval_dataset"]
)
universality_runs_histories_tall_df.rename(
    columns={
        "attack_dataset": "Attack Dataset (Train Split)",
        "eval_dataset": "Eval Dataset (Val Split)",
    },
    inplace=True,
)
# Convert metrics to nice strings.
universality_runs_histories_tall_df[
    "Original Metric"
] = universality_runs_histories_tall_df["Metric"]
universality_runs_histories_tall_df["Metric"] = universality_runs_histories_tall_df[
    "Metric"
].map(lambda k: src.globals.METRICS_TO_TITLE_STRINGS_DICT.get(k, k))

# Convert datasets to nice strings.
universality_runs_histories_tall_df[
    "Attack Dataset (Train Split)"
] = universality_runs_histories_tall_df["Attack Dataset (Train Split)"].map(
    lambda k: src.globals.DATASETS_TO_NICE_STRINGS_DICT.get(k, k)
)


plt.close()
g = sns.relplot(
    data=universality_runs_histories_tall_df,
    kind="line",
    x="optimizer_step_counter_epoch",
    y="Score",
    col="Metric",
    col_order=unique_metrics_nice_strings_order,
    col_wrap=2,
    style="Same Data Distribution",
    style_order=[True, False],
    hue="Attack Dataset (Train Split)",
    hue_order=unique_datasets_nice_strings_order,
    facet_kws={"margin_titles": True, "sharey": False},
)
g.set(xlim=(0, 50000))
g.set_axis_labels("Gradient Step")
g.fig.suptitle("Universality of Image Jailbreaks", y=1.0)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
for ax, key in zip(g.axes.flat, unique_metrics_order):
    ax.set_ylabel(src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[key])
    ax.set_ylim(src.globals.METRICS_TO_BOUNDS_DICT[key])
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"score_vs_optimizer_step_by_same_data_distribution_by_attack_dataset_split_metric_lineplot",
)
# plt.show()


plt.close()
g = sns.relplot(
    data=universality_runs_histories_tall_df,
    kind="line",
    x="optimizer_step_counter_epoch",
    y="Score",
    col="Metric",
    col_order=unique_metrics_nice_strings_order,
    col_wrap=2,
    style="Same Data Distribution",
    style_order=[True, False],
    hue="Attack Dataset (Train Split)",
    hue_order=unique_datasets_nice_strings_order,
    facet_kws={"margin_titles": True, "sharey": False},
)
g.set(xlim=(0, 50000))
g.set_axis_labels("Gradient Step")
g.fig.suptitle("Universality of Image Jailbreaks", y=1.0)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
for ax, key in zip(g.axes.flat, unique_metrics_order):
    ax.set_ylabel(src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[key])
    ax.set_xlim(250, 50000)
    ax.set_yscale("log")
    ax.set_xscale("log")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"score_vs_optimizer_step_by_same_data_distribution_by_attack_dataset_split_metric_lineplot",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"score_log_vs_optimizer_step_log_by_same_data_distribution_by_attack_dataset_split_metric_lineplot",
)
# plt.show()


print("Finished notebooks/01_universality!")
