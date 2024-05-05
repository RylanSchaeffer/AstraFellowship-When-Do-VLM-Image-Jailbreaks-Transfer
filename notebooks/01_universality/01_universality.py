import ast
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb

import src.analyze
import src.plot


# refresh = True
refresh = False

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "jb02fx4o",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH
]

eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=False,
)
# This data wasn't consistently logged due to changing code, so let's retrieve it from the attack W&B runs.
eval_runs_configs_df.drop(columns=["models_to_attack"], inplace=True)
eval_runs_configs_df["eval_dataset"] = eval_runs_configs_df["data"].apply(
    lambda x: x["dataset"] if isinstance(x, dict) else ast.literal_eval(x)["dataset"]
)
eval_runs_configs_df["split"] = eval_runs_configs_df["data"].apply(
    lambda x: x["split"] if isinstance(x, dict) else ast.literal_eval(x)["split"]
)
# Keep only the eval data (previously we measured train and eval).
eval_runs_configs_df = eval_runs_configs_df[eval_runs_configs_df["split"] == "eval"]

attack_wandb_run_ids = eval_runs_configs_df["wandb_run_id"].unique()
api = wandb.Api(timeout=600)
attack_wandb_sweep_ids = np.unique(
    [
        api.run(f"rylan/universal-vlm-jailbreak/{run_id}").sweep.id
        for run_id in attack_wandb_run_ids
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

# Add metadata about evaluations.
# TODO: Fix this shit naming of wandb_run_id.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[
        [
            "attack_run_id",
            "attack_dataset",
            "models_to_attack",  # (we now no longer need this because we updated the eval run on W&B.)
        ]
    ],
    how="left",
    left_on="wandb_run_id",
    right_on="attack_run_id",
)

universality_runs_configs_df = eval_runs_configs_df[
    eval_runs_configs_df["models_to_attack"] == eval_runs_configs_df["model_to_eval"]
]

universality_runs_configs_tall_df = universality_runs_configs_df.melt(
    id_vars=["attack_dataset", "eval_dataset"],
    value_vars=src.analyze.metrics_to_nice_strings_dict.keys(),
    var_name="Metric",
    value_name="Score",
)
universality_runs_configs_tall_df["Same Data Distribution"] = (
    universality_runs_configs_tall_df["attack_dataset"]
    == universality_runs_configs_tall_df["eval_dataset"]
)
universality_runs_configs_tall_df.rename(
    columns={
        "attack_dataset": "Attack Dataset (Train Split)",
        "eval_dataset": "Eval Dataset (Val Split)",
    },
    inplace=True,
)
# Convert metrics to nice strings.
universality_runs_configs_tall_df["Metric"] = universality_runs_configs_tall_df[
    "Metric"
].replace(src.analyze.metrics_to_nice_strings_dict)


plt.close()
g = sns.catplot(
    data=universality_runs_configs_tall_df,
    kind="violin",
    x="Attack Dataset (Train Split)",
    order=["advbench", "rylan_anthropic_hhh"],
    y="Score",
    hue="Eval Dataset (Val Split)",
    hue_order=["advbench", "rylan_anthropic_hhh"],
    col="Metric",
    col_order=src.analyze.metrics_to_nice_strings_dict.values(),
    sharey=False,
    inner="point",
)
g.fig.suptitle("Universality of Image Jailbreaks", y=1.0)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
for ax, ylim in zip(g.axes[0, :], src.analyze.metrics_to_bounds_dict.values()):
    ax.set_ylim(ylim)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"score_vs_attack_dataset_by_eval_dataset_split_metric_violinplot",
)
# plt.show()

plt.close()
g = sns.catplot(
    data=universality_runs_configs_tall_df,
    kind="violin",
    x="Same Data Distribution",
    order=[True, False],
    y="Score",
    hue="Attack Dataset (Train Split)",
    hue_order=["advbench", "rylan_anthropic_hhh"],
    col="Metric",
    col_order=src.analyze.metrics_to_nice_strings_dict.values(),
    sharey=False,
    inner="point",
)
g.fig.suptitle("Universality of Image Jailbreaks", y=1.0)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
for ax, ylim in zip(g.axes[0, :], src.analyze.metrics_to_bounds_dict.values()):
    ax.set_ylim(ylim)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"score_vs_same_data_distribution_by_attack_dataset_split_metric_kdeplot",
)
# plt.show()


eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=10000,
)
# This data wasn't consistently logged due to changing code, so let's retrieve it from the attack W&B runs.
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)


eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "run_id",
            "attack_run_id",
            "model_to_eval",
            "models_to_attack",
            "attack_dataset",
            "eval_dataset",
            "split",
        ]
    ],
    how="inner",
    left_on="run_id",
    right_on="run_id",
)

universality_runs_histories_df = eval_runs_histories_df[
    eval_runs_histories_df["models_to_attack"]
    == eval_runs_histories_df["model_to_eval"]
]

universality_runs_histories_tall_df = universality_runs_histories_df.melt(
    id_vars=["attack_dataset", "eval_dataset", "optimizer_step_counter_epoch"],
    value_vars=src.analyze.metrics_to_nice_strings_dict.keys(),
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
universality_runs_histories_tall_df["Metric"] = universality_runs_histories_tall_df[
    "Metric"
].replace(src.analyze.metrics_to_nice_strings_dict)


plt.close()
g = sns.relplot(
    data=universality_runs_histories_tall_df,
    kind="line",
    x="optimizer_step_counter_epoch",
    y="Score",
    col="Metric",
    col_order=src.analyze.metrics_to_nice_strings_dict.values(),
    hue="Same Data Distribution",
    hue_order=[True, False],
    split="Attack Dataset (Train Split)",
    facet_kws={"margin_titles": True},
)
g.set_axis_labels("Gradient Step")
g.fig.suptitle("Universality of Image Jailbreaks", y=1.0)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
for ax, ylim in zip(g.axes[0, :], src.analyze.metrics_to_bounds_dict.values()):
    ax.set_ylim(ylim)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"score_vs_optimizer_step_by_same_data_distribution_by_attack_dataset_split_metric_lineplot",
)
plt.show()

print("Finished notebooks/01_universality!")
