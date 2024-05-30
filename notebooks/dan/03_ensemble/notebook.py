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
    "mybo6x8i",  # Prismatic with N-Choose-8 Jailbreaks trained on 5% generated, AdvBench & Rylan Anthropic HHH & Generated
]

eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=True,
    filetype="csv",
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

attack_wandb_attack_run_ids = eval_runs_configs_df["wandb_attack_run_id"].unique()
api = wandb.Api(timeout=600)
attack_wandb_sweep_ids = np.unique(
    [
        api.run(f"danvalentine256/universal-vlm-jailbreak/{run_id}").sweep.id
        for run_id in attack_wandb_attack_run_ids
    ]
).tolist()
attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak",
    data_dir=data_dir,
    sweep_ids=attack_wandb_sweep_ids,
    refresh=refresh,
    finished_only=False,
    filetype="csv",
)
attack_runs_configs_df["attack_dataset"] = attack_runs_configs_df["data"].apply(
    lambda x: x["dataset"] if isinstance(x, dict) else ast.literal_eval(x)["dataset"]
)
attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)

# Add metadata about evaluations.
# TODO: Fix this shit naming of wandb_attack_run_id.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[
        [
            "attack_run_id",
            "attack_dataset",
            "models_to_attack",  # (we now no longer need this because we updated the eval run on W&B.)
        ]
    ],
    how="left",
    left_on="wandb_attack_run_id",
    right_on="attack_run_id",
)

# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=10000,
    filetype="csv",
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

eval_runs_histories_df["Same Data Distribution"] = (
    eval_runs_histories_df["attack_dataset"] == eval_runs_histories_df["eval_dataset"]
)
eval_runs_histories_df.rename(
    columns={
        "attack_dataset": "Attack Dataset (Train Split)",
        "eval_dataset": "Eval Dataset (Val Split)",
        "model_to_eval": "Evaluated Model",
    },
    inplace=True,
)

eval_runs_histories_tall_df = eval_runs_histories_df.melt(
    id_vars=[
        "Attack Dataset (Train Split)",
        "Eval Dataset (Val Split)",
        "Same Data Distribution",
        "optimizer_step_counter_epoch",
    ],
    value_vars=src.analyze.metrics_to_nice_strings_dict.keys(),
    var_name="Metric",
    value_name="Score",
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].replace(
    src.analyze.metrics_to_nice_strings_dict
)

unique_and_ordered_eval_model_strs = np.sort(
    eval_runs_histories_df["Evaluated Model"].unique()
).tolist()


plt.close()
g = sns.relplot(
    data=eval_runs_histories_tall_df,
    kind="line",
    x="optimizer_step_counter_epoch",
    y="Score",
    col="Metric",
    col_order=src.analyze.metrics_to_nice_strings_dict.values(),
    style="Same Data Distribution",
    style_order=[True, False],
    hue="Attack Dataset (Train Split)",
    hue_order=["advbench", "rylan_anthropic_hhh", "generated"],
    facet_kws={"margin_titles": True, "sharey": False},
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
# plt.show()

for (
    metric,
    universality_metric_runs_histories_tall_df,
) in eval_runs_histories_tall_df.groupby("Original Metric"):
    plt.close()
    g = sns.relplot(
        data=universality_metric_runs_histories_tall_df,
        kind="line",
        x="optimizer_step_counter_epoch",
        y="Score",
        col="Metric",
        style="Same Data Distribution",
        style_order=[True, False],
        hue="Attack Dataset (Train Split)",
        hue_order=["advbench", "rylan_anthropic_hhh", "generated"],
        facet_kws={"margin_titles": True, "sharey": False},
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
        plot_title=f"score={metric.replace('/', '_')}_vs_optimizer_step_by_same_data_distribution_by_attack_dataset_lineplot",
    )
    # plt.show()

learning_curves_by_attack_model_dir = os.path.join(
    results_dir, "learning_curves_by_attack_model"
)
os.makedirs(learning_curves_by_attack_model_dir, exist_ok=True)


for metric in src.analyze.metrics_to_nice_strings_dict:
    plt.close()
    # I stupidly used the wrong column name for the LM eval scoring. One has "epoch" and others do not.
    x = (
        "optimizer_step_counter_epoch"
        if metric == "loss/avg_epoch"
        else "optimizer_step_counter"
    )
    g = sns.relplot(
        data=eval_runs_histories_df,
        kind="line",
        x=x,
        y=metric,
        hue="Attack Dataset (Train Split)",
        hue_order=["advbench", "rylan_anthropic_hhh", "generated"],
        style="Eval Dataset (Val Split)",
        style_order=["advbench", "rylan_anthropic_hhh", "generated"],
        col="models_to_attack",
        row="Evaluated Model",
        row_order=unique_and_ordered_eval_model_strs,
        facet_kws={"margin_titles": True},
    )
    # plt.show()
    g.set_axis_labels(
        "Gradient Step",
        src.analyze.metrics_to_nice_strings_dict[metric],
    )
    g.fig.suptitle("Attacked Model(s)", y=1.0)
    g.set_titles(col_template="", row_template="{row_name}")
    # g.set_titles(col_template="{col_name}", row_template="{row_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
    g.set(ylim=src.analyze.metrics_to_bounds_dict[metric])
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"{metric[5:]}_vs_gradient_step_cols=attack_models_rows=eval_models",
    )
    if metric == "loss/avg_epoch":
        g.set(
            xscale="log",
            yscale="log",
            ylim=(0.95 * eval_runs_histories_df["loss/avg_epoch"].min(), None),
        )
        src.plot.save_plot_with_multiple_extensions(
            plot_dir=results_dir,
            plot_title=f"{metric[5:]}_log_vs_gradient_step_log_cols=attack_models_rows=eval_models",
        )
    # plt.show()

    plt.close()
    # I stupidly used the wrong column name for the LM eval scoring. One has "epoch" and others do not.
    x = (
        "optimizer_step_counter_epoch"
        if metric == "loss/avg_epoch"
        else "optimizer_step_counter"
    )
    g = sns.relplot(
        data=eval_runs_histories_df,
        kind="line",
        x=x,
        y=metric,
        hue="Evaluated Model",
        hue_order=unique_and_ordered_eval_model_strs,
        style="Attack Dataset (Train Split)",
        style_order=["advbench", "rylan_anthropic_hhh", "generated"],
        col="models_to_attack",
        row="Eval Dataset (Val Split)",
        facet_kws={"margin_titles": True},
    )
    # plt.show()
    g.set_axis_labels(
        "Gradient Step",
        src.analyze.metrics_to_nice_strings_dict[metric],
    )
    g.fig.suptitle("Attacked Model(s)", y=1.0)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
    g.set(ylim=src.analyze.metrics_to_bounds_dict[metric])
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models",
    )
    if metric == "loss/avg_epoch":
        g.set(
            xscale="log",
            yscale="log",
            ylim=(0.95 * eval_runs_histories_df["loss/avg_epoch"].min(), None),
        )
        src.plot.save_plot_with_multiple_extensions(
            plot_dir=results_dir,
            plot_title=f"{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=eval_datasets",
        )
    # plt.show()


idx = 0
for (
    models_to_attack,
    eval_runs_histories_by_attack_df,
) in eval_runs_histories_df.groupby("models_to_attack"):
    for metric in src.analyze.metrics_to_nice_strings_dict:
        plt.close()
        g = sns.relplot(
            data=eval_runs_histories_by_attack_df,
            kind="line",
            x="optimizer_step_counter_epoch",
            y=metric,
            hue="Evaluated Model",
            hue_order=unique_and_ordered_eval_model_strs,
            style="Attack Dataset (Train Split)",
            style_order=["advbench", "rylan_anthropic_hhh", "generated"],
            col="models_to_attack",
            row="Eval Dataset (Val Split)",
            facet_kws={"margin_titles": True},
        )
        g.set_axis_labels(
            "Gradient Step", src.analyze.metrics_to_nice_strings_dict[metric]
        )
        g.fig.suptitle("Attacked Model(s)", y=1.0)
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        # g._legend.set_title("Evaluated Model")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
        g.set(ylim=src.analyze.metrics_to_bounds_dict[metric])
        src.plot.save_plot_with_multiple_extensions(
            plot_dir=learning_curves_by_attack_model_dir,
            plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={idx}",
        )
        idx += 1
        if metric == "loss/avg_epoch":
            g.set(
                xscale="log",
                yscale="log",
                ylim=(0.95 * eval_runs_histories_df["loss/avg_epoch"].min(), None),
            )
            src.plot.save_plot_with_multiple_extensions(
                plot_dir=learning_curves_by_attack_model_dir,
                plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={idx}",
            )
        # plt.show()
        idx += 1


print("Finished notebooks/03_transfer_attack_prismatic_n=8!")
