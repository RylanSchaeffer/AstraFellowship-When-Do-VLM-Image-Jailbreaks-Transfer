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
finished_only = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "tpyxrgxu",  # Prismatic with N-Choose-8 Jailbreaks, AdvBench & Rylan Anthropic HHH
]


wandb_username = "rylan"
eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
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

eval_runs_configs_df["one_minus_score_model=claude3opus"] = (
    1.0 - eval_runs_configs_df["loss/score_model=claude3opus"]
)
eval_runs_configs_df["one_minus_score_model=harmbench"] = (
    1.0 - eval_runs_configs_df["loss/score_model=harmbench"]
)
eval_runs_configs_df["one_minus_score_model=llamaguard2"] = (
    1.0 - eval_runs_configs_df["loss/score_model=llamaguard2"]
)

attack_failure_rate_heatmaps_dir = os.path.join(
    results_dir, "attack_failure_rate_heatmaps"
)
os.makedirs(attack_failure_rate_heatmaps_dir, exist_ok=True)
for eval_dataset in eval_runs_configs_df["eval_dataset"].unique():
    attack_failure_rate_heatmaps_eval_dataset_dir = os.path.join(
        attack_failure_rate_heatmaps_dir, f"eval_dataset={eval_dataset}"
    )
    os.makedirs(attack_failure_rate_heatmaps_eval_dataset_dir, exist_ok=True)
    for attack_dataset in eval_runs_configs_df["attack_dataset"].unique():
        attack_failure_rate_heatmaps_eval_dataset_attack_dataset_dir = os.path.join(
            attack_failure_rate_heatmaps_eval_dataset_dir,
            f"attack_dataset={attack_dataset}",
        )
        os.makedirs(
            attack_failure_rate_heatmaps_eval_dataset_attack_dataset_dir, exist_ok=True
        )
        for metric in [
            "loss/avg_epoch",
            "one_minus_score_model=claude3opus",
            "one_minus_score_model=harmbench",
            "one_minus_score_model=llamaguard2",
        ]:
            eval_runs_configs_subset_df = eval_runs_configs_df[
                (eval_runs_configs_df["attack_dataset"] == attack_dataset)
                & (eval_runs_configs_df["eval_dataset"] == eval_dataset)
            ][["models_to_attack", "model_to_eval", metric]]

            if len(eval_runs_configs_subset_df) == 0:
                print(
                    f"No data for attack_dataset={attack_dataset} and eval_dataset={eval_dataset} and metric={metric}."
                )
                continue

            num_rows_before_dropping_duplicates = eval_runs_configs_subset_df.shape[0]
            print(
                f"Number of rows before dropping duplicates: {num_rows_before_dropping_duplicates}"
            )
            eval_runs_configs_subset_df.drop_duplicates(
                subset=["models_to_attack", "model_to_eval"], inplace=True
            )
            num_rows_after_dropping_duplicates = eval_runs_configs_subset_df.shape[0]
            print(
                f"Number of rows after dropping duplicates: {num_rows_after_dropping_duplicates}"
            )

            eval_runs_configs_subset_pivoted_df = eval_runs_configs_subset_df.pivot(
                index="model_to_eval", columns="models_to_attack", values=metric
            )

            # Switch the rows (model_to_eval) to nice strings.
            eval_runs_configs_subset_pivoted_df.index = (
                eval_runs_configs_subset_pivoted_df.index.map(
                    src.analyze.map_string_set_of_models_to_nice_string
                )
            )

            # Reorder the rows to be alphabetical.
            eval_runs_configs_subset_pivoted_df.sort_index(inplace=True)

            eval_runs_configs_subset_pivoted_df.columns = (
                eval_runs_configs_subset_pivoted_df.columns.map(
                    src.analyze.map_string_set_of_models_to_nice_string
                )
            )

            metric_bounds = src.globals.METRICS_TO_BOUNDS_DICT.get(metric, None)
            plt.close()
            plt.figure(figsize=(30, 15))  # Adjust the size as needed
            g = sns.heatmap(
                data=eval_runs_configs_subset_pivoted_df,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                cbar=True,
                vmin=metric_bounds[0] if metric_bounds else None,
                vmax=metric_bounds[1] if metric_bounds else None,
                cbar_kws={
                    "label": src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[metric]
                },
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=0)
            # Add a black border around the heatmap.
            ax = plt.gca()
            ax.add_patch(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    lw=5,
                    transform=ax.transAxes,
                )
            )
            plt.title("Jailbreaking Ensembles of $N=8$ VLMs", fontsize=50)
            plt.xlabel("Attacked Ensemble of $N=8$ VLMs", fontsize=50)
            plt.ylabel("Evaluated VLM", fontsize=50)
            metric_as_filename = src.globals.METRICS_TO_FILENAME_STRINGS_DICT[metric]
            src.plot.save_plot_with_multiple_extensions(
                plot_dir=attack_failure_rate_heatmaps_eval_dataset_attack_dataset_dir,
                plot_title=f"eval_data={eval_dataset}_attack_data={attack_dataset}_metric={metric_as_filename}_heatmap",
            )
            # plt.show()

print()


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
    value_vars=src.analyze.METRICS_TO_NICE_STRINGS_DICT.keys(),
    var_name="Metric",
    value_name="Score",
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].replace(
    src.analyze.METRICS_TO_NICE_STRINGS_DICT
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
    col_order=src.analyze.METRICS_TO_NICE_STRINGS_DICT.values(),
    style="Same Data Distribution",
    style_order=[True, False],
    hue="Attack Dataset (Train Split)",
    hue_order=["advbench", "rylan_anthropic_hhh"],
    facet_kws={"margin_titles": True, "sharey": False},
)
g.set_axis_labels("Gradient Step")
g.fig.suptitle("Universality of Image Jailbreaks", y=1.0)
g.set_titles(col_template="{col_name}")
# Set the y-lim per axis
for ax, ylim in zip(g.axes[0, :], src.analyze.METRICS_TO_BOUNDS_DICT.values()):
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
        hue_order=["advbench", "rylan_anthropic_hhh"],
        facet_kws={"margin_titles": True, "sharey": False},
    )
    g.set_axis_labels("Gradient Step")
    g.fig.suptitle("Universality of Image Jailbreaks", y=1.0)
    g.set_titles(col_template="{col_name}")
    # Set the y-lim per axis
    for ax, ylim in zip(g.axes[0, :], src.analyze.METRICS_TO_BOUNDS_DICT.values()):
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


for metric in src.analyze.METRICS_TO_NICE_STRINGS_DICT:
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
        hue_order=["advbench", "rylan_anthropic_hhh"],
        style="Eval Dataset (Val Split)",
        style_order=["advbench", "rylan_anthropic_hhh"],
        col="models_to_attack",
        row="Evaluated Model",
        row_order=unique_and_ordered_eval_model_strs,
        facet_kws={"margin_titles": True, "sharey": False},
    )
    # plt.show()
    g.set_axis_labels(
        "Gradient Step",
        src.analyze.METRICS_TO_NICE_STRINGS_DICT[metric],
    )
    g.fig.suptitle("Attacked Model(s)", y=1.0)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
    g.set(ylim=src.analyze.METRICS_TO_BOUNDS_DICT[metric])
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
        style_order=["advbench", "rylan_anthropic_hhh"],
        col="models_to_attack",
        row="Eval Dataset (Val Split)",
        facet_kws={"margin_titles": True},
    )
    # plt.show()
    g.set_axis_labels(
        "Gradient Step",
        src.analyze.METRICS_TO_NICE_STRINGS_DICT[metric],
    )
    g.fig.suptitle("Attacked Model(s)", y=1.0)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
    g.set(ylim=src.analyze.METRICS_TO_BOUNDS_DICT[metric])
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
    for metric in src.analyze.METRICS_TO_NICE_STRINGS_DICT:
        plt.close()
        g = sns.relplot(
            data=eval_runs_histories_by_attack_df,
            kind="line",
            x="optimizer_step_counter_epoch",
            y=metric,
            hue="Evaluated Model",
            hue_order=unique_and_ordered_eval_model_strs,
            style="Attack Dataset (Train Split)",
            style_order=["advbench", "rylan_anthropic_hhh"],
            col="models_to_attack",
            row="Eval Dataset (Val Split)",
            facet_kws={"margin_titles": True},
        )
        g.set_axis_labels(
            "Gradient Step", src.analyze.METRICS_TO_NICE_STRINGS_DICT[metric]
        )
        g.fig.suptitle("Attacked Model(s)", y=1.0)
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        # g._legend.set_title("Evaluated Model")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
        g.set(ylim=src.analyze.METRICS_TO_BOUNDS_DICT[metric])
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
