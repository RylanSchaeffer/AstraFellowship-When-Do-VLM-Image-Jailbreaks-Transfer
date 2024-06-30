import ast
import numpy as np
import matplotlib.pyplot as plt
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

# Switch attack_model_names and eval_model_name to nice strings.
eval_runs_configs_df["model_to_eval"] = eval_runs_configs_df["model_to_eval"].apply(
    src.analyze.map_string_set_of_models_to_nice_string
)
eval_runs_configs_df["models_to_attack"] = eval_runs_configs_df[
    "models_to_attack"
].apply(src.analyze.map_string_set_of_models_to_nice_string)

# Add a column to indicate whether the eval model is in the attack models.
eval_runs_configs_df["Eval VLM in\nAttacked Ensemble"] = eval_runs_configs_df.apply(
    lambda row: row["model_to_eval"] in row["models_to_attack"], axis=1
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

attack_failure_rate_heatmaps_dir = os.path.join(
    results_dir, "attack_failure_rate_heatmap"
)
attack_failure_rate_scatterplots_dir = os.path.join(
    results_dir, "attack_failure_rate_scatter"
)
os.makedirs(attack_failure_rate_heatmaps_dir, exist_ok=True)
os.makedirs(attack_failure_rate_scatterplots_dir, exist_ok=True)
for eval_dataset in eval_runs_configs_df["eval_dataset"].unique():
    attack_failure_rate_heatmaps_eval_dataset_dir = os.path.join(
        attack_failure_rate_heatmaps_dir, f"eval_dataset={eval_dataset}"
    )
    attack_failure_rate_scatterplots_eval_dataset_dir = os.path.join(
        attack_failure_rate_scatterplots_dir, f"eval_dataset={eval_dataset}"
    )
    os.makedirs(attack_failure_rate_heatmaps_eval_dataset_dir, exist_ok=True)
    os.makedirs(attack_failure_rate_scatterplots_eval_dataset_dir, exist_ok=True)
    for attack_dataset in eval_runs_configs_df["attack_dataset"].unique():
        attack_failure_rate_heatmaps_eval_dataset_attack_dataset_dir = os.path.join(
            attack_failure_rate_heatmaps_eval_dataset_dir,
            f"attack_dataset={attack_dataset}",
        )
        attack_failure_rate_scatterplots_eval_dataset_attack_dataset_dir = os.path.join(
            attack_failure_rate_scatterplots_eval_dataset_dir,
            f"attack_dataset={attack_dataset}",
        )
        os.makedirs(
            attack_failure_rate_heatmaps_eval_dataset_attack_dataset_dir,
            exist_ok=True,
        )
        os.makedirs(
            attack_failure_rate_scatterplots_eval_dataset_attack_dataset_dir,
            exist_ok=True,
        )
        for metric in [
            "loss/avg_epoch",
            "loss/score_model=claude3opus",
        ]:
            metric_as_filename = src.globals.METRICS_TO_FILENAME_STRINGS_DICT[metric]
            eval_runs_configs_subset_df = eval_runs_configs_df[
                (eval_runs_configs_df["attack_dataset"] == attack_dataset)
                & (eval_runs_configs_df["eval_dataset"] == eval_dataset)
            ][
                [
                    "models_to_attack",
                    "model_to_eval",
                    "Eval VLM in\nAttacked Ensemble",
                    metric,
                ]
            ]

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

            # Make a duplicate without the newline.
            eval_runs_configs_subset_df[
                "Eval VLM in Attacked VLMs Ensemble"
            ] = eval_runs_configs_subset_df["Eval VLM in\nAttacked Ensemble"]

            eval_runs_configs_subset_df[
                "Attacked VLMs Ensemble"
            ] = eval_runs_configs_subset_df["models_to_attack"]

            # Sort based on the evaluated VLMs.
            eval_runs_configs_subset_df.sort_values(by="model_to_eval", inplace=True)

            # Create a categorical type with the sorted order
            eval_runs_configs_subset_df["model_to_eval_categorical"] = pd.Categorical(
                eval_runs_configs_subset_df["model_to_eval"],
                categories=sorted(
                    eval_runs_configs_subset_df["model_to_eval"].unique()
                ),
                ordered=True,
            )

            plt.close()
            plt.figure(figsize=(24, 16))
            g = sns.scatterplot(
                data=eval_runs_configs_subset_df,
                x="model_to_eval_categorical",
                y=metric,
                hue="Eval VLM in Attacked VLMs Ensemble",
                style="Attacked VLMs Ensemble",
                s=500,
            )
            plt.ylim(src.globals.METRICS_TO_BOUNDS_DICT[metric])
            plt.xlabel("Evaluated VLMs")
            plt.ylabel(src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[metric])
            sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            plt.title(
                f"{src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]} Scores of Attacking Ensemble of $N=8$ VLMs",
                fontsize=50,
            )
            # Make space for the title.
            plt.subplots_adjust(top=0.9)
            plt.xticks(rotation=45, ha="right")
            src.plot.save_plot_with_multiple_extensions(
                plot_dir=attack_failure_rate_scatterplots_eval_dataset_attack_dataset_dir,
                plot_title=f"metric={metric_as_filename}_vs_eval_vlm_by_inclusion_by_attack_ensemble_scatter",
            )
            # plt.show()

            plt.close()
            # Create a categorical type with the sorted order
            eval_runs_configs_subset_df["model_to_eval_categorical"] = pd.Categorical(
                eval_runs_configs_subset_df["model_to_eval"],
                categories=sorted(
                    eval_runs_configs_subset_df["model_to_eval"].unique()
                ),
                ordered=True,
            )
            g = sns.relplot(
                data=eval_runs_configs_subset_df,
                x=metric,
                y="model_to_eval_categorical",
                col="models_to_attack",
                hue="Eval VLM in\nAttacked Ensemble",
                hue_order=[False, True],
                style="models_to_attack",
                height=13,
                aspect=0.55,
                s=150,
                legend=False,
            )
            g.set(
                xlim=src.globals.METRICS_TO_BOUNDS_DICT[metric], ylabel="Evaluated VLMs"
            )
            g.set_axis_labels(
                x_var=src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[metric]
            )
            g.set_titles(col_template="{col_name}")
            # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            # g.fig.suptitle("Jailbreaking Ensembles of $N=8$ VLMs", fontsize=50)
            g.fig.suptitle(
                f"{src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]} Scores of Attacking $N=8$ Ensembled VLMs"
            )
            # Make space for the title.
            plt.subplots_adjust(top=0.9)
            src.plot.save_plot_with_multiple_extensions(
                plot_dir=attack_failure_rate_scatterplots_eval_dataset_attack_dataset_dir,
                plot_title=f"eval_vlm_vs_metric={metric_as_filename}_split_attack_ensemble_scatter",
            )
            # plt.show()

            eval_runs_configs_subset_pivoted_df = (
                eval_runs_configs_subset_df.pivot_table(
                    index="model_to_eval", columns="models_to_attack", values=metric
                )
            )

            # Reorder the rows to be alphabetically ordered.
            eval_runs_configs_subset_pivoted_df.sort_index(inplace=True)

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
            src.plot.save_plot_with_multiple_extensions(
                plot_dir=attack_failure_rate_heatmaps_eval_dataset_attack_dataset_dir,
                plot_title=f"eval_vlm_vs_attack_ensemble_by_metric={metric_as_filename}_heatmap",
            )
            # plt.show()


# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_username,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=1000000,
    filetype="csv",
)
# This col is not populated on this df.
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
            "Eval VLM in\nAttacked Ensemble",
        ]
    ],
    how="inner",
    on="eval_run_id",
)

unique_metrics_order = [
    "loss/avg_epoch",
    "loss/score_model=claude3opus",
]

eval_runs_histories_tall_df = eval_runs_histories_df.melt(
    id_vars=[
        "attack_dataset",
        "eval_dataset",
        "eval_run_id",
        "Eval VLM in\nAttacked Ensemble",
        "optimizer_step_counter_epoch",
    ],
    value_vars=unique_metrics_order,
    var_name="Metric",
    value_name="Score",
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].map(
    lambda k: src.globals.METRICS_TO_TITLE_STRINGS_DICT.get(k, k)
)

# This is an insanely stupid hack. I can't figure out how to exclude the style from the legend
# and then I discovered that preappending a "_" will remove it.
# https://blog.rtwilson.com/easily-hiding-items-from-the-legend-in-matplotlib/
eval_runs_histories_tall_df["Single VLM Eval Run"] = eval_runs_histories_tall_df[
    "eval_run_id"
].apply(lambda x: f"_{x}")

learning_curves_results_dir = os.path.join(results_dir, "learning_curves")
os.makedirs(learning_curves_results_dir, exist_ok=True)
for eval_dataset in eval_runs_histories_tall_df["eval_dataset"].unique():
    learning_curves_eval_dataset_results_dir = os.path.join(
        learning_curves_results_dir, f"eval_dataset={eval_dataset}"
    )
    os.makedirs(learning_curves_eval_dataset_results_dir, exist_ok=True)
    for attack_dataset in eval_runs_histories_tall_df["attack_dataset"].unique():
        learning_curves_eval_dataset_attack_dataset_results_dir = os.path.join(
            learning_curves_eval_dataset_results_dir,
            f"attack_dataset={attack_dataset}",
        )
        os.makedirs(
            learning_curves_eval_dataset_attack_dataset_results_dir, exist_ok=True
        )
        eval_runs_histories_tall_subset_df = eval_runs_histories_tall_df[
            (eval_runs_histories_tall_df["attack_dataset"] == attack_dataset)
            & (eval_runs_histories_tall_df["eval_dataset"] == eval_dataset)
        ]

        if len(eval_runs_histories_tall_subset_df) == 0:
            print(
                f"No data for attack_dataset={attack_dataset} and eval_dataset={eval_dataset}."
            )
            continue
        plt.close()
        g = sns.relplot(
            data=eval_runs_histories_tall_subset_df,
            kind="line",
            x="optimizer_step_counter_epoch",
            y="Score",
            col="Metric",
            col_order=["Cross Entropy Loss", "Claude 3 Opus"],
            col_wrap=2,
            hue="Eval VLM in\nAttacked Ensemble",
            hue_order=[False, True],
            aspect=0.75,
            linewidth=3,
            style="Single VLM Eval Run",
            facet_kws={"margin_titles": True, "sharey": False},
        )
        g.set(xlim=(0, 50000))
        g.set_axis_labels("Gradient Step")
        g.fig.suptitle("Transfer From Ensemble of VLMs to New VLM", y=1.0)
        g.set_titles(col_template="{col_name}")
        # Set the y-lim per axis
        for ax, key in zip(g.axes.flat, unique_metrics_order):
            ax.set_ylabel(src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[key])
            ax.set_ylim(src.globals.METRICS_TO_BOUNDS_DICT[key])
        sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
        src.plot.save_plot_with_multiple_extensions(
            plot_dir=learning_curves_eval_dataset_attack_dataset_results_dir,
            plot_title=f"score_vs_optimizer_step_by_in_ensemble_split_metric_lineplot",
        )
        # plt.show()


print("Finished notebooks/03_transfer_attack_prismatic_n=8!")
