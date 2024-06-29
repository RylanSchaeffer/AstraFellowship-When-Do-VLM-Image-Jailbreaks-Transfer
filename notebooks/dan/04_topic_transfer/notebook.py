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


# TODO: enable cross-user analysis
wandb_user = "danvalentine256"
refresh = False

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,  # delete existing results if they exist (data is not deleted)
)

sweep_ids = [
    "y83erjhv",  # Topic to topic xfer
]


# Download wandb results for all runs in the given sweep.
eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,  # will use local cache if this is false
    finished_only=True,
    wandb_username=wandb_user,
    filetype="csv",
)


# parse data config blob into cols
def parse_data_col(df, col_name, config_name=None):
    if not config_name:
        config_name = col_name

    df[col_name] = df["data"].apply(
        lambda x: x[config_name]
        if isinstance(x, dict)
        else ast.literal_eval(x)[config_name]
    )
    return df


eval_runs_configs_df = parse_data_col(eval_runs_configs_df, "eval_dataset", "dataset")
eval_runs_configs_df = parse_data_col(eval_runs_configs_df, "split")
eval_runs_configs_df = parse_data_col(eval_runs_configs_df, "eval_subset", "subset")

# Keep only the eval data (previously we measured train and eval).
eval_runs_configs_df = eval_runs_configs_df[eval_runs_configs_df["split"] == "eval"]

# download attack runs
attack_run_ids = eval_runs_configs_df["wandb_attack_run_id"].unique()

attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs_by_run_ids(
    wandb_project_path="universal-vlm-jailbreak",
    wandb_username=wandb_user,
    data_dir=data_dir,
    run_ids=attack_run_ids,
    refresh=refresh,
    finished_only=True,
    filetype="csv",
)

attack_runs_configs_df = parse_data_col(
    attack_runs_configs_df, "attack_dataset", "dataset"
)
# TODO: this will break if subset was not specified (ie on all non-topic exps)
attack_runs_configs_df = parse_data_col(
    attack_runs_configs_df, "attack_subset", "subset"
)

attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)


# Add needed attack run data to evals df
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[["attack_run_id", "attack_dataset", "attack_subset"]],
    how="left",
    left_on="wandb_attack_run_id",
    right_on="attack_run_id",
)

# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_user,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    # finished_only=True,
    wandb_run_history_samples=1000000,
    filetype="csv",
)
# This col is not populated on this df
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)


eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "run_id",
            "attack_run_id",
            "model_to_eval",
            "models_to_attack",
            "attack_dataset",
            "attack_subset",
            "eval_dataset",
            "eval_subset",
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
eval_runs_histories_df["Same Topic"] = (
    eval_runs_histories_df["attack_subset"] == eval_runs_histories_df["eval_subset"]
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
        # "attack_dataset": "Attack Dataset (Train Split)",
        # "eval_dataset": "Eval Dataset (Val Split)",
        "attack_subset": "Attack Topic (Train Split)",
        "eval_subset": "Eval Topic (Val Split)",
        "model_to_eval": "Evaluated Model",
    },
    inplace=True,
)


eval_runs_histories_tall_df = eval_runs_histories_df.melt(
    id_vars=[
        # "Attack Dataset (Train Split)",
        # "Eval Dataset (Val Split)",
        # "Same Data Distribution",
        "Attack Topic (Train Split)",
        "Eval Topic (Val Split)",
        "Same Topic",
        "optimizer_step_counter_epoch",
    ],
    value_vars=src.globals.METRICS_TO_TITLE_STRINGS_DICT.keys(),
    var_name="Metric",
    value_name="Score",
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].replace(
    src.globals.METRICS_TO_TITLE_STRINGS_DICT
)

learning_curves_by_attack_model_dir = os.path.join(
    results_dir, "learning_curves_by_attack_model"
)
os.makedirs(learning_curves_by_attack_model_dir, exist_ok=True)
topics = np.sort(eval_runs_histories_df["Eval Topic (Val Split)"].unique()).tolist()

learning_curves_by_eval_model_dir = os.path.join(
    results_dir, "learning_curves_by_eval_model"
)
heatmaps_dir = os.path.join(results_dir, "heatmaps")
os.makedirs(heatmaps_dir, exist_ok=True)
plt.close()

# idx = 0
# for (
#     topics_to_attack,
#     eval_runs_histories_by_attack_topic_df,
# ) in eval_runs_histories_df.groupby("Attack Topic (Train Split)"):
#     for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT:
#         breakpoint()


# Initialize a dictionary to store the minimum values for each metric
min_values = {
    metric: pd.DataFrame(index=topics, columns=topics)
    for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT
}

# Double loop to find minimum values
for attack_topic in topics:
    attack_df = eval_runs_histories_df[
        eval_runs_histories_df["Attack Topic (Train Split)"] == attack_topic
    ]

    for eval_topic in topics:
        eval_df = attack_df[attack_df["Eval Topic (Val Split)"] == eval_topic]

        for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT.keys():
            if eval_df.empty:
                min_values[metric].loc[attack_topic, eval_topic] = np.nan
            else:
                # Convert to numeric, coercing errors to NaN
                numeric_values = pd.to_numeric(eval_df[metric], errors="coerce")

                # Calculate min, ignoring NaN values
                min_value = numeric_values.min()

                min_values[metric].loc[attack_topic, eval_topic] = min_value

# Create heatmaps
for metric, df in min_values.items():
    # Convert DataFrame to numeric type, replacing any remaining non-numeric values with NaN
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # Increase figure size for larger boxes
    plt.figure(figsize=(20, 16))

    # Create a custom colormap from blue to green to yellow
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap.set_bad(color="lightgray")

    # Use robust=True to compute color scale robustly to outliers
    sns.heatmap(
        df_numeric,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        robust=True,
        mask=df_numeric.isnull(),
        cbar_kws={"label": "Minimum Value"},
        square=True,
        linewidths=0.5,
        annot_kws={"size": 14},
    )

    plt.title(
        f"Minimum {src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]} Values",
        fontsize=20,
    )
    plt.xlabel("Eval Topic", fontsize=20)
    plt.ylabel("Attack Topic", fontsize=20)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)

    # Add more padding
    plt.tight_layout(pad=3.0)

    plt.savefig(
        f'{heatmaps_dir}/{metric.replace("/", "_")}.png', dpi=300, bbox_inches="tight"
    )
    plt.close()

print("Heatmaps have been generated and saved in the results directory.")

# Diagnostic information
print("\nDiagnostic Information:")
for metric, df in min_values.items():
    print(f"\nMetric: {metric}")
    print("Data types:")
    print(df.dtypes)
    print("\nNumber of NaN values:")
    print(df.isna().sum())
    print("\nSample data:")
    print(df.head())
    print(
        "\nPercentage of missing values:", (df.isna().sum().sum() / df.size) * 100, "%"
    )
# idx = 0
#
# for (
#     topics_to_attack,
#     eval_runs_histories_by_attack_topic_df,
# ) in eval_runs_histories_df.groupby("Attack Topic (Train Split)"):
#     for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT:
#         print(metric)
#         plt.close()
#         g = sns.relplot(
#             data=eval_runs_histories_by_attack_topic_df,
#             kind="line",
#             x="optimizer_step_counter_epoch",
#             y=metric,
#             hue="Eval Topic (Val Split)",
#             hue_order=unique_and_ordered_eval_topic_strs,
#             style="Same Topic",
#             style_order=[False, True],
#             col="Attack Topic (Train Split)",
#             row="models_to_attack",
#             facet_kws={"margin_titles": True},
#         )
#         g.set_axis_labels(
#             "Gradient Step", src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]
#         )
#         g.fig.suptitle("Attacked Topics", y=1.0)
#         g.set_titles(col_template="{col_name}", row_template="{row_name}")
#         # g._legend.set_title("Evaluated Model")
#         sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
#         g.set(ylim=src.globals.METRICS_TO_BOUNDS_DICT[metric])
#         src.plot.save_plot_with_multiple_extensions(
#             plot_dir=learning_curves_by_attack_model_dir,
#             plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={idx}",
#         )
#         idx += 1
#         # if metric == "loss/avg_epoch":
#         g.set(
#             xscale="log",
#             yscale="log",
#             ylim=(0.95 * eval_runs_histories_df[metric].min(), None),
#         )
#         src.plot.save_plot_with_multiple_extensions(
#             plot_dir=learning_curves_by_attack_model_dir,
#             plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={idx}",
#         )
#         # plt.show()
#         idx += 1

# idx = 0
# for (
#     topics_to_eval,
#     eval_runs_histories_by_eval_topic_df,
# ) in eval_runs_histories_df.groupby("Eval Topic (Val Split)"):
#     for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT:
#         print(metric)
#         plt.close()
#         g = sns.relplot(
#             data=eval_runs_histories_by_eval_topic_df,
#             kind="line",
#             x="optimizer_step_counter_epoch",
#             y=metric,
#             hue="Attack Topic (Train Split)",
#             hue_order=unique_and_ordered_eval_topic_strs,
#             style="Same Topic",
#             style_order=[False, True],
#             col="Eval Topic (Val Split)",
#             row="models_to_attack",
#             facet_kws={"margin_titles": True},
#         )
#         g.set_axis_labels(
#             "Gradient Step", src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]
#         )
#         g.fig.suptitle("Evaluated Topics", y=1.0)
#         g.set_titles(col_template="{col_name}", row_template="{row_name}")
#         # g._legend.set_title("Evaluated Model")
#         sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
#         g.set(ylim=src.globals.METRICS_TO_BOUNDS_DICT[metric])
#         src.plot.save_plot_with_multiple_extensions(
#             plot_dir=learning_curves_by_eval_model_dir,
#             plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={idx}",
#         )
#         idx += 1
#         # if metric == "loss/avg_epoch":
#         g.set(
#             xscale="log",
#             yscale="log",
#             ylim=(0.95 * eval_runs_histories_df[metric].min(), None),
#         )
#         src.plot.save_plot_with_multiple_extensions(
#             plot_dir=learning_curves_by_eval_model_dir,
#             plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={idx}",
#         )
#         # plt.show()
#         idx += 1
