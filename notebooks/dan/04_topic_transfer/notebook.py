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

max_values = {
    metric: pd.DataFrame(index=topics, columns=topics)
    for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT
}
# Double loop to find max values
for attack_topic in topics:
    attack_df = eval_runs_histories_df[
        eval_runs_histories_df["Attack Topic (Train Split)"] == attack_topic
    ]

    for eval_topic in topics:
        eval_df = attack_df[attack_df["Eval Topic (Val Split)"] == eval_topic]

        for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT.keys():
            if eval_df.empty:
                max_values[metric].loc[attack_topic, eval_topic] = np.nan
            else:
                # Convert to numeric, coercing errors to NaN
                numeric_values = pd.to_numeric(eval_df[metric], errors="coerce")

                # Calculate min, ignoring NaN values
                max_value = numeric_values.max()

                max_values[metric].loc[attack_topic, eval_topic] = max_value

# Append mean values to the DataFrame
loss_df_with_means = min_values["loss/avg_epoch"].copy()
print("1")
loss_df_with_means.loc["Mean per Eval"] = loss_df_with_means.mean(axis=0)
print("2")
loss_df_with_means["Mean per Attack"] = loss_df_with_means.mean(axis=1)
print("3")
loss_df_with_means.iloc[-1, -1] = np.nan
# Convert the updated DataFrame to numeric, handling non-numeric values
df_numeric_loss = loss_df_with_means.apply(pd.to_numeric, errors="coerce")
print("4")
# Step 1: Filter out rows where 'loss' is NaN
cleaned_df = eval_runs_histories_df.dropna(subset=["loss/avg_epoch"])

# Step 2: Group by 'eval_topic' and find the minimum '_step' for each topic where 'loss' is not NaN
grouped = cleaned_df.groupby("Eval Topic (Val Split)")
min_step_per_topic = grouped["_step"].min().reset_index()

# Step 3: Merge to get rows that match the minimum step for each topic
min_step_rows = pd.merge(
    cleaned_df, min_step_per_topic, on=["Eval Topic (Val Split)", "_step"]
)

# Step 4: Calculate the mean loss for these rows if there are duplicates
baseline_losses = min_step_rows.groupby("Eval Topic (Val Split)")[
    "loss/avg_epoch"
].mean()

# Convert baseline_losses to a DataFrame and transpose it for compatibility with df_numeric_loss
baseline_row = pd.DataFrame(baseline_losses).transpose()
baseline_row.index = ["Baseline"]  # Naming the index as 'Baseline'

# Step 5: Append the baseline row to the df_numeric_loss DataFrame
df_numeric_loss_with_baseline = pd.concat([df_numeric_loss, baseline_row])

# Ensure columns match, if df_numeric_loss is summarized differently, adjust column names
# df_numeric_loss_with_baseline = df_numeric_loss_with_baseline[df_numeric_loss.columns.tolist()]

# Now you can plot df_numeric_loss_with_baseline with the baseline included
breakpoint()

# Now you can plot df_numeric_loss_with_baseline with the baseline included
# Increase figure size for better visibility
plt.figure(figsize=(22, 18))

# Define a custom color map
cmap = sns.color_palette("YlGnBu", as_cmap=True)
cmap.set_bad(color="white")

# Create a mask to separate mean scores visually
# mask = np.zeros_like(df_numeric_loss, dtype=bool)
# mask[-1, :] = True
# mask[:, -1] = True

print("5")
# Create the heatmap with a custom mask for the mean values
ax = sns.heatmap(
    df_numeric_loss,
    annot=True,
    cmap=cmap,
    fmt=".2f",
    robust=True,
    # mask=mask,
    cbar_kws={"label": "Cross-Entropy Loss"},
    linewidths=0.5,
    linecolor="black",
    square=True,
    annot_kws={"size": 16},
)
ax.hlines(len(loss_df_with_means) - 1, *ax.get_xlim(), color="white", linewidth=6)
ax.vlines(
    len(loss_df_with_means.columns) - 1, *ax.get_ylim(), color="white", linewidth=6
)
xticks_labels = [tick.get_text() for tick in ax.get_xticklabels()]
xticks_labels[-1] = r"$\mathbf{\underline{Mean~per~Attack}}$"
yticks_labels = [tick.get_text() for tick in ax.get_yticklabels()]
yticks_labels[-1] = r"$\mathbf{\underline{Mean~per~Eval}}$"
ax.set_xticklabels(xticks_labels)
ax.set_yticklabels(yticks_labels)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
print("6")
# # Set thicker lines to separate means
# for edge, spine in plt.gca().spines.items():
#     spine.set_visible(True)
#     spine.set_linewidth(2)
# for spine in ax.spines.values():
#     spine.set_visible(False)
print("7")
plt.title("Minimum Attained Cross-Entropy Loss", fontsize=24)
plt.xlabel("Topic Evaluated", fontsize=20)
plt.ylabel("Topic Attacked", fontsize=20)
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)
print("a")
# Save and show plot
plt.tight_layout()
plt.savefig(f"{heatmaps_dir}/loss_with_means.png", dpi=300)
print("b")
# plt.show()
plt.close()

print("8")
# Claude heatmap
# Convert DataFrame to numeric type, replacing any remaining non-numeric values with NaN
# claude_df = max_values["loss/score_model=claude3opus"]
# df_numeric = claude_df.apply(pd.to_numeric, errors="coerce")

# # Increase figure size for larger boxes
# plt.figure(figsize=(20, 16))

# # Create a custom colormap from blue to green to yellow
# cmap = sns.color_palette("YlGnBu", as_cmap=True)
# cmap.set_bad(color="lightgray")

# # Use robust=True to compute color scale robustly to outliers
# sns.heatmap(
#     df_numeric,
#     annot=True,
#     cmap=cmap,
#     fmt=".2f",
#     robust=True,
#     mask=df_numeric.isnull(),
#     cbar_kws={"label": "Harmful-Yet-Helpful Score"},
#     square=True,
#     linewidths=0.5,
#     annot_kws={"size": 14},
# )

# plt.title(
#     f"Maximum Attained Harmful-Yet-Helpful Score",
#     fontsize=20,
# )
# plt.xlabel("Topic Evaluated", fontsize=20)
# plt.ylabel("Topic Attacked", fontsize=20)
# plt.xticks(rotation=45, ha="right", fontsize=16)
# plt.yticks(fontsize=16)

# # Add more padding
# plt.tight_layout(pad=3.0)

# plt.savefig(f"{heatmaps_dir}/claude.pdf", dpi=300, bbox_inches="tight")
# plt.close()


# loss_df = min_values["loss/avg_epoch"]
# df_numeric = loss_df.apply(pd.to_numeric, errors="coerce")

# # Increase figure size for larger boxes
# plt.figure(figsize=(20, 16))

# # Create a custom colormap from blue to green to yellow
# cmap = sns.color_palette("YlGnBu", as_cmap=True)
# cmap.set_bad(color="lightgray")

# # Use robust=True to compute color scale robustly to outliers
# sns.heatmap(
#     df_numeric,
#     annot=True,
#     cmap=cmap,
#     fmt=".2f",
#     robust=True,
#     mask=df_numeric.isnull(),
#     cbar_kws={"label": "Cross-Entropy Loss"},
#     square=True,
#     linewidths=0.5,
#     annot_kws={"size": 14},
# )

# plt.title(
#     f"Minimum Attained Cross-Entropy Loss",
#     fontsize=20,
# )
# plt.xlabel("Topic Evaluated", fontsize=20)
# plt.ylabel("Topic Attacked", fontsize=20)
# plt.xticks(rotation=45, ha="right", fontsize=16)
# plt.yticks(fontsize=16)

# # Add more padding
# plt.tight_layout(pad=3.0)

# plt.savefig(f"{heatmaps_dir}/loss.pdf", dpi=300, bbox_inches="tight")
# plt.close()
# print("Heatmaps have been generated and saved in the results directory.")

# # Diagnostic information
# print("\nDiagnostic Information:")
# for metric, df in min_values.items():
#     print(f"\nMetric: {metric}")
#     print("Data types:")
#     print(df.dtypes)
#     print("\nNumber of NaN values:")
#     print(df.isna().sum())
#     print("\nSample data:")
#     print(df.head())
#     print(
#         "\nPercentage of missing values:", (df.isna().sum().sum() / df.size) * 100, "%"
#     )

# # Compute mean scores per attack topic for Minimum Loss and Maximum Claude
# mean_scores_attack_topic = {
#     "Min Loss": min_values["loss/avg_epoch"].mean(
#         axis=1
#     ),  # Mean across columns (eval topics)
#     "Max Claude": max_values["loss/score_model=claude3opus"].mean(axis=1),
# }

# # Compute mean scores per eval topic for Minimum Loss and Maximum Claude
# mean_scores_eval_topic = {
#     "Min Loss": min_values["loss/avg_epoch"].mean(
#         axis=0
#     ),  # Mean across rows (attack topics)
#     "Max Claude": max_values["loss/score_model=claude3opus"].mean(axis=0),
# }

# # Convert the mean scores dictionaries to DataFrame for better tabular presentation
# mean_scores_attack_topic_df = pd.DataFrame(mean_scores_attack_topic)
# mean_scores_eval_topic_df = pd.DataFrame(mean_scores_eval_topic)

# # Print the mean scores tables
# print("Mean Scores per Attack Topic:")
# print(mean_scores_attack_topic_df)
# print("\nMean Scores per Eval Topic:")
# print(mean_scores_eval_topic_df)

# # # Function to calculate mean scores
# # def calculate_mean_scores(data, metric, value_type):
# #     # Calculate means for attack topics and eval topics
# #     attack_means = data[metric].mean(axis=1)
# #     eval_means = data[metric].mean(axis=0)

# #     # Calculate statistics of these means
# #     attack_stats = pd.Series(
# #         {
# #             "Mean of Means": attack_means.mean(),
# #             "Std.Dev of Means": attack_means.std(),
# #             "Variance of Means": attack_means.var(),
# #         }
# #     )

# #     eval_stats = pd.Series(
# #         {
# #             "Mean of Means": eval_means.mean(),
# #             "Std.Dev of Means": eval_means.std(),
# #             "Variance of Means": eval_means.var(),
# #         }
# #     )

# #     print(f"\n{value_type} {metric} - Statistics of Attack Topic Means:")
# #     print(attack_stats.to_string())
# #     print(f"\n{value_type} {metric} - Statistics of Eval Topic Means:")
# #     print(eval_stats.to_string())

# #     # Optional: Print all means for reference
# #     print(f"\nAll Attack Topic Means:")
# #     print(attack_means.sort_values(ascending=False).to_string())
# #     print(f"\nAll Eval Topic Means:")
# #     print(eval_means.sort_values(ascending=False).to_string())


# # # Calculate for minimum loss
# # calculate_mean_scores(min_values, "loss/avg_epoch", "Minimum")

# # # Calculate for maximum Claude score
# # calculate_mean_scores(
# #     max_values,
# #     "loss/score_model=claude3opus",
# #     "Maximum",
# # )
# # idx = 0
# #
# # for (
# #     topics_to_attack,
# #     eval_runs_histories_by_attack_topic_df,
# # ) in eval_runs_histories_df.groupby("Attack Topic (Train Split)"):
# #     for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT:
# #         print(metric)
# #         plt.close()
# #         g = sns.relplot(
# #             data=eval_runs_histories_by_attack_topic_df,
# #             kind="line",
# #             x="optimizer_step_counter_epoch",
# #             y=metric,
# #             hue="Eval Topic (Val Split)",
# #             hue_order=unique_and_ordered_eval_topic_strs,
# #             style="Same Topic",
# #             style_order=[False, True],
# #             col="Attack Topic (Train Split)",
# #             row="models_to_attack",
# #             facet_kws={"margin_titles": True},
# #         )
# #         g.set_axis_labels(
# #             "Gradient Step", src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]
# #         )
# #         g.fig.suptitle("Attacked Topics", y=1.0)
# #         g.set_titles(col_template="{col_name}", row_template="{row_name}")
# #         # g._legend.set_title("Evaluated Model")
# #         sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
# #         g.set(ylim=src.globals.METRICS_TO_BOUNDS_DICT[metric])
# #         src.plot.save_plot_with_multiple_extensions(
# #             plot_dir=learning_curves_by_attack_model_dir,
# #             plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={idx}",
# #         )
# #         idx += 1
# #         # if metric == "loss/avg_epoch":
# #         g.set(
# #             xscale="log",
# #             yscale="log",
# #             ylim=(0.95 * eval_runs_histories_df[metric].min(), None),
# #         )
# #         src.plot.save_plot_with_multiple_extensions(
# #             plot_dir=learning_curves_by_attack_model_dir,
# #             plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={idx}",
# #         )
# #         # plt.show()
# #         idx += 1

# # idx = 0
# # for (
# #     topics_to_eval,
# #     eval_runs_histories_by_eval_topic_df,
# # ) in eval_runs_histories_df.groupby("Eval Topic (Val Split)"):
# #     for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT:
# #         print(metric)
# #         plt.close()
# #         g = sns.relplot(
# #             data=eval_runs_histories_by_eval_topic_df,
# #             kind="line",
# #             x="optimizer_step_counter_epoch",
# #             y=metric,
# #             hue="Attack Topic (Train Split)",
# #             hue_order=unique_and_ordered_eval_topic_strs,
# #             style="Same Topic",
# #             style_order=[False, True],
# #             col="Eval Topic (Val Split)",
# #             row="models_to_attack",
# #             facet_kws={"margin_titles": True},
# #         )
# #         g.set_axis_labels(
# #             "Gradient Step", src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]
# #         )
# #         g.fig.suptitle("Evaluated Topics", y=1.0)
# #         g.set_titles(col_template="{col_name}", row_template="{row_name}")
# #         # g._legend.set_title("Evaluated Model")
# #         sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
# #         g.set(ylim=src.globals.METRICS_TO_BOUNDS_DICT[metric])
# #         src.plot.save_plot_with_multiple_extensions(
# #             plot_dir=learning_curves_by_eval_model_dir,
# #             plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={idx}",
# #         )
# #         idx += 1
# #         # if metric == "loss/avg_epoch":
# #         g.set(
# #             xscale="log",
# #             yscale="log",
# #             ylim=(0.95 * eval_runs_histories_df[metric].min(), None),
# #         )
# #         src.plot.save_plot_with_multiple_extensions(
# #             plot_dir=learning_curves_by_eval_model_dir,
# #             plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={idx}",
# #         )
# #         # plt.show()
# #         idx += 1
