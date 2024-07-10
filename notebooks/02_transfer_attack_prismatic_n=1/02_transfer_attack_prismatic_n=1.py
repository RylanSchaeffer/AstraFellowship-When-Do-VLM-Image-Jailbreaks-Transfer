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
    "zyf0lb9y",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 1)
    "s754hflc",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 2)
    "jl9as45o",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 3)
    "1yoxmmrk",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 4)
    "bjg1o5ko",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 5)
    "8nrhoa2q",  # Prismatic with N-Choose-1 Jailbreaks, AdvBench & Rylan Anthropic HHH (Part 6)
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

# Download attack runs.
unique_attack_run_ids = eval_runs_configs_df["attack_run_id"].unique()
print("Attack Run IDs: ", unique_attack_run_ids.tolist())
attack_runs_configs_df = src.analyze.download_wandb_project_runs_configs_by_run_ids(
    wandb_project_path="universal-vlm-jailbreak",
    wandb_username=wandb_username,
    data_dir=data_dir,
    run_ids=unique_attack_run_ids,
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
attack_runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=attack_runs_configs_df,
    col_name="image_kwargs",
    key_in_dict="image_initialization",
    new_col_name="image_initialization",
)
attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)
attack_runs_configs_df["image_initialization"] = attack_runs_configs_df[
    "image_initialization"
].map(src.globals.IMAGE_INITIALIZATION_TO_STRINGS_DICT)

# Join attack run data into to evals df.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[
        ["attack_run_id", "attack_dataset", "image_initialization"]
    ],
    how="left",
    left_on="attack_run_id",
    right_on="attack_run_id",
)

eval_runs_configs_df["Attacked"] = eval_runs_configs_df.apply(
    lambda row: row["model_to_eval"] in row["models_to_attack"], axis=1
)

# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_username,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=1000000,
    # nrows_to_read=500000,
    # nrows_to_read=5000000,
    filetype="csv",
    # filetype="feather",
    # filetype="parquet",
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
            "image_initialization",
            "Attacked",
        ]
    ],
    how="inner",
    on="eval_run_id",
)

unique_metrics_order = [
    "loss/score_model=claude3opus",
]

eval_runs_histories_tall_df = eval_runs_histories_df.melt(
    id_vars=[
        "eval_run_id",
        "attack_run_id",
        "attack_dataset",
        "eval_dataset",
        "model_to_eval",
        "models_to_attack",
        "optimizer_step_counter_epoch",
        "image_initialization",
        "Attacked",
    ],
    value_vars=unique_metrics_order,
    var_name="Metric",
    value_name="Score",
)

eval_runs_histories_tall_df.rename(
    columns={
        "model_to_eval": "Eval VLM",
        "image_initialization": "Image Initialization",
    },
    inplace=True,
)

sorted_unique_attacked_models = list(
    sorted(eval_runs_histories_tall_df["models_to_attack"].unique())
)

# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].map(
    lambda k: src.globals.METRICS_TO_TITLE_STRINGS_DICT.get(k, k)
)

# Obtain the first optimizer_step_counter_epoch per eval_run_id.
first_optimizer_step = (
    eval_runs_histories_tall_df.groupby("eval_run_id")["optimizer_step_counter_epoch"]
    .min()
    .reset_index()
)
last_optimizer_step = (
    eval_runs_histories_tall_df.groupby("eval_run_id")["optimizer_step_counter_epoch"]
    .max()
    .reset_index()
)


# Merge these with the original dataframe to get the corresponding rows
first_optimizer_step_rows_df = (
    pd.merge(
        eval_runs_histories_tall_df,
        first_optimizer_step,
        on=["eval_run_id", "optimizer_step_counter_epoch"],
        how="inner",
    )
    .rename(columns={"Score": "Initial Score"})
    .drop(columns=["optimizer_step_counter_epoch"])
)

last_optimizer_step_rows_df = (
    pd.merge(
        eval_runs_histories_tall_df,
        last_optimizer_step,
        on=["eval_run_id", "optimizer_step_counter_epoch"],
        how="inner",
    )
    .rename(columns={"Score": "Final Score"})
    .drop(columns=["optimizer_step_counter_epoch"])
)

# Combine first and last rows into a single dataframe
first_and_last_optimizer_step_df = pd.merge(
    first_optimizer_step_rows_df,
    last_optimizer_step_rows_df,
    on=[
        "eval_run_id",
        "attack_run_id",
        "attack_dataset",
        "eval_dataset",
        "Eval VLM",
        "models_to_attack",
        "Image Initialization",
        "Attacked",
        "Metric",
        "Original Metric",
    ],
    how="inner",
)


plt.close()
g = sns.relplot(
    data=first_and_last_optimizer_step_df,
    kind="scatter",
    x="Initial Score",
    y="Final Score",
    col="models_to_attack",
    col_order=sorted_unique_attacked_models,
    style="Attacked",
    style_order=[False, True],
    hue="Eval VLM",
    col_wrap=5,
    s=250,
    aspect=0.75,
)
line = np.linspace(0.0, 1.0, 100)
for ax in g.axes.flat:
    ax.plot(line, line, "k--")
g.set_axis_labels("Harmful-Yet-Helpful (Initial)", "Harmful-Yet-Helpful (Final)")
g.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
# Add identity line to each axis.
g.set_titles(col_template="{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
g.fig.suptitle("Transfer From Single VLM to New VLM", y=1.0, fontsize=60)
# Make space for the title.
plt.subplots_adjust(top=0.9)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title=f"final_score_vs_initial_score_by_attacked_split_models_to_attack",
)
# plt.show()


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
            col="models_to_attack",
            col_order=sorted_unique_attacked_models,
            style="Attacked",
            style_order=[False, True],
            hue="Eval VLM",
            col_wrap=5,
            linewidth=3,
            aspect=0.75,
        )
        g.set_axis_labels("Gradient Step", "Harmful-Yet-Helpful")
        g.set(xlim=(0, 50000), ylim=(0.0, 1.0))
        g.set_titles(col_template="{col_name}")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
        g.fig.suptitle("Transfer From Single VLM to New VLM", y=1.0, fontsize=60)
        # Make space for the title.
        plt.subplots_adjust(top=0.9)
        src.plot.save_plot_with_multiple_extensions(
            plot_dir=learning_curves_eval_dataset_attack_dataset_results_dir,
            plot_title=f"score_vs_optimizer_step_by_attacked_split_models_to_attack",
        )
        # plt.show()


print("Finished notebooks/02_transfer_attack_prismatic_n=1!")
