import ast
import math
from matplotlib.lines import Line2D
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
finished_only = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,  # delete existing results if they exist (data is not deleted)
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


sweep_ids = [
    "u3bmmese",  # n=8
    "7mqmxgm1",
    "q0dk9m5p",
    "x714akbx",  # n=2
    "b5oob18s",  # n=2
    "wuxm0jp4",  # n=2
    "x38im6cm",  # n=2
    "kkypbhgu",  # n=2
]


# Download wandb results for all runs in the given sweep.
eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,  # will use local cache if this is false
    finished_only=finished_only,
    wandb_username=wandb_user,
    filetype="csv",
)


eval_runs_configs_df = parse_data_col(eval_runs_configs_df, "eval_dataset", "dataset")
eval_runs_configs_df = parse_data_col(eval_runs_configs_df, "split")
# eval_runs_configs_df = parse_data_col(eval_runs_configs_df, "eval_subset", "subset")

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
    finished_only=finished_only,
    filetype="csv",
)

attack_runs_configs_df = parse_data_col(
    attack_runs_configs_df, "attack_dataset", "dataset"
)
# TODO: this will break if subset was not specified (ie on all non-topic exps)
# attack_runs_configs_df = parse_data_col(
#     attack_runs_configs_df, "attack_subset", "subset"
# )

attack_runs_configs_df.rename(
    columns={"run_id": "attack_run_id"},
    inplace=True,
)


# Add needed attack run data to evals df
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[["attack_run_id", "attack_dataset"]],
    how="left",
    left_on="wandb_attack_run_id",
    right_on="attack_run_id",
)


def check_in_ensemble(row):
    model_to_eval = ast.literal_eval(row["model_to_eval"])
    models_to_attack = ast.literal_eval(row["models_to_attack"])
    return any(model in models_to_attack for model in model_to_eval)


eval_runs_configs_df["In ensemble"] = eval_runs_configs_df.apply(
    check_in_ensemble, axis=1
)
# unique_and_ordered_eval_model_strs = np.sort(
#     eval_runs_configs_df["model_to_eval"].unique()
# ).tolist()

# unique_and_ordered_attack_model_strs = np.sort(
#     eval_runs_configs_df["models_to_attack"].unique()
# ).tolist()

eval_runs_configs_df["num_attack_models"] = eval_runs_configs_df[
    "models_to_attack"
].apply(lambda x: len(ast.literal_eval(x)))

eval_runs_configs_df["Evaluated Model"] = eval_runs_configs_df["model_to_eval"].apply(
    src.analyze.map_string_set_of_models_to_nice_string
)
eval_runs_configs_df["Attacked Model"] = eval_runs_configs_df["models_to_attack"].apply(
    src.analyze.map_string_set_of_models_to_nice_string
)

eval_runs_configs_df["Same Data Distribution"] = (
    eval_runs_configs_df["attack_dataset"] == eval_runs_configs_df["eval_dataset"]
)
eval_runs_configs_df["Same Model"] = (
    eval_runs_configs_df["models_to_attack"] == eval_runs_configs_df["model_to_eval"]
)
# Load the heftier runs' histories dataframe.
# Not entirely sure on the specifics here - high level it is metric samples from the history of each eval run
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_user,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    # finished_only=finished_only,
    wandb_run_history_samples=1000000,
    filetype="csv",
)
# This col is not populated on this df
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)
print(1)

# breakpoint()
eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "run_id",
            "Sweep",
            "attack_run_id",
            "Evaluated Model",
            "Attacked Model",
            "attack_dataset",
            # "attack_subset",
            "eval_dataset",
            # "eval_subset",
            "split",
            "In ensemble",
            "Same Data Distribution",
            "Same Model",
            "num_attack_models",
        ]
    ],
    how="inner",
    left_on="run_id",
    right_on="run_id",
)


print(2)


# eval_runs_histories_df["Same Topic"] = (
#     eval_runs_histories_df["attack_subset"] == eval_runs_histories_df["eval_subset"]
# )

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
        # "model_to_eval": "Evaluated Model",
        # "models_to_attack": "Attacked Model",
    },
    inplace=True,
)

print(3)

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

print(4)
# Convert metrics to nice strings.
eval_runs_histories_tall_df["Original Metric"] = eval_runs_histories_tall_df["Metric"]
eval_runs_histories_tall_df["Metric"] = eval_runs_histories_tall_df["Metric"].replace(
    src.globals.METRICS_TO_TITLE_STRINGS_DICT
)

# eval_runs_histories_df["Eval Model Type"] = eval_runs_histories_df["Evaluated Model"]
# eval_runs_histories_df["Attack Model Type"] = eval_runs_histories_df["Attacked Model"]
# eval_runs_histories_df["Eval Model Type"] = eval_runs_histories_df[
#     "Eval Model Type"
# ].replace(models_to_types)
# eval_runs_histories_df["Attack Model Type"] = eval_runs_histories_df[
#     "Attack Model Type"
# ].replace(models_to_types)
learning_curves_from_base_dir = os.path.join(results_dir, "learning_curves_from_base")

learning_curves_to_base_dir = os.path.join(results_dir, "learning_curves_to_base")
generalisation_dir = os.path.join(results_dir, "generalisation")


os.makedirs(learning_curves_from_base_dir, exist_ok=True)
os.makedirs(learning_curves_to_base_dir, exist_ok=True)
os.makedirs(generalisation_dir, exist_ok=True)


model_order = [
    "One-Stage Training (1S)",
    "Two-Stage Training",
    "1S, 1.25 Epochs",
    "1S, 1.5 Epochs",
    "1S, 2 Epochs",
    "1S, 3 Epochs",
    "1S + LVIS-Instruct-4V",
    "1S + LRV-Instruct",
    "1S + LVIS + LRV",
]
plt.close()


from_base_df = eval_runs_histories_df[eval_runs_histories_df["Sweep"] == "wkxrq2t2"]
to_base_df = eval_runs_histories_df[eval_runs_histories_df["Sweep"] == "e813ex2n"]

print(5)
metrics = ["loss/avg_epoch", "loss/score_model=claude3opus"]
idx = 0
df8 = eval_runs_histories_df[eval_runs_histories_df["num_attack_models"] == 8]
attack_models = df8["Attacked Model"].unique()

for metric in metrics:
    n_rows = math.ceil(len(attack_models) / 3)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=3, figsize=(30, 8 * n_rows), squeeze=False
    )
    fig.suptitle(
        f"n=8 Ensemble Transfer - {src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]}",
        fontsize=35,
        y=1.02,
    )

    # Get unique values for legend
    unique_ensemble = df8["In ensemble"].unique()

    # Create a color palette
    color_palette = sns.color_palette("husl", n_colors=9)

    for idx, attack_model in enumerate(attack_models):
        df_by_model_type = df8[df8["Attacked Model"] == attack_model]
        non_ensemble_model = df_by_model_type[df_by_model_type["In ensemble"] == False][
            "Evaluated Model"
        ].iloc[0]
        non_ensemble_model = non_ensemble_model.strip("{'}")  # Remove extra characters

        idx = model_order.index(non_ensemble_model)
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        sns_plot = sns.lineplot(
            data=df_by_model_type,
            x="optimizer_step_counter_epoch",
            y=metric,
            hue="Evaluated Model",
            hue_order=model_order,
            style="In ensemble",
            style_order=[False, True],
            palette=color_palette,
            ax=ax,
            legend=False,  # Turn off individual legends
        )

        ax.set_title(f"{non_ensemble_model}", fontsize=28)
        ax.set_xlabel("Gradient Step", fontsize=28)
        ax.set_ylabel(src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric], fontsize=28)
        ax.set_ylim(src.globals.METRICS_TO_BOUNDS_DICT[metric])

        ax.tick_params(axis="x", rotation=45, labelsize=24)
        ax.tick_params(axis="y", labelsize=24)

    # Remove any unused subplots
    for idx in range(len(attack_models), n_rows * 3):
        row = idx // 3
        col = idx % 3
        fig.delaxes(axes[row, col])

    # Manually create legend elements
    legend_elements = []

    # Add color/hue legend elements
    for model, color in zip(model_order, color_palette):
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=model.strip("{'" "}'"))
        )

    # Add style legend elements
    for ensemble in unique_ensemble:
        style = "-" if ensemble else "--"
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                lw=2,
                linestyle=style,
                label=f"{'In' if ensemble else 'Not in'} ensemble",
            )
        )

    # Add the manual legend
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=24,
        title="Evaluated Model",
        title_fontsize=28,
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, hspace=0.5, wspace=0.3)

    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"n=8_{metric[5:]}",
    )
    plt.close()

idx = 0
df2 = eval_runs_histories_df[eval_runs_histories_df["num_attack_models"] == 2]
attack_models = df2["Attacked Model"].unique()

for metric in metrics:
    n_rows = math.ceil(len(attack_models) / 3)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=3, figsize=(30, 8 * n_rows), squeeze=False
    )
    fig.suptitle(
        f"n=2 Ensemble Transfer - {src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]}",
        fontsize=35,
        y=1.02,
    )

    # Get unique values for legend
    unique_ensemble = df8["In ensemble"].unique()

    # Create a color palette
    color_palette = sns.color_palette(
        "husl", n_colors=len(df2["Evaluated Model"].unique())
    )

    for idx, attack_model in enumerate(attack_models):
        df_by_model_type = df2[df2["Attacked Model"] == attack_model]
        eval_models = df_by_model_type["Evaluated Model"].unique()

        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        sns_plot = sns.lineplot(
            data=df_by_model_type,
            x="optimizer_step_counter_epoch",
            y=metric,
            hue="Evaluated Model",
            hue_order=model_order,
            palette=color_palette,
            ax=ax,
            legend=False,  # Turn off individual legends
        )

        att = attack_model.replace("\n", " + ")
        print(att)
        ax.set_title(
            f"Attack: {att}\nEval: {', '.join(eval_models)}",
            fontsize=28,
        )
        ax.set_xlabel("Gradient Step", fontsize=28)
        ax.set_ylabel(src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric], fontsize=28)
        ax.set_ylim(src.globals.METRICS_TO_BOUNDS_DICT[metric])

        ax.tick_params(axis="x", rotation=45, labelsize=24)
        ax.tick_params(axis="y", labelsize=24)

    # Remove any unused subplots
    for idx in range(len(attack_models), n_rows * 3):
        row = idx // 3
        col = idx % 3
        fig.delaxes(axes[row, col])

    # Manually create legend elements
    legend_elements = []

    # Add color/hue legend elements
    for model, color in zip(model_order, color_palette):
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=model.strip("{'" "}'"))
        )

    # Add the manual legend
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=24,
        title="Evaluated Model",
        title_fontsize=28,
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, hspace=0.5, wspace=0.3)

    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"n=2_{metric[5:]}",
    )
    plt.close()
# ooe = eval_runs_histories_df[eval_runs_histories_df["In ensemble"] == False]
# if attack_models == "Base":
#     print("skipping base")
#     continue
# base = from_base_df[from_base_df["Eval Model Type"] == "Base"]
# df = pd.concat([df_by_model_type, base], ignore_index=True)
# for metric in src.globals.METRICS_TO_TITLE_STRINGS_DICT:
#     plt.close()
#     g = sns.relplot(
#         data=ooe,
#         kind="line",
#         x="optimizer_step_counter_epoch",
#         y=metric,
#         hue="Evaluated Model",
#         hue_order=unique_and_ordered_eval_model_strs,
#         # style="Same Model",
#         # style_order=[False, True],
#         col="In ensemble",
#         # row="Eval Dataset (Val Split)",
#         facet_kws={"margin_titles": True},
#     )
#     g.set_axis_labels(
#         "Gradient Step", src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]
#     )
#     g.fig.suptitle("Ensemble Transfer", y=1.0)
#     # g.set_titles(col_template="{col_name}", row_template="{row_name}")
#     # g._legend.set_title("Evaluated Model")
#     sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
#     g.set(ylim=src.globals.METRICS_TO_BOUNDS_DICT[metric])
#     src.plot.save_plot_with_multiple_extensions(
#         plot_dir=generalisation_dir,
#         plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={idx}_model_type=ooe",
#     )
#     idx += 1
#     # if metric == "loss/avg_epoch":
#     g.set(
#         xscale="log",
#         yscale="log",
#         ylim=(0.95 * eval_runs_histories_df[metric].min(), None),
#     )
#     src.plot.save_plot_with_multiple_extensions(
#         plot_dir=generalisation_dir,
#         plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={idx}_model_type=ooe",
#     )
#     # plt.show()
#     idx += 1
