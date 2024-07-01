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
    "wkxrq2t2",  # n=1 from base
    "e813ex2n",  # n=1 to base
    "x714akbx",  # n=2
    "b5oob18s",  # n=2
    "wuxm0jp4",  # n=2
    "x38im6cm",  # n=2
    "kkypbhgu",  # n=2
    "u3bmmese",  # n=8
    "7mqmxgm1",  # n=8
    "q0dk9m5p",  # n=8
]


wandb_username = "danvalentine256"
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

print(1)

eval_runs_configs_df["num_attack_models"] = eval_runs_configs_df[
    "models_to_attack"
].apply(lambda x: len(ast.literal_eval(x)))
# Load the heftier runs' histories dataframe.
eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    wandb_username=wandb_username,
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    # finished_only=True,
    wandb_run_history_samples=1000000,
    filetype="csv",
)
print(2)
# This col is not populated on this df
eval_runs_histories_df.drop(columns=["models_to_attack"], inplace=True)


eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "eval_run_id",
            "attack_run_id",
            "model_to_eval",
            "models_to_attack",
            "num_attack_models",
            "attack_dataset",
            # "attack_subset",
            "eval_dataset",
            # "eval_subset",
            # "split",
        ]
    ],
    how="inner",
    left_on="run_id",
    right_on="eval_run_id",
)

print(3)
eval_runs_histories_df["Same Data Distribution"] = (
    eval_runs_histories_df["attack_dataset"] == eval_runs_histories_df["eval_dataset"]
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
# eval_runs_histories_df.rename(
#     columns={
#         # "attack_dataset": "Attack Dataset (Train Split)",
#         # "eval_dataset": "Eval Dataset (Val Split)",
#         # "attack_subset": "Attack Topic (Train Split)",
#         # "eval_subset": "Eval Topic (Val Split)",
#         "model_to_eval": "Evaluated Model",
#     },
#     inplace=True,
# )

print(4)
for metric in [
    "loss/avg_epoch",
    "loss/score_model=claude3opus",
]:
    metric_as_filename = src.globals.METRICS_TO_FILENAME_STRINGS_DICT[metric]

    df = eval_runs_configs_df[
        ["models_to_attack", "model_to_eval", metric, "num_attack_models"]
    ]

    df = eval_runs_histories_df.copy()
    df = df.dropna(subset=[metric])
    df = (
        df.groupby("eval_run_id")
        .apply(
            lambda x: x.nlargest(max(1, int(np.ceil(0.20 * len(x)))), "_step")
        )  # limit to the final 20% of eval steps
        .reset_index(drop=True)
        .groupby("eval_run_id")
        .agg(
            {
                metric: "mean",
                "model_to_eval": "first",
                "models_to_attack": "first",
                "num_attack_models": "first",
            }
        )
    )

    # Switch attack_model_names and eval_model_name to nice strings.
    df["model_to_eval"] = df["model_to_eval"].apply(
        src.analyze.map_string_set_of_models_to_nice_string
    )
    df["models_to_attack"] = df["models_to_attack"].apply(
        src.analyze.map_string_set_of_models_to_nice_string
    )

    # Add a column to indicate whether the eval model is in the attack models.
    df["Eval VLM in\nAttacked Ensemble"] = df.apply(
        lambda row: row["model_to_eval"] in row["models_to_attack"], axis=1
    )
    # Make a duplicate without the newline.
    df["Eval VLM in Attacked VLMs Ensemble"] = df["Eval VLM in\nAttacked Ensemble"]

    df["Attacked VLMs Ensemble"] = df["models_to_attack"]
    # for the n=8's, each model has an in-ensemble result from each of 8 attack model combos. aggregate these to reduce clutter.
    condition = (df["num_attack_models"] == 8) & (
        df["Eval VLM in Attacked VLMs Ensemble"]
    )
    subset_to_aggregate = df[condition]
    subset_not_to_aggregate = df[~condition]
    aggregation_methods = {
        col: "first"
        for col in subset_to_aggregate.columns
        if col
        not in [
            "loss/avg_epoch",
            "loss/score_model=claude3opus",
        ]
    }
    aggregation_methods.update(
        {
            metric: "mean",
        }
    )

    aggregated_subset = subset_to_aggregate.groupby(
        "model_to_eval", as_index=False
    ).agg(aggregation_methods)
    aggregated_subset["aggregated"] = True
    subset_not_to_aggregate["aggregated"] = False

    model_order = [
        "One Stage Training",
        "LLAVAv1.5 7B + CLIP (Repro)",
        "1.25 Epochs",
        "1.5 Epochs",
        "2 Epochs",
        "3 Epochs",
        "LVIS4V",
        "LRV",
        "LVIS4V+LRV",
    ]
    final_df = pd.concat(
        [aggregated_subset, subset_not_to_aggregate], ignore_index=True
    )

    # Sort based on the evaluated VLMs.
    final_df.sort_values(by="model_to_eval", inplace=True)

    # Create a categorical type with the sorted order
    final_df["model_to_eval_categorical"] = pd.Categorical(
        final_df["model_to_eval"],
        categories=model_order,
        ordered=True,
    )

    plt.close()
    plt.figure(figsize=(24, 16))
    g = sns.scatterplot(
        data=final_df,
        x="model_to_eval_categorical",
        y=metric,
        hue="Eval VLM in Attacked VLMs Ensemble",
        style="num_attack_models",
        markers=["o", "X", "D"],
        s=500,
    )
    plt.ylim(src.globals.METRICS_TO_BOUNDS_DICT[metric])
    plt.xlabel("Evaluated VLMs")
    plt.ylabel(src.globals.METRICS_TO_LABELS_NICE_STRINGS_DICT[metric])
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.title(
        f"{src.globals.METRICS_TO_TITLE_STRINGS_DICT[metric]} Scores of Attacking Similar VLMs with ensembles of different sizes"
    )
    # g.tick_params("x", rotation=90)
    plt.xticks(rotation=45, ha="right")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"metric={metric_as_filename}_scatter",
    )
    # plt.show()

    plt.close()
