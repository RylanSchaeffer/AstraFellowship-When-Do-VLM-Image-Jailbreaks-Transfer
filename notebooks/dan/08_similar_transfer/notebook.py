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

plt.close()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 16))

metrics = [
    "loss/avg_epoch",
    "loss/score_model=claude3opus",
]

bounds = {
    "loss/avg_epoch": (0.90, 1.85),
    "loss/score_model=claude3opus": (0.0, 1.0),
}

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

for i, (metric, ax) in enumerate(zip(metrics, [ax1, ax2])):
    metric_as_filename = src.globals.METRICS_TO_FILENAME_STRINGS_DICT[metric]

    aggregation_cols = {
        metric: "mean",
        "model_to_eval": "first",
        "models_to_attack": "first",
        "num_attack_models": "first",
    }
    df = eval_runs_histories_df.copy()
    df = df.dropna(subset=[metric])
    df = (
        df.groupby("eval_run_id")
        .apply(lambda x: x.nlargest(max(1, int(np.ceil(0.20 * len(x)))), "_step"))
        .reset_index(drop=True)
        .groupby("eval_run_id")
        .agg(aggregation_cols)
    )

    df["Eval VLM in Attacked VLMs Ensemble"] = df.apply(
        lambda row: list(ast.literal_eval(row["model_to_eval"]))[0]
        in ast.literal_eval(row["models_to_attack"]),
        axis=1,
    )
    aggregation_cols.update({"Eval VLM in Attacked VLMs Ensemble": "first"})

    df["model_to_eval"] = df["model_to_eval"].apply(
        src.analyze.map_string_set_of_models_to_nice_string
    )
    df["models_to_attack"] = df["models_to_attack"].apply(
        src.analyze.map_string_set_of_models_to_nice_string
    )
    df["Attacked VLMs Ensemble"] = df["models_to_attack"]
    df = df.groupby(
        ["model_to_eval", "num_attack_models", "Eval VLM in Attacked VLMs Ensemble"],
        as_index=False,
    ).agg(aggregation_cols)

    df.sort_values(by="model_to_eval", inplace=True)

    df["Number of Attacked Models"] = df["num_attack_models"]
    df["Evaluated Model"] = pd.Categorical(
        df["model_to_eval"],
        categories=model_order,
        ordered=True,
    )

    out_of_ensemble = df[df["Eval VLM in Attacked VLMs Ensemble"] == False]

    g = sns.lineplot(
        data=out_of_ensemble,
        x="Number of Attacked Models",
        y=metric,
        hue="Evaluated Model",
        linewidth=2.0,
        ax=ax,
    )
    g.set_xscale("log", base=2)
    # if "loss" in metric:
    #     g.set_yscale("log")
    # g.set_ylim(bounds[metric])
    g.set_ylabel("Cross Entropy Loss" if i == 0 else "Harmful-Yet-Helpful", fontsize=36)
    g.set_title("Cross Entropy Loss" if i == 0 else "Claude 3 Opus", fontsize=40)
    g.set_xlabel("Number of Attacked Models", fontsize=36)
    if i == 1:
        g.legend(
            fontsize=32,
            title_fontsize=32,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Target VLM",
        )
    else:
        g.legend().remove()

    tick_locations = [1, 2, 4, 8]
    g.set_xticks(tick_locations)
    g.set_xticklabels(tick_locations)

    sns.scatterplot(
        data=out_of_ensemble,
        x="Number of Attacked Models",
        y=metric,
        hue="Evaluated Model",
        s=400,
        legend=False,
        ax=ax,
    )
# plt.subplots_adjust(wspace=1)
# plt.tight_layout()
fig.suptitle("Transfer When Attacking Highly-Similar Ensembles", fontsize=50, y=1.05)

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="combined_metrics_scatter",
)
plt.close()
