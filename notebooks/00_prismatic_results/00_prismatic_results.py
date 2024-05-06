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


refresh = True
# refresh = False
metrics = src.analyze.metrics_to_nice_strings_dict

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

attack_wandb_attack_run_ids = eval_runs_configs_df["wandb_attack_run_id"].unique()
api = wandb.Api(timeout=600)
attack_wandb_sweep_ids = np.unique(
    [
        api.run(f"rylan/universal-vlm-jailbreak/{run_id}").sweep.id
        for run_id in attack_wandb_attack_run_ids
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

pairwise_metrics_dir = os.path.join(results_dir, "pairwise_metrics")
os.makedirs(pairwise_metrics_dir, exist_ok=True)
for metric_x in metrics:
    for metric_y in metrics:
        if metric_x == metric_y:
            continue
        plt.close()
        g = sns.relplot(
            eval_runs_configs_df,
            x=metric_x,
            y=metric_y,
            col="attack_dataset",
            row="eval_dataset",
            # style="attack_dataset",
            # style_order=["advbench", "rylan_anthropic_hhh"],
            facet_kws={"margin_titles": True},
        )
        g.set(
            xlim=src.analyze.metrics_to_bounds_dict[metric_x],
            ylim=src.analyze.metrics_to_bounds_dict[metric_y],
        )
        g.set_axis_labels(
            x_var=src.analyze.metrics_to_nice_strings_dict[metric_x],
            y_var=src.analyze.metrics_to_nice_strings_dict[metric_y],
        )
        g.set_titles(
            col_template="Attack Dataset: {col_name}",
            row_template="Eval Dataset: {row_name}",
        )
        src.plot.save_plot_with_multiple_extensions(
            plot_dir=pairwise_metrics_dir,
            plot_title=f"pairwise_metrics_{metric_y[5:]}_vs_{metric_x[5:]}",  # Strip off "loss/".
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


unique_and_ordered_eval_model_strs = np.sort(
    eval_runs_histories_df["model_to_eval"].unique()
)

# Modify columns names and values.
eval_runs_histories_df.rename(
    columns={
        "model_to_eval": "Evaluated Model",
        "eval_dataset": "Eval Dataset",
        "attack_dataset": "Attack Dataset",
    },
    inplace=True,
)
# vlm_metadata_df = pd.read_csv(
#     os.path.join("configs", "vlm_metadata.csv"),
# )

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
        data=eval_runs_histories_df[eval_runs_histories_df["split"] == "eval"],
        kind="line",
        x=x,
        y=metric,
        hue="Evaluated Model",
        hue_order=unique_and_ordered_eval_model_strs,
        style="Attack Dataset",
        style_order=["advbench", "rylan_anthropic_hhh"],
        col="models_to_attack",
        row="Eval Dataset",
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
            plot_title=f"{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models",
        )
    # plt.show()


for (
    models_to_attack,
    eval_runs_histories_by_attack_df,
) in eval_runs_histories_df.groupby("models_to_attack"):
    for metric in src.analyze.metrics_to_nice_strings_dict:
        plt.close()
        g = sns.relplot(
            data=eval_runs_histories_by_attack_df[
                eval_runs_histories_by_attack_df["split"] == "eval"
            ],
            kind="line",
            x="optimizer_step_counter_epoch",
            y=metric,
            hue="Evaluated Model",
            hue_order=unique_and_ordered_eval_model_strs,
            style="Attack Dataset",
            style_order=["advbench", "rylan_anthropic_hhh"],
            col="models_to_attack",
            row="Eval Dataset",
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
            plot_title=f"prismatic_{metric[5:]}_vs_gradient_step_cols=eval_models_rows=attack_models={models_to_attack}",
        )
        if metric == "loss/avg_epoch":
            g.set(
                xscale="log",
                yscale="log",
                ylim=(0.95 * eval_runs_histories_df["loss/avg_epoch"].min(), None),
            )
            src.plot.save_plot_with_multiple_extensions(
                plot_dir=learning_curves_by_attack_model_dir,
                plot_title=f"prismatic_{metric[5:]}_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={models_to_attack}",
            )
        # plt.show()


print("Finished notebooks/00_prismatic_results!")
