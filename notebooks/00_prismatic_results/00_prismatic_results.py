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
    "omurwifa",  # Prismatic with N-Choose-1 Jailbreaks
    # "",  # Prismatic with N-Choose-2 Jailbreaks
]

eval_runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=False,
)
eval_runs_configs_df["dataset"] = eval_runs_configs_df["data"].apply(
    lambda x: x["dataset"] if isinstance(x, dict) else ast.literal_eval(x)["dataset"]
)
eval_runs_configs_df["split"] = eval_runs_configs_df["data"].apply(
    lambda x: x["split"] if isinstance(x, dict) else ast.literal_eval(x)["split"]
)

attack_wandb_run_ids = eval_runs_configs_df["wandb_run_id"].unique()
api = wandb.Api(timeout=600)
attack_wandb_sweep_ids = np.unique(
    [
        api.run(f"rylan/universal-vlm-jailbreak/{run_id}").sweep.id
        for run_id in attack_wandb_run_ids
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

eval_runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=10000,
)

# Add metadata about evaluations.
# TODO: Fix this shit naming of wandb_run_id.
eval_runs_configs_df = eval_runs_configs_df.merge(
    right=attack_runs_configs_df[
        ["attack_run_id", "models_to_attack", "attack_dataset"]
    ],
    how="left",
    left_on="wandb_run_id",
    right_on="attack_run_id",
)

eval_runs_histories_df = eval_runs_histories_df.merge(
    right=eval_runs_configs_df[
        [
            "run_id",
            "attack_run_id",
            "model_to_eval",
            "models_to_attack",
            "attack_dataset",
            "dataset",
            "split",
        ]
    ],
    how="inner",
    left_on="run_id",
    right_on="run_id",
)

# TODO: Get run_configs of each attack run to have those hyperparameters.
# vlm_metadata_df = pd.read_csv(
#     os.path.join("configs", "vlm_metadata.csv"),
# )
unique_and_ordered_eval_model_strs = np.sort(
    eval_runs_histories_df["model_to_eval"].unique()
)

plt.close()
g = sns.relplot(
    data=eval_runs_histories_df[eval_runs_histories_df["split"] == "eval"],
    kind="line",
    x="optimizer_step_counter_epoch",
    y="loss/avg_epoch",
    hue="model_to_eval",
    hue_order=unique_and_ordered_eval_model_strs,
    style="attack_dataset",
    col="models_to_attack",
    row="dataset",
    facet_kws={"margin_titles": True},
)
# plt.show()
g.set_axis_labels(
    "Gradient Step", "Eval Cross Entropy of\n" + r"P(Target $\lvert$ Prompt, Image)"
)
g.fig.suptitle("Attacked Model(s)", y=1.0)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
# Set legend title to "Evaluated Model".
g._legend.set_title("Evaluated Model")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
g.set(ylim=(-0.05, None))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="prismatic_loss_vs_gradient_step_cols=eval_models_rows=attack_models",
)
g.set(
    xscale="log",
    yscale="log",
    ylim=(0.95 * eval_runs_histories_df["loss/avg_epoch"].min(), None),
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="prismatic_loss_log_vs_gradient_step_log_cols=eval_models_rows=attack_models",
)
# plt.show()


for (
    models_to_attack,
    eval_runs_histories_by_attack_df,
) in eval_runs_histories_df.groupby("models_to_attack"):
    plt.close()
    g = sns.relplot(
        data=eval_runs_histories_by_attack_df[
            eval_runs_histories_by_attack_df["split"] == "eval"
        ],
        kind="line",
        x="optimizer_step_counter_epoch",
        y="loss/avg_epoch",
        hue="model_to_eval",
        hue_order=unique_and_ordered_eval_model_strs,
        style="attack_dataset",
        col="models_to_attack",
        row="dataset",
        facet_kws={"margin_titles": True},
    )
    g.set_axis_labels(
        "Gradient Step", "Eval Cross Entropy of\n" + r"P(Target $\lvert$ Prompt, Image)"
    )
    g.fig.suptitle("Attacked Model(s)", y=1.0)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g._legend.set_title("Evaluated Model")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
    g.set(ylim=(-0.05, None))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"prismatic_loss_vs_gradient_step_cols=eval_models_rows=attack_models={models_to_attack}",
    )
    g.set(
        xscale="log",
        yscale="log",
        ylim=(0.95 * eval_runs_histories_df["loss/avg_epoch"].min(), None),
    )
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_title=f"prismatic_loss_log_vs_gradient_step_log_cols=eval_models_rows=attack_models={models_to_attack}",
    )
    # plt.show()


# plt.close()
# g = sns.relplot(
#     data=runs_histories_df,
#     kind="line",
#     x="n_gradient_steps",
#     y="eval/generation_begins_with_target",
#     hue="model_to_eval",
#     hue_order=unique_and_ordered_eval_model_strs,
#     col="models_to_attack",
#     # col_order=unique_and_ordered_attack_models_strs,
#     # row="model_to_eval",
#     # row_order=unique_and_ordered_eval_model_strs,
# )
# g.set_axis_labels("Gradient Step", "Exact Match of Generation \& Target")
# g.fig.suptitle("Attacked Model(s)", y=1.0)
# g.set_titles(col_template="{col_name}", row_template="Eval: {row_name}")
# g.set(ylim=(-0.05, 1.05))
# g._legend.set_title("Evaluated Model")
# sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
# # g.fig.text(
# #     1.01,
# #     0.5,
# #     "Evaluated Model",
# #     rotation=-90,
# #     va="center",
# #     ha="left",
# #     transform=g.fig.transFigure,
# # )
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_title="prismatic_exact_match_vs_gradient_step_cols=eval_models_rows=attack_models",
# )
# # plt.show()


print("Finished notebooks/00_prismatic_results.py!")
