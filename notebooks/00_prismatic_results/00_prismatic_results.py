import ast
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot


# refresh = True
refresh = False

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "tc10qy1l",  # Prismatic with N-Choose-1 Jailbreaks
    # "cewqh39e",  # Prismatic with N-Choose-2 Jailbreaks
]

runs_configs_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    finished_only=False,
)

vlm_metadata_df = pd.read_csv(
    os.path.join("configs", "vlm_metadata.csv"),
)
unique_and_ordered_eval_model_strs = vlm_metadata_df["Name"]
# Keep only if the eval_model_str is in the unique_and_ordered_eval_model_strs.
unique_and_ordered_eval_model_strs = [
    eval_model_str
    for eval_model_str in unique_and_ordered_eval_model_strs
    if eval_model_str in runs_configs_df["eval_model_str"].unique()
]

runs_histories_df = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="universal-vlm-jailbreak-eval",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_run_history_samples=10000,
)

plt.close()
g = sns.relplot(
    data=runs_histories_df,
    kind="line",
    x="loss/optimizer_step_counter_step",
    y="loss/avg_step",
    hue="eval_model_str",
    hue_order=unique_and_ordered_eval_model_strs,
    col="attack_models_str",
    facet_kws={"margin_titles": True},
)
plt.show()
g.set_axis_labels("Gradient Step", r"Cross Entropy of P(Target $\lvert$ Prompt, Image)")
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
    ylim=(0.95 * runs_histories_df["loss/avg_step"].min(), None),
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="prismatic_loss_log_vs_gradient_step_log_cols=eval_models_rows=attack_models",
)
plt.show()


plt.close()
g = sns.relplot(
    data=runs_histories_df,
    kind="line",
    x="n_gradient_steps",
    y="eval/generation_begins_with_target",
    hue="eval_model_str",
    hue_order=unique_and_ordered_eval_model_strs,
    col="attack_models_str",
    # col_order=unique_and_ordered_attack_models_strs,
    # row="eval_model_str",
    # row_order=unique_and_ordered_eval_model_strs,
)
g.set_axis_labels("Gradient Step", "Exact Match of Generation \& Target")
g.fig.suptitle("Attacked Model(s)", y=1.0)
g.set_titles(col_template="{col_name}", row_template="Eval: {row_name}")
g.set(ylim=(-0.05, 1.05))
g._legend.set_title("Evaluated Model")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
# g.fig.text(
#     1.01,
#     0.5,
#     "Evaluated Model",
#     rotation=-90,
#     va="center",
#     ha="left",
#     transform=g.fig.transFigure,
# )
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="prismatic_exact_match_vs_gradient_step_cols=eval_models_rows=attack_models",
)
# plt.show()


print("Finished notebooks/00_prismatic_results.py!")
