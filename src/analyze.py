from collections import OrderedDict
import joblib
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple
import wandb
from tqdm import tqdm


metrics_to_nice_strings_dict = OrderedDict(
    {
        "loss/avg_epoch": "Cross Entropy of\n" + r"P(Target $\lvert$ Prompt, Image)",
        "loss/score_model=harmbench": "HarmBench Score",
        "loss/score_model=llamaguard2": "LlamaGuard2 Score",
    }
)

metrics_to_bounds_dict = OrderedDict(
    {
        "loss/avg_epoch": (0.0, None),
        "loss/score_model=harmbench": (0.0, 1.0),
        "loss/score_model=llamaguard2": (0.0, 1.0),
    }
)


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: str = None,
) -> pd.DataFrame:
    runs_configs_df_path = os.path.join(
        data_dir, "sweeps=" + ",".join(sweep_ids) + "_runs_configs.csv"
    )
    if refresh or not os.path.isfile(runs_configs_df_path):
        # Download sweep results
        api = wandb.Api(timeout=600)

        if wandb_username is None:
            wandb_username = api.viewer.username

        sweep_results_list = []
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")
            for run in tqdm(sweep.runs):
                # .summary contains the output keys/values for metrics like accuracy.
                #  We call ._json_dict to omit large files
                summary = run.summary._json_dict

                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                summary.update(
                    {k: v for k, v in run.config.items() if not k.startswith("_")}
                )

                summary.update(
                    {
                        "State": run.state,
                        "Sweep": run.sweep.id if run.sweep is not None else None,
                        "run_id": run.id,
                    }
                )
                # .name is the human-readable name of the run.
                summary.update({"run_name": run.name})
                sweep_results_list.append(summary)

            runs_configs_df = pd.DataFrame(sweep_results_list)

            # Save to disk.
            runs_configs_df.to_csv(runs_configs_df_path, index=False)
            print(f"Wrote {runs_configs_df_path} to disk.")
    else:
        runs_configs_df = pd.read_csv(runs_configs_df_path)
        print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {100.0 * finished_runs.mean()} ({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_histories(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    wandb_run_history_samples: int = 10000,
    refresh: bool = False,
    keys: List[str] = None,
    wandb_username: str = None,
    filetype: str = "csv",
) -> pd.DataFrame:
    if keys is None:
        keys = ["losses_train/loss_epoch", "losses_val/loss"]

    assert filetype in {"csv", "parquet", "feather"}

    runs_histories_df_path = os.path.join(
        data_dir, "sweeps=" + ",".join(sweep_ids) + f"_runs_histories.{filetype}"
    )
    if refresh or not os.path.isfile(runs_histories_df_path):
        # Download sweep results
        api = wandb.Api(timeout=6000)

        if wandb_username is None:
            wandb_username = api.viewer.username

        runs_histories_list = []
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")
            for run in tqdm(sweep.runs):
                # https://community.wandb.ai/t/run-history-returns-different-values-on-almost-each-call/2431/4
                history = run.history(
                    samples=wandb_run_history_samples,
                )
                if history.empty:
                    continue
                history["run_id"] = run.id
                runs_histories_list.append(history)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)

        runs_histories_df.reset_index(inplace=True, drop=True)

        if filetype == "csv":
            runs_histories_df.to_csv(runs_histories_df_path, index=False)
        elif filetype == "feather":
            runs_histories_df.reset_index(inplace=True)
            runs_histories_df.to_feather(runs_histories_df_path)
        elif filetype == "parquet":
            runs_histories_df.to_parquet(runs_histories_df_path, index=False)
        else:
            raise ValueError(f"Invalid filetype: {filetype}")
        print(f"Wrote {runs_histories_df_path} to disk")
    else:
        if filetype == "csv":
            runs_histories_df = pd.read_csv(runs_histories_df_path)
        elif filetype == "feather":
            runs_histories_df = pd.read_feather(runs_histories_df_path)
        elif filetype == "parquet":
            runs_histories_df = pd.read_parquet(runs_histories_df_path)
            runs_histories_df = pd.read_parquet(runs_histories_df_path)
        else:
            raise ValueError(f"Invalid filetype: {filetype}")
        print(f"Loaded {runs_histories_df_path} from disk.")

    return runs_histories_df


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir
