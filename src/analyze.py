import ast
import os
import pandas as pd
import pyarrow
import requests
import time
from typing import Dict, List, Optional, Tuple
import wandb
from tqdm import tqdm


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: str = None,
    filetype: str = "csv",
) -> pd.DataFrame:
    assert filetype in {"csv", "feather", "parquet"}
    runs_configs_df_path = os.path.join(
        data_dir, "sweeps=" + ",".join(sweep_ids) + f"_runs_configs.{filetype}"
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
        runs_configs_df.reset_index(inplace=True, drop=True)

        # Save to disk.
        runs_configs_df.to_csv(
            runs_configs_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_configs_df.to_feather(
                runs_configs_df_path.replace(filetype, "feather")
            )
        except BaseException:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        try:
            runs_configs_without_model_generations_kwargs_df = runs_configs_df.drop(
                columns=["model_generation_kwargs"]
            )
            runs_configs_without_model_generations_kwargs_df.to_parquet(
                runs_configs_df_path.replace(filetype, "parquet"), index=False
            )
        except BaseException:
            # pyarrow.lib.ArrowNotImplementedError: Cannot write struct type 'model_generation_kwargs' with no child field to Parquet. Consider adding a dummy child field.
            pass

        print(f"Regenerated and wrote {runs_configs_df_path} to disk.")
        del runs_configs_df

    print(f"Reading {runs_configs_df_path} from disk.")
    if filetype == "csv":
        runs_configs_df = pd.read_csv(runs_configs_df_path)
    elif filetype == "feather":
        runs_configs_df = pd.read_feather(runs_configs_df_path)
    elif filetype == "parquet":
        runs_configs_df = pd.read_parquet(runs_configs_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
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


def download_wandb_project_runs_configs_by_run_ids(
    wandb_project_path: str,
    data_dir: str,
    run_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: str = None,
    filetype: str = "csv",
) -> pd.DataFrame:
    assert filetype in {"csv", "feather", "parquet"}
    runs_configs_df_path = os.path.join(
        data_dir, "runs=" + ",".join(run_ids) + f"_runs_configs.{filetype}"
    )
    if refresh or not os.path.isfile(runs_configs_df_path):
        # Download sweep results
        api = wandb.Api(timeout=600)

        if wandb_username is None:
            wandb_username = api.viewer.username

        run_results_list = []
        for run_id in tqdm(run_ids):
            run = api.run(f"{wandb_username}/{wandb_project_path}/{run_id}")
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
            run_results_list.append(summary)

        runs_configs_df = pd.DataFrame(run_results_list)
        runs_configs_df.reset_index(inplace=True, drop=True)

        # Save to disk.
        runs_configs_df.to_csv(
            runs_configs_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_configs_df.to_feather(
                runs_configs_df_path.replace(filetype, "feather")
            )
        except BaseException:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        try:
            runs_configs_without_model_generations_kwargs_df = runs_configs_df.drop(
                columns=["model_generation_kwargs"]
            )
            runs_configs_without_model_generations_kwargs_df.to_parquet(
                runs_configs_df_path.replace(filetype, "parquet"), index=False
            )
        except BaseException:
            # pyarrow.lib.ArrowNotImplementedError: Cannot write struct type 'model_generation_kwargs' with no child field to Parquet. Consider adding a dummy child field.
            pass
        except pyarrow.lib.ArrowInvalid:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass

        print(f"Regenerated and wrote {runs_configs_df_path} to disk.")
        del runs_configs_df

    print(f"Reading {runs_configs_df_path} from disk.")
    if filetype == "csv":
        runs_configs_df = pd.read_csv(runs_configs_df_path)
    elif filetype == "feather":
        runs_configs_df = pd.read_feather(runs_configs_df_path)
    elif filetype == "parquet":
        runs_configs_df = pd.read_parquet(runs_configs_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
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

    assert filetype in {"csv", "feather", "parquet"}

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

                history = None
                for num_attempts in range(5):
                    try:
                        history = run.history(
                            samples=wandb_run_history_samples,
                        )
                        break
                    except (requests.exceptions.HTTPError, wandb.errors.CommError):
                        print("Retrying...")
                        time.sleep(3)

                # Skip this run.
                if history is None or history.empty:
                    continue

                # Drop generation columns because these are meaty.
                generation_columns = [
                    col for col in history.columns if "generation" in col
                ]
                history.drop(columns=generation_columns, inplace=True)
                history["run_id"] = run.id
                runs_histories_list.append(history)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)
        runs_histories_df.reset_index(inplace=True, drop=True)

        # Save all three because otherwise this is a pain in the ass.
        runs_histories_df.to_csv(
            runs_histories_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_histories_df.to_feather(
                runs_histories_df_path.replace(filetype, "feather")
            )
        except BaseException:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        try:
            runs_histories_df.to_parquet(
                runs_histories_df_path.replace(filetype, "parquet"), index=False
            )
        except pyarrow.lib.ArrowInvalid:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        print(f"Wrote {runs_histories_df_path} to disk")
        del runs_histories_df

    print(f"Loading {runs_histories_df_path} from disk.")
    if filetype == "csv":
        runs_histories_df = pd.read_csv(runs_histories_df_path)
    elif filetype == "feather":
        runs_histories_df = pd.read_feather(runs_histories_df_path)
    elif filetype == "parquet":
        runs_histories_df = pd.read_parquet(runs_histories_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_histories_df_path} from disk.")

    return runs_histories_df


# parse data config blob into cols
def extract_key_value_from_df_col(
    df: pd.DataFrame,
    col_name: str,
    key_in_dict: Optional[str] = None,
    new_col_name: Optional[str] = None,
):
    if new_col_name is None:
        new_col_name = key_in_dict

    df[new_col_name] = df[col_name].apply(
        lambda x: x[key_in_dict]
        if isinstance(x, dict)
        else ast.literal_eval(x)[key_in_dict]
    )
    return df


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
