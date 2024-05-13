import pandas as pd
import pydantic
import os
from slist import Slist
from typing import Sequence
import wandb

from src.openai_utils.client import encode_image
from src.prompts_and_targets import PromptAndTarget


class JailbreakData(pydantic.BaseModel):
    file_path: str
    wandb_run_id: str
    optimizer_step_counter: int
    attack_models_str: str
    image_base_64: str


def load_jailbreak_list_v2(
    wandb_run_id: str | None = None,
    wandb_sweep_id: str | None = None,
    data_dir_path: str = "eval_data",
) -> Slist[JailbreakData]:
    os.makedirs(data_dir_path, exist_ok=True)

    print("Downloading jailbreak images...")

    api = wandb.Api()
    if wandb_sweep_id is None and wandb_run_id is not None:
        run = api.run(f"universal-vlm-jailbreak/{wandb_run_id}")
        runs = [run]
    elif wandb_sweep_id is not None and wandb_run_id is None:
        sweep = api.sweep(f"universal-vlm-jailbreak/{wandb_run_id}")
        runs = list(sweep.runs)
    else:
        raise ValueError(
            "Invalid wandb_sweep_id and wandb_run_id: "
            f"{wandb_sweep_id}, {wandb_run_id}"
        )
    runs_jailbreak_dict_list = []
    for run in runs:
        for file in run.files():
            file_name = str(file.name)
            if not file_name.endswith(".png"):
                continue
            file_dir_path = os.path.join(data_dir_path, run.id)
            os.makedirs(file_dir_path, exist_ok=True)
            file.download(root=file_dir_path, replace=True)
            # Example:
            #   'eval_data/sweep=7v3u4uq5/dz2maypg/media/images/jailbreak_image_step=500_0_6bff027c89aa794cfb3b.png'
            # becomes
            #   500
            optimizer_step_counter = int(file_name.split("_")[2][5:])
            file_path = os.path.join(file_dir_path, file_name)
            runs_jailbreak_dict_list.append(
                JailbreakData(
                    file_path=file_path,
                    wandb_run_id=run.id,
                    optimizer_step_counter=optimizer_step_counter,
                    attack_models_str=run.config["models_to_attack"],
                    image_base_64=encode_image(file_path),
                )
            )

            print(
                "Downloaded jailbreak image for run: ",
                run.id,
                " at optimizer step: ",
                optimizer_step_counter,
            )

        # Sort runs_jailbreak_dict_list based on wandb_run_id and then n_gradient_steps.
        runs_jailbreak_dict_list = sorted(
            runs_jailbreak_dict_list,
            key=lambda x: (x.wandb_run_id, x.optimizer_step_counter),
        )

    return Slist(runs_jailbreak_dict_list)


def load_prompts_and_targets_v2(
    dataset: str,
    split: str = "train",
    prompts_and_targets_dir: str = "prompts_and_targets",
) -> Sequence[PromptAndTarget]:

    if dataset == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", f"{split}.csv"
        )
        os.path.join(prompts_and_targets_dir, "advbench", "tokenized")
    elif dataset == "rylan_anthropic_hhh":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", f"{split}.csv"
        )
        os.path.join(prompts_and_targets_dir, "anthropic_hhh", "tokenized")
    elif dataset == "mmlu":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "mmlu", f"{split}.csv"
        )
    elif dataset == "mmlu_d":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "mmlu_d", f"{split}.csv"
        )
    elif dataset == "survival":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "survival", f"{split}.csv"
        )
    elif dataset == "wealth":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "wealth", f"{split}.csv"
        )

    elif dataset == "power_seeking":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "power_seeking", f"{split}.csv"
        )

    elif dataset == "all_model_generated_evals":
        return (
            Slist(["survival", "wealth", "power_seeking"])
            .map(
                lambda dataset: load_prompts_and_targets_v2(
                    dataset=dataset,
                    split=split,
                    prompts_and_targets_dir=prompts_and_targets_dir,
                )
            )
            .flatten_list()
            .shuffle("42")
        )

    else:
        raise ValueError("Invalid prompts_and_targets_str: {}".format(dataset))

    df = pd.read_csv(prompts_and_targets_path)
    prompts, targets = df["prompt"].tolist(), df["target"].tolist()
    assert len(prompts) == len(targets)
    assert len(prompts) > 0
    return [PromptAndTarget(prompt, target) for prompt, target in zip(prompts, targets)]
