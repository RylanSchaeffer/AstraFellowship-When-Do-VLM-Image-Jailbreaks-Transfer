# Note: Some of this code came from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt4/common/dist_utils.py.
from accelerate import Accelerator
import ast
import getpass
import joblib
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.distributed
import torch.utils.data
from torchvision import transforms
from typing import Any, Dict, List, Tuple
import wandb

from src.attacks.base import JailbreakAttacker
from src.data import VLMEnsembleTextDataset, VLMEnsembleTextDataModule
from src.models.ensemble import VLMEnsemble
from src.image_handling import get_list_image


def calc_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def create_attacker(
    wandb_config: Dict[str, Any],
    vlm_ensemble: VLMEnsemble,
) -> JailbreakAttacker:
    if wandb_config["attack_kwargs"]["attack_name"] == "pgd":
        from src.attacks.pgd import PGDAttacker

        attacker = PGDAttacker(
            vlm_ensemble=vlm_ensemble,
            attack_kwargs=wandb_config["attack_kwargs"],
        )
    elif wandb_config["attack_kwargs"]["attack_name"] == "ssa_common_weakness":
        # from old.how_robust_is_bard.src.attacks.SpectrumSimulationAttack import (
        #     SpectrumSimulationCommonWeaknessAttack,
        # )
        #
        # attacker = SpectrumSimulationCommonWeaknessAttack(
        #     models_to_attack_dict=models_to_attack_dict,
        #     models_to_eval_dict=models_to_eval_dict,
        #     criterion=GPT4AttackCriterion(),
        #     **wandb_config["attack_kwargs"],
        # )
        raise NotImplementedError
    elif wandb_config["attack_kwargs"]["attack_name"] == "eval":
        from src.attacks.base import JailbreakAttacker

        attacker = JailbreakAttacker(
            vlm_ensemble=vlm_ensemble,
            attack_kwargs=wandb_config["attack_kwargs"],
        )
    else:
        raise NotImplementedError(
            f"Invalid attack_name: {wandb_config['attack_kwargs']['attack_name']}"
        )

    return attacker


def create_text_dataloader(
    vlm_ensemble: VLMEnsemble,
    prompt_and_targets_kwargs: Dict[str, Any],
    wandb_config: Dict[str, Any],
    split: str = "train",
    load_prompts_and_targets_kwargs: Dict[str, Any] = {},
) -> Tuple[Dict[str, List[str]], torch.utils.data.DataLoader]:
    prompts_and_targets_dict: Dict[str, List[str]] = load_prompts_and_targets(
        prompts_and_targets_kwargs=prompt_and_targets_kwargs,
        split=split,
        **load_prompts_and_targets_kwargs,
    )

    dataset = VLMEnsembleTextDataset(
        vlm_ensemble=vlm_ensemble,
        prompts_and_targets_dict=prompts_and_targets_dict,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=wandb_config["attack_kwargs"]["batch_size"],
        shuffle=True if split == "train" else False,
        num_workers=wandb_config["data"]["num_workers"],
        drop_last=True if split == "train" else False,
    )

    return prompts_and_targets_dict, dataloader


def create_text_datamodule(
    vlm_ensemble: VLMEnsemble,
    prompt_and_targets_kwargs: Dict[str, Any],
    wandb_config: Dict[str, Any],
    split: str = "train",
    load_prompts_and_targets_kwargs: Dict[str, Any] = {},
    prompts_and_targets_dir: str = "prompts_and_targets",
) -> VLMEnsembleTextDataModule:
    text_datamodule = VLMEnsembleTextDataModule(
        vlm_ensemble=vlm_ensemble,
        prompts_and_targets_dict=prompts_and_targets_dict,
        wandb_config=wandb_config,
    )
    return text_datamodule


def create_initial_image(image_kwargs: Dict[str, Any]) -> torch.Tensor:
    if image_kwargs["image_initialization"] == "NIPS17":
        image = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        # resizer = transforms.Resize((224, 224))
        # images = torch.stack(
        #     [resizer(i).unsqueeze(0).to(torch.float16) for i in images]
        # )
        # # Only use one image for one attack.
        # images: torch.Tensor = images[image_kwargs["datum_index"]].unsqueeze(0)
        raise NotImplementedError
    elif image_kwargs["image_initialization"] == "random":
        image_size = image_kwargs["image_size"]
        image: torch.Tensor = torch.rand((1, 3, image_size, image_size))
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                image_kwargs["image_initialization_str"]
            )
        )
    assert len(image.shape) == 4
    return image


def instantiate_vlm_ensemble(
    model_strs: List[str],
    model_generation_kwargs: Dict[str, Dict[str, Any]],
    accelerator: Accelerator,
) -> VLMEnsemble:
    # TODO: This function is probably overengineered and should be deleted.
    vlm_ensemble = VLMEnsemble(
        model_strs=model_strs,
        model_generation_kwargs=model_generation_kwargs,
        accelerator=accelerator,
    )
    vlm_ensemble = accelerator.prepare([vlm_ensemble])[0]
    return vlm_ensemble


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def load_jailbreak_dicts_list(
    wandb_sweep_id: str,
    data_dir_path: str = "eval_data",
    refresh: bool = False,
) -> List[Dict[str, Any]]:
    os.makedirs(data_dir_path, exist_ok=True)
    runs_jailbreak_dict_list_path = os.path.join(
        data_dir_path,
        f"runs_jailbreak_dict_list_sweep={wandb_sweep_id}.joblib",
    )
    if refresh or not os.path.exists(runs_jailbreak_dict_list_path):
        print("Downloading jailbreak images...")

        api = wandb.Api()
        sweep = api.sweep(f"universal-vlm-jailbreak/{wandb_sweep_id}")
        runs = list(sweep.runs)
        runs_jailbreak_dict_list = []
        for run in runs:
            for file in run.files():
                file_name = str(file.name)
                if not file_name.endswith(".png"):
                    continue
                file_dir_path = os.path.join(
                    data_dir_path, f"sweep={wandb_sweep_id}", run.id
                )
                os.makedirs(file_dir_path, exist_ok=True)
                file.download(root=file_dir_path, replace=True)
                # Example:
                #   'eval_data/sweep=7v3u4uq5/dz2maypg/media/images/jailbreak_image_step=500_0_6bff027c89aa794cfb3b.png'
                # becomes
                #   500
                wandb_logging_step = int(file_name.split("_")[2][5:])
                n_gradient_steps = wandb_logging_step
                file_path = os.path.join(file_dir_path, file_name)
                runs_jailbreak_dict_list.append(
                    {
                        "file_path": file_path,
                        "wandb_run_id": run.id,
                        "wandb_logging_step": wandb_logging_step,
                        "n_gradient_steps": n_gradient_steps,
                        "wandb_run_train_indices": np.array(
                            ast.literal_eval(run.config["train_indices"])
                        ),
                        "attack_models_str": run.config["models_to_attack"],
                    }
                )

                print(
                    "Downloaded jailbreak image for run: ",
                    run.id,
                    " at step: ",
                    n_gradient_steps,
                )

        # Sort runs_jailbreak_dict_list based on wandb_run_id and then n_gradient_steps.
        runs_jailbreak_dict_list = sorted(
            runs_jailbreak_dict_list,
            key=lambda x: (x["wandb_run_id"], x["n_gradient_steps"]),
        )

        joblib.dump(
            value=runs_jailbreak_dict_list,
            filename=runs_jailbreak_dict_list_path,
        )

        print("Saved runs_jailbreak_dict_list to: ", runs_jailbreak_dict_list_path)

    else:
        runs_jailbreak_dict_list = joblib.load(runs_jailbreak_dict_list_path)

        print("Loaded runs_jailbreak_dict_list from: ", runs_jailbreak_dict_list_path)

    return runs_jailbreak_dict_list


def load_prompts_and_targets(
    prompts_and_targets_kwargs: Dict[str, Any],
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> Dict[str, List[str]]:
    prompts_and_targets_str = prompts_and_targets_kwargs[f"dataset_{split}"]

    if prompts_and_targets_str == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", "harmful_behaviors.csv"
        )
    elif prompts_and_targets_str == "rylan_anthropic_hhh":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", "red_team_attempts.csv"
        )
    elif prompts_and_targets_str == "robust_bard":
        raise NotImplementedError
    else:
        raise ValueError(
            "Invalid prompts_and_targets_str: {}".format(prompts_and_targets_str)
        )

    df = pd.read_csv(prompts_and_targets_path)
    prompts, targets = df["prompt"].tolist(), df["target"].tolist()

    if split == "train":
        n_unique_prompts_and_targets = prompts_and_targets_kwargs[
            "n_unique_prompts_and_targets"
        ]

        if n_unique_prompts_and_targets != -1:
            unique_indices = np.random.choice(
                len(prompts), n_unique_prompts_and_targets, replace=False
            )
        else:
            unique_indices = np.arange(len(prompts))
            wandb.config.update({"n_unique_prompts_and_targets": len(prompts)})
    elif split == "eval":
        # TODO: Fix this in the future.
        # assert "train_indices" in kwargs
        # train_indices = kwargs["train_indices"]
        # unique_indices = np.setdiff1d(np.arange(len(prompts)), train_indices)
        unique_indices = np.arange(len(prompts))
    else:
        raise ValueError(f"Invalid split: {split}")

    prompts = [prompts[idx] for idx in unique_indices]
    targets = [targets[idx] for idx in unique_indices]

    assert len(prompts) == len(targets)
    assert len(prompts) > 0

    prompts_and_targets_dict = {
        "prompts": prompts,
        "targets": targets,
        "split": split,
        "indices": unique_indices,
    }
    return prompts_and_targets_dict


def retrieve_wandb_username() -> str:
    # system_username = getpass.getuser()
    # if system_username == "rschaef":
    #     wandb_username = "rylan"
    # else:
    #     raise ValueError(f"Unknown W&B username: {system_username}")
    import wandb

    api = wandb.Api()
    wandb_username = api.viewer.username
    return wandb_username


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = False
        cudnn.deterministic = True
    except ImportError:
        pass
