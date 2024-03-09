# Note: Some of this code came from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt4/common/dist_utils.py.
from accelerate import Accelerator
import getpass
import numpy as np
import os
import pandas as pd
import random
import torch
from torchvision import transforms
import torch.distributed
from typing import Any, Dict, List, Tuple

from src.attacks.base import AdversarialAttacker
from src.models.ensemble import VLMEnsemble
from src.image_handling import get_list_image


def calc_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def create_attacker(
    wandb_config: Dict[str, Any],
    models_to_attack_dict: Dict[str, torch.nn.Module],
    models_to_eval_dict: Dict[str, torch.nn.Module],
) -> AdversarialAttacker:
    if wandb_config["attack_kwargs"]["attack_name"] == "pgd":
        from src.attacks.pgd import PGDAttacker

        attacker = PGDAttacker(
            models_to_attack_dict=models_to_attack_dict,
            models_to_eval_dict=models_to_eval_dict,
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
    else:
        raise ValueError(
            "Invalid attack_name: {}".format(
                wandb_config["attack_kwargs"]["attack_name"]
            )
        )
    return attacker


def create_or_load_images(image_kwargs: Dict[str, Any]) -> List[torch.Tensor]:
    if image_kwargs["image_initialization_str"] == "NIPS17":
        images_list = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        resizer = transforms.Resize((224, 224))
        images_list = [resizer(i).unsqueeze(0).to(torch.float16) for i in images_list]
        # Only use one image for one attack.
        images_list: List[torch.Tensor] = [images_list[image_kwargs["datum_index"]]]
    elif image_kwargs["image_initialization_str"] == "random":
        image_size = image_kwargs["image_size"]
        images_list: List[torch.Tensor] = [torch.rand((1, 3, image_size, image_size))]
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                image_kwargs["image_initialization_str"]
            )
        )
    return images_list


def instantiate_vlm_ensemble(
    models_to_attack: List[str],
    models_to_eval: List[str],
    model_generation_kwargs: Dict[str, Dict[str, Any]],
    accelerator: Accelerator,
) -> VLMEnsemble:
    vlm_ensemble = VLMEnsemble(
        models_to_attack=models_to_attack,
        models_to_eval=models_to_eval,
        model_generation_kwargs=model_generation_kwargs,
        accelerator=accelerator,
    )

    return vlm_ensemble


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def load_prompts_and_targets(
    prompts_and_targets_kwargs: Dict[str, Any],
    prompts_and_targets_dir: str = "prompts_and_targets",
) -> Tuple[List[str], List[str]]:
    prompts_and_targets_str = prompts_and_targets_kwargs["prompts_and_targets"]
    n_unique_prompts_and_targets = prompts_and_targets_kwargs[
        "n_unique_prompts_and_targets"
    ]

    if prompts_and_targets_str == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", "harmful_behaviors.csv"
        )
    elif prompts_and_targets_str == "rylan_anthropic_hhh":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", "red_team_attempts.csv"
        )
    elif prompts_and_targets_str == "robust_bard":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "coco", "default.csv"
        )
        raise NotImplementedError
    else:
        raise ValueError(
            "Invalid prompts_and_targets_str: {}".format(prompts_and_targets_str)
        )

    df = pd.read_csv(prompts_and_targets_path)
    prompts, targets = df["prompt"].tolist(), df["target"].tolist()

    if n_unique_prompts_and_targets != -1:
        unique_indices = np.random.choice(
            len(prompts), n_unique_prompts_and_targets, replace=False
        )
        prompts = [prompts[i] for i in unique_indices]
        targets = [targets[i] for i in unique_indices]

    return prompts, targets


def retrieve_wandb_username() -> str:
    system_username = getpass.getuser()
    if system_username == "rschaef":
        wandb_username = "rylan"
    else:
        raise ValueError(f"Unknown W&B username: {system_username}")
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
