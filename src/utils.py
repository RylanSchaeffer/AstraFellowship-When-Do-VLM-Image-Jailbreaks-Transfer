# Note: Some of this code came from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt4/common/dist_utils.py.

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
        from old.how_robust_is_bard.src.attacks.SpectrumSimulationAttack import (
            SpectrumSimulationCommonWeaknessAttack,
        )

        attacker = SpectrumSimulationCommonWeaknessAttack(
            models_to_attack_dict=models_to_attack_dict,
            models_to_eval_dict=models_to_eval_dict,
            criterion=GPT4AttackCriterion(),
            **wandb_config["attack_kwargs"],
        )
    else:
        raise ValueError(
            "Invalid attack_name: {}".format(
                wandb_config["attack_kwargs"]["attack_name"]
            )
        )
    return attacker


def create_or_load_images(
    image_initialization_str: str, data_kwargs: Dict[str, Any]
) -> List[torch.Tensor]:
    if image_initialization_str == "NIPS17":
        images_list = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        resizer = transforms.Resize((224, 224))
        images_list = [resizer(i).unsqueeze(0).to(torch.float16) for i in images_list]
        # Only use one image for one attack.
        images_list: List[torch.Tensor] = [images_list[data_kwargs["datum_index"]]]
    elif image_initialization_str == "random":
        image_size = data_kwargs["image_size"]
        images_list: List[torch.Tensor] = [torch.rand((1, 3, image_size, image_size))]
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(image_initialization_str)
        )
    return images_list


def instantiate_models(
    model_strs: List[str],
    model_generation_kwargs: Dict[str, Dict[str, Any]],
    split: str = "train",
) -> Dict[str, torch.nn.Module]:
    models_dict = {}
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu_id = 0
    else:
        gpu_id = None
    for model_str in model_strs:
        # Load BLIP2 models.
        if model_str.startswith("blip2"):
            from old.how_robust_is_bard.src.models.blip2 import Blip2VisionLanguageModel

            if model_str.endswith("flan-t5-xxl"):
                huggingface_name = "Salesforce/blip2-flan-t5-xxl"
            elif model_str.endswith("opt-2.7b"):
                huggingface_name = "Salesforce/blip2-opt-2.7b"
            elif model_str.endswith("opt-6.7b"):
                huggingface_name = "Salesforce/blip2-opt-6.7b"
            else:
                raise ValueError("Invalid model_str: {}".format(model_str))

            model = Blip2VisionLanguageModel(
                huggingface_name=huggingface_name,
                split=split,
            )

        # Load Instruct BLIP models.
        elif model_str.startswith("instructblip"):
            from old.how_robust_is_bard.src.models.instructblip import (
                InstructBlipVisionLanguageModel,
            )

            if model_str.endswith("flan-t5-xxl"):
                huggingface_name = "Salesforce/instructblip-flan-t5-xxl"
            elif model_str.endswith("vicuna-7b"):
                huggingface_name = "Salesforce/instructblip-vicuna-7b"
            elif model_str.endswith("vicuna-13b"):
                huggingface_name = "Salesforce/instructblip-vicuna-13b"
            else:
                raise ValueError("Invalid model_str: {}".format(model_str))

            model = InstructBlipVisionLanguageModel(
                huggingface_name=huggingface_name,
                split=split,
            )

        # # Load MiniGPT4 model.
        # elif model_str.startswith("gpt4"):
        #     from src.models.minigpt4v import get_gpt4_image_model
        #
        #     model = get_gpt4_image_model(targets=targets)

        elif model_str.startswith("llava"):
            from src.models.llava import LlavaVisionLanguageModel

            if model_str.endswith("v1.5-vicuna7b"):
                huggingface_name = "liuhaotian/llava-v1.5-7b"
            elif model_str.endswith("v1.6-hermes-yi-34b"):
                huggingface_name = "liuhaotian/llava-v1.6-34b"
            elif model_str.endswith("v1.6-mistral7b"):
                # Lots of bugs. They needed to be patched here.
                # https://huggingface.co/Trelis/llava-v1.6-mistral-7b-PATCHED.
                huggingface_name = "Trelis/llava-v1.6-mistral-7b-PATCHED"
            elif model_str.endswith("v1.6-vicuna7b"):
                huggingface_name = "liuhaotian/llava-v1.6-vicuna-7b"
            elif model_str.endswith("v1.6-vicuna13b"):
                huggingface_name = "liuhaotian/llava-v1.6-vicuna-13b"
            else:
                raise ValueError("Invalid model_str: {}".format(model_str))

            model = LlavaVisionLanguageModel(
                huggingface_name=huggingface_name,
                split=split,
                generation_kwargs=model_generation_kwargs[model_str],
                gpu_id=gpu_id,
            )

        elif model_str.startswith("prism"):
            from src.models.prismatic import PrismaticVisionLanguageModel

            model = PrismaticVisionLanguageModel(
                model_str=model_str,
                split=split,
                generation_kwargs=model_generation_kwargs[model_str],
            )

        else:
            raise ValueError("Invalid model_str: {}".format(model_str))

        models_dict[model_str] = model
        if gpu_id is not None:
            gpu_id += 1

    return models_dict


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def load_prompts_and_targets(
    prompts_and_targets_str: str,
    prompts_and_targets_dir: str = "prompts_and_targets",
) -> Tuple[List[str], List[str]]:
    if prompts_and_targets_str == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", "harmful_behaviors.csv"
        )
    elif prompts_and_targets_str == "anthropic_hhh":
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
    return df["prompt"].tolist(), df["target"].tolist()


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
