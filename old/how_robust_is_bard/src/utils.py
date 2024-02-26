import getpass
import numpy as np
import os
import pandas as pd
import random
import torch
from typing import Any, Dict, List, Tuple

from old.how_robust_is_bard.src.attacks.base import AdversarialImageAttacker

from old.how_robust_is_bard.src.models.blip2 import Blip2VisionLanguageModel
from old.how_robust_is_bard.src.models.instructblip import (
    InstructBlipVisionLanguageModel,
)


class GPT4AttackCriterion:
    def __init__(self):
        self.count = 0

    def __call__(self, loss, *args):
        self.count += 1
        if self.count % 120 == 0:
            print(loss)
        return -loss


def create_attacker(
    wandb_config: Dict[str, Any],
    models_to_attack_dict: Dict[str, torch.nn.Module],
    models_to_eval_dict: Dict[str, torch.nn.Module],
) -> AdversarialImageAttacker:
    if wandb_config["attack_kwargs"]["attack_name"] == "sgd":
        from old.how_robust_is_bard.src.attacks.sgd import SGDAttack

        attacker = SGDAttack(
            models_to_attack_dict=models_to_attack_dict,
            models_to_eval_dict=models_to_eval_dict,
            **wandb_config["attack_kwargs"],
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


def instantiate_models(
    model_strs: List[str], prompts: List[str], targets: List[str], split: str = "train"
) -> Dict[str, torch.nn.Module]:
    models_dict = {}
    for model_str in model_strs:
        # Load BLIP2 models.
        if model_str.startswith("blip2"):
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

        # Load MiniGPT4 model.
        elif model_str.startswith("gpt4"):
            from old.how_robust_is_bard.src.models.minigpt4v import get_gpt4_image_model

            model = get_gpt4_image_model(targets=targets)
        else:
            raise ValueError("Invalid model_str: {}".format(model_str))

        models_dict[model_str] = model

    return models_dict


def load_prompts_and_targets(
    prompts_and_targets_str: str,
    prompts_and_targets_dir: str = "prompts_and_targets",
) -> Tuple[List[str], List[str]]:
    if prompts_and_targets_str == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", "harmful_behaviors.csv"
        )
    elif prompts_and_targets_str == "robust_bard":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "coco", "default.csv"
        )

    df = pd.read_csv(prompts_and_targets_path)
    return df["prompt"].tolist(), df["target"].tolist()


def retrieve_wandb_username() -> str:
    system_username = getpass.getuser()
    if system_username == "rschaef":
        wandb_username = "rylan"
    else:
        raise ValueError(f"Unknown system username: {system_username}")
    return wandb_username


def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
