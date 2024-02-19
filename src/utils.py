import getpass
import numpy as np
import os
import pandas as pd
import random
import torch
from typing import Any, Dict, List, Tuple

from src.attacks import AdversarialInputAttacker, SSA_CommonWeakness
from src.models.blip2 import Blip2VisionModel
from src.models.instruct_blip import InstructBlipVisionModel


class GPT4AttackCriterion:
    def __init__(self):
        self.count = 0

    def __call__(self, loss, *args):
        self.count += 1
        if self.count % 120 == 0:
            print(loss)
        return -loss


def create_attacker(
    wandb_config: Dict[str, Any], models_list: List[torch.nn.Module]
) -> AdversarialInputAttacker:
    if wandb_config["attack_kwargs"]["attack_name"] == "ssa_common_weakness":
        attacker = SSA_CommonWeakness(
            models_list=models_list,
            epsilon=wandb_config["attack_kwargs"]["epsilon"],
            step_size=wandb_config["attack_kwargs"]["step_size"],
            total_step=wandb_config["attack_kwargs"]["total_steps"],
            criterion=GPT4AttackCriterion(),
        )
    else:
        raise ValueError(
            "Invalid attack_name: {}".format(
                wandb_config["attack_kwargs"]["attack_name"]
            )
        )
    return attacker


def instantiate_models(
    wandb_config: Dict[str, Any], prompts: List[str], targets: List[str]
) -> List[torch.nn.Module]:
    models_list = []
    for model_str in wandb_config["models_to_attack"]:
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

            model = Blip2VisionModel(
                prompts=prompts,
                targets_=targets,
                huggingface_name=huggingface_name,
            )

        # Load Instruct BLIP models.
        elif model_str.startswith("instructblip"):
            if model_str.endswith("flan-t5-xxl"):
                huggingface_name = "Salesforce/instructblip-flan-t5-xxl"
            elif model_str.endswith("opt-6.7b"):
                huggingface_name = "Salesforce/instructblip-vicuna-7b"
            elif model_str.endswith("vicuna-13b"):
                huggingface_name = "Salesforce/instructblip-vicuna-13b"
            else:
                raise ValueError("Invalid model_str: {}".format(model_str))

            model = InstructBlipVisionModel(
                prompts=prompts,
                targets=targets,
                huggingface_name=huggingface_name,
            )

        elif model_str == "gpt4":
            from src.models.minigpt4 import get_gpt4_image_model

            model = get_gpt4_image_model(targets=targets)
        else:
            raise ValueError("Invalid model_str: {}".format(model_str))

        models_list.append(model)
    return models_list


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
