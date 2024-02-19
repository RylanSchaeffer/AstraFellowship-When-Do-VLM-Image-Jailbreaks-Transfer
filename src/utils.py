import getpass
import numpy as np
import os
import pandas as pd
import random
import torch
from typing import List, Tuple


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
